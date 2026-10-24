"""Self-play PPO, hyperparameters from Wang, "Winning at Pokémon Random Battles Using
Reinforcement Learning" (MIT MEng thesis, 2024), Table 3.1.

    scripts/start_servers.sh 8          # Showdown servers on ports 8000-8007
    python -m porygon.train --n-battles 48 --ports 8000-8007 --total-steps 150_000_000
"""

import argparse
import time
from functools import partial
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from porygon.async_env import AsyncBattles, AsyncPPO, pool_battle, selfplay_battle
from porygon.env import ConcatVecEnv, make_selfplay_vec_env
from porygon.evaluate import Evaluator
from porygon.model import MaskedPolicy, load_policy
from porygon.pool import make_pool_vec_env, save_snapshot


def thesis_lr(progress_remaining: float) -> float:
    """l(x) = 10^-4.23 / (8x + 1)^1.5, x = fraction of training done. Annealing was the
    single largest win-rate factor in the thesis (55% -> 80% vs SimpleHeuristicsPlayer)."""
    x = 1.0 - progress_remaining
    return 10**-4.23 / (8 * x + 1) ** 1.5


class WallClockLimit(BaseCallback):
    """Stops training after `hours`; learn() then returns and final_model is saved."""

    def __init__(self, hours: float):
        super().__init__()
        self.deadline = time.time() + hours * 3600

    def _on_step(self) -> bool:
        return time.time() < self.deadline


class SnapshotCallback(BaseCallback):
    """Adds the current policy to the opponent pool every `every` steps."""

    def __init__(self, pool_dir: Path, every: int):
        super().__init__()
        self.pool_dir, self.every = pool_dir, every

    def _on_training_start(self) -> None:
        self.next = self.num_timesteps + self.every

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps >= self.next:
            self.next += self.every
            save_snapshot(self.model.policy, self.pool_dir, f"{self.num_timesteps:011d}")


class EvalCallback(BaseCallback):
    """Every `every` steps: win rate vs baselines (and the anchor checkpoint, if given) ->
    tensorboard; keep the best checkpoint, judged by the anchor score when there is one."""

    def __init__(self, battle_format: str, port: int, every: int, n_battles: int,
                 opponents: tuple[str, ...], save_dir: Path, anchor: Path | None = None):
        super().__init__()
        self.battle_format, self.port, self.every, self.n_battles = battle_format, port, every, n_battles
        self.opponent_names, self.save_dir, self.anchor = opponents, save_dir, anchor
        self.next_eval, self.best = every, -1.0
        self.evaluator: Evaluator | None = None

    def _on_training_start(self) -> None:
        self.next_eval = self.num_timesteps + self.every  # don't eval every rollout after --resume

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps < self.next_eval:
            return
        self.next_eval += self.every
        if self.evaluator is None:
            anchor = load_policy(self.anchor, str(self.model.device)) if self.anchor else None
            self.evaluator = Evaluator(self.model.policy, self.battle_format, self.port,
                                       self.opponent_names, anchor=anchor)
        self.model.policy.set_training_mode(False)
        start = time.time()
        results = self.evaluator.run(self.n_battles)
        for name, rate in results.items():
            self.logger.record(f"eval/win_rate_{name}", rate)
        self.logger.record("eval/seconds", time.time() - start)
        score = results["anchor" if self.anchor else self.opponent_names[0]]
        print(f"[eval @ {self.num_timesteps:,}] " + "  ".join(f"{k}={v:.1%}" for k, v in results.items()))
        if score > self.best:
            self.best = score
            self.model.save(self.save_dir / "best_model")


def parse_ports(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = map(int, spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(p) for p in spec.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--n-battles", type=int, default=39, help="concurrent self-play battles")
    p.add_argument("--pool-battles", type=int, default=0,
                   help="extra concurrent battles vs the opponent pool (past snapshots + baseline bots)")
    p.add_argument("--pool-dir", type=Path, help="opponent pool directory (default: RUN_DIR/pool)")
    p.add_argument("--p-baseline", type=float, default=0.2, help="share of pool battles vs baseline bots")
    p.add_argument("--snapshot-every", type=int, default=2_500_000, help="add policy to the pool every N steps")
    p.add_argument("--n-cpus", type=int, default=None, help="env worker processes (default: one per battle)")
    p.add_argument("--ports", default="8000", help="Showdown server ports, e.g. 8000-8007")
    p.add_argument("--async-envs", action="store_true",
                   help="battles step independently (porygon.async_env) instead of in lockstep")
    p.add_argument("--overlap-update", action="store_true",
                   help="with --async-envs: keep battles running during the PPO update (data <= 1 update old)")
    p.add_argument("--total-steps", type=int, default=150_000_000)
    p.add_argument("--rollout", type=int, default=78 * 512, help="transitions per PPO update")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-epochs", type=int, default=7)
    p.add_argument("--gamma", type=float, default=0.9999)
    p.add_argument("--gae-lambda", type=float, default=0.754)
    p.add_argument("--clip-range", type=float, default=0.0829)
    p.add_argument("--clip-range-vf", type=float, default=0.0184)
    # Thesis: 0.0588, tuned for its 495-way ID action space; with <=9 legal slot actions it kept
    # our first run near-uniform (entropy 1.72 -> 1.56 nats over 7M steps).
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.4375)
    p.add_argument("--max-grad-norm", type=float, default=0.543)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    # Change in (fainted, hp, status) balance per step. It is a difference of state values, i.e.
    # potential-based shaping, so with gamma ~ 1 it speeds up credit assignment without changing
    # the optimal policy. Thesis: none (pass 0 0 0).
    p.add_argument("--shaping", type=float, nargs=3, metavar=("FAINTED", "HP", "STATUS"),
                   default=(0.05, 0.05, 0.0))
    p.add_argument("--eval-every", type=int, default=2_500_000)
    p.add_argument("--eval-battles", type=int, default=200)
    # The baseline bots saturate around 85%; a fixed earlier checkpoint keeps showing progress.
    p.add_argument("--anchor", type=Path, help="checkpoint (.zip or pool .pt) to also evaluate against")
    p.add_argument("--checkpoint-every", type=int, default=5_000_000)
    p.add_argument("--run-dir", type=Path, default=Path("runs") / time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--resume", type=Path, help="checkpoint .zip to continue from")
    p.add_argument("--max-hours", type=float, help="stop (and save) after this much wall-clock time")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")  # TF32 on Ampere+: faster PPO updates
    ports = parse_ports(args.ports)
    shaping = dict(zip(("fainted_value", "hp_value", "status_value"), args.shaping))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = args.pool_dir or args.run_dir / "pool"
    if args.pool_battles:
        pool_dir.mkdir(parents=True, exist_ok=True)
    if args.async_envs:  # same battles and ports as the lockstep envs below, one worker process each
        env = AsyncBattles(
            [(partial(selfplay_battle, ports[i % len(ports)], args.format, shaping), "fork")
             for i in range(args.n_battles)]
            + [(partial(pool_battle, ports[::-1][i % len(ports)], args.format, shaping, pool_dir, args.p_baseline),
                "spawn") for i in range(args.pool_battles)])
        algo = AsyncPPO
    else:
        env = make_selfplay_vec_env(args.n_battles, ports, args.format, shaping, args.n_cpus)
        if args.pool_battles:
            env = ConcatVecEnv([env, make_pool_vec_env(args.pool_battles, ports[::-1], args.format, pool_dir,
                                                       shaping, args.p_baseline)])
        env = VecMonitor(env)
        algo = PPO

    if args.resume:
        model = algo.load(args.resume, env=env, device=args.device)
        model.learning_rate, model.lr_schedule = thesis_lr, thesis_lr  # schedules aren't pickled reliably
        model.tensorboard_log = str(args.run_dir / "tb")  # not the folder of the run the checkpoint came from
        n_steps = max(1, args.rollout // env.num_envs)  # keep ~args.rollout transitions per update
        if n_steps != model.n_steps:
            model.n_steps = n_steps
            model.rollout_buffer = model.rollout_buffer_class(
                n_steps, model.observation_space, model.action_space, device=model.device, gamma=model.gamma,
                gae_lambda=model.gae_lambda, n_envs=env.num_envs, **model.rollout_buffer_kwargs)
    else:
        model = algo(
            MaskedPolicy, env,
            learning_rate=thesis_lr,
            n_steps=max(1, args.rollout // env.num_envs),
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=dict(features_extractor_kwargs=dict(d_model=args.d_model, n_layers=args.n_layers)),
            tensorboard_log=str(args.run_dir / "tb"),
            device=args.device,
            verbose=1,
        )
    if args.overlap_update:
        assert args.async_envs, "--overlap-update needs --async-envs"
        model.overlap_update = True
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"{'async' if args.async_envs else 'lockstep'}{' + overlapped update' if args.overlap_update else ''}: {args.n_battles} self-play + {args.pool_battles} pool battles "
          f"on ports {ports}, n_steps={model.n_steps} x {env.num_envs} envs, params={n_params:,}, device={model.device}")

    callbacks = CallbackList([
        CheckpointCallback(max(1, args.checkpoint_every // env.num_envs), str(args.run_dir / "checkpoints"),
                           name_prefix="ppo"),
        EvalCallback(args.format, ports[0], args.eval_every, args.eval_battles, ("heuristic", "max_power"),
                     args.run_dir, args.anchor),
    ] + ([WallClockLimit(args.max_hours)] if args.max_hours else [])
      + ([SnapshotCallback(pool_dir, args.snapshot_every)] if args.pool_battles else []))
    try:
        model.learn(
            total_timesteps=args.total_steps - model.num_timesteps,
            callback=callbacks,
            reset_num_timesteps=not args.resume,
            tb_log_name="ppo",
        )
    finally:
        model.save(args.run_dir / "final_model")
        env.close()


if __name__ == "__main__":
    main()
