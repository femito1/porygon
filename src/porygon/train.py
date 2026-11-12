"""Self-play PPO, hyperparameters from Wang, "Winning at Pokémon Random Battles Using
Reinforcement Learning" (MIT MEng thesis, 2024), Table 3.1.

    scripts/start_servers.sh 8          # Showdown servers on ports 8000-8007
    python -m porygon.train --n-battles 48 --ports 8000-8007 --total-steps 150_000_000
"""

import argparse
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from porygon.async_env import AsyncBattles, AsyncPPO, pool_battle, selfplay_battle
from porygon.env import SHAPING, ConcatVecEnv, make_selfplay_vec_env
from porygon.evaluate import Evaluator
from porygon.model import MaskedPolicy, load_policy, set_value_trunk_grad
from porygon.pool import make_pool_vec_env, save_snapshot


EMBEDDINGS = ("species", "items", "abilities", "moves")


def boost_embedding_steps(policy, mult: float) -> None:
    """Multiply every Adam update of the token embeddings by `mult` (and nothing else: no change to gradients,
    clipping or other layers, no extra param group, so checkpoints load as before).

    The embeddings start at N(0, 1), the linear layers at ~0.03, and Adam moves each weight by ~lr per step, so in
    500M steps the embeddings moved ~0.4%; `mult` gives them a learning rate that is not 30x too small relative to
    their scale. Measured: x30 / x300 for 40M steps from 600M did not make the network stronger (50.6% / 48.5%
    vs its start, unboosted 50.0%, 4000 games each)."""
    if mult == 1.0:
        return
    fe = policy.features_extractor
    emb = [getattr(fe, n).weight for n in EMBEDDINGS]
    before: list[torch.Tensor] = []

    def pre(opt, args, kwargs):
        before[:] = [p.detach().clone() for p in emb]

    def post(opt, args, kwargs):
        with torch.no_grad():
            for p, old in zip(emb, before):
                p.add_((p - old) * (mult - 1.0))

    policy.optimizer.register_step_pre_hook(pre)
    policy.optimizer.register_step_post_hook(post)


def thesis_lr(progress_remaining: float) -> float:
    """l(x) = 10^-4.23 / (8x + 1)^1.5, x = fraction of training done. Annealing was the
    single largest win-rate factor in the thesis (55% -> 80% vs SimpleHeuristicsPlayer)."""
    x = 1.0 - progress_remaining
    return 10**-4.23 / (8 * x + 1) ** 1.5


class WarmupConstantLR:
    """Warmup-stable(-decay) learning rate on absolute env steps: linear warmup from `start` over `warmup` steps
    to `peak`, constant after that, and (optional) cosine decay to `floor` x peak from `decay_start` to `total`.
    The thesis schedule spends most of a run far below its peak (its mean is 1/6 of the peak and it ends at
    1/27); here the rate stays at the peak until a deliberate cooldown. Reads `model.num_timesteps`, so set
    `.model` once the PPO object exists (before that it returns the peak, which SB3 only uses to build Adam)."""

    def __init__(self, peak: float, warmup: int, start: int = 0, decay_start: int | None = None,
                 total: int | None = None, floor: float = 0.1):
        self.peak, self.warmup, self.start = peak, warmup, start
        self.decay_start, self.total, self.floor = decay_start, total, floor
        self.model = None

    def __getstate__(self) -> dict:  # SB3 pickles the schedule into checkpoints: never the model with it
        return {k: v for k, v in self.__dict__.items() if k != "model"} | {"model": None}

    def __call__(self, progress_remaining: float) -> float:
        if self.model is None:
            return self.peak
        t = self.model.num_timesteps
        lr = self.peak * min(1.0, max(t - self.start, 0) / max(self.warmup, 1))
        if self.decay_start is not None and self.total and t > self.decay_start:
            x = min(1.0, (t - self.decay_start) / max(self.total - self.decay_start, 1))
            lr *= self.floor + (1 - self.floor) * 0.5 * (1 + math.cos(math.pi * x))
        return lr


class GradientDiagnostics(BaseCallback):
    """After each rollout, on `n` random transitions of it: norms of the policy-loss, value-loss (as weighted
    in the PPO loss, with the value-to-trunk gradient scale applied) and entropy-loss gradients w.r.t. the shared
    trunk (the transformer features extractor), and their ratio, logged as diag/* (MissingNo.Zero's alarm:
    a value/policy ratio of ~50 meant the trunk learned to predict the winner instead of how to play)."""

    def __init__(self, n: int = 2048):
        super().__init__()
        self.n = n

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        model, buf = self.model, self.model.rollout_buffer
        if not buf.full:
            return
        policy = model.policy
        size = buf.buffer_size * buf.n_envs
        idx = np.random.default_rng().choice(size, min(self.n, size), replace=False)
        flat = lambda a: a.reshape(size, *a.shape[2:])[idx]  # noqa: E731
        obs = {k: torch.as_tensor(flat(v), device=policy.device) for k, v in buf.observations.items()}
        actions = torch.as_tensor(flat(buf.actions), device=policy.device).long().flatten()
        adv = torch.as_tensor(flat(buf.advantages), device=policy.device).flatten()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = torch.as_tensor(flat(buf.returns), device=policy.device).flatten()
        policy.set_training_mode(True)
        values, log_prob, entropy = policy.evaluate_actions(obs, actions)
        policy.set_training_mode(False)
        losses = {
            "policy": -(adv * log_prob).mean(),  # the clipped PPO loss's gradient at ratio 1
            "value": model.vf_coef * torch.nn.functional.mse_loss(returns, values.flatten()),
            "entropy": -model.ent_coef * entropy.mean(),
        }
        trunk = [p for p in policy.features_extractor.parameters() if p.requires_grad]
        norms = {}
        for i, (name, loss) in enumerate(losses.items()):
            grads = torch.autograd.grad(loss, trunk, retain_graph=i < len(losses) - 1, allow_unused=True)
            norms[name] = torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None)).item()
        ratio = norms["value"] / max(norms["policy"], 1e-12)
        for name, v in norms.items():
            self.logger.record(f"diag/trunk_grad_{name}", v)
        self.logger.record("diag/trunk_grad_value_over_policy", ratio)
        print(f"[diag @ {self.num_timesteps:,}] trunk grad norms policy={norms['policy']:.3g} "
              f"value={norms['value']:.3g} entropy={norms['entropy']:.3g} value/policy={ratio:.3g} "
              f"(value-to-trunk scale {getattr(policy, 'value_trunk_grad', 1.0)})", flush=True)


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
    p.add_argument("--fastenv", metavar="WORKERSxBATTLES",
                   help="in-process Showdown battles (porygon.fastenv), e.g. 28x64; no servers; self-play only")
    p.add_argument("--seed", type=int, default=0, help="--fastenv team/battle seed")
    p.add_argument("--fastenv-lockstep", action="store_true",
                   help="--fastenv through SB3's lockstep PPO (fastenv_vec.FastVecEnv) instead of AsyncPPO")
    p.add_argument("--fastenv-stale-wait", action="store_true",
                   help="--fastenv-lockstep with the old approximate wait steps (previous obs, reward one step late): "
                        "only to reproduce diag 2; it under-credits KOs and cost ~10 points at 2.5M (gap Test 1)")
    p.add_argument("--autocast", choices=["bf16"], help="async PPO: bf16 autocast in the update (~1.4x on A16)")
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
                   default=SHAPING)
    p.add_argument("--eval-every", type=int, default=2_500_000)
    p.add_argument("--eval-battles", type=int, default=200)
    # The baseline bots saturate around 85%; a fixed earlier checkpoint keeps showing progress.
    p.add_argument("--anchor", type=Path, help="checkpoint (.zip or pool .pt) to also evaluate against")
    p.add_argument("--checkpoint-every", type=int, default=5_000_000)
    p.add_argument("--run-dir", type=Path, default=Path("runs") / time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--resume", type=Path, help="checkpoint .zip to continue from")
    p.add_argument("--max-hours", type=float, help="stop (and save) after this much wall-clock time")
    p.add_argument("--embedding-lr-mult", type=float, default=1.0,
                   help="multiply the token embeddings' Adam updates (they barely train at 1: boost_embedding_steps)")
    p.add_argument("--lr-schedule", choices=["thesis", "wsd"], default="thesis",
                   help="thesis: 10^-4.23/(8x+1)^1.5 over --total-steps; wsd: warmup to --lr, constant, optional cooldown")
    p.add_argument("--lr", type=float, default=10**-4.23, help="wsd: peak learning rate")
    p.add_argument("--lr-warmup", type=int, default=1_000_000, help="wsd: warmup steps (from the start/resume step)")
    p.add_argument("--lr-decay-start", type=int, help="wsd: cosine cooldown from this step to --total-steps")
    p.add_argument("--value-trunk-grad", type=float, default=1.0,
                   help="share of the value loss's gradient that reaches the shared trunk (MissingNo.Zero: << 1)")
    p.add_argument("--no-vf-clip", action="store_true",
                   help="no value clipping (also on --resume); SB3's clip zeroes the value gradient beyond the band")
    p.add_argument("--grad-diagnostics", action="store_true",
                   help="log trunk gradient norms of the policy / value / entropy losses after every rollout")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")  # TF32 on Ampere+: faster PPO updates
    ports = parse_ports(args.ports)
    shaping = dict(zip(("fainted_value", "hp_value", "status_value"), args.shaping))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = args.pool_dir or args.run_dir / "pool"
    if args.pool_battles:
        pool_dir.mkdir(parents=True, exist_ok=True)
    if args.fastenv:
        from porygon.fastenv import FastEnv
        from porygon.fastenv_sb3 import FastBattles
        assert not args.pool_battles, "--fastenv is self-play only (pool opponents: not yet)"
        n_workers, per_worker = map(int, args.fastenv.lower().split("x"))
        # lockstep = poke-env step semantics, whose wait steps carry the waiting side's fresh obs and reward
        fast = FastEnv(args.format, n_workers, per_worker, seed=args.seed, shaping=tuple(args.shaping),
                       wait_rows=args.fastenv_lockstep and not args.fastenv_stale_wait)
        args.n_battles = n_workers * per_worker  # --ports: only for --eval-every
        if args.fastenv_lockstep:  # SB3's own rollout code over poke-env-style lockstep sub-envs (fastenv_vec)
            from porygon.fastenv_vec import FastVecEnv
            from porygon.async_env import DiagPPO
            env, algo = VecMonitor(FastVecEnv(fast)), DiagPPO
            assert not args.overlap_update, "--fastenv-lockstep: no --overlap-update"
        else:
            env, algo = FastBattles(fast), AsyncPPO
            args.async_envs = True
    elif args.async_envs:  # same battles and ports as the lockstep envs below, one worker process each
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

    schedule = thesis_lr
    if args.lr_schedule == "wsd":
        schedule = WarmupConstantLR(args.lr, args.lr_warmup, decay_start=args.lr_decay_start, total=args.total_steps)
    if args.resume:
        model = algo.load(args.resume, env=env, device=args.device)
        model.learning_rate, model.lr_schedule = schedule, schedule  # schedules aren't pickled reliably
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
            learning_rate=schedule,
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
    if isinstance(schedule, WarmupConstantLR):
        schedule.model, schedule.start = model, model.num_timesteps
    if args.no_vf_clip:
        model.clip_range_vf = None
    set_value_trunk_grad(model.policy, args.value_trunk_grad)
    boost_embedding_steps(model.policy, args.embedding_lr_mult)
    if args.autocast:
        assert args.async_envs, "--autocast needs --async-envs or --fastenv"
        model.autocast = args.autocast
    if args.overlap_update:
        assert args.async_envs, "--overlap-update needs --async-envs"
        model.overlap_update = True
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"{'fastenv ' + args.fastenv + ' ' if args.fastenv else ''}{'async' if args.async_envs else 'lockstep'}{' + overlapped update' if args.overlap_update else ''}: {args.n_battles} self-play + {args.pool_battles} pool battles "
          f"on ports {ports}, n_steps={model.n_steps} x {env.num_envs} envs, params={n_params:,}, device={model.device}")

    callbacks = CallbackList([
        CheckpointCallback(max(1, args.checkpoint_every // env.num_envs), str(args.run_dir / "checkpoints"),
                           name_prefix="ppo"),
    ] + ([EvalCallback(args.format, ports[0], args.eval_every, args.eval_battles, ("heuristic", "max_power"),
                       args.run_dir, args.anchor)] if args.eval_every > 0 else [])
      + ([WallClockLimit(args.max_hours)] if args.max_hours else [])
      + ([GradientDiagnostics()] if args.grad_diagnostics else [])
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
