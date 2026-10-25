"""Expert iteration, one round: distil the search player (`porygon.search`) into the network.

`generate` plays search-vs-search games on local Showdown servers and records, for every searched
decision of both sides, the observation, the legal-action mask, the network's prior, the move
distribution search played (mean visit share + prior_weight * prior, as `SearchPlayer.combine`
scores it) and the shaped return PPO would assign from that state (final state value minus the
state value now, with the training shaping weights; gamma ~ 1). `distill` trains a PPO checkpoint
on those targets: cross-entropy to the search distribution sharpened by `--temperature` (raw visit
shares are much flatter than the move search plays: entropy ~1.2 vs the network's ~0.25 nats), MSE
of the value head to the return,
and a small KL to the original prior so states search never visits don't drift. The output is a
normal PPO checkpoint (load_policy, tournament, search, `train.py --resume`).

    scripts/start_servers.sh 4 8100
    python -m porygon.expert generate runs/<run>/final_model.zip --games 2000 --out data/round1 --ports 8100-8103
    python -m porygon.expert distill runs/<run>/final_model.zip data/round1 --out runs/round1/distilled.zip
"""

import argparse
import asyncio
import logging
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, Battle
from poke_env.environment import SinglesEnv
from stable_baselines3 import PPO
from torch.nn import functional as F

from porygon.encoder import embed_battle
from porygon.env import SHAPING, local_server
from porygon.model import load_policy
from porygon.search import SearchPlayer
from porygon.train import parse_ports


def state_value(battle: AbstractBattle, shaping: tuple[float, float, float] = SHAPING) -> float:
    """poke-env's shaping state value (what SelfPlayEnv.calc_reward differences), +-1 once won/lost."""
    fainted, hp, status = shaping
    holder = SimpleNamespace(_reward_buffer={})  # empty buffer -> the helper returns the state value itself
    return SinglesEnv.reward_computing_helper(holder, battle, fainted_value=fainted, hp_value=hp,
                                              status_value=status, victory_value=1.0)


def search_target(shares: np.ndarray, probs: np.ndarray, mask: np.ndarray, prior_weight: float) -> np.ndarray:
    """The distribution SearchPlayer plays the argmax of, normalised over legal actions."""
    t = np.where(mask > 0, shares + prior_weight * probs, 0.0)
    return (t / t.sum()).astype(np.float32)


class ShardWriter:
    """Collects finished games' records (one game per player side) and writes one compressed .npz
    per `games_per_shard` of them."""

    FIELDS = ("obs", "mask", "prior", "target", "ret")

    def __init__(self, out: Path, games_per_shard: int):
        self.out, self.games_per_shard = out, games_per_shard
        out.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self.games = self.total_games = self.total_records = self.shards = 0

    def add_game(self, records: list[dict]) -> None:
        self.records += records
        self.games += 1
        self.total_games += 1
        if self.games >= self.games_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        path = self.out / f"shard-{time.strftime('%Y%m%d-%H%M%S')}-{self.shards:04d}.npz"
        np.savez_compressed(path, **{k: np.stack([r[k] for r in self.records]) for k in self.FIELDS})
        self.total_records += len(self.records)
        self.shards += 1
        self.records, self.games = [], 0


class RecordingSearchPlayer(SearchPlayer):
    """SearchPlayer that records every searched decision as a training example."""

    def __init__(self, *args, writer: ShardWriter, shaping=SHAPING, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer, self.shaping = writer, shaping
        self.pending: dict[str, list[dict]] = {}

    def on_decision(self, battle: Battle, probs: np.ndarray, mask: np.ndarray, shares: np.ndarray) -> None:
        self.pending.setdefault(battle.battle_tag, []).append({
            "obs": embed_battle(battle).astype(np.float16),
            "mask": mask.astype(np.int8),
            "prior": probs.astype(np.float32),
            "target": search_target(shares, probs, mask, self.prior_weight),
            "ret": -state_value(battle, self.shaping),  # final value added when the battle ends
        })

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        super()._battle_finished_callback(battle)
        records = self.pending.pop(battle.battle_tag, [])
        final = state_value(battle, self.shaping)
        for r in records:
            r["ret"] = np.float32(r["ret"] + final)
        self.writer.add_game(records)


async def generate(args: argparse.Namespace) -> None:
    policy = load_policy(args.model)
    writer = ShardWriter(args.out, 2 * args.shard_games)  # both sides of every battle record
    tag, pairs = random.randint(0, 10**5), []
    for i, port in enumerate(parse_ports(args.ports)):
        common = dict(battle_format=args.format, server_configuration=local_server(port),
                      max_concurrent_battles=args.concurrency, log_level=40, writer=writer,
                      ms=args.ms, samples=args.samples, prior_weight=args.prior_weight, workers=args.workers)
        pairs.append(tuple(RecordingSearchPlayer(policy, account_configuration=AccountConfiguration(f"x{s}{i}x{tag}", None),
                                                 **common) for s in "ab"))
    per_pair = -(-args.games // len(pairs))
    start = time.time()

    async def report():
        while True:
            await asyncio.sleep(args.report_every)
            n, dt = writer.total_records + len(writer.records), time.time() - start
            print(f"[{dt / 3600:.2f} h] {writer.total_games // 2} battles, {n} records ({n / dt:.1f}/s), "
                  f"{writer.shards} shards", flush=True)

    reporter = asyncio.create_task(report())
    await asyncio.gather(*(a.battle_against(b, n_battles=per_pair) for a, b in pairs))
    reporter.cancel()
    writer.flush()
    stats = [p.stats for pair in pairs for p in pair]
    print(f"done: {writer.total_games // 2} battles, {writer.total_records} records in {writer.shards} shards "
          f"({time.time() - start:.0f}s); decisions {sum(s['decisions'] for s in stats)}, "
          f"searched {sum(s['searched'] for s in stats)}, fallbacks {sum(s['fallbacks'] for s in stats)}")


def load_shards(paths: list[Path]) -> dict[str, np.ndarray]:
    files = sorted(f for p in paths for f in ([p] if p.is_file() else p.glob("*.npz")))
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([d[k] for d in parts]) for k in ShardWriter.FIELDS}


def distill(args: argparse.Namespace) -> None:
    data = load_shards(args.data)
    sharp = data["target"] ** (1 / args.temperature)
    data["target"] = (sharp / sharp.sum(1, keepdims=True)).astype(np.float32)
    n = len(data["ret"])
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val, train = idx[:n_val], idx[n_val:]
    print(f"{n} records from {len(args.data)} path(s): {len(train)} train / {n_val} validation", flush=True)

    model = PPO.load(args.model, device=args.device)
    policy, device = model.policy, model.device
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch(ix):
        t = lambda k, dt=torch.float32: torch.as_tensor(data[k][ix], dtype=dt, device=device)  # noqa: E731
        return {"observation": t("obs"), "action_mask": t("mask")}, t("target"), t("prior"), t("ret")

    def losses(ix):
        obs, target, prior, ret = batch(ix)
        logp = policy.get_distribution(obs).distribution.logits  # log-probs, illegal actions ~ -inf
        ce = -(target * logp).sum(1)
        kl = (prior * (torch.log(prior.clamp_min(1e-8)) - logp)).sum(1)
        v = policy.predict_values(obs).squeeze(1)
        agree = (logp.argmax(1) == target.argmax(1)).float()
        return ce.mean(), kl.mean(), F.mse_loss(v, ret), agree.mean()

    def evaluate() -> str:
        policy.set_training_mode(False)
        sums, count = np.zeros(4), 0
        with torch.no_grad():
            for i in range(0, n_val, args.batch_size):
                ix = val[i:i + args.batch_size]
                sums += np.array([x.item() for x in losses(ix)]) * len(ix)
                count += len(ix)
        ce, kl, vmse, agree = sums / count
        return f"val ce {ce:.4f}  kl-to-prior {kl:.4f}  value mse {vmse:.4f}  top-1 = search {agree:.1%}"

    print(f"before: {evaluate()}", flush=True)
    for epoch in range(args.epochs):
        policy.set_training_mode(True)
        order = rng.permutation(train)
        for i in range(0, len(order), args.batch_size):
            ce, kl, vmse, _ = losses(order[i:i + args.batch_size])
            loss = ce + args.vf_coef * vmse + args.beta * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            opt.step()
        print(f"epoch {epoch + 1}: {evaluate()}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="search-vs-search games -> training shards")
    g.add_argument("model")
    g.add_argument("--games", type=int, default=1000)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--ports", default="8000", help="one pair of players per Showdown server, e.g. 8100-8103")
    g.add_argument("--concurrency", type=int, default=4, help="simultaneous battles per pair")
    g.add_argument("--ms", type=int, default=300, help="MCTS time per determinization per decision")
    g.add_argument("--samples", type=int, default=4)
    g.add_argument("--prior-weight", type=float, default=0.25)
    g.add_argument("--workers", type=int, default=None, help="engine processes (default: cores - 1)")
    g.add_argument("--shard-games", type=int, default=200, help="battles per shard")
    g.add_argument("--report-every", type=float, default=600, help="seconds between progress lines")
    g.add_argument("--format", default="gen4randombattle")
    d = sub.add_parser("distill", help="train a checkpoint on generated shards")
    d.add_argument("model")
    d.add_argument("data", nargs="+", type=Path, help="shard files or directories")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--epochs", type=int, default=3)
    d.add_argument("--lr", type=float, default=3e-5)
    d.add_argument("--batch-size", type=int, default=1024)
    d.add_argument("--vf-coef", type=float, default=0.4375)
    d.add_argument("--beta", type=float, default=0.1, help="weight of KL(prior || policy)")
    d.add_argument("--temperature", type=float, default=0.25, help="target ~ target^(1/T); 1 = raw shares")
    d.add_argument("--weight-decay", type=float, default=0.0)
    d.add_argument("--max-grad-norm", type=float, default=0.543)
    d.add_argument("--val-frac", type=float, default=0.05)
    d.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "generate":
        asyncio.run(generate(args))
    else:
        distill(args)


if __name__ == "__main__":
    main()
