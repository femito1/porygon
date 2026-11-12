"""Head-to-head of two policies on in-process Showdown battles (porygon.fastenv), duplicate pairs.

    python -m porygon.fastenv_eval --format gen4randombattle --a runs/x/final_model.zip --b random --games 2000

Side one (p1) is always A, side two B; each slot plays duplicate pairs (same battle seed, teams swapped), so a
pair's team luck cancels. Reports A's win rate (ties count half) with a 95% interval from pair means. A policy is
an SB3 checkpoint (.zip) / pool snapshot (.pt) acting stochastically, `random` (uniform over legal actions), or
(B only) `heuristic` / `max_power` (poke-env's SimpleHeuristicsPlayer / MaxBasePowerPlayer, ported to JS with
0 mismatches on 22.8k decisions, played inside the env), or a netv2 model `v2:PATH[,temperature=T]` (spec-v2
observations from the JS encoder; sampled from softmax(logits / T) over legal actions; needs netv2's porygon.v2).
v1 (SB3) and v2 nets cannot meet in one run (one encoder per env).
Fast and exact Showdown mechanics, but strength claims still go through real Showdown benchmarks (README).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from porygon.fastenv import FastEnv, random_actions


def make_actor(spec: str, device: str):
    if spec == "random":
        rng = np.random.default_rng(12345)
        return lambda obs, mask: random_actions(mask, rng)
    if spec.startswith("v2:"):
        return make_v2_actor(spec, device)
    from porygon.model import load_policy
    policy = load_policy(spec, device)

    def act(obs, mask):
        batch = {"observation": torch.as_tensor(obs, device=device),
                 "action_mask": torch.as_tensor(mask.astype(np.int8), device=device)}
        with torch.no_grad():
            actions, _, _ = policy(batch)
        return actions.cpu().numpy()
    return act


def make_v2_actor(spec: str, device: str):
    """`v2:PATH[,temperature=T]` -> act(obs fields dict, mask) sampling softmax(logits / T) over legal actions."""
    from porygon.v2.model import batch_to_tensors, load, model_keys
    path, *opts = spec[3:].split(",")
    temp = 1.0
    for o in opts:
        k, _, v = o.partition("=")
        assert k == "temperature", f"unknown v2 option {o!r}"
        temp = float(v)
    net = load(path, device)
    keys = model_keys(net.gen)
    gen = torch.Generator(device=device).manual_seed(4321)

    def act(obs, mask):
        t = batch_to_tensors({k: obs[k] for k in keys}, device)
        with torch.no_grad():
            logits = net(t, aux=False)["logits"].float()
        m = torch.as_tensor(mask, device=device)
        logits = torch.where(m, logits / temp, torch.finfo(torch.float32).min)
        return torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1).cpu().numpy()
    act.encoder = "v2"
    return act


ENV_BOTS = ("heuristic", "max_power")  # played inside the env (porygon.fastenv bot=...)


def evaluate(fmt: str, a: str, b: str, games: int, workers: int, battles: int, seed: int, device: str) -> dict:
    act = [make_actor(a, device), None if b in ENV_BOTS else make_actor(b, device)]
    bot = dict(bot=b, bot_battles=battles) if b in ENV_BOTS else {}
    v2 = [x.startswith("v2:") for x in (a, b)]
    v1 = [x not in ("random", *ENV_BOTS) and not x.startswith("v2:") for x in (a, b)]
    assert not (any(v2) and any(v1)), "a v1 (SB3) and a v2 net cannot share an env (one encoder per env)"
    bot["encoder"] = "v2" if any(v2) else "v1"
    pairs: dict[tuple[int, int], list[float]] = {}  # (slot, pair index) -> A's scores
    played = np.zeros(workers * battles, dtype=np.int64)  # games finished per battle position
    with FastEnv(fmt, workers, battles, seed=seed, duplicate=True, **bot) as env:
        batch = env.reset()
        t0 = time.time()
        while sum(len(v) for v in pairs.values()) < games:
            na = batch.needs_action
            actions = np.zeros(int(na.sum()), dtype=np.int64)
            idx = np.flatnonzero(na)
            for side in (0, 1):
                sel = (batch.slot[idx] % 2) == side
                if sel.any():
                    rows = idx[sel]
                    obs = batch.obs
                    obs = {k: v[rows] for k, v in obs.items()} if isinstance(obs, dict) else obs[rows]
                    actions[sel] = act[side](obs, batch.mask[rows])
            done = np.flatnonzero(batch.done & (batch.slot % 2 == 0))
            for i in done:
                pos = int(batch.slot[i]) // 2
                pairs.setdefault((pos, int(played[pos]) // 2), []).append(0.5 + 0.5 * float(batch.win[i]))
                played[pos] += 1
            batch = env.step(actions)
        secs = time.time() - t0
    full = [v for v in pairs.values() if len(v) == 2]
    means = np.array([np.mean(v) for v in full])
    n = len(means)
    rate = float(means.mean()) if n else float("nan")
    half = 1.96 * float(means.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
    return {"a": a, "b": b, "format": fmt, "pairs": n, "win_rate": rate, "ci95": half, "seconds": secs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--a", required=True)
    p.add_argument("--b", default="random")
    p.add_argument("--games", type=int, default=2000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--battles", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    r = evaluate(a.format, a.a, a.b, a.games, a.workers, a.battles, a.seed, a.device)
    print(f"{r['a']} vs {r['b']} ({r['format']}): {100 * r['win_rate']:.1f}% ± {100 * r['ci95']:.1f}% "
          f"({r['pairs']} pairs, {r['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
