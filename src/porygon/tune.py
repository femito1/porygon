"""SPSA tuning of the engine's leaf evaluation for playing strength, in engine-only self-play (`porygon.selfplay`).

A good outcome predictor need not make the strongest search (UCT's exploration interacts with the scale of leaf
values; correlation is not the effect of a move), so after the offline fit (scripts/fit_eval.py) a few knobs
are tuned on what matters, results:
  scale         log multiplier of the leaf sigmoid's scale (a temperature: how sharply leaf values differ)
  root_weight   the leaf value is sigmoid(scale * (eval - root_weight * eval(root))), clamped to [0, 1]
  g:<group>     log multiplier of a feature group's weights (GROUPS)
Each SPSA step perturbs every knob by +-c (Rademacher), plays one duplicate pair (the same teams, swapped) of
theta+ against theta-, and moves theta by a * (result) / c in the perturbation's direction (Spall; the
schedule follows Stockfish fishtest's: alpha 0.602, gamma 0.101, A = 10% of the steps, a = r_end * c_end^2).
Steps run asynchronously on a process pool (theta can be a few dozen results stale; harmless for SPSA).

    python -m porygon.tune --base runs/evalfit/r1/full.json --teams teams.jsonl --out runs/tune/x \
        --knobs scale root_weight g:material g:status g:boosts --pairs 20000 --iterations 20000
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import poke_engine as pe

from porygon.selfplay import EvalConfig, game_timeout, play

GROUPS = {
    "stock": ["stock"],
    "material": ["alive_", "hp_", "active_hp"],
    "status": ["burn", "freeze", "sleep", "rest", "paralyze", "toxic", "poison", "status_good"],
    "items": ["item", "sash"],
    "hazards": ["sr_", "spikes", "tspikes"],
    "volatiles": ["substitute", "leech_seed", "confusion", "taunt", "yawn", "perish", "curse", "ingrain",
                  "locked_move", "must_recharge"],
    "boosts": ["atk_boost", "spa_boost", "def_boost", "spd_boost", "spe_boost", "acc_boost", "eva_boost",
               "off_boost"],
    "tempo": ["faster"],
    "field": ["reflect", "light_screen", "safeguard", "tailwind", "mist", "lucky_chant", "wish", "future_sight"],
}
C_END = {"scale": 0.2, "root_weight": 0.15}  # g:* default 0.2


def group_masks() -> dict[str, np.ndarray]:
    names = pe.eval_feature_names()
    out = {}
    for g, prefixes in GROUPS.items():
        out[g] = np.array([any(n == p or n.startswith(p) for p in prefixes) for n in names])
    assert sum(m.astype(int) for m in out.values()).max() == 1, "feature groups overlap"
    return out


def configure(base: EvalConfig, knobs: list[str], theta: np.ndarray, masks: dict[str, np.ndarray]) -> EvalConfig:
    w = np.array(base.weights if base.weights is not None else [100.0] + [0.0] * (len(pe.eval_feature_names()) - 1))
    scale, root = base.scale, base.root_weight
    for k, t in zip(knobs, theta):
        if k == "scale":
            scale *= math.exp(t)
        elif k == "root_weight":
            root = min(1.0, max(0.0, root + t))
        else:
            w = np.where(masks[k[2:]], w * math.exp(t), w)
    return EvalConfig(tuple(float(x) for x in w), float(scale), float(root))


def _pair(task):
    plus, minus, teams, iterations, seed = task
    signal.alarm(2 * game_timeout(iterations))  # default action: the worker dies, the pool replaces it, the task is lost
    rng = random.Random(seed)
    score = 0.0
    try:
        for t in (teams, teams[::-1]):  # theta+ plays both teams, always as side one
            g = play(t, plus, minus, iterations, rng)
            if g is None:
                return None
            score += g["winner"] == 1
    except BaseException:  # engine panic
        return None
    finally:
        signal.alarm(0)
    return score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="stock", help="the evaluation config theta = 0 stands for")
    p.add_argument("--teams", required=True, help="team pairs (scripts/gen_teams.js), cycled")
    p.add_argument("--out", required=True)
    p.add_argument("--knobs", nargs="+", default=["scale", "root_weight"])
    p.add_argument("--pairs", type=int, default=20000, help="SPSA steps (one duplicate pair each)")
    p.add_argument("--iterations", type=int, default=20000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--r-end", type=float, default=0.002)
    p.add_argument("--c-end", type=float, default=0.2, help="perturbation at the end for g:* knobs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    base, masks = EvalConfig.load(args.base), group_masks()
    for k in args.knobs:
        assert k in C_END or (k.startswith("g:") and k[2:] in GROUPS), k
    teams = [json.loads(line) for line in open(args.teams)]
    d, n = len(args.knobs), args.pairs
    c_end = np.array([C_END.get(k, args.c_end) for k in args.knobs])
    alpha, gamma, big_a = 0.602, 0.101, 0.1 * n
    c0 = c_end * n ** gamma
    a0 = args.r_end * c_end ** 2 * (big_a + n) ** alpha
    rng = np.random.default_rng(args.seed)
    theta = np.zeros(d)
    pending: dict[int, tuple] = {}  # step -> (async result, delta, c_k, start time)
    done = lost = submitted = 0
    log = open(out / "theta.jsonl", "a")
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        while done + lost < n:
            while submitted < n and len(pending) < 2 * args.workers:
                k = submitted
                ck = c0 / (k + 1) ** gamma
                delta = rng.choice([-1.0, 1.0], size=d)
                plus = configure(base, args.knobs, theta + ck * delta, masks)
                minus = configure(base, args.knobs, theta - ck * delta, masks)
                task = (plus, minus, teams[k % len(teams)], args.iterations, args.seed * 10**7 + k)
                pending[k] = (pool.apply_async(_pair, (task,)), delta, ck, time.time())
                submitted += 1
            time.sleep(0.2)
            for k, (res, delta, ck, t0) in list(pending.items()):
                if res.ready():
                    del pending[k]
                    score = res.get()
                    if score is None:
                        lost += 1
                        continue
                    ak = a0 / (big_a + done + 1) ** alpha
                    theta += ak * (score - 1.0) * delta / ck
                    done += 1
                    if done % 50 == 0 or done == n:
                        rec = {"done": done, "lost": lost, "theta": dict(zip(args.knobs, map(float, theta)))}
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                elif time.time() - t0 > 2 * game_timeout(args.iterations) + 60:  # its worker died
                    del pending[k]
                    lost += 1
    final = configure(base, args.knobs, theta, masks)
    (out / "tuned.json").write_text(json.dumps({"weights": list(final.weights), "scale": final.scale,
                                                "root_weight": final.root_weight}))
    print(json.dumps({"done": done, "lost": lost, "theta": dict(zip(args.knobs, map(float, theta)))}))


if __name__ == "__main__":
    main()
