"""Throughput of porygon.fastenv with random legal actions (env cost only), or with a policy (--model).

    python scripts/fastenv/bench_fastenv.py --workers 4 --battles 64 --seconds 30 --format gen9randombattle
Reports decisions/s (rows that needed an action), games/s, CPU per decision of the node workers and of this
process. --async uses poll/send (act on whichever workers are ready) instead of lockstep step().
"""
import argparse
import os
import time

import numpy as np

from porygon.fastenv import FastEnv, random_actions


def child_cpu(pids):
    tck = os.sysconf("SC_CLK_TCK")
    tot = 0.0
    for p in pids:
        try:
            with open(f"/proc/{p}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            tot += (int(parts[11]) + int(parts[12])) / tck
        except FileNotFoundError:
            pass
    return tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--battles", type=int, default=64)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--encoder", default="v1")
    p.add_argument("--async", dest="use_async", action="store_true")
    p.add_argument("--policy", help="act with a network instead of random: 'init' (fresh MaskedPolicy) or a checkpoint")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    pick = lambda b, rows: random_actions(b.mask[rows], rng)  # noqa: E731
    if a.policy:
        import torch
        from gymnasium import spaces
        from porygon.model import MaskedPolicy, load_policy
        n_act = 26 if a.format.startswith("gen9") else 10
        if a.policy == "init":
            from porygon.encoder import OBS_DIM
            pol = MaskedPolicy(spaces.Dict({"observation": spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
                                            "action_mask": spaces.Box(0, 1, (n_act,), np.int8)}),
                               spaces.Discrete(n_act), lambda _: 1e-4).to(a.device).eval()
        else:
            pol = load_policy(a.policy, a.device)

        def pick(b, rows):
            x = {"observation": torch.as_tensor(b.obs[rows], device=a.device),
                 "action_mask": torch.as_tensor(b.mask[rows].astype(np.int8), device=a.device)}
            with torch.no_grad():
                return pol(x)[0].cpu().numpy()
    with FastEnv(a.format, a.workers, a.battles, seed=1, encoder=a.encoder) as env:
        pids = [w.proc.pid for w in env.workers]
        batch = env.reset()
        c0, p0, t0 = child_cpu(pids), time.process_time(), time.time()
        dec = games = rows = rejected = 0
        if a.use_async:
            for w, b in [(i, env._pending[i]) for i in range(a.workers)]:
                env.send(w, pick(b, b.needs_action))
            while time.time() - t0 < a.seconds:
                ready = env.poll()
                for w, b in ready:
                    dec += int(b.needs_action.sum()); games += int(b.done.sum()) // 2; rows += len(b)
                    rejected += int(((b.flags & 8) > 0).sum())
                if a.policy:  # one forward for every ready worker, as the trainer does
                    from porygon.fastenv import Batch
                    big = Batch.concat([b for _, b in ready])
                    acts = pick(big, big.needs_action)
                    k = 0
                    for w, b in ready:
                        m = int(b.needs_action.sum())
                        env.send(w, acts[k:k + m])
                        k += m
                else:
                    for w, b in ready:
                        env.send(w, pick(b, b.needs_action))
        else:
            while time.time() - t0 < a.seconds:
                batch = env.step(pick(batch, batch.needs_action))
                dec += int(batch.needs_action.sum()); games += int(batch.done.sum()) // 2; rows += len(batch)
                rejected += int(((batch.flags & 8) > 0).sum())
        wall = time.time() - t0
        c1, p1 = child_cpu(pids), time.process_time()
    print(f"{a.format} encoder={a.encoder} policy={a.policy or 'random'} {a.workers}x{a.battles} {'async' if a.use_async else 'lockstep'}: "
          f"{dec / wall:,.0f} decisions/s, {games / wall:.1f} games/s ({dec / max(games, 1):.0f} decisions/game), "
          f"node CPU {1e6 * (c1 - c0) / dec:.0f} us/decision, python CPU {1e6 * (p1 - p0) / dec:.0f} us/decision, "
          f"rejected {rejected / max(dec, 1):.2%}")


if __name__ == "__main__":
    main()
