"""Profile the current training environment (local Showdown server + websockets + poke-env + Python encoder).

Runs W worker processes, each one SelfPlayEnv battle loop with random legal actions (both sides), for S seconds,
and reports env steps/s plus CPU-seconds per step split by process: Python workers (poke-env + encoder) vs the
Node server (all node processes: server + its simulator children). Worker 0 also runs under cProfile, and the
top cumulative entries are printed so the Python share can be split (embed_battle vs poke-env parsing vs I/O).

    python scripts/fastenv/profile_showdown_env.py --port 8300 --workers 4 --seconds 60 --format gen9randombattle
"""
import argparse
import cProfile
import io
import multiprocessing as mp
import os
import pstats
import time

import numpy as np


def node_cpu() -> float:
    tot = 0.0
    tck = os.sysconf("SC_CLK_TCK")
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read()
            if b"pokemon-showdown" not in cmd and b"sim" not in cmd.split(b"\0")[0]:
                continue
            if not cmd.startswith(b"node") and b"/node" not in cmd.split(b"\0")[0]:
                continue
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            tot += (int(parts[11]) + int(parts[12])) / tck
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return tot


def worker(i, port, fmt, seconds, q, prof):
    from porygon.env import SelfPlayEnv, local_server, SHAPING
    import porygon.encoder as enc
    shaping = dict(zip(("fainted_value", "hp_value", "status_value"), SHAPING))
    env = SelfPlayEnv(battle_format=fmt, server_configuration=local_server(port), shaping=shaping,
                      log_level=40, open_timeout=None, challenge_timeout=None, strict=False)
    # time embed_battle separately
    t_embed = [0.0, 0]
    orig = enc.embed_battle

    def timed(b):
        t = time.perf_counter()
        r = orig(b)
        t_embed[0] += time.perf_counter() - t
        t_embed[1] += 1
        return r
    import porygon.env as penv
    penv.embed_battle = timed
    rng = np.random.default_rng(i)
    pr = cProfile.Profile() if prof else None
    obs, _ = env.reset()
    agents = env.possible_agents
    steps = games = 0
    c0 = time.process_time()
    t0 = time.time()
    if pr:
        pr.enable()
    while time.time() - t0 < seconds:
        acts = {}
        for a in agents:
            m = obs[a]["action_mask"]
            acts[a] = np.int64(rng.choice(np.flatnonzero(m)))
        obs, rew, term, trunc, _ = env.step(acts)
        steps += len(agents)
        if any(term.values()) or any(trunc.values()):
            games += 1
            obs, _ = env.reset()
    if pr:
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
        print(s.getvalue()[:6000])
    q.put((steps, games, time.process_time() - c0, t_embed[0], t_embed[1]))
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    n0, w0 = node_cpu(), time.time()
    procs = [ctx.Process(target=worker, args=(i, a.port, a.format, a.seconds, q, a.profile and i == 0))
             for i in range(a.workers)]
    for pr in procs:
        pr.start()
    res = [q.get() for _ in procs]
    wall = time.time() - w0
    n1 = node_cpu()
    for pr in procs:
        pr.join()
    steps = sum(r[0] for r in res)
    games = sum(r[1] for r in res)
    py_cpu = sum(r[2] for r in res)
    emb, nemb = sum(r[3] for r in res), sum(r[4] for r in res)
    print(f"{a.format}: {a.workers} workers, {wall:.0f}s wall, {steps} steps, {games} games "
          f"({steps / max(games, 1):.0f} steps/game)")
    print(f"steps/s {steps / wall:.0f}; per step: python CPU {1e3 * py_cpu / steps:.2f} ms "
          f"(embed_battle {1e3 * emb / max(nemb, 1):.2f} ms/call, {nemb / steps:.2f} calls/step), "
          f"node CPU {1e3 * (n1 - n0) / steps:.2f} ms")


if __name__ == "__main__":
    main()
