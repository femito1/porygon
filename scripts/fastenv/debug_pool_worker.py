"""Run one netv2 battle worker (porygon.v2.workers._battle_loop) against a running Showdown server and dump every
thread's stack if its first message does not arrive in time (the pool-battle hang, 2026-11-09).

    PYTHONPATH=<tree with porygon.v2> python scripts/fastenv/debug_pool_worker.py PORT [pool|selfplay] [FORMAT] [fresh]

Default: the parent imports porygon.v2.workers (hence poke_env and its POKE_LOOP thread) before forking the worker,
as netv2's forkserver preload does. `fresh`: the child imports it itself (what removing the preload does).
Finding (2026-11-09): a forked child has no POKE_LOOP thread, so poke-env's Player.__init__ (create_in_poke_loop)
blocks forever in PoolOpponent's SimpleHeuristicsPlayer / MaxBasePowerPlayer: every pool worker hangs.
"""
import faulthandler
import multiprocessing as mp
import sys
import tempfile
import time

import numpy as np


def child(conn, spec, dump):
    with open(dump, "w") as f:
        faulthandler.dump_traceback_later(40, repeat=False, file=f)
        from porygon.v2.workers import WorkerSpec, _battle_loop
        if isinstance(spec, tuple):  # fresh: build the spec here (first import of poke_env in this process)
            spec = WorkerSpec(*spec)
        _battle_loop(conn, spec)


def main():
    port, kind = int(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 else "pool")
    fmt = sys.argv[3] if len(sys.argv) > 3 else "gen9randombattle"
    fresh = len(sys.argv) > 4 and sys.argv[4] == "fresh"
    pool_dir = tempfile.mkdtemp(prefix="pooldbg-")
    spec = (kind, port, fmt, {}, pool_dir if kind == "pool" else None, 0.2)
    if not fresh:
        from porygon.v2.workers import WorkerSpec
        spec = WorkerSpec(*spec)
    dump = tempfile.mktemp(prefix="pooldbg-stack-")
    a, b = mp.Pipe()
    p = mp.get_context("fork").Process(target=child, args=(b, spec, dump), daemon=True)
    p.start()
    t0 = time.time()
    steps = 0
    while time.time() - t0 < 60:
        if not a.poll(50):
            print(f"{kind}: no message for 50 s after {steps} steps")
            break
        obs, rewards, done, info, opp = a.recv()
        steps += 1
        if steps == 1:
            print(f"{kind}: first message after {time.time() - t0:.1f} s: sides {[o is not None for o in obs]}")
        if info is not None:
            print(f"{kind}: episode done after {steps} steps: {info}")
            break
        a.send(([0] * len(obs), 0))  # action 0 for every side (legal or not: poke-env strict=False)
    time.sleep(1)
    try:
        print(open(dump).read()[-6000:])
    except OSError:
        pass
    p.kill()


if __name__ == "__main__":
    main()
