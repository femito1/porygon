"""netv2 BattleWorkers with pool specs (forkserver, as porygon.v2.train): does every worker send its first message?
    PYTHONPATH=<tree with porygon.v2> python scripts/fastenv/test_battleworkers_pool.py PORT
with patches/netv2-pool-worker-fix.patch: all workers within seconds; without it the pool workers never answer."""
import sys
import tempfile
import time
from multiprocessing.connection import wait


def main():
    from porygon.v2.workers import BattleWorkers, WorkerSpec  # parent imports poke_env first, as porygon.v2.train
    port = int(sys.argv[1])
    d = tempfile.mkdtemp()
    specs = ([WorkerSpec("pool", port, "gen9randombattle", {}, d, 0.2) for _ in range(2)]
             + [WorkerSpec("selfplay", port, "gen9randombattle", {})])
    t0 = time.time()
    bw = BattleWorkers(specs)
    got = set()
    while len(got) < len(specs) and time.time() - t0 < 90:
        for c in wait([bw.conns[i] for i in range(len(specs)) if i not in got], timeout=5):
            i = bw.conns.index(c)
            c.recv()
            got.add(i)
    print(f"first messages from workers {sorted(got)} of {len(specs)} (0-1 pool, 2 self-play) in {time.time() - t0:.1f} s")
    for i in range(len(specs)):
        bw.kill(i)


if __name__ == "__main__":
    main()
