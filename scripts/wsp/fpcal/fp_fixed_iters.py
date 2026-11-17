"""Run Foul Play (GPL-3.0, not vendored; run from its own directory) with every MCTS search fixed to an iteration
count instead of a wall-clock budget, so its strength does not depend on the machine or on CPU contention.

    cd $FP && $FP/.venv-gen9/bin/python fp_fixed_iters.py --iters-per-s N run.py [Foul Play's run.py arguments]

Each search Foul Play asks for with a budget of D ms runs round(N · D / 1000) iterations (poke-engine's own
`iterations` argument of monte_carlo_tree_search, which ignores the duration when > 0; it rounds up to a multiple of
1000). So Foul Play's schedule is kept: 2 states × N iterations normally, 4 × N/2 early in the game. Our code only:
it patches the name fp.search.main.monte_carlo_tree_search in this process before run.py starts; the search worker
processes are forked (Python 3.12 on Linux) and inherit the patch. Foul Play logs "Iterations i: V" per state, which
the harness checks against the requested counts.
"""
import os
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[1] != "--iters-per-s":
        raise SystemExit(__doc__)
    n = int(sys.argv[2])
    script = sys.argv[3]
    if n <= 0:
        raise SystemExit("--iters-per-s must be > 0")
    sys.path.insert(0, os.getcwd())  # Foul Play's package (fp/) lives in its directory
    import multiprocessing

    import poke_engine
    import fp.search.main as fp_search

    if multiprocessing.get_start_method() != "fork":
        raise SystemExit(f"start method {multiprocessing.get_start_method()}: search workers would not inherit the patch")
    original = poke_engine.monte_carlo_tree_search

    def fixed(state, duration_ms=1000, iterations=0, threads=1):
        return original(state, duration_ms, iterations=max(1, round(n * duration_ms / 1000)), threads=threads)

    fp_search.monte_carlo_tree_search = fixed
    print(f"fp_fixed_iters: {n} iterations per 1000 ms of nominal budget", file=sys.stderr, flush=True)
    sys.argv = [script] + sys.argv[4:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
