"""Run Foul Play (GPL-3.0, not vendored; run from its own directory) with every MCTS search fixed to an iteration
count instead of a wall-clock budget, so its strength does not depend on the machine or on CPU contention.

    cd $FP && $FP/.venv/bin/python fp_fixed_iters.py --iters-per-s N run.py [Foul Play's run.py arguments]

wsp P3 (2026-11-24): the fpcal launcher (scripts/wsp/fpcal/fp_fixed_iters.py, md5 f9c32cdf, gen9) extended to gen4.
Gen4 Foul Play runs from its own venv ($FP/.venv: the stock poke-engine 0.0.48 built with the gen4 feature), gen9
Foul Play from $FP/.venv-gen9 with gloria's .engine-gen9 on PYTHONPATH; this file is the same for both. The gen4
additions are guards only, so a build without the argument, or a Foul Play that calls another binding, fails loudly
instead of silently running on the clock:
  - the loaded poke_engine.monte_carlo_tree_search must take an `iterations` argument;
  - fp.search.main must call that very function (the name patched below);
  - the engine module path (and its dist version when it has one) is printed, so each Foul Play log names the build.
Each search Foul Play asks for with a budget of D ms runs round(N · D / 1000) iterations (poke-engine's own
`iterations` argument of monte_carlo_tree_search, which ignores the duration when > 0; it rounds up to a multiple of
1000). So Foul Play's schedule is kept: 2 states × N iterations normally, 4 × N/2 early in the game. Our code only:
it patches the name fp.search.main.monte_carlo_tree_search in this process before run.py starts; the search worker
processes are forked (Python 3.12 on Linux) and inherit the patch. Foul Play logs "Iterations i: V" per state, which
the harness checks against the requested counts (fp_iters_check.py).
"""
import inspect
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
    if "iterations" not in inspect.signature(original).parameters:
        raise SystemExit(f"{poke_engine.__file__}: monte_carlo_tree_search has no iterations argument")
    if fp_search.monte_carlo_tree_search is not original:
        raise SystemExit("fp.search.main.monte_carlo_tree_search is not poke_engine's: nothing would be fixed")
    try:
        from importlib.metadata import version
        build = f"poke-engine {version('poke-engine')}"
    except Exception:  # an engine on PYTHONPATH without dist-info (gloria's .engine-gen9)
        build = "poke-engine (no dist-info)"

    def fixed(state, duration_ms=1000, iterations=0, threads=1):
        return original(state, duration_ms, iterations=max(1, round(n * duration_ms / 1000)), threads=threads)

    fp_search.monte_carlo_tree_search = fixed
    print(f"fp_fixed_iters: {n} iterations per 1000 ms of nominal budget; engine {build} at {poke_engine.__file__}",
          file=sys.stderr, flush=True)
    sys.argv = [script] + sys.argv[4:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
