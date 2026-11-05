"""Label logged decisions with an oracle search, for offline network studies.

Every decision logged with `porygon.search --log-oracle` carries the position converted with the opponent's whole
true team. This runs poke-engine's MCTS on that position (perfect information, the default tuned evaluation, a
fixed iteration budget so results do not depend on the CPU) and records, per poke-env action index 0-9, the
oracle's visit share and its average score (Q, NaN if unvisited). The oracle knows hidden sets, so it is an
upper bound on what any player could know, not the best move given what was visible: use its Q gaps to find
decisions where an action is clearly bad whatever the hidden information.

Output: <out>/labels-<worker>.jsonl, one line per decision: {"key": [file, line], "shares", "q", "iterations"}.
Workers are restartable (records already written are skipped) and a stuck search is killed by SIGALRM.

    python scripts/oracle_labels.py runs/search-*/decisions.jsonl --out runs/oracle-labels --iterations 1000000
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT = 600  # seconds per decision


def records(files: list[str], worker: int, workers: int):
    n = 0
    for f in files:
        with open(f) as fh:
            for i, line in enumerate(fh):
                if n % workers == worker:
                    yield [f, i], line
                n += 1


def label(rec: dict, iterations: int, cfg: dict) -> dict | None:
    import numpy as np
    import poke_engine as pe
    o = rec.get("oracle")
    if not o or "error" in o:
        return None
    pe.set_eval(cfg.get("weights"), cfg.get("scale", 0.0125), cfg.get("root_weight", 1.0))
    res = pe.monte_carlo_tree_search(pe.State.from_string(o["state"]), iterations=iterations)
    shares, q = np.zeros(10), np.full(10, np.nan)
    total = sum(r.visits for r in res.side_one) or 1
    for r in res.side_one:
        c = r.move_choice
        if c == "No Move":
            continue
        if c.startswith("switch "):
            name = c[len("switch "):]
            a = o["switch_ids"].index(name) if name in o["switch_ids"] else None
        else:
            a = 6 + o["move_ids"].index(c) if c in o["move_ids"] else None
        if a is None:
            return {"error": f"unmappable {c!r}"}
        shares[a] += r.visits / total
        if r.visits:
            q[a] = r.total_score / r.visits
    return {"shares": np.round(shares, 4).tolist(), "q": [None if np.isnan(x) else round(float(x), 4) for x in q],
            "iterations": res.total_visits if hasattr(res, "total_visits") else iterations}


def worker(args) -> None:
    from porygon.search import DEFAULT_EVAL
    cfg = json.loads(Path(args.eval_config or DEFAULT_EVAL).read_text()) if args.eval_config != "stock" else {}
    out = Path(args.out) / f"labels-{args.worker}.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(tuple(json.loads(line)["key"]))
            except (json.JSONDecodeError, KeyError):
                continue
    started = Path(args.out) / f"started-{args.worker}.txt"
    skip = set(started.read_text().split("\n")) if started.exists() else set()
    with open(out, "a") as sink, open(started, "a") as log:
        for key, line in records(args.files, args.worker, args.workers):
            k = f"{key[0]}:{key[1]}"
            if tuple(key) in done or k in skip:  # done, or the search that killed this worker last time
                continue
            log.write(k + "\n")
            log.flush()
            signal.alarm(TIMEOUT)
            try:
                lab = label(json.loads(line), args.iterations, cfg)
            except BaseException as e:  # engine panics
                lab = {"error": f"{type(e).__name__}: {e}"[:200]}
            signal.alarm(0)
            if lab is not None:
                sink.write(json.dumps({"key": key} | lab) + "\n")
                sink.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--iterations", type=int, default=1_000_000)
    p.add_argument("--eval-config", default=None, help="default: porygon.search's tuned evaluation")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--worker", type=int, default=-1)
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.worker >= 0:
        worker(args)
        return
    cmd = [sys.executable, __file__, *sys.argv[1:]]
    procs = {i: subprocess.Popen(cmd + ["--worker", str(i)]) for i in range(args.workers)}
    restarts = 0
    while procs:
        time.sleep(5)
        for i, proc in list(procs.items()):
            if proc.poll() is None:
                continue
            del procs[i]
            if proc.returncode != 0 and restarts < 50 * args.workers:
                restarts += 1
                procs[i] = subprocess.Popen(cmd + ["--worker", str(i)])
    n = sum(1 for f in Path(args.out).glob("labels-*.jsonl") for _ in open(f))
    print(f"{n} labels, {restarts} restarts")


if __name__ == "__main__":
    main()
