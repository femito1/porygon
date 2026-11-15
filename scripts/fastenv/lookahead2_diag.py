"""Override diagnostics of a lookahead2 run (informational, never a decision): from a netsearch_slot.sh run directory,
arm K's gate logs (one line per searched decision) and client results (one line per game).

    python lookahead2_diag.py RUN [--arm 1] [--json OUT]
Prints: override rate by the kind of the net's draw and of the lookahead's pick (switch / move / tera), by gap bin and
by turn; overrides per game and A's game score by that count (confounded: long games have more decisions); leaves and
seconds per decision.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def kind(a: int) -> str:
    return "switch" if a < 6 else "tera" if a >= 22 else "move"


def load(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def diag(run: Path, arm: int = 1) -> dict:
    gate, games = [], {}
    for b in sorted((run / f"arm-{arm}").glob("block-*"), key=lambda p: int(p.name.split("-")[1])):
        gate += load(b / "gate.jsonl")
        for r in load(b / "results.jsonl"):
            games[r["battle"]] = {"win": 1.0, "loss": 0.0, "tie": 0.5}.get(r.get("status"), math.nan)
    out = {"decisions": len(gate), "games": len(games)}
    if not gate:
        return out
    over = [g for g in gate if g.get("override")]
    out["override_rate"] = len(over) / len(gate)
    out["by_kind"] = {f"{k1}->{k2}": n for (k1, k2), n in Counter((kind(g["net"]), kind(g["search"])) for g in over).most_common()}
    out["net_kind_rate"] = {k: sum(1 for g in over if kind(g["net"]) == k) / max(1, sum(1 for g in gate if kind(g["net"]) == k))
                            for k in ("move", "switch", "tera")}
    bins = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]
    gaps = np.array([g["gap"] for g in over if g.get("gap") is not None])
    out["gap_bins"] = {f"{lo}-{hi}": int(((gaps > lo) & (gaps <= hi)).sum()) for lo, hi in zip(bins, bins[1:])}
    turns = defaultdict(lambda: [0, 0])
    for g in gate:
        t = min(int(g["turn"]) // 10 * 10, 40)
        turns[t][0] += 1
        turns[t][1] += bool(g.get("override"))
    out["by_turn"] = {f"{t}-{t + 9 if t < 40 else 'end'}": round(o / n, 4) for t, (n, o) in sorted(turns.items())}
    per_game = Counter(g["battle"] for g in over)
    buckets = defaultdict(list)
    for battle, score in games.items():
        n = per_game.get(battle, 0)
        buckets[min(n, 6)].append(score)
    out["score_by_overrides_per_game"] = {f"{k}{'+' if k == 6 else ''}": {"games": len(v), "score": round(float(np.nanmean(v)), 4)}
                                          for k, v in sorted(buckets.items())}

    def q(key, p):
        xs = [g[key] for g in gate if g.get(key) is not None]
        return round(float(np.quantile(xs, p)), 3) if xs else None
    out["per_decision"] = {k: {"median": q(k, .5), "q90": q(k, .9), "max": q(k, 1.0)}
                           for k in ("leaves", "t", "t_fwd", "cpu_expand", "worlds", "cols")}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--arm", type=int, default=1)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    d = diag(a.run, a.arm)
    print(json.dumps(d, indent=1))
    if a.json:
        a.json.write_text(json.dumps(d, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
