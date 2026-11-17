"""fpcal reading (PREREG $WSP/fpcal/PREREG.txt): raw net vs Foul Play at several Foul Play budgets on the SAME team
pairs (one duplicate-server seed). Per run: pair-mean win rate of our net with a 95% interval; per pair of runs: the
paired difference over pairs whose two teams match in both runs (species lists of both sides), 95% interval; and the
least-squares slope of the win rate on log4(budget), with an interval from the per-run standard errors.

    python fpcal_read.py LABEL=BUDGET=RUN_DIR [LABEL=BUDGET=RUN_DIR ...]   # BUDGET = iterations per state (1000 ms)
"""
import json
import math
import sys
from pathlib import Path


def load(run: Path) -> dict:
    """pair -> {"score": our mean over the pair's games, "key": the teams, "n": games}"""
    pairs: dict = {}
    for line in (run / "battles.jsonl").open():
        r = json.loads(line)
        ours = r["p1"] if not r["p1"].startswith("fp") else r["p2"]
        s = 0.5 if not r.get("winner") else float(r["winner"] == ours)
        key = tuple(sorted(tuple(sorted(m["species"] for m in r[k])) for k in ("p1team", "p2team")))
        p = pairs.setdefault(r["pair"], {"s": [], "key": key})
        p["s"].append(s)
        if p["key"] != key:
            p["key"] = None  # the two games of one pair should share the teams
    return {k: {"score": sum(v["s"]) / len(v["s"]), "key": v["key"], "n": len(v["s"])} for k, v in pairs.items()}


def ci(xs: list) -> tuple:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, 1.96 * sd / math.sqrt(n), n


def main() -> None:
    runs = []
    for arg in sys.argv[1:]:
        label, budget, d = arg.split("=", 2)
        runs.append((label, float(budget), Path(d), load(Path(d))))
    stats = []
    for label, budget, d, P in runs:
        full = [v["score"] for v in P.values() if v["n"] == 2]
        m, h, n = ci(full)
        stats.append((label, budget, m, h / 1.96))
        print(f"{label}: Foul Play {budget:,.0f} iterations/state: {100 * m:.1f}% ± {100 * h:.1f}% ({n} complete pairs) [{d}]")
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            (la, _, _, A), (lb, _, _, B) = runs[i], runs[j]
            common = [p for p in A if p in B and A[p]["n"] == 2 and B[p]["n"] == 2]
            same = [p for p in common if A[p]["key"] is not None and A[p]["key"] == B[p]["key"]]
            if not same:
                print(f"{la} - {lb}: no pair with matching teams ({len(common)} common indices)")
                continue
            m, h, n = ci([A[p]["score"] - B[p]["score"] for p in same])
            print(f"{la} - {lb}: {100 * m:+.1f} points, 95% [{100 * (m - h):+.1f}, {100 * (m + h):+.1f}] "
                  f"(paired over {n} pairs with the same teams; {len(common)} common indices)")
    if len(stats) >= 2:
        xs = [math.log(b, 4) for _, b, _, _ in stats]
        ys = [m for _, _, m, _ in stats]
        ws = [1 / se ** 2 for _, _, _, se in stats]
        sw = sum(ws); xm = sum(w * x for w, x in zip(ws, xs)) / sw; ym = sum(w * y for w, y in zip(ws, ys)) / sw
        sxx = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
        b = sum(w * (x - xm) * (y - ym) for w, x, y in zip(ws, xs, ys)) / sxx
        se = math.sqrt(1 / sxx)
        print(f"slope: {100 * b:+.1f} points per 4x Foul Play budget, 95% [{100 * (b - 1.96 * se):+.1f}, "
              f"{100 * (b + 1.96 * se):+.1f}] (weighted least squares on log4 budget; runs treated as independent)")


if __name__ == "__main__":
    main()
