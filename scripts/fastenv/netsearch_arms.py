"""The Stage 2 margins by the pre-registered rule (NETSEARCH-DESIGN.md §3.1): delta_1 < delta_2 < delta_3 are the
margins where the judged loss L(delta) = -(hybrid - pure search) first reaches 0.0005, 0.001 and 0.002 Q per decision,
by linear interpolation on the margin grid of a netsearch_analyze.py --json output (judge R; judge B if R is
missing); a level not reached up to delta = 0.2 gives 0.2.

    python netsearch_arms.py s2all.json [--levels 0.0005 0.001 0.002] [--cap 0.2] [--json OUT]
"""
import argparse
import json
import math
import sys


def curve(analysis: dict, judge: str | None = None) -> tuple[str, list[tuple[float, float, float]]]:
    """(judge, [(margin, L, override rate at T 0.5)]) over the finite margins >= 0, in increasing order."""
    rows = analysis["margins"]
    judges = [judge] if judge else ["R", "B"]
    for j in judges:
        pts = []
        for r in rows:
            jr = r.get(f"judge_{j}") or {}
            hs = jr.get("hybrid_minus_search")
            if math.isfinite(r["margin"]) and r["margin"] >= 0 and hs is not None and math.isfinite(hs):
                pts.append((float(r["margin"]), -float(hs), float(r["override_T05"])))
        if pts:
            return j, sorted(pts)
    raise SystemExit(f"no judge {judges} in the margin grid")


def reach(pts: list[tuple[float, float, float]], level: float, cap: float) -> tuple[float, float]:
    """(margin, interpolated override rate) where L first reaches `level` (scanning margins upwards)."""
    for i, (d, loss, ov) in enumerate(pts):
        if d > cap:
            break
        if loss >= level:
            if i == 0:
                return d, ov
            d0, l0, o0 = pts[i - 1]
            f = (level - l0) / (loss - l0)
            return d0 + f * (d - d0), o0 + f * (ov - o0)
    capped = [p for p in pts if p[0] <= cap]
    return cap, capped[-1][2] if capped else math.nan


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("analysis")
    p.add_argument("--levels", type=float, nargs="+", default=[0.0005, 0.001, 0.002])
    p.add_argument("--cap", type=float, default=0.2)
    p.add_argument("--judge", choices=["R", "B"])
    p.add_argument("--json")
    a = p.parse_args()
    d = json.load(open(a.analysis))
    d = d.get("all", d)
    judge, pts = curve(d, a.judge)
    arms = []
    for lv in a.levels:
        m, ov = reach(pts, lv, a.cap)
        arms.append({"level": lv, "margin": round(m, 4), "override_T05": round(ov, 3)})
    print(f"{a.analysis}: judge {judge}, {d.get('positions')} positions; L(delta) = -(hybrid - search):")
    print("  " + "  ".join(f"{m:g}: {loss:+.4f}" for m, loss, _ in pts))
    for x in arms:
        print(f"  L = {x['level']:g} at delta = {x['margin']:.4f} (override rate at T 0.5 ≈ {x['override_T05']:.1%})")
    if a.json:
        json.dump({"analysis": a.analysis, "judge": judge, "arms": arms, "curve": pts}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
