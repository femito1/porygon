"""wsp P3 reading (PREREG $W/council/p3-gen4/PREREG.txt): arm (i) the incumbent gen4 bot and arm (ii) raw R1b-F-cd,
each vs Foul Play gen4 at a FIXED 200,000 iterations per state, on the SAME duplicate-server team pairs (one seed).

    python p3_read.py pairs RUN_DIR                  # prints the number of complete pairs (both games logged)
    python p3_read.py samepairs RUN_A RUN_B          # prints "<common complete pairs> <same teams> <different teams>"
    python p3_read.py verdict ARM_I_DIR ARM_II_DIR [NOTE ...]   # VERDICT text on stdout (NOTEs appended as lines)

Scores come from the server's duplicate log RUN_DIR/battles.jsonl (one line per finished battle: pair, game, winner,
players, turns, both true teams). Our side = the player whose name does not start with "fp"; a win 1, a loss 0, a
tie 0.5. A pair is complete when both of its games are logged; its score is the mean of the two.
  - each arm: the mean over its complete pairs, 95% interval mean ± 1.96 sd / √n (pair means);
  - Δ = (ii) − (i) per team pair, over the pairs complete in both arms whose teams match (species of both sides),
    95% interval from the per-pair differences;
  - rule (verbatim): BUILD-LA2-GEN4 iff Δ ≥ −2.0 (point estimate); else STOP-GEN4-NET;
  - informational: arm (ii) per game, games < 25 turns vs ≥ 25 turns (games treated as independent); arm (i) too.
"""
import json
import math
import sys
from pathlib import Path

MARGIN = -2.0  # points: BUILD-LA2-GEN4 iff Δ >= MARGIN (point estimate)
SHORT = 25     # turns: a short game has < 25 turns


def load(run: str) -> dict:
    """pair -> {"s": [our score per game], "t": [turns per game], "g": {game indices}, "key": teams or None}"""
    pairs: dict = {}
    for line in (Path(run) / "battles.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        ours = r["p2"] if r["p1"].startswith("fp") else r["p1"]
        s = 0.5 if not r.get("winner") else float(r["winner"] == ours)
        key = tuple(sorted(tuple(sorted(m["species"] for m in r[k])) for k in ("p1team", "p2team")))
        p = pairs.setdefault(r["pair"], {"s": [], "t": [], "g": set(), "key": key})
        p["s"].append(s)
        p["t"].append(int(r.get("turns") or 0))
        p["g"].add(r.get("game"))
        if p["key"] != key:
            p["key"] = None  # the two games of one pair should share the teams
    return pairs


def complete(P: dict) -> dict:
    return {k: v for k, v in P.items() if len(v["s"]) == 2 and v["g"] == {0, 1}}


def ci(xs: list) -> tuple:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, 1.96 * sd / math.sqrt(n), n


def pct(m: float, h: float) -> str:
    return f"{100 * m:.2f}% [{100 * (m - h):.2f}, {100 * (m + h):.2f}]"


def split(P: dict) -> str:
    out = []
    for label, keep in ((f"short (< {SHORT} turns)", lambda t: t < SHORT), (f"long (≥ {SHORT})", lambda t: t >= SHORT)):
        xs = [s for v in P.values() for s, t in zip(v["s"], v["t"]) if keep(t)]
        out.append(f"{label} {pct(*ci(xs)[:2])} ({len(xs)} games)" if len(xs) > 1 else f"{label}: {len(xs)} games")
    return "; ".join(out)


def common(A: dict, B: dict) -> tuple:
    ca, cb = complete(A), complete(B)
    both = [p for p in ca if p in cb]
    same = [p for p in both if ca[p]["key"] is not None and ca[p]["key"] == cb[p]["key"]]
    return ca, cb, both, same


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "pairs":
        print(len(complete(load(sys.argv[2]))))
        return
    if len(sys.argv) >= 4 and sys.argv[1] == "samepairs":
        _, _, both, same = common(load(sys.argv[2]), load(sys.argv[3]))
        print(len(both), len(same), len(both) - len(same))
        return
    if len(sys.argv) < 4 or sys.argv[1] != "verdict":
        raise SystemExit(__doc__)
    ri, rii, notes = sys.argv[2], sys.argv[3], sys.argv[4:]
    A, B = load(ri), load(rii)
    ca, cb, both, same = common(A, B)
    if len(same) < 2:
        raise SystemExit(f"only {len(same)} common team pairs ({len(both)} common complete indices): no verdict")
    diffs = [sum(cb[p]["s"]) / 2 - sum(ca[p]["s"]) / 2 for p in same]  # multiples of 0.25: exact in binary
    d, dh, dn = ci(diffs)
    # the rule on the point estimate, decided exactly: 100 · Σdiffs / n ≥ MARGIN  ⟺  100 · Σdiffs ≥ MARGIN · n
    verdict = "BUILD-LA2-GEN4" if 100 * sum(diffs) >= MARGIN * dn else "STOP-GEN4-NET"
    mi, hi, ni = ci([sum(v["s"]) / 2 for v in ca.values()])
    mii, hii, nii = ci([sum(v["s"]) / 2 for v in cb.values()])
    print(verdict)
    print(f"Δ = (ii) − (i) = {100 * d:+.2f} points, 95% [{100 * (d - dh):+.2f}, {100 * (d + dh):+.2f}] (paired over {dn} "
          f"team pairs complete in both arms with the same teams; {len(both)} common complete pair indices); rule: "
          f"BUILD-LA2-GEN4 iff Δ ≥ {MARGIN:.1f} (point estimate), else STOP-GEN4-NET")
    print(f"(i) incumbent gen4 bot (search + fitted eval + v1@500M prior, fp4-04 config) vs Foul Play gen4 at a FIXED "
          f"200,000 iterations per state: {pct(mi, hi)} ({ni} complete pairs, {2 * ni} games; pair-mean 95%)")
    print(f"(ii) raw R1b-F-cd at T 0.5 vs Foul Play gen4 at a FIXED 200,000 iterations per state: {pct(mii, hii)} "
          f"({nii} complete pairs, {2 * nii} games; pair-mean 95%)")
    print(f"(ii) by game length, informational (per game, games treated as independent): {split(cb)}")
    print(f"(i) by game length, context (per game, games treated as independent): {split(ca)}")
    print(f"runs: (i) {ri}  (ii) {rii}")
    for n in notes:
        print(n)


if __name__ == "__main__":
    main()
