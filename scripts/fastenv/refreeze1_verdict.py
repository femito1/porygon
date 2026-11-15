"""Re-freeze #1 verdict (REFREEZE-PREREG.md md5 ad1911d2 §4), applied as written to a finished slot run R:
void iff HARNESS_FAIL or > 2% search fallbacks on either side (replay with the same seeds after the fix); else ADOPT
iff arm 1's lower 95% bound (pair means) > 50%, NOT ADOPTED otherwise. Also reported (not part of the rule): the
paired diff H(cand) - H(G9p-cd) per duplicate pair, which is 2 * pair mean - 1 (both bots play both team sides of
every pair), so its 95% interval is the arm's interval doubled; n below the fixed 500 pairs is flagged.

    python refreeze1_verdict.py RUN_DIR [--pairs 500]   # prints the verdict; exit 0 (ADOPT / NOT ADOPTED), 3 (VOID)
"""
import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path)
    p.add_argument("--pairs", type=int, default=500)
    a = p.parse_args()
    r = json.loads((a.run / "result.json").read_text())
    if len(r["arms"]) != 1:
        raise SystemExit(f"expected one arm, got {len(r['arms'])}")
    s = r["arms"][0]
    harness_fail = (a.run / "HARNESS_FAIL").exists()
    fb = [s.get("fallback_rate_a"), s.get("fallback_rate_b")]
    void_fb = any(x is not None and x > 0.02 for x in fb)
    n, m, hw = s["pairs"], s["mean"], s["hw"]
    lo = s["lo"]
    if harness_fail or void_fb:
        verdict = "VOID (" + ", ".join(w for w, c in (("HARNESS_FAIL", harness_fail), ("fallbacks > 2%", void_fb))
                                       if c) + "): replay with the same seeds after the fix"
        rc = 3
    else:
        verdict = "ADOPT" if (n > 0 and math.isfinite(lo) and lo > 0.5) else "NOT ADOPTED"
        rc = 0
    flag = "" if n == a.pairs else f" [n = {n} pairs, not the fixed {a.pairs}: orchestrator to confirm]"
    fbs = "/".join("-" if x is None else f"{x:.3%}" for x in fb)
    ov = s.get("override_rate")
    ovo = s.get("override_rate_opp")
    lines = [
        f"RE-FREEZE #1 R VERDICT: {verdict}{flag}",
        f"  H(cand) vs H(G9p-cd): {m:.2%} ± {hw:.2%} [{lo:.2%}, {s['hi']:.2%}] ({n} pairs, {s['games']} games; "
        f"won both {s['won_both']}, split {s['split']}, lost both {s['lost_both']}; incomplete {s['incomplete_pairs']})",
        f"  rule: adopt iff lower 95% > 50% -> lower {lo:.2%}",
        f"  paired diff H(cand) - H(G9p-cd) per pair: {2 * m - 1:+.1%} ± {2 * hw:.1%} "
        f"(= 2 x pair mean - 1; points of win rate)",
        f"  overrides ours {'-' if ov is None else f'{ov:.1%}'} of {s['gate']['decisions']}, opponent "
        f"{'-' if ovo is None else f'{ovo:.1%}'} of {s['gate_opp']['decisions']}; fallbacks ours/theirs {fbs}; "
        f"{s['seconds'] / 3600:.2f} h of play; HARNESS_FAIL {'yes' if harness_fail else 'no'}",
        f"  spec: {s['spec']}",
        f"  run: {a.run}",
    ]
    out = "\n".join(lines)
    print(out)
    (a.run / "VERDICT.txt").write_text(out + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
