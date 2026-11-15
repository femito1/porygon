"""Verdict of the Bet 1 main run (lookahead2 vs the same raw net; PREREG runs/la2-bet1/PREREG.txt), read mechanically
from the run directory netsearch_slot.sh wrote: the sequential decision (arm-1/SEQ_DECISION, from vendor/seq_monitor.py
on the pre-registered design), the slot report (netsearch_slot.py report: pairs, naive 95% interval, overrides,
fallbacks, void flag) and the gate log (leaves and seconds per searched decision).

    python lookahead2_verdict.py RUN_DIR [--json OUT]     # prints VERDICT.txt; line 1 = GO | NO-GO | VOID: ...
GO iff the design's SUCCESS bound was crossed (the repeated 95% CI excludes 50% from below at the stopping look).
NO-GO iff its FUTILITY bound was crossed (or the last look did not reach SUCCESS). VOID iff HARNESS_FAIL, more than 2%
fallbacks on either side, or no sequential decision (the chain then stops; replay with the same seeds after the fix).
"""
import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def slot_module():
    spec = importlib.util.spec_from_file_location("netsearch_slot", HERE / "netsearch_slot.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gate_stats(run: Path) -> dict:
    """Per searched decision of arm 1 (all blocks): leaves, wall seconds, forward seconds, worker CPU seconds."""
    rows = []
    for f in sorted((run / "arm-1").glob("block-*/gate.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return {"decisions": 0}

    def q(key, p):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return float(np.quantile(xs, p)) if xs else math.nan

    over = [r for r in rows if r.get("override")]
    return {"decisions": len(rows), "override_rate": len(over) / len(rows),
            "gap_of_overrides_median": float(np.median([r["gap"] for r in over])) if over else None,
            "leaves_median": q("leaves", .5), "leaves_q90": q("leaves", .9), "leaves_max": q("leaves", 1.0),
            "seconds_median": q("t", .5), "seconds_q90": q("t", .9), "seconds_max": q("t", 1.0),
            "forward_median": q("t_fwd", .5), "cpu_median": q("cpu_expand", .5), "cpu_q90": q("cpu_expand", .9),
            "failed_worlds": int(sum(r.get("failed_worlds", 0) for r in rows)),
            "engine_errors": int(sum(r.get("errors", 0) for r in rows)),
            "undefined_gap": sum(r.get("gap") is None for r in rows)}


def verdict(run: Path) -> tuple[list[str], dict]:
    SL = slot_module()
    rep = SL.report(run, go=0.515, tie=0.01, void_fallbacks=0.02)
    arm = rep["arms"][0]
    seq = arm.get("seq_decision")
    looks = sorted((run / "arm-1").glob("seq-look-*.txt"), key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    last = looks[-1].read_text().splitlines() if looks else []
    est = next((ln.strip() for ln in last if ln.strip().startswith("estimate:")), None)
    harness_fail = (run / "HARNESS_FAIL").exists()
    if harness_fail:
        head = "VOID: harness failure (HARNESS_FAIL in the run dir); replay with the same seeds after the fix"
    elif arm["void"]:
        head = (f"VOID: fallbacks above 2% (ours {arm['fallback_rate_a']}, theirs {arm['fallback_rate_b']}); replay "
                "with the same seeds after the fix")
    elif not seq:
        head = f"VOID: no sequential decision after {arm['pairs']} pairs ({arm['blocks']} blocks); see arm-1/seq-look-*.txt"
    elif seq.startswith("DECISION SUCCESS"):
        head = "GO: lookahead2 beats the same raw net (the design's repeated 95% CI excludes 50% from below)"
    elif seq.startswith("DECISION FUTILITY"):
        head = "NO-GO: lookahead2 does not beat the same raw net by the design's standard (futility bound crossed)"
    else:
        head = f"VOID: unreadable sequential decision {seq!r}"
    g = gate_stats(run)
    wr = f"{arm['mean']:.2%} ± {arm['hw']:.2%}" if arm["pairs"] and math.isfinite(arm["hw"]) else "-"
    lines = [head,
             f"  sequential: {seq or '-'}",
             f"  {est or 'estimate: -'}",
             f"  naive (fixed-n style, informational): A {wr} [{arm['lo']:.2%}, {arm['hi']:.2%}] on {arm['pairs']} pairs "
             f"({arm['games']} games, {arm['blocks']} blocks; won both {arm['won_both']}, split {arm['split']}, lost both "
             f"{arm['lost_both']}; incomplete {arm['incomplete_pairs']}); {arm['seconds'] / 3600:.2f} h",
             f"  overrides {g.get('override_rate', math.nan):.1%} of {g['decisions']} searched decisions (gap of overrides "
             f"median {g.get('gap_of_overrides_median')}); fallbacks ours/theirs {arm['fallback_rate_a']}/{arm['fallback_rate_b']}",
             f"  per searched decision: leaves median {g.get('leaves_median', math.nan):.0f} (q90 {g.get('leaves_q90', math.nan):.0f}, "
             f"max {g.get('leaves_max', math.nan):.0f}); wall s median {g.get('seconds_median', math.nan):.2f} (q90 "
             f"{g.get('seconds_q90', math.nan):.2f}, max {g.get('seconds_max', math.nan):.2f}); forward s median "
             f"{g.get('forward_median', math.nan):.3f}; worker CPU s median {g.get('cpu_median', math.nan):.2f} (q90 "
             f"{g.get('cpu_q90', math.nan):.2f}); failed worlds {g.get('failed_worlds')}, engine errors {g.get('engine_errors')}",
             f"  A = {arm['spec']}",
             f"  B = {rep['opponent']}",
             f"  run {run}"]
    return lines, {"verdict": head, "report": rep, "gate": g, "seq_decision": seq, "estimate": est}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    lines, data = verdict(a.run)
    print("\n".join(lines))
    if a.json:
        a.json.write_text(json.dumps(data, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
