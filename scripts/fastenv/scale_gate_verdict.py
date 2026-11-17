"""Verdict of the scale gate (docs/netsearch/SCALE-GATE-PREREG.md; runs/scale-gate/PREREG.txt), read mechanically from
the two run directories netsearch_slot.sh wrote:
  A-run: arm 1 = D4 vs D1 (T1);   B-run: arm 1 = lookahead2 vs D1 (T2), arm 2 = raw vs D1 (T3, informational).

    python scale_gate_verdict.py part A|B RUN --json OUT.json      # one chain's part: line 1 = T1|T2 status
    python scale_gate_verdict.py combine A.json [B.json]            # line 1 = GO | MIXED | FLAT | VOID: ...
T1 SUCCESS = the design's SUCCESS bound crossed by arm 1 of A (D4 beats D1); FUTILITY (or the last look) = not.
T2 SUCCESS = lookahead2 beats D1 ("1x WORSE"). GO iff T1 SUCCESS and T2 not SUCCESS; MIXED iff both SUCCESS; FLAT iff
T1 not SUCCESS; VOID iff HARNESS_FAIL, > 2% fallbacks on any searching side, or no decision (T1; T2 when T1 SUCCESS).
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _rows(files) -> list[dict]:
    out = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def side_stats(rows: list[dict]) -> dict:
    if not rows:
        return {"decisions": 0}

    def q(key, p):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.quantile(xs, p)), 4) if xs else None

    over = [r for r in rows if r.get("override")]
    s = {"decisions": len(rows), "override_rate": round(len(over) / len(rows), 4),
         "gap_of_overrides_median": float(np.median([r["gap"] for r in over])) if over else None,
         "failed_worlds": int(sum(r.get("failed_worlds", 0) for r in rows)),
         "engine_errors": int(sum(r.get("errors", 0) for r in rows))}
    for k in ("leaves", "extra_leaves", "depth", "t", "t_fwd"):
        s[k] = {"median": q(k, .5), "q90": q(k, .9), "max": q(k, 1.0)}
    return s


def part(mode: str, run: Path) -> tuple[list[str], dict]:
    import netsearch_slot as SL
    rep = SL.report(run, go=0.515, tie=0.01, void_fallbacks=0.02)
    harness_fail = (run / "HARNESS_FAIL").exists()
    arms = []
    for a in rep["arms"]:
        k = a["arm"]
        seq = a.get("seq_decision")
        looks = sorted((run / f"arm-{k}").glob("seq-look-*.txt"), key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
        last = looks[-1].read_text().splitlines() if looks else []
        est = next((ln.strip() for ln in last if ln.strip().startswith("estimate:")), None)
        status = ("SUCCESS" if seq and seq.startswith("DECISION SUCCESS") else
                  "FUTILITY" if seq and seq.startswith("DECISION FUTILITY") else None)
        arms.append({"arm": k, "spec": a["spec"], "status": status, "seq": seq, "estimate": est,
                     "pairs": a["pairs"], "games": a["games"], "mean": a["mean"], "lo": a["lo"], "hi": a["hi"],
                     "won_both": a["won_both"], "split": a["split"], "lost_both": a["lost_both"],
                     "fallback_rate_a": a["fallback_rate_a"], "fallback_rate_b": a["fallback_rate_b"],
                     "void": a["void"], "stopped": a["stopped"], "hours": a["seconds"] / 3600,
                     "side": side_stats(_rows(sorted((run / f"arm-{k}").glob("block-*/gate.jsonl")))),
                     "opponent_side": side_stats(_rows(sorted((run / f"arm-{k}").glob("block-*/gate-opp.jsonl"))))})
    void = harness_fail or any(a["void"] for a in arms)
    key = {"A": "T1", "B": "T2"}[mode]
    st = arms[0]["status"] if arms else None
    if harness_fail:
        head = f"{key} VOID: HARNESS_FAIL in {run}"
    elif void:
        head = f"{key} VOID: fallbacks above 2% on a searching side"
    elif st is None:
        head = f"{key} VOID: no sequential decision on arm 1"
    else:
        head = {("A", "SUCCESS"): "T1 SUCCESS: D4 beats D1", ("A", "FUTILITY"): "T1 FUTILITY: D4 not shown better than D1",
                ("B", "SUCCESS"): "T2 SUCCESS: lookahead2 beats D1 (1x WORSE)",
                ("B", "FUTILITY"): "T2 FUTILITY: 1x not shown worse than lookahead2"}[(mode, st)]
    lines = [head]
    for a in arms:
        lines.append(f"  arm {a['arm']}: {a['status'] or '-'} | {a['seq'] or '-'} | {a['estimate'] or 'estimate: -'}")
        lines.append(f"    naive {a['mean']:.2%} [{a['lo']:.2%}, {a['hi']:.2%}] on {a['pairs']} pairs ({a['games']} games; won "
                     f"both {a['won_both']}, split {a['split']}, lost both {a['lost_both']}); fallbacks "
                     f"{a['fallback_rate_a']}/{a['fallback_rate_b']}; {a['hours']:.2f} h | {a['spec']}")
        for nm in ("side", "opponent_side"):
            s = a[nm]
            if s.get("decisions"):
                lines.append(f"    {nm}: {s['decisions']} searched, overrides {s['override_rate']:.1%} (gap median "
                             f"{s['gap_of_overrides_median']}), leaves {s['leaves']}, extra {s['extra_leaves']}, depth "
                             f"{s['depth']}, wall s {s['t']}, forward s {s['t_fwd']}, failed worlds {s['failed_worlds']}")
    lines.append(f"  opponent {rep['opponent']}; run {run}")
    return lines, {"mode": mode, "head": head, "status": st, "void": void or st is None, "arms": arms}


def combine(a: dict, b: dict | None) -> str:
    if a["void"]:
        return f"VOID: {a['head']}"
    if a["status"] != "SUCCESS":
        return f"FLAT: 4x does not beat 1x by the design's standard ({a['head']}); stop this route"
    if b is None:
        return "PENDING: T1 SUCCESS, chain B not finished"
    if b["void"]:
        return f"VOID: T1 SUCCESS but {b['head']}"
    if b["status"] == "SUCCESS":
        return f"MIXED: T1 SUCCESS (D4 beats D1) and T2 SUCCESS (lookahead2 beats D1); main decides"
    return "GO: D4 beats D1 (T1 SUCCESS) and 1x is not shown worse than lookahead2 (T2 not SUCCESS)"


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("part")
    q.add_argument("mode", choices=["A", "B"])
    q.add_argument("run", type=Path)
    q.add_argument("--json", type=Path, required=True)
    c = sub.add_parser("combine")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path, nargs="?")
    x = p.parse_args()
    if x.cmd == "part":
        lines, data = part(x.mode, x.run)
        x.json.write_text(json.dumps(data, indent=1, default=str) + "\n")
        print("\n".join(lines))
        return 0
    a = json.loads(x.a.read_text())
    b = json.loads(x.b.read_text()) if x.b and x.b.exists() else None
    print(combine(a, b))
    print(f"  {a['head']}")
    if b:
        print(f"  {b['head']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
