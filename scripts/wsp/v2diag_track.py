"""Append one row per tracked checkpoint to a markdown table (+ flags) from v2diag_decay / v2diag_switch outputs.

    python scripts/wsp/v2diag_track.py --decay D/decay.json --switch D/switch_sensitivity.json --ckpt NAME
        --table TABLE.md --flags FLAGS.txt
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

HEADER = ("| step | entropy | P(top ≥ 0.9) | top1−top2 gap | voluntary switch / attack / status | all switch / attack / status "
          "| threat / offence slope | pg mean (frac > 0, n upd) in the 2.5M window | value AUC | top-1 = 7.5M | KL vs 7.5M | flags | written (UTC) |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")


def pg_window(segments, step, width=2.5e6):
    """Mean PPO surrogate pg over this run's updates with steps in (step - width, step]. `segments`: list of
    (train.log path, since, until) — the run's lines are those with since <= log time < until in that file (a run can
    move between job directories)."""
    xs = []
    for path, since, until in segments:
        try:
            for line in open(path):
                m = re.match(r"(\S+ \S+) upd \d+ steps ([\d,]+) .* pg ([-+0-9.e]+) ", line)
                if not m or not (since <= m.group(1) < until):
                    continue
                st = int(m.group(2).replace(",", ""))
                if step - width < st <= step:
                    xs.append(float(m.group(3)))
        except OSError:
            continue
    if not xs:
        return "—"
    return f"{sum(xs) / len(xs):+.4f} ({sum(x > 0 for x in xs) / len(xs):.2f}, {len(xs)})"


def main():
    ap = argparse.ArgumentParser()
    for k in ("--decay", "--switch", "--ckpt", "--table", "--flags"):
        ap.add_argument(k, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--trainlog", action="append", default=[],
                    help="PATH|SINCE|UNTIL (log-time bounds 'YYYY-MM-DD HH:MM:SS'); repeat for each segment of the run")
    ap.add_argument("--boundary", default="", help="STEP:TEXT — write a marker row before the first row at or after STEP")
    ap.add_argument("--ref", default="", help="reference checkpoint name in decay.json (relative flags)")
    ap.add_argument("--status-abs", type=float, default=0.2)
    ap.add_argument("--top09-abs", type=float, default=0.2)
    ap.add_argument("--entropy-rise", type=float, default=0.3)
    ap.add_argument("--auc-drop", type=float, default=0.03)
    ap.add_argument("--relative-top09", action="store_true", help="gen9: P(top>=0.9) threshold = half the reference's")
    ap.add_argument("--collapse", action="store_true",
                    help="also flag collapse: entropy < ref - 0.6, P(top>=0.9) > max(3 x ref, 0.4), voluntary switch < ref / 2")
    ap.add_argument("--auc-min-step", type=float, default=0, help="no value-AUC flag before this step (fresh value head)")
    ap.add_argument("--since", default="0000", help="segment start, 'YYYY-MM-DD HH:MM:SS' as in train.log")
    a = ap.parse_args()
    d = json.load(open(a.decay))[a.ckpt]
    s = json.load(open(a.switch))[a.ckpt]
    fl = []
    r = json.load(open(a.decay)).get(a.ref) if a.ref else None
    st_thr = a.status_abs if r is None else max(a.status_abs, r["mass_voluntary"]["status"] + 0.10)
    if d["mass_voluntary"]["status"] > st_thr:
        fl.append(f"voluntary status mass {d['mass_voluntary']['status']:.2f} > {st_thr:.2f}")
    top_thr = 0.5 * r["p_top_ge_0.9"] if (r is not None and a.relative_top09) else a.top09_abs
    if d["p_top_ge_0.9"] < top_thr:
        fl.append(f"P(top>=0.9) {d['p_top_ge_0.9']:.2f} < {top_thr:.2f}")
    if r is not None and d["entropy"] > r["entropy"] + a.entropy_rise:
        fl.append(f"entropy {d['entropy']:.2f} > ref {r['entropy']:.2f} + {a.entropy_rise}")
    if r is not None and a.collapse:
        if d["entropy"] < r["entropy"] - 0.6:
            fl.append(f"COLLAPSE entropy {d['entropy']:.2f} < ref {r['entropy']:.2f} - 0.6")
        thr = max(3 * r["p_top_ge_0.9"], 0.4)
        if d["p_top_ge_0.9"] > thr:
            fl.append(f"COLLAPSE P(top>=0.9) {d['p_top_ge_0.9']:.2f} > {thr:.2f}")
        sw = r["mass_voluntary"]["switch"]
        if d["mass_voluntary"]["switch"] < 0.5 * sw:
            fl.append(f"voluntary switch mass {d['mass_voluntary']['switch']:.2f} < half the ref's {sw:.2f}")
    if r is not None and a.step >= a.auc_min_step and d["value_auc"] < r["value_auc"] - a.auc_drop:
        fl.append(f"value AUC {d['value_auc']:.3f} < ref {r['value_auc']:.3f} - {a.auc_drop}")
    mv, ma = d["mass_voluntary"], d["mass_all"]
    row = (f"| {a.step / 1e6:.2f}M | {d['entropy']:.2f} | {d['p_top_ge_0.9']:.2f} | {d['top_gap']:.2f} | "
           f"{mv['switch']:.2f} / {mv['attack']:.2f} / {mv['status']:.2f} | {ma['switch']:.2f} / {ma['attack']:.2f} / {ma['status']:.2f} | "
           f"{s['threat_slope']:+.2f} / {s['offence_slope']:+.2f} | {pg_window([tuple((seg.split("|") + ["", "9999"])[:3]) for seg in a.trainlog], a.step)} | {d['value_auc']:.3f} | {d['top1_eq_ref']:.2f} | "
           f"{d['KL_ref_to_ck']:.2f} | {'; '.join(fl) or '—'} | {datetime.now(timezone.utc):%m-%d %H:%M} |\n")
    t = Path(a.table)
    if not t.exists():
        t.write_text(HEADER)
    if a.boundary:
        bstep, btext = a.boundary.split(":", 1)
        if a.step >= float(bstep) and f"BOUNDARY {btext}" not in t.read_text():
            with open(t, "a") as f:
                f.write(f"| BOUNDARY {btext} |" + " |" * 12 + "\n")
    with open(t, "a") as f:
        f.write(row)
    if fl:
        with open(a.flags, "a") as f:
            f.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ} step {a.step:,}: {'; '.join(fl)}\n")
    print(row.strip())


if __name__ == "__main__":
    main()
