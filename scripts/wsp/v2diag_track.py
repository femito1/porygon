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


def pg_window(path, since, step, width=2.5e6):
    """Mean PPO surrogate pg over updates of this segment (log time >= since) with steps in (step - width, step]."""
    xs = []
    try:
        for line in open(path):
            m = re.match(r"(\S+ \S+) upd \d+ steps ([\d,]+) .* pg ([-+0-9.e]+) ", line)
            if not m or m.group(1) < since:
                continue
            st = int(m.group(2).replace(",", ""))
            if step - width < st <= step:
                xs.append(float(m.group(3)))
    except OSError:
        return "—"
    if not xs:
        return "—"
    return f"{sum(xs) / len(xs):+.4f} ({sum(x > 0 for x in xs) / len(xs):.2f}, {len(xs)})"


def main():
    ap = argparse.ArgumentParser()
    for k in ("--decay", "--switch", "--ckpt", "--table", "--flags"):
        ap.add_argument(k, required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--trainlog", default="")
    ap.add_argument("--since", default="0000", help="segment start, 'YYYY-MM-DD HH:MM:SS' as in train.log")
    a = ap.parse_args()
    d = json.load(open(a.decay))[a.ckpt]
    s = json.load(open(a.switch))[a.ckpt]
    fl = []
    if d["mass_voluntary"]["status"] > 0.2:
        fl.append(f"voluntary status mass {d['mass_voluntary']['status']:.2f} > 0.2")
    if d["p_top_ge_0.9"] < 0.2:
        fl.append(f"P(top>=0.9) {d['p_top_ge_0.9']:.2f} < 0.2")
    mv, ma = d["mass_voluntary"], d["mass_all"]
    row = (f"| {a.step / 1e6:.2f}M | {d['entropy']:.2f} | {d['p_top_ge_0.9']:.2f} | {d['top_gap']:.2f} | "
           f"{mv['switch']:.2f} / {mv['attack']:.2f} / {mv['status']:.2f} | {ma['switch']:.2f} / {ma['attack']:.2f} / {ma['status']:.2f} | "
           f"{s['threat_slope']:+.2f} / {s['offence_slope']:+.2f} | {pg_window(a.trainlog, a.since, a.step)} | {d['value_auc']:.3f} | {d['top1_eq_ref']:.2f} | "
           f"{d['KL_ref_to_ck']:.2f} | {'; '.join(fl) or '—'} | {datetime.now(timezone.utc):%m-%d %H:%M} |\n")
    t = Path(a.table)
    if not t.exists():
        t.write_text(HEADER)
    with open(t, "a") as f:
        f.write(row)
    if fl:
        with open(a.flags, "a") as f:
            f.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ} step {a.step:,}: {'; '.join(fl)}\n")
    print(row.strip())


if __name__ == "__main__":
    main()
