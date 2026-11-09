"""Summarise a v2 train.log (porygon.v2.train._line format) in windows of env steps and flag alarms.

    python scripts/wsp/v2diag_log.py RUN_DIR/train.log [--window 2.5e6]
"""

import argparse
import re

import numpy as np

KEYS = ["sps", "lr", "pg", "vf", "ent", "kl", "clip", "ev", "gn", "ratio", "cos_pg", "cos_vf", "len"]


def parse(path):
    rows = []
    for line in open(path):
        if " upd " not in f" {line}":
            continue
        line = line[line.index("upd "):]
        m = re.search(r"steps ([\d,]+)", line)
        if not m:
            continue
        r = {"steps": int(m.group(1).replace(",", ""))}
        for k in KEYS:
            mm = re.search(rf"(?:^| ){re.escape(k)} ([-+0-9.eE]+|nan)(?= |$)", line)
            if mm:
                try:
                    r[k] = float(mm.group(1))
                except ValueError:
                    pass
        aux = line.split("|", 1)[1] if "|" in line else ""
        for kv in aux.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    r["aux:" + k] = float(v)
                except ValueError:
                    pass
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--window", type=float, default=2.5e6)
    a = ap.parse_args()
    rows = parse(a.log)
    if not rows:
        print("no update lines")
        return
    keys = KEYS + sorted({k for r in rows for k in r if k.startswith("aux:")})
    w = int(a.window)
    buckets = {}
    for r in rows:
        buckets.setdefault(r["steps"] // w, []).append(r)
    print("steps(M) " + " ".join(f"{k:>9s}" for k in keys))
    for b, rs in sorted(buckets.items()):
        vals = []
        for k in keys:
            x = [r[k] for r in rs if k in r and np.isfinite(r[k])]
            vals.append(f"{np.mean(x):9.4g}" if x else f"{'-':>9s}")
        print(f"{(b + 1) * w / 1e6:8.1f} " + " ".join(vals))
    last = buckets[max(buckets)]
    m = lambda k: np.nanmean([r.get(k, np.nan) for r in last])  # noqa: E731
    fl = []
    if rows[-1]["steps"] >= 5e6:
        if m("ev") < 0.1:
            fl.append(f"EV ~0: {m('ev'):.3f}")
        if m("ent") < 0.1:
            fl.append(f"ENTROPY COLLAPSE: {m('ent'):.3f}")
        if m("cos_pg") < 0.005:
            fl.append(f"POLICY GRADIENT PURE NOISE: cos {m('cos_pg'):.4f}")
        if m("kl") > 0.05:
            fl.append(f"KL HIGH: {m('kl'):.4f}")
    print("last step", f"{rows[-1]['steps']:,}", "FLAGS:", fl or "none")


if __name__ == "__main__":
    main()
