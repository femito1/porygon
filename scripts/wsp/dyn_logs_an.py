"""dyn: tables from OUT_DIR/scalars.csv (dyn_logs.py): per-segment lr -> implied TOTAL, and every training
scalar averaged in ±2.5M-step windows on a 50M grid (plus 640M for the arms).

    python scripts/wsp/dyn_logs_an.py OUT_DIR
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np

L0 = 10 ** -4.23


def implied_total(step, lr):
    x = ((L0 / lr) ** (2 / 3) - 1) / 8
    return step / x if x > 0 else float("nan")


def main():
    out = sys.argv[1]
    rows = defaultdict(list)  # tag -> [(step, value, source)]
    for r in csv.DictReader(open(f"{out}/scalars.csv")):
        if r["step"] in ("", "None"):
            continue
        tag = r["tag"].replace("log/", "logtxt/")
        rows[tag].append((int(float(r["step"])), float(r["value"]), r["source"]))
    print("tags:", {k: len(v) for k, v in rows.items()})
    res = {}
    # 1. segments from the train.log lr lines and the tb learning_rate (per event file)
    seg = defaultdict(list)
    for s, v, src in rows["logtxt/learning_rate"] + rows.get("train/learning_rate", []):
        seg[src].append((s, v))
    segs = []
    for src, pts in seg.items():
        pts.sort()
        tot = [implied_total(s, v) for s, v in pts if v > 0]
        segs.append(dict(source=src, start=pts[0][0], end=pts[-1][0], lr_first=pts[0][1], lr_last=pts[-1][1],
                         lr_min=min(v for _, v in pts), lr_max=max(v for _, v in pts),
                         total_median=float(np.median(tot)), total_iqr=[float(np.percentile(tot, 25)),
                                                                     float(np.percentile(tot, 75))], n=len(pts)))
    segs.sort(key=lambda d: (d["start"], d["source"]))
    print("\nsegment                                                         start→end (M)     lr first→last      implied TOTAL (M)")
    for d in segs:
        print(f"{d['source'][:62]:62s} {d['start']/1e6:7.1f}→{d['end']/1e6:6.1f}  {d['lr_first']:.3g}→{d['lr_last']:.3g}  "
              f"{d['total_median']/1e6:8.1f} [{d['total_iqr'][0]/1e6:.1f}, {d['total_iqr'][1]/1e6:.1f}]  n={d['n']}")
    res["segments"] = segs
    # 2. 50M grid of every scalar (tb and train.log merged, train.log for runs without tb)
    tags = ["train/learning_rate", "train/approx_kl", "train/clip_fraction", "train/entropy_loss",
            "train/explained_variance", "train/value_loss", "train/policy_gradient_loss", "train/loss",
            "rollout/ep_len_mean", "rollout/ep_rew_mean", "eval/win_rate_heuristic", "eval/win_rate_max_power",
            "eval/win_rate_anchor", "eval/heuristic", "eval/max_power", "eval/anchor"]
    alias = {"train/learning_rate": "logtxt/learning_rate", "train/approx_kl": "logtxt/approx_kl",
             "train/clip_fraction": "logtxt/clip_fraction", "train/entropy_loss": "logtxt/entropy_loss",
             "train/explained_variance": "logtxt/explained_variance", "train/value_loss": "logtxt/value_loss",
             "train/policy_gradient_loss": "logtxt/policy_gradient_loss", "rollout/ep_len_mean": "logtxt/ep_len_mean",
             "rollout/ep_rew_mean": "logtxt/ep_rew_mean"}
    grid = [2_500_000, 10_000_000, 25_000_000] + list(range(50_000_000, 650_000_000, 50_000_000)) + [620_000_000, 640_000_000]
    table = {}
    for t in tags:
        pts = [(s, v, src) for s, v, src in rows.get(t, []) if not src.startswith("arm-")]
        pts += [(s, v, src) for s, v, src in rows.get(alias.get(t, "_"), []) if not src.startswith("arm-")]
        arm = defaultdict(list)
        for s, v, src in rows.get(t, []) + rows.get(alias.get(t, "_"), []):
            if src.startswith("arm-"):
                arm[src.split("-")[1]].append((s, v))
        if not pts and not arm:
            continue
        S = np.array([p[0] for p in pts]) if pts else np.zeros(0)
        V = np.array([p[1] for p in pts]) if pts else np.zeros(0)
        col = {}
        for g in grid:
            w = 2_500_000 if g >= 50_000_000 else max(1_000_000, g // 5)
            sel = np.abs(S - g) <= w
            if g > 600_002_347:
                for a, ap in arm.items():
                    aS, aV = np.array([p[0] for p in ap]), np.array([p[1] for p in ap])
                    m = np.abs(aS - g) <= (2_500_000 if g == 620_000_000 else 5_000_000)
                    if m.any():
                        col[f"{g//1_000_000}M-{a}"] = (float(aV[m].mean()), float(aV[m].std() / max(1, np.sqrt(m.sum()))), int(m.sum()))
            elif sel.any():
                col[f"{g//1_000_000}M" if g >= 10_000_000 else f"{g/1e6:.1f}M"] = (float(V[sel].mean()), float(V[sel].std() / np.sqrt(sel.sum())), int(sel.sum()))
        table[t] = col
    res["grid"] = table
    keys = sorted({k for c in table.values() for k in c}, key=lambda k: (float(k.split("M")[0]), k))
    print("\n" + "tag".ljust(28) + "".join(k.rjust(13) for k in keys))
    for t, c in table.items():
        print(t.ljust(28) + "".join((f"{c[k][0]:.4g}".rjust(13) if k in c else "".rjust(13)) for k in keys))
    print("\n(n per cell)")
    for t, c in table.items():
        print(t.ljust(28) + "".join((str(c[k][2]).rjust(13) if k in c else "".rjust(13)) for k in keys))
    # 3. lr vs step on a 5M grid (one value per 5M: median of logged lr within ±2.5M; chain sources only)
    lr = [(s, v) for s, v, src in rows["logtxt/learning_rate"] + rows.get("train/learning_rate", []) if not src.startswith("arm-")]
    S, V = np.array([p[0] for p in lr]), np.array([p[1] for p in lr])
    lrgrid = []
    for g in range(0, 605_000_000, 5_000_000):
        m = np.abs(S - g) <= 2_500_000
        if m.any():
            lrgrid.append((g, float(np.median(V[m])), float(V[m].min()), float(V[m].max())))
    res["lr_grid"] = lrgrid
    # cumulative Σ lr per rollout ~ total "distance budget"; integrate over steps (per 39,936-step rollout)
    cum, tot = [], 0.0
    for g, v, _, _ in lrgrid:
        tot += v * 5_000_000 / 39_936 * 273
        cum.append((g, tot))
    res["cum_lr_adam"] = cum
    print("\nlr on the 5M grid (step M: median lr; cumulative Σ lr·Adam steps):")
    print("  ".join(f"{g/1e6:.0f}:{v:.2e}" for g, v, _, _ in lrgrid[::4]))
    print("  ".join(f"{g/1e6:.0f}:{c:.3g}" for g, c in cum[::10]))
    json.dump(res, open(f"{out}/logs_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
