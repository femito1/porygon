"""Tables from att_patterns.py outputs -> $WSP/out/att/patterns_report.md (+ json of key numbers).

    python scripts/wsp/att_report_patterns.py
"""
import json

import numpy as np

from att_common import NR, OUT, ROLES, boot_games

SHORT = ["fld", "oA", "oB", "oF", "pA", "pB", "pF", "pU", "mL", "mI"]
QROLES = [0, 1, 2, 4, 5, 7, 8]  # query roles shown


def load(name):
    z = np.load(OUT / f"patterns_{name}.npz")
    return {k: z[k] for k in z.files}


def tables(z):
    mass = z["mass_sum"] / np.maximum(z["mass_cnt"], 1)[None, None, :, None]          # 3,8,q,k
    perkey = z["perkey_sum"] / np.maximum(z["perkey_cnt"], 1)[None, None]             # 3,8,q,k
    ent = z["ent_sum"] / np.maximum(z["mass_cnt"], 1)[None, None]                     # 3,8,q
    ent_all = z["ent_sum"].sum(-1) / z["mass_cnt"].sum()
    selfm = np.einsum("lhtt->lht", z["pos_mean"])                                      # 3,8,17
    return mass, perkey, ent, ent_all, selfm


def main():
    out, js = [], {}
    names = [n for n in ("500M", "400M", "600M", "2.5M") if (OUT / f"patterns_{n}.npz").exists()]
    Z = {n: load(n) for n in names}
    T = {n: tables(Z[n]) for n in names}
    mass, perkey, ent, ent_all, selfm = T["500M"]
    z = Z["500M"]
    out.append("# Attention patterns by token role (500M, 48,518 decisions)\n")
    out.append("Roles: " + ", ".join(f"{s}={r}" for s, r in zip(SHORT, ROLES)) + "\n")
    out.append("Mean tokens per decision: " + ", ".join(f"{SHORT[i]} {z['mass_cnt'][i] / len(z['game']):.2f}"
                                                        for i in range(NR)) + "\n")
    out.append("## Entropy per head (nats, mean over all queries; uniform over 17 = 2.83)\n")
    out.append("| layer | " + " | ".join(f"H{h}" for h in range(8)) + " |\n|---|" + "---|" * 8)
    for i in range(3):
        out.append(f"| L{i} | " + " | ".join(f"{ent_all[i, h]:.2f}" for h in range(8)) + " |")
    js["entropy"] = ent_all.tolist()
    out.append("\n## Mass (attention summed over keys of a role, averaged over queries of a role)\n")
    out.append("Each head: rows = query role, cols = key role; `self` = mean weight on own token.\n")
    for i in range(3):
        for h in range(8):
            out.append(f"\n### L{i}H{h}  (entropy {ent_all[i, h]:.2f})\n")
            out.append("| q \\ k | " + " | ".join(SHORT) + " |\n|---|" + "---|" * NR)
            for q in QROLES:
                out.append(f"| {SHORT[q]} (H={ent[i, h, q]:.2f}) | " + " | ".join(f"{mass[i, h, q, k]:.2f}" for k in range(NR)) + " |")
            out.append("\nself weight by slot: " + " ".join(f"{selfm[i, h, t]:.2f}" for t in range(17)))
    js["mass"] = mass.tolist(); js["perkey"] = perkey.tolist()
    # compact summary: top per-key enrichment per head (per-key weight × 17 = ratio to uniform)
    out.append("\n## Compact: top-3 (query→key) pairs by mass, for queries oA / mL / fld\n")
    out.append("| head | ent | oA→ (top 3) | mL→ (top 3) | fld→ (top 3) | self(oA) |\n|---|---|---|---|---|---|")
    for i in range(3):
        for h in range(8):
            cells = []
            for q in (1, 8, 0):
                o = np.argsort(-mass[i, h, q])[:3]
                cells.append(", ".join(f"{SHORT[k]} {mass[i, h, q, k]:.2f}" for k in o))
            # self weight of own active: mass to own_act from own_act query
            out.append(f"| L{i}H{h} | {ent_all[i, h]:.2f} | " + " | ".join(cells) + f" | {mass[i, h, 1, 1]:.2f} |")
    # conditioning
    out.append("\n## Conditioning: attention mass vs position features (Spearman ρ over decisions; "
               "Δ = mean(top tercile) − mean(bottom tercile), 95% game bootstrap)\n")
    feats = [k[6:] for k in z if k.startswith("feats_")]
    game = z["game"]

    def rank(a):
        return np.argsort(np.argsort(a)).astype(np.float64)
    js["cond"] = {}
    for ck in ("ownact_oppact", "moves_oppact", "field_oppact", "oppact_ownact", "ownact_ownbench"):
        c = z[f"cond_{ck}"]
        out.append(f"\n### {ck}: mean mass per head\n")
        out.append("| layer | " + " | ".join(f"H{h}" for h in range(8)) + " |\n|---|" + "---|" * 8)
        for i in range(3):
            out.append(f"| L{i} | " + " | ".join(f"{c[:, i, h].mean():.3f}" for h in range(8)) + " |")
        out.append(f"\nSpearman ρ of {ck} with features (rows head, cols feature):\n")
        out.append("| head | " + " | ".join(feats) + " |\n|---|" + "---|" * len(feats))
        rho = np.zeros((3, 8, len(feats)))
        for i in range(3):
            for h in range(8):
                rc = rank(c[:, i, h])
                for j, f in enumerate(feats):
                    rf = rank(z[f"feats_{f}"] + 1e-9 * np.random.default_rng(0).standard_normal(len(game)))
                    rho[i, h, j] = np.corrcoef(rc, rf)[0, 1]
                out.append(f"| L{i}H{h} | " + " | ".join(f"{rho[i, h, j]:+.2f}" for j in range(len(feats))) + " |")
        js["cond"][ck] = {"mean": c.mean(0).tolist(), "rho": rho.tolist(), "feats": feats}
    # threat tercile contrast for own_act -> opp_act, the requested test
    out.append("\n### own_act→opp_act: high vs low opp_threat (known moves bp×eff×STAB vs our active), only "
               "decisions where the opp active has ≥1 revealed damaging move\n")
    th = z["feats_opp_threat"]; ok = th > 0
    lo, hi = np.percentile(th[ok], [33.3, 66.7])
    out.append(f"terciles: low ≤ {lo:.0f}, high ≥ {hi:.0f}; n = {ok.sum()}\n")
    out.append("| head | low | high | Δ [95%] |\n|---|---|---|---|")
    c = z["cond_ownact_oppact"]
    js["threat_contrast"] = {}
    for i in range(3):
        for h in range(8):
            v = c[:, i, h]
            a = v[ok & (th <= lo)].mean(); b = v[ok & (th >= hi)].mean()
            # bootstrap the difference by games: per-decision signed contribution
            sel = ok & ((th <= lo) | (th >= hi))
            g = game[sel]; ishi = th[sel] >= hi; vv = v[sel]
            ug, gi = np.unique(g, return_inverse=True)
            sh = np.bincount(gi, vv * ishi, len(ug)); ch = np.bincount(gi, ishi.astype(float), len(ug))
            sl = np.bincount(gi, vv * ~ishi, len(ug)); cl = np.bincount(gi, (~ishi).astype(float), len(ug))
            rng = np.random.default_rng(1); idx = rng.integers(0, len(ug), (1000, len(ug)))
            d = sh[idx].sum(1) / ch[idx].sum(1) - sl[idx].sum(1) / cl[idx].sum(1)
            out.append(f"| L{i}H{h} | {a:.3f} | {b:.3f} | {b - a:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |")
            js["threat_contrast"][f"L{i}H{h}"] = [float(a), float(b), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    # checkpoints: pattern similarity per head vs 500M, entropy
    if len(names) > 1:
        out.append("\n## Checkpoint comparison: per-head similarity of the role-mass table to 500M "
                   "(cosine of flattened q×k mass, query roles weighted equally) and entropy\n")
        out.append("| head | " + " | ".join(f"cos {n} | ent {n}" for n in names) + " |\n|---|" + "---|---|" * len(names))
        js["ckpt_cos"] = {}
        for i in range(3):
            for h in range(8):
                row = []
                for n in names:
                    a = T[n][0][i, h][QROLES].ravel(); b = mass[i, h][QROLES].ravel()
                    cs = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
                    row.append(f"{cs:.3f} | {T[n][3][i, h]:.2f}")
                    js["ckpt_cos"].setdefault(n, {})[f"L{i}H{h}"] = cs
                out.append(f"| L{i}H{h} | " + " | ".join(row) + " |")
        for n in names:
            out.append(f"\n### compact top-3 for {n}\n")
            out.append("| head | oA→ | mL→ | fld→ |\n|---|---|---|---|")
            m_ = T[n][0]
            for i in range(3):
                for h in range(8):
                    cells = []
                    for q in (1, 8, 0):
                        o = np.argsort(-m_[i, h, q])[:3]
                        cells.append(", ".join(f"{SHORT[k]} {m_[i, h, q, k]:.2f}" for k in o))
                    out.append(f"| L{i}H{h} | " + " | ".join(cells) + " |")
    # head-pair similarity within layer (500M)
    out.append("\n## Within-layer head-pair pattern cosine (mean over decisions, flattened 17×17), 500M\n")
    pc = z["pair_cos"]
    for i in range(3):
        out.append(f"\nL{i}\n\n| | " + " | ".join(f"H{h}" for h in range(8)) + " |\n|---|" + "---|" * 8)
        for h in range(8):
            out.append(f"| H{h} | " + " | ".join(f"{pc[i, h, g]:.2f}" for g in range(8)) + " |")
    js["pair_cos"] = pc.tolist()
    (OUT / "patterns_report.md").write_text("\n".join(out) + "\n")
    (OUT / "patterns_summary.json").write_text(json.dumps(js))
    print("\n".join(out))


if __name__ == "__main__":
    main()
