"""Tables from att_ablate.py outputs -> $WSP/out/att/ablate_report.md + ablate_summary.json.

    python scripts/wsp/att_report_ablate.py
Outcome / game ids come from the current decisions.npz (regenerated 2026-11-08 14:50: games keyed by run+tag);
Δ value-MSE is recomputed from the stored ΔV, clean V and Φ with that outcome.
"""
import json

import numpy as np

from att_common import OUT, boot_games, data

PAT = None


def load(tag):
    p = OUT / f"{tag}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def conds(z):
    return sorted({k.split("|")[0] for k in z if "|" in k and not k.startswith("base")},
                  key=lambda s: (s[0] if s[0] in "zm" and "." in s else "~", s))


def summarise(z, game, y):
    out = {}
    n = len(z["phi"]); game, y = game[:n], y[:n]
    v0, phi = z["base|v"], z["phi"]
    sq0 = (v0 + phi - y) ** 2
    out["base"] = {"agree_search": boot_games(z["base|agree_search"], game), "share": boot_games(z["base|share"], game),
                   "mse": boot_games(sq0, game), "abs_v": float(np.abs(v0).mean())}
    for c in conds(z):
        dv = z[f"{c}|dv"]
        dsq = (v0 + dv + phi - y) ** 2 - sq0
        out[c] = {"kl": boot_games(z[f"{c}|kl"].astype(np.float64), game),
                  "disagree": boot_games(1 - z[f"{c}|agree"].astype(np.float64), game),
                  "d_agree_search": boot_games(z[f"{c}|agree_search"] - z["base|agree_search"], game),
                  "d_share": boot_games(z[f"{c}|share"] - z["base|share"], game),
                  "abs_dv": boot_games(np.abs(dv).astype(np.float64), game),
                  "d_mse": boot_games(dsq.astype(np.float64), game)}
    return out


def f3(t, scale=1.0, fmt="{:.3f}"):
    m, lo, hi = (x * scale for x in t)
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    d = data()
    game = d["game"]; y = np.where(d["won"] > .5, 1.0, -1.0)
    out, js = [], {}
    z = load("ablate_500M")
    S = summarise(z, game, y)
    js["500M"] = S
    b = S["base"]
    out.append("# Ablations (500M, 48,518 decisions, 95% game-bootstrap intervals)\n")
    out.append(f"Clean: top-1 = search argmax {f3(b['agree_search'])}; search share of the net argmax "
               f"{f3(b['share'])}; value MSE (V+Φ vs ±1) {f3(b['mse'])}; mean |V| {b['abs_v']:.3f}\n")
    out.append("| condition | KL(clean‖abl) | top-1 changed | Δ agree w/ search | Δ search share | |ΔV| | Δ MSE |")
    out.append("|---|---|---|---|---|---|---|")
    for c in conds(z):
        s = S[c]
        out.append(f"| {c} | {f3(s['kl'])} | {f3(s['disagree'])} | {f3(s['d_agree_search'])} | {f3(s['d_share'])} | "
                   f"{f3(s['abs_dv'])} | {f3(s['d_mse'], fmt='{:+.4f}')} |")
    # importance ranking (mean ablation)
    heads = [f"{i}.{h}" for i in range(3) for h in range(8)]
    kl = np.array([S[f"m{x}"]["kl"][0] for x in heads]); dv = np.array([S[f"m{x}"]["abs_dv"][0] for x in heads])
    dis = np.array([S[f"m{x}"]["disagree"][0] for x in heads]); dmse = np.array([S[f"m{x}"]["d_mse"][0] for x in heads])
    klz = np.array([S[f"z{x}"]["kl"][0] for x in heads]); dvz = np.array([S[f"z{x}"]["abs_dv"][0] for x in heads])
    out.append("\n## Head importance (mean ablation), ranked by policy KL\n")
    out.append("| rank | head | KL | top-1 changed | |ΔV| | Δ MSE | policy share | value share | policy/value |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    pk, pv = kl / kl.sum(), dv / dv.sum()
    for r, k in enumerate(np.argsort(-kl)):
        out.append(f"| {r + 1} | L{heads[k].replace('.', 'H')} | {kl[k]:.4f} | {dis[k]:.3f} | {dv[k]:.4f} | "
                   f"{dmse[k]:+.4f} | {pk[k]:.3f} | {pv[k]:.3f} | {pk[k] / pv[k]:.2f} |")
    out.append(f"\nSpearman(policy KL, |ΔV|) over heads = {spearman(kl, dv):+.2f}; "
               f"zero vs mean ablation: Spearman KL {spearman(kl, klz):+.2f}, |ΔV| {spearman(dv, dvz):+.2f}")
    kl_full, dv_full = kl.copy(), dv.copy()
    js["rank"] = {"heads": heads, "kl": kl.tolist(), "dv": dv.tolist(), "disagree": dis.tolist(), "dmse": dmse.tolist(),
                  "kl_zero": klz.tolist(), "dv_zero": dvz.tolist()}
    # dead heads: compare with each layer's total
    out.append("\n## Head effect relative to its whole layer (mean ablation)\n")
    out.append("| layer | attn KL | attn |ΔV| | MLP KL | MLP |ΔV| | heads with KL < 2% and |ΔV| < 2% of layer attn |")
    out.append("|---|---|---|---|---|---|")
    for i in range(3):
        la, lv = S[f"ma{i}"]["kl"][0], S[f"ma{i}"]["abs_dv"][0]
        dead = [f"H{h}" for h in range(8) if kl[i * 8 + h] < .02 * la and dv[i * 8 + h] < .02 * lv]
        out.append(f"| L{i} | {la:.3f} | {lv:.3f} | {S[f'mm{i}']['kl'][0]:.3f} | {S[f'mm{i}']['abs_dv'][0]:.3f} | "
                   f"{', '.join(dead) or 'none'} |")
    # pairs
    zp = load("ablate_500M_pairs")
    if zp is not None:
        SP = summarise(zp, game, y)
        js["pairs"] = SP
        np_ = len(zp["phi"])
        zs = {k: (v[:np_] if v.ndim and len(v) == len(z["phi"]) else v) for k, v in z.items()}
        S1 = summarise(zs, game, y)
        kl = np.array([S1[f"m{x}"]["kl"][0] for x in heads]); dv = np.array([S1[f"m{x}"]["abs_dv"][0] for x in heads])
        out.append("\n## Pairs (joint mean ablation, same layer): superadditivity\n")
        out.append("ratio = KL(pair) / (KL(h) + KL(g)); >1 = the two heads back each other up (redundant), "
                   "<1 = overlapping / saturating effects. Same for |ΔV|. Pattern cos from att_patterns.\n")
        pc = np.load(OUT / "patterns_500M.npz")["pair_cos"]
        rows = []
        for c in conds(zp):
            _, i, h, g = c[:2], int(c[2]), int(c.split(".")[1]), int(c.split(".")[2])
            a = kl[i * 8 + h] + kl[i * 8 + g]; av = dv[i * 8 + h] + dv[i * 8 + g]
            rows.append((SP[c]["kl"][0] / a, f"L{i}H{h}+H{g}", SP[c]["kl"][0], a, SP[c]["abs_dv"][0] / av,
                         SP[c]["disagree"][0], pc[i, h, g]))
        rows.sort(key=lambda r: -r[0])
        out.append("| pair | KL pair | KL sum | ratio KL | ratio |ΔV| | top-1 changed | pattern cos |")
        out.append("|---|---|---|---|---|---|---|")
        for r in rows:
            out.append(f"| {r[1]} | {r[2]:.4f} | {r[3]:.4f} | {r[0]:.2f} | {r[4]:.2f} | {r[5]:.3f} | {r[6]:.2f} |")
        ratios = np.array([r[0] for r in rows]); cosv = np.array([r[6] for r in rows])
        out.append(f"\nmedian ratio KL {np.median(ratios):.2f}; Spearman(ratio, pattern cos) = {spearman(ratios, cosv):+.2f}")
    # checkpoints
    out.append("\n## Head importance across checkpoints (mean ablation, single heads)\n")
    rows = {"500M": (kl_full, dv_full)}
    for ck in ("400M", "600M", "2.5M"):
        zc = load(f"ablate_{ck}_heads")
        if zc is None:
            continue
        SC = summarise(zc, game, y)
        js[ck] = SC
        nc = len(zc["phi"])
        if "500M_sub" not in rows:
            zs = {k: (v[:nc] if v.ndim and len(v) == len(z["phi"]) else v) for k, v in z.items()}
            S5 = summarise(zs, game, y)
            rows["500M_sub"] = (np.array([S5[f"m{x}"]["kl"][0] for x in heads]),
                                np.array([S5[f"m{x}"]["abs_dv"][0] for x in heads]))
        rows[ck] = (np.array([SC[f"m{x}"]["kl"][0] for x in heads]), np.array([SC[f"m{x}"]["abs_dv"][0] for x in heads]))
    out.append("| ckpt | Spearman KL vs 500M | Spearman |ΔV| vs 500M | top-5 policy heads | top-5 value heads | Σ head KL | Σ |ΔV| |")
    out.append("|---|---|---|---|---|---|---|")
    kl5, dv5 = rows.get("500M_sub", rows["500M"])
    for ck, (a_, b_) in rows.items():
        kl, dv = (kl5, dv5) if ck not in ("500M",) else rows["500M"]
        tp = ", ".join(f"L{heads[k].replace('.', 'H')}" for k in np.argsort(-a_)[:5])
        tv = ", ".join(f"L{heads[k].replace('.', 'H')}" for k in np.argsort(-b_)[:5])
        out.append(f"| {ck} | {spearman(a_, kl):+.2f} | {spearman(b_, dv):+.2f} | {tp} | {tv} | {a_.sum():.3f} | {b_.sum():.3f} |")
    out.append("\nPer-head KL / |ΔV| by checkpoint:\n")
    out.append("| head | " + " | ".join(f"KL {c} | |ΔV| {c}" for c in rows) + " |\n|---|" + "---|---|" * len(rows))
    for k, hname in enumerate(heads):
        out.append(f"| L{hname.replace('.', 'H')} | " + " | ".join(f"{rows[c][0][k]:.4f} | {rows[c][1][k]:.4f}" for c in rows) + " |")
    js["ckpt_rank"] = {c: {"kl": a_.tolist(), "dv": b_.tolist()} for c, (a_, b_) in rows.items()}
    (OUT / "ablate_report.md").write_text("\n".join(out) + "\n")
    (OUT / "ablate_summary.json").write_text(json.dumps(js))
    print("\n".join(out))


if __name__ == "__main__":
    main()
