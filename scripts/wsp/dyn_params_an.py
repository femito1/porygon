"""dyn: tables from dyn_params.py outputs (params_lineage.json, params_arm_*.json).

    python scripts/wsp/dyn_params_an.py OUT_DIR
"""
import json
import sys

import numpy as np

GROUPS = ["all", "embeddings", "non_embedding", "emb.species", "emb.items", "emb.abilities", "emb.moves", "pos",
          "move_proj", "mon_proj", "field_proj", "action_move_proj", "L0.attn", "L0.mlp", "L0.norm", "L1.attn",
          "L1.mlp", "L1.norm", "L2.attn", "L2.mlp", "L2.norm", "final_norm", "policy_head", "value_mlp", "value_head"]


def f(x, p=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}g}"


def main():
    out = sys.argv[1]
    L = json.load(open(f"{out}/params_lineage.json"))
    rows = L["rows"]
    res = {}
    print("## lineage per checkpoint (all / embeddings / non-embedding)")
    print("steps(M) adam_step lr | |Δ|/|θ| all emb non | cos_prev all emb non | R all emb non | rho all emb non | upd_rms/lr non")
    for r in rows:
        g = r["groups"]
        print(f"{r['steps']/1e6:7.1f} {r['adam_step']:9.0f} {r['lr']:.3g} | " + " ".join(f(g[k].get('d_rel')) for k in ("all", "embeddings", "non_embedding")) + " | "
              + " ".join(f(g[k].get('cos_prev'), 2) for k in ("all", "embeddings", "non_embedding")) + " | "
              + " ".join(f(g[k].get('R')) for k in ("all", "embeddings", "non_embedding")) + " | "
              + " ".join(f(g[k]['rho']) for k in ("all", "embeddings", "non_embedding")) + f" | {g['non_embedding']['upd_rms']/r['lr']:.3f}")
    # late window per group: 5M blocks with both ends in [300M, 600M]
    late = [r for r in rows if r["steps"] > 300_100_000 and r["steps"] <= 600_002_347 and "K" in r]
    early = [r for r in rows if 60_000_000 < r["steps"] <= 140_000_000 and "K" in r]
    mid = [r for r in rows if 140_000_000 < r["steps"] <= 300_100_000 and "K" in r]
    first = {s: next(r for r in rows if r["steps"] == s) for s in (21261312, 300028848, 400029852, 500036720, 600002347)}
    print("\n## per group, means over 5M blocks (early 60-140M | mid 140-300M | late 300-600M)")
    print("group n | |θ|@600 | |Δ5M|/|θ| early mid late | cos_prev early mid late | R early mid late | rho@100M rho@600M | "
          "|θ600-θ300| / Σ|Δ5M| (efficiency; RW≈1/√n, drift≈1) | |θ600-θ300| / √Σ|Δ5M|² (RW≈1) | |θ600-θ21M|/|θ21M| | |θ600-θ300|/|θ300|")
    tab = {}
    r100 = min(rows, key=lambda r: abs(r["steps"] - 100e6))
    for g in GROUPS:
        def m(rs, k):
            v = [r["groups"][g].get(k) for r in rs if r["groups"][g].get(k) is not None]
            return float(np.mean(v)) if v else None
        dl = np.array([r["groups"][g]["d_norm"] for r in late])
        d300 = first[600002347]["groups"][g].get("dist_300028848")
        d21 = first[600002347]["groups"][g].get("dist_21261312")
        n21 = first[21261312]["groups"][g]["norm"]
        n300 = first[300028848]["groups"][g]["norm"]
        row = dict(n=rows[-1]["groups"][g]["n"], norm600=rows[-1]["groups"][g]["norm"],
                   drel=[m(early, "d_rel"), m(mid, "d_rel"), m(late, "d_rel")],
                   cos=[m(early, "cos_prev"), m(mid, "cos_prev"), m(late, "cos_prev")],
                   R=[m(early, "R"), m(mid, "R"), m(late, "R")],
                   rho=[r100["groups"][g]["rho"], rows[-1]["groups"][g]["rho"]],
                   eff=d300 / dl.sum() if d300 else None, rw=d300 / np.sqrt((dl ** 2).sum()) if d300 else None,
                   n_blocks=len(dl), tot21=d21 / n21 if d21 else None, tot300=d300 / n300 if d300 else None,
                   upd_rms_over_lr=rows[-1]["groups"][g]["upd_rms"] / rows[-1]["lr"])
        tab[g] = row
        print(f"{g:16s} {row['n']:8d} | {row['norm600']:.3g} | " + " ".join(f(x) for x in row["drel"]) + " | "
              + " ".join(f(x, 2) for x in row["cos"]) + " | " + " ".join(f(x) for x in row["R"]) + " | "
              + " ".join(f(x) for x in row["rho"]) + f" | {f(row['eff'])} | {f(row['rw'])} | {f(row['tot21'])} | {f(row['tot300'])}"
              + f" | upd/lr {row['upd_rms_over_lr']:.3f}")
    res["groups"] = tab
    # Gram lag structure
    G = L.get("gram")
    if G:
        iv = G["intervals"]
        print(f"\n## cosine between 5M Δs in 300-600M ({len(iv)} intervals) by lag; mean [min, max]")
        lag = {}
        for g in GROUPS:
            C = np.array(G["cos"][g])
            n = len(C)
            row = {}
            for k in (1, 2, 3, 5, 10, 20, 40):
                if k < n:
                    d = np.diag(C, k)
                    row[k] = (float(d.mean()), float(d.std() / np.sqrt(len(d))))
            lag[g] = row
            print(f"{g:16s} " + "  ".join(f"lag{k}: {v[0]:+.3f}±{v[1]:.3f}" for k, v in row.items()))
        res["lag_cos"] = lag
    # Adam / SNR timeline at selected checkpoints
    print("\n## Adam state over time: rho = Σm²/Σv (iid noise 0.053, constant gradient 1); upd_rms/lr")
    sel = [r for r in rows if any(abs(r["steps"] - s) < 3e6 for s in (2.5e6, 10e6, 21e6, 66e6, 100e6, 136e6, 200e6, 273e6, 300e6, 400e6, 500e6, 600e6))]
    print("steps(M) " + " ".join(f"{g[:10]:>10s}" for g in GROUPS))
    tl = {}
    for r in sel:
        print(f"{r['steps']/1e6:7.1f}  " + " ".join(f"{r['groups'][g]['rho']:10.3f}" for g in GROUPS))
        tl[r["steps"]] = {g: dict(rho=r["groups"][g]["rho"], upd=r["groups"][g]["upd_rms"] / r["lr"], lr=r["lr"],
                                  v_rms=r["groups"][g]["v_rms"], n_active=r["groups"][g]["n_active"]) for g in GROUPS}
    print("upd_rms/lr:")
    for r in sel:
        print(f"{r['steps']/1e6:7.1f}  " + " ".join(f"{r['groups'][g]['upd_rms']/r['lr']:10.3f}" for g in GROUPS))
    print("n_active/n:")
    for r in sel:
        print(f"{r['steps']/1e6:7.1f}  " + " ".join(f"{r['groups'][g]['n_active']/r['groups'][g]['n']:10.3f}" for g in GROUPS))
    print("lr at checkpoints:", [(round(r["steps"] / 1e6, 1), f"{r['lr']:.3g}") for r in sel])
    res["adam_timeline"] = tl
    # distance from references
    print("\n## distance from references |θ - θ_ref| / |θ_ref| (all, embeddings, non_embedding)")
    for r in rows[::4] + [rows[-1]]:
        g = r["groups"]
        print(f"{r['steps']/1e6:7.1f} " + " | ".join(
            f"ref{s/1e6:.0f}: " + " ".join(f(g[k].get(f'dist_{s}', 0) / first[s]['groups'][k]['norm']) for k in ("all", "embeddings", "non_embedding"))
            for s in first if f"dist_{s}" in g["all"]))
    # arms
    print("\n## arms 600M→640M")
    arms = {}
    for a in ("ctrl", "x30", "x300"):
        A = json.load(open(f"{out}/params_arm_{a}.json"))["rows"]
        last = A[-1]
        arms[a] = {g: dict(tot=last["groups"][g].get("dist_600002347", 0) / A[0]["groups"][g]["norm"],
                           drel=float(np.mean([r["groups"][g]["d_rel"] for r in A[1:]])),
                           cos=float(np.mean([r["groups"][g]["cos_prev"] for r in A[2:] if "cos_prev" in r["groups"][g]])),
                           R=float(np.mean([r["groups"][g]["R"] for r in A[1:]])),
                           eff=last["groups"][g].get("dist_600002347", 0) / sum(r["groups"][g]["d_norm"] for r in A[1:]))
                   for g in GROUPS}
    print("group            | |θ640-θ600|/|θ600| ctrl x30 x300 | mean cos_prev ctrl x30 x300 | efficiency ctrl x30 x300")
    for g in GROUPS:
        print(f"{g:16s} | " + " ".join(f(arms[a][g]["tot"]) for a in arms) + " | "
              + " ".join(f(arms[a][g]["cos"], 2) for a in arms) + " | " + " ".join(f(arms[a][g]["eff"]) for a in arms))
    res["arms"] = arms
    json.dump(res, open(f"{out}/params_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
