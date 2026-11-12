"""Tables from att_circuits.py -> $WSP/out/att/circuits_report_<CKPT>.md + circuits_summary_<CKPT>.json.

    python scripts/wsp/att_report_circuits.py [CKPT]
"""
import json
import sys

import numpy as np

from att_common import OUT, ROLES

SHORT = ["fld", "oA", "oB", "oF", "pA", "pB", "pF", "pU", "mL", "mI"]
QSHOW = [0, 1, 8]   # field, own active, legal moves as queries


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "500M"
    z = np.load(OUT / f"circuits_{ck}.npz")
    keys = list(z["keys"])
    K = {k: n for n, k in enumerate(keys)}
    dS, dSa, pdS = z["dS_key"], z["dS_key_abs"], z["pred_dS"]          # F,3,H,NR
    dh, pdh, hn = z["dhead"], z["pred_dhead"], z["headnorm"]          # F,3,H,NR
    out, js = [], {}
    out.append(f"# QK / OV circuits ({ck}, 2,048 decisions)\n")
    out.append("## Weight statistics per head (LN1 gain folded in)\n")
    out.append("| head | ‖W_OV‖_F | σ₁(W_OV) | eff. rank OV | ‖W_QK‖_F | clean score std |")
    out.append("|---|---|---|---|---|---|")
    for i in range(3):
        for h in range(8):
            out.append(f"| L{i}H{h} | {z['w_ov_fro'][i, h]:.2f} | {z['w_ov_top'][i, h]:.2f} | {z['w_ov_eff_rank'][i, h]:.1f} | "
                       f"{z['w_qk_fro'][i, h]:.2f} | {z['w_score_std'][i, h]:.2f} |")
    # validation: predicted vs measured key-side score change, signed features only
    signed = [n for n, k in enumerate(keys) if not k.endswith(("species_swap", "pos_slot"))]
    out.append("\n## Read-the-weights test: predicted vs measured Δ score (target token as key)\n")
    out.append("pred = q̄_role · W_K Δā / √32 from the weights + mean query; measured = mean change of the actual "
               "pre-softmax score. Over all (signed feature, head, query role ∈ {fld, oA, oB, pA, mL}).\n")
    out.append("| layer | Pearson r | slope (meas/pred) | sign agreement where |meas| > 0.25 | n |")
    out.append("|---|---|---|---|---|")
    js["validate_qk"] = {}
    for i in range(3):
        a = pdS[signed][:, i][..., [0, 1, 2, 4, 8]].ravel(); b = dS[signed][:, i][..., [0, 1, 2, 4, 8]].ravel()
        r = np.corrcoef(a, b)[0, 1]; sl = (a @ b) / (a @ a)
        big = np.abs(b) > .25
        sg = float((np.sign(a[big]) == np.sign(b[big])).mean()) if big.any() else float("nan")
        out.append(f"| L{i} | {r:.3f} | {sl:.2f} | {sg:.3f} | {big.sum()} |")
        js["validate_qk"][i] = [float(r), float(sl), sg]
    out.append("\n## Read-the-weights test for OV: predicted vs measured ‖Δ head output‖ at other tokens\n")
    out.append("pred = clean attention to the target × ‖W_O W_V Δa‖ (per decision); measured = actual ‖Δ head_out‖.\n")
    out.append("| layer | Pearson r | slope | n |\n|---|---|---|---|")
    for i in range(3):
        a = pdh[:, i][..., [0, 1, 2, 4, 8]].ravel(); b = dh[:, i][..., [0, 1, 2, 4, 8]].ravel()
        out.append(f"| L{i} | {np.corrcoef(a, b)[0, 1]:.3f} | {(a @ b) / (a @ a):.2f} | {len(a)} |")
    # per head: key features that most change the score from own-active and legal-move queries
    out.append("\n## What each head's queries look for: top key features (measured Δ score, nats; "
               "`target|feature`, flag 0→1 or lo→hi; abs for species swap / pos)\n")
    out.append("| head | from own active (oA) | from legal moves (mL) | from field |")
    out.append("|---|---|---|---|")
    js["top_key"] = {}
    for i in range(3):
        for h in range(8):
            cells = []
            for q in (1, 8, 0):
                v = np.where([k.endswith(("species_swap", "pos_slot")) for k in keys], dSa[:, i, h, q], dS[:, i, h, q])
                # skip features targeting the query role itself (own_act features from own-active query)
                o = np.argsort(-np.abs(v))[:4]
                cells.append(", ".join(f"{keys[k]} {v[k]:+.2f}" for k in o))
                js["top_key"][f"L{i}H{h}|{SHORT[q]}"] = [(keys[k], float(v[k])) for k in o]
            out.append(f"| L{i}H{h} | " + " | ".join(cells) + " |")
    # OV: what each head moves into own active / move tokens / field (measured ‖Δ head out‖ / clean ‖head out‖)
    out.append("\n## What each head moves: top source features by measured ‖Δ head output‖ at the destination "
               "(relative to the clean output norm there)\n")
    out.append("| head | into own active | into legal moves | into field |")
    out.append("|---|---|---|---|")
    js["top_ov"] = {}
    for i in range(3):
        for h in range(8):
            cells = []
            for q in (1, 8, 0):
                v = dh[:, i, h, q] / np.maximum(hn[:, i, h, q], 1e-6)
                # exclude features whose target is the destination role itself
                tgt = {1: "own_act|", 8: "move0|", 0: "field|"}[q]
                v = np.where([k.startswith(tgt) for k in keys], 0, v)
                o = np.argsort(-v)[:4]
                cells.append(", ".join(f"{keys[k]} {v[k]:.2f}" for k in o))
                js["top_ov"][f"L{i}H{h}|{SHORT[q]}"] = [(keys[k], float(v[k])) for k in o]
            out.append(f"| L{i}H{h} | " + " | ".join(cells) + " |")
    # specific question: opp-active features -> move tokens, per layer (sum over heads)
    out.append("\n## Opp-active information reaching the legal move tokens (Σ over heads of measured ‖Δ head out‖ "
               "/ clean ‖head out‖), per layer, selected features\n")
    feats = ["hp", "species_swap", "type_FIRE", "type_WATER", "type_GROUND", "spe_stat", "boost_atk", "substitute", "par",
             "unknown", "pos_slot"]
    out.append("| feature | " + " | ".join(f"L{i}" for i in range(3)) + " | top heads (L2) |\n|---|---|---|---|---|")
    for f in feats:
        k = K.get(f"opp_act|{f}")
        if k is None:
            continue
        v = dh[k][..., 8] / np.maximum(hn[k][..., 8], 1e-6)      # 3,H
        top = np.argsort(-v[2])[:3]
        out.append(f"| {f} | " + " | ".join(f"{v[i].sum():.3f}" for i in range(3)) + " | "
                   + ", ".join(f"H{t} {v[2, t]:.3f}" for t in top) + " |")
    # positional: slot identity vs content as a key feature
    out.append("\n## Slot identity (pos row → group mean) vs content, as a key feature: mean |Δ score| (nats)\n")
    out.append("| head | own_bench pos (oA q) | own_bench species (oA q) | own_bench hp (oA q) | own_bench pos (mL q) | "
               "opp_bench pos (oA q) | opp_bench species (oA q) |")
    out.append("|---|---|---|---|---|---|---|")
    js["pos_vs_content"] = {}
    for i in range(3):
        for h in range(8):
            r = [dSa[K["own_bench|pos_slot"], i, h, 1], dSa[K["own_bench|species_swap"], i, h, 1],
                 dSa[K["own_bench|hp"], i, h, 1], dSa[K["own_bench|pos_slot"], i, h, 8],
                 dSa[K["opp_bench|pos_slot"], i, h, 1], dSa[K["opp_bench|species_swap"], i, h, 1]]
            js["pos_vs_content"][f"L{i}H{h}"] = [float(x) for x in r]
            out.append(f"| L{i}H{h} | " + " | ".join(f"{x:.3f}" for x in r) + " |")
    # positional QK at layer 0 from weights: slot-dependent spread across own slots
    pr, pe = z["pos_qk_real"], z["pos_qk_eq"]                 # 8,17,17
    out.append("\n## Layer-0 QK on (mean content + pos) per slot, from the weights: how much the score to own "
               "slots 1–6 varies with slot (std over own key slots, mean over queries), real pos vs equalised pos\n")
    out.append("| head | std real | std equalised | own-slot score row for own-active-like query (slot 1) |")
    out.append("|---|---|---|---|")
    for h in range(8):
        sr = pr[h][:, 1:7].std(1).mean(); se = pe[h][:, 1:7].std(1).mean()
        out.append(f"| L0H{h} | {sr:.3f} | {se:.3f} | " + " ".join(f"{x:+.2f}" for x in pr[h][1, 1:7]) + " |")
    # behaviour per feature
    out.append("\n## Behaviour effect of each feature (KL of policy hi vs lo, |ΔV|), top 25 by KL\n")
    out.append("| target|feature | KL | |ΔV| | n |\n|---|---|---|---|")
    for k in np.argsort(-z["kl"])[:25]:
        out.append(f"| {keys[k]} | {z['kl'][k]:.3f} | {z['dv'][k]:.3f} | {z['n'][k]} |")
    js["behaviour"] = {keys[k]: [float(z["kl"][k]), float(z["dv"][k])] for k in range(len(keys))}
    (OUT / f"circuits_report_{ck}.md").write_text("\n".join(out) + "\n")
    (OUT / f"circuits_summary_{ck}.json").write_text(json.dumps(js))
    print("\n".join(out))


if __name__ == "__main__":
    main()
