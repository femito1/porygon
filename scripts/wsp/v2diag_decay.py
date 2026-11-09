"""What changed in the policy between a reference checkpoint and later ones (R1 degradation after ~7.5M):
per checkpoint on the fixed positions, probability mass by action class (switch / attack move / status move),
entropy, logit scale (std of legal logits, top-1 − top-2 gap, per-class mean relative logit), the best global
temperature mapping the reference to it (and the KL left after that: ~0 means a pure scale collapse), argmax
agreement with the reference, value AUC.

    python scripts/wsp/v2diag_decay.py --ref REF.pt --pos v2diag/pos CKPT [CKPT ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import porygon.v2.model as M
from porygon.v2.spec import MON_NUM
from porygon.v2.tables import move_raw
from v2diag import auc, load_pos

COL = {n: i for i, n in enumerate(MON_NUM)}


def logits_values(net, rows, bs=512):
    L, V = [], []
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            o = net(M.obs_to_tensors([r["obs"] for r in rows[s:s + bs]], "cpu"), aux=False)
            L.append(o["logits"][:, :10].numpy().astype(np.float64))
            V.append(o["value"].numpy())
    return np.concatenate(L), np.concatenate(V)


def softmax(lg, mask, T=1.0):
    z = np.where(mask, lg / T, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.where(mask, np.exp(z), 0.0)
    return p / p.sum(1, keepdims=True)


def kl(p, q):
    return float(np.mean(np.sum(np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(q, 1e-300))), 0), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12000)
    a = ap.parse_args()
    rows = load_pos(a.pos)
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.n, len(rows)), replace=False))]
    mask = np.stack([r["obs"]["action_mask"][:10] for r in rows]) > 0
    cat = move_raw(4)["category"]
    cls = np.zeros((len(rows), 10), np.int64)  # 0 switch, 1 attack, 2 status, 3 unknown move
    for i, r in enumerate(rows):
        mn = r["obs"]["mon_num"]
        act = mn[:6, COL["active"]] * mn[:6, COL["present"]]
        ids = r["obs"]["moves"][int(act.argmax()), :4] if act.max() > 0 else np.zeros(4, int)
        for j in range(4):
            c = cat[int(ids[j])] if 0 <= int(ids[j]) < len(cat) else 0
            cls[i, 6 + j] = 1 if c in (1, 2) else (2 if c == 3 else 3)
    vol = mask[:, :6].any(1) & mask[:, 6:].any(1)
    won = np.array([r["won"] for r in rows])
    nleg = mask.sum(1)
    ref_l, _ = logits_values(M.load(a.ref), rows)
    p_ref = softmax(ref_l, mask)
    res = {}
    names = ["switch", "attack", "status", "move?"]
    print(f"{'ckpt':34s} {'H':>5s} {'H/Hmax':>6s} {'std':>5s} {'gap':>5s} | mass all: " + " ".join(f"{n:>7s}" for n in names)
          + " | mass vol: " + " ".join(f"{n:>7s}" for n in names[:3]) + " | rel logit: " + " ".join(f"{n:>7s}" for n in names[:3])
          + f" | {'T*':>5s} {'KL':>6s} {'KL|T*':>6s} {'top1=ref':>8s} {'AUC':>5s}")
    for ck in [a.ref] + a.ckpts:
        lg, v = logits_values(M.load(ck), rows)
        p = softmax(lg, mask)
        H = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), 1)
        lm = np.where(mask, lg, np.nan)
        rel = lg - np.nanmean(lm, 1, keepdims=True)
        std = np.nanstd(lm, 1)
        srt = np.sort(np.where(mask, lg, -np.inf), 1)
        gap = np.where(nleg >= 2, srt[:, -1] - srt[:, -2], np.nan)
        mass = [float(np.mean(np.sum(p * (cls == c), 1))) for c in range(4)]
        massv = [float(np.mean(np.sum(p[vol] * (cls[vol] == c), 1))) for c in range(3)]
        relc = [float(np.nanmean(np.where(mask & (cls == c), rel, np.nan))) for c in range(3)]
        Ts = np.exp(np.linspace(np.log(0.2), np.log(5), 61))
        kls = [kl(p, softmax(ref_l, mask, T)) for T in Ts]
        Tb = float(Ts[int(np.argmin(kls))])
        st = {"entropy": float(H.mean()), "entropy_over_max": float(np.mean(H[nleg >= 2] / np.log(nleg[nleg >= 2]))),
              "logit_std": float(np.nanmean(std)), "top_gap": float(np.nanmean(gap)),
              "mass_all": dict(zip(names, mass)), "mass_voluntary": dict(zip(names[:3], massv)),
              "rel_logit": dict(zip(names[:3], relc)), "T_star": Tb, "KL_ref_to_ck": kl(p, p_ref),
              "KL_after_T": float(min(kls)), "top1_eq_ref": float(np.mean(p.argmax(1) == p_ref.argmax(1))),
              "value_auc": auc(v, won), "value_std": float(v.std())}
        # where the entropy rose: by reference sharpness and by decision type
        dH = H - (-np.sum(np.where(p_ref > 0, p_ref * np.log(np.maximum(p_ref, 1e-300)), 0), 1))
        forced = ~mask[:, 6:].any(1)
        st["dH_by_type"] = {"forced switch": float(dH[forced].mean()), "voluntary": float(dH[vol].mean()),
                            "moves only": float(dH[~mask[:, :6].any(1)].mean())}
        res[Path(ck).name] = st
        print(f"{Path(ck).name:34s} {st['entropy']:5.3f} {st['entropy_over_max']:6.3f} {st['logit_std']:5.2f} {st['top_gap']:5.2f} | "
              + " ".join(f"{x:7.3f}" for x in mass) + " | " + " ".join(f"{x:7.3f}" for x in massv) + " | "
              + " ".join(f"{x:+7.2f}" for x in relc)
              + f" | {Tb:5.2f} {st['KL_ref_to_ck']:6.3f} {st['KL_after_T']:6.3f} {st['top1_eq_ref']:8.3f} {st['value_auc']:5.3f}")
        print(f"{'':34s} ΔH vs ref by type: {st['dH_by_type']}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "decay.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
