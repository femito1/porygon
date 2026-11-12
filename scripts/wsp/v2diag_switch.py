"""Over-switching check for v2 (the v1 type-chart defence heuristic): swap the opponent's active mon (all of its
per-mon fields) with another position's opponent active and measure how each legal switch candidate's logit
(relative to the mean legal logit) moves with
  Δ log2 threat  = change of the opponent's best type multiplier on the candidate (max over its types, product over
                   the candidate's types; immunities floored at 1/4), and
  Δ log2 offence = change of the candidate's best type multiplier on the opponent (same definition, reversed).
OLS slope per checkpoint, 95% interval by bootstrap over games. v1 500M (part1-beh, same swap design, offence from
its moves): −1.90 per doubling of threat, +0.67 per doubling of offence.

    python scripts/wsp/v2diag_switch.py --pos v2diag/pos --out v2diag/r1/out CKPT [CKPT ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from poke_env.data import GenData

import porygon.v2.model as M
from porygon.v2.spec import MON_NUM, TYPES

COL = {n: i for i, n in enumerate(MON_NUM)}
from v2diag import load_pos


def chart(gen):
    tid = {t: i + 1 for i, t in enumerate(TYPES)}
    c = np.ones((len(TYPES) + 1, len(TYPES) + 1), np.float32)  # [attacking, defending], 0 = none
    for dname, r in GenData.from_gen(gen).type_chart.items():
        for aname, v in r.items():
            a, d = tid.get(aname.lower()), tid.get(dname.lower())
            if a and d:
                c[a, d] = v
    return c


def best_mult(att_types, def_types, c):
    best = 0.0
    for a in att_types:
        if a <= 0:
            continue
        m = 1.0
        for d in def_types:
            if d > 0:
                m *= c[a, d]
        best = max(best, m)
    return max(best, 0.25) if best > 0 else 1.0


def opp_active(o):
    mn = o["mon_num"]
    a = mn[6:, COL["active"]] * mn[6:, COL["present"]] * (1 - mn[6:, COL["fainted"]])
    return 6 + int(a.argmax()) if a.max() > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--pos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--gen", type=int, default=4)
    a = ap.parse_args()
    c = chart(a.gen)
    rows = load_pos(a.pos)
    rng = np.random.default_rng(0)
    cand = [i for i, r in enumerate(rows) if r["obs"]["action_mask"][:6].sum() >= 2 and opp_active(r["obs"]) is not None]
    rec = rng.choice(cand, min(a.n, len(cand)), replace=False)
    donors = rng.choice(cand, len(rec))
    base, swap, meta = [], [], []
    for i, j in zip(rec, donors):
        o, d = rows[i]["obs"], rows[j]["obs"]
        si, sj = opp_active(o), opp_active(d)
        if o["species"][si] == d["species"][sj]:
            continue
        s = {k: v.copy() for k, v in o.items()}
        for k, v in s.items():
            if v.ndim >= 1 and v.shape[0] == 12:
                v[si] = d[k][sj]
        legal = np.where(o["action_mask"][:6] > 0)[0]
        for k in legal:
            ct = o["types"][k]
            dthr = np.log2(best_mult(d["types"][sj], ct, c)) - np.log2(best_mult(o["types"][si], ct, c))
            doff = np.log2(best_mult(ct, d["types"][sj], c)) - np.log2(best_mult(ct, o["types"][si], c))
            meta.append((len(base), int(k), dthr, doff, hash(rows[i]["gid"]) % (2**31)))
        base.append(o)
        swap.append(s)
    meta = np.array(meta, dtype=np.float64)
    res = {"n_positions": len(base), "n_candidates": len(meta)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ck in a.ckpts:
        net = M.load(ck)
        def rel(obs):
            outl = []
            with torch.no_grad():
                for s in range(0, len(obs), 256):
                    t = M.obs_to_tensors(obs[s:s + 256], "cpu")
                    lg = net(t, aux=False)["logits"][:, :10].numpy()
                    m = t["action_mask"][:, :10].numpy() > 0
                    mean = np.where(m, lg, 0).sum(1) / m.sum(1)
                    outl.append(lg - mean[:, None])
            return np.concatenate(outl)
        r0, r1 = rel(base), rel(swap)
        p, k = meta[:, 0].astype(int), meta[:, 1].astype(int)
        y = r1[p, k] - r0[p, k]
        X = np.column_stack([meta[:, 2], meta[:, 3], np.ones(len(meta))])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        g = meta[:, 4]
        ug = np.unique(g)
        idx_by_g = {u: np.where(g == u)[0] for u in ug}
        bs = []
        for _ in range(300):
            sel = np.concatenate([idx_by_g[u] for u in rng.choice(ug, len(ug))])
            bs.append(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0][:2])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        res[Path(ck).name] = {"threat_slope": float(coef[0]), "threat_ci": [float(lo[0]), float(hi[0])],
                              "offence_slope": float(coef[1]), "offence_ci": [float(lo[1]), float(hi[1])],
                              "ratio_threat_over_offence": float(-coef[0] / coef[1]) if coef[1] != 0 else None}
        print(f"{ck}: threat {coef[0]:+.3f} [{lo[0]:+.3f}, {hi[0]:+.3f}]  offence {coef[1]:+.3f} [{lo[1]:+.3f}, {hi[1]:+.3f}]")
    (out / "switch_sensitivity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
