"""edw step 2c (TRAIN games only, reading): is the net's defence weighting wrong in its STATIC part (how type-vulnerable
a candidate is against the opponents it will meet in general) or in its DYNAMIC part (the matchup against THIS
opponent)? static_s = E over the train distribution of opponent-active type sets of the incoming STAB threat on
candidate s; dyn_s = dfn_s − static_s (same split for off: own best effectiveness). Within-decision ranking
regressions (centred) of the net logit, the search's log(share+0.01) and the oracle Q on
[static dfn, dyn dfn, static off, dyn off, spd, hp], voluntary and forced; and how the two readouts (observational,
swap-fitted) load on each part.
    python scripts/wsp/edw_decomp.py 500M
"""
import json
import sys

import numpy as np

from beh_common import CAT, BP, IMMUNE, MTYPE, MULT, TYP, decode, load_local
from edw_common import LOCAL, OUT, LOGM, IMM, log2_stab, matchup, split, M

tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
d = load_local()
N = len(d["mask"]); mask = d["mask"] > 0
tr = split(d["game"])
f = matchup(d["obs"])
o = decode(d["obs"])
nums, mv = o["nums"], o["mv"]
r = np.arange(N)
oppT = nums[r, f["opp_a"], TYP:TYP + 18] > 0.5
ownT = nums[:, :6, TYP:TYP + 18] > 0.5
# distribution of opponent-active type sets (train rows with an opponent active)
keys, inv, cnt = np.unique(oppT[tr & f["has_opp"]], axis=0, return_inverse=True, return_counts=True)
wts = cnt / cnt.sum()
P_ = []
def Pp(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); P_.append(s)
Pp(f"# edw decomposition ({tag}); {len(keys)} distinct opponent type sets in train")
# static dfn: E_opp[log2_stab(opp types, candidate types)]
stat_d = np.zeros((N, 6))
for kk, w in zip(keys, wts):
    stat_d += w * log2_stab(np.broadcast_to(kk, (N, 6, 18)), ownT)
# static off: E_opp[max over damaging moves of log2 eff vs that type set]
om = mv[:, :6]
dmg = (om[..., CAT + 2] < 0.5) & (om[..., BP] > 0) & (om[..., MTYPE:MTYPE + 18].sum(-1) > 0.5)
mt = om[..., MTYPE:MTYPE + 18].argmax(-1)                     # [N,6,4] move type
stat_o = np.zeros((N, 6))
for kk, w in zip(keys, wts):
    tj = np.flatnonzero(kk)
    lm = np.zeros(mt.shape)
    for j in tj:
        lm = lm + np.where(M[mt, j] == 0, -30, LOGM[mt, j])
    lm = np.clip(lm, -3, 2)
    best = np.where(dmg, lm, -np.inf).max(-1)
    stat_o += w * np.where(np.isfinite(best), best, 0.0)
dyn_d, dyn_o = f["dfn"] - stat_d, f["off"] - stat_o
full = np.load(LOCAL / f"{tag}_full.npz")
lgr = np.clip(full["logits"], -80, 80)
sh = d["shares"].astype(np.float64)
Q = np.load("/vol/bitbucket/fr1226/porygon-ws-wsp/out/beh/derived_r2-300k.npz", allow_pickle=True)["lab_q"]
ro = np.load(OUT / f"readouts_{tag}.npz")
fin = full["fin"]
hat = {nm: (fin[:, 1:7].astype(np.float64) - ro["mu"]) @ ro[f"dfn:{nm}"] for nm in ("observational", "swap-fitted", "combined")}
hat.update({f"off {nm}": (fin[:, 1:7].astype(np.float64) - ro["mu"]) @ ro[f"off:{nm}"] for nm in ("observational", "swap-fitted", "combined")})
legsw, legmv = mask[:, :6], mask[:, 6:]
FE = {"static dfn": stat_d, "dyn dfn": dyn_d, "static off": stat_o, "dyn off": dyn_o, "spd": f["spd"], "hp": f["hp"]}
res = {}
for kind, sel in (("voluntary", legsw.any(1) & legmv.any(1)), ("forced", legsw.any(1) & ~legmv.any(1))):
    rows = np.flatnonzero(sel & tr & f["has_opp"] & (legsw.sum(1) >= 2))
    ri, si = np.nonzero(legsw[rows]); ri = rows[ri]
    cn = np.bincount(ri, minlength=N).astype(float)
    cen = lambda v: v - np.bincount(ri, weights=v, minlength=N)[ri] / cn[ri]  # noqa: E731
    X = np.c_[[cen(v[ri, si]) for v in FE.values()]].T
    Pp(f"\n## {kind}: {len(rows)} decisions; sd within decision: " + ", ".join(f"{k} {X[:, i].std():.2f}" for i, k in enumerate(FE)))
    ys = {"net logit": lgr[ri, si], "search log(s+0.01)": np.log(sh[ri, si] + 1e-2), "oracle Q": Q[ri, si]}
    ys.update({f"readout {k}": v[ri, si] for k, v in hat.items()})
    for nm, y in ys.items():
        ok = np.isfinite(y)
        bad = np.bincount(ri, weights=~ok, minlength=N)[ri] > 0
        yc = cen(np.where(ok, y, 0))
        c = np.linalg.lstsq(X[~bad], yc[~bad], rcond=None)[0]
        Pp(f"{nm:28s}: " + ", ".join(f"{k} {c[i]:+.3f}" for i, k in enumerate(FE)))
        res.setdefault(kind, {})[nm] = dict(zip(FE, map(float, c)))
    net, srch, orc = (np.array(list(res[kind][k].values())) for k in ("net logit", "search log(s+0.01)", "oracle Q"))
    Pp("search/net ratio : " + ", ".join(f"{k} {srch[i]/net[i]:.3f}" for i, k in enumerate(FE)))
    Pp("oracle/net ratio : " + ", ".join(f"{k} {orc[i]/net[i]:.4f}" for i, k in enumerate(FE)))
json.dump(res, open(OUT / f"decomp_{tag}.json", "w"), indent=1)
(OUT / f"decomp_{tag}.txt").write_text("\n".join(P_) + "\n")
