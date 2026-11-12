"""edw step 3, design v2 (TRAIN games only). v1 (edw_design.py) added δ·(dfn readout); the decomposition
(edw_decomp.py) showed why that cannot work: the linear dfn readout mostly reads a candidate's STATIC type
vulnerability (E over opponents), which the net already UNDER-weights, while the over-weighted part is the
DYNAMIC matchup against this opponent. v2 therefore writes a combination of six readouts
(dfn observational, dfn swap-fitted, off observational, off swap-fitted, spd probe, hp probe) chosen to move the six
within-decision ranking coefficients [static dfn, dyn dfn, static off, dyn off, spd, hp] to the search's
(× the net/search scale on spd), per context (voluntary V, forced F), solving in closed form
    min_δ Σ_i var_i·(b_net,i + (Rδ)_i − T_i)² + δᵀ Σ_res δ
(R = regression of each readout on the components, var_i = within-decision variance of component i, Σ_res =
covariance of the readouts' unexplained parts: both terms are in logit² per candidate). Gates, units and the ρ_V
rule as in v1. Strengths v2s1/v2s2/v2s3 = 1/3, 2/3, 1 of δ.
    python scripts/wsp/edw_design2.py 500M
"""
import json
import sys

import numpy as np
import torch

from beh_common import CAT, BP, MTYPE, TYP, decode, load_local
from edw_common import LOCAL, OUT, LOGM, M, head_logits, head_weights, log2_stab, log_check, matchup, softmax, split
from edw_lib import gate_direction, make_W, unit_usage
from wsp_common import CKPTS, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
d = load_local()
N = len(d["mask"]); mask = d["mask"] > 0
tr = split(d["game"])
f = matchup(d["obs"])
full = np.load(LOCAL / f"{tag}_full.npz")
fin = full["fin"]
lgr = np.clip(full["logits"], -80, 80)
sh = d["shares"].astype(np.float64)
pol = load(CKPTS.get(tag, tag)); W = head_weights(pol)
txt, res = [], {"tag": tag}


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


# components (as edw_decomp.py)
o = decode(d["obs"]); nums, mv = o["nums"], o["mv"]; r = np.arange(N)
oppT = nums[r, f["opp_a"], TYP:TYP + 18] > 0.5; ownT = nums[:, :6, TYP:TYP + 18] > 0.5
keys, cnt = np.unique(oppT[tr & f["has_opp"]], axis=0, return_counts=True); wts = cnt / cnt.sum()
stat_d = np.zeros((N, 6)); stat_o = np.zeros((N, 6))
om = mv[:, :6]
dmg = (om[..., CAT + 2] < 0.5) & (om[..., BP] > 0) & (om[..., MTYPE:MTYPE + 18].sum(-1) > 0.5)
mt = om[..., MTYPE:MTYPE + 18].argmax(-1)
for kk, w in zip(keys, wts):
    stat_d += w * log2_stab(np.broadcast_to(kk, (N, 6, 18)), ownT)
    lm = np.zeros(mt.shape)
    for j in np.flatnonzero(kk):
        lm = lm + np.where(M[mt, j] == 0, -30, LOGM[mt, j])
    best = np.where(dmg, np.clip(lm, -3, 2), -np.inf).max(-1)
    stat_o += w * np.where(np.isfinite(best), best, 0.0)
COMP = {"static dfn": stat_d, "dyn dfn": f["dfn"] - stat_d, "static off": stat_o, "dyn off": f["off"] - stat_o,
        "spd": f["spd"], "hp": f["hp"]}
# readouts (all as r·(c − μ))
ro = np.load(OUT / f"readouts_{tag}.npz"); pr = np.load(OUT / f"probes_{tag}.npz")
mu = ro["mu"]
RW = {"dfn obs": ro["dfn:observational"], "dfn swap": ro["dfn:swap-fitted"], "off obs": ro["off:observational"],
      "off swap": ro["off:swap-fitted"], "spd": pr["fin:spd:w"], "hp": pr["fin:hp:w"]}
C = fin[:, 1:7].astype(np.float64) - mu
RO = {k: C @ w for k, w in RW.items()}
legsw, legmv = mask[:, :6], mask[:, 6:]
ctx_kind = {"V": legsw.any(1) & legmv.any(1), "F": legsw.any(1) & ~legmv.any(1)}
delta = {}
for kk, sel in ctx_kind.items():
    rows = np.flatnonzero(sel & tr & f["has_opp"] & (legsw.sum(1) >= 2))
    ri, si = np.nonzero(legsw[rows]); ri = rows[ri]
    cn = np.bincount(ri, minlength=N).astype(float)
    cen = lambda v: v - np.bincount(ri, weights=v, minlength=N)[ri] / cn[ri]  # noqa: E731
    X = np.c_[[cen(v[ri, si]) for v in COMP.values()]].T
    Y = np.c_[[cen(v[ri, si]) for v in RO.values()]].T
    var = X.var(0)
    b_net = np.linalg.lstsq(X, cen(lgr[ri, si]), rcond=None)[0]
    b_s = np.linalg.lstsq(X, cen(np.log(sh[ri, si] + 1e-2)), rcond=None)[0]
    scale = b_net[4] / b_s[4]
    T = scale * b_s
    R = np.linalg.lstsq(X, Y, rcond=None)[0]                     # [6 comps, 6 readouts]
    E = Y - X @ R; Sres = E.T @ E / len(E)
    D = np.diag(var)
    dl = np.linalg.solve(R.T @ D @ R + Sres, R.T @ D @ (T - b_net))
    pred = b_net + R @ dl
    noise = float(dl @ Sres @ dl)
    P(f"[{kk}] {len(rows)} decisions; components " + ", ".join(COMP))
    P(f"[{kk}] net    {np.round(b_net, 3)}")
    P(f"[{kk}] target {np.round(T, 3)}  (search × {scale:.2f})")
    P(f"[{kk}] full-move δ over readouts {dict(zip(RW, np.round(dl, 3)))} → predicted coefs {np.round(pred, 3)}; "
      f"added unexplained logit sd per candidate {np.sqrt(noise):.3f} (net centred-logit sd {cen(lgr[ri, si]).std():.2f})")
    delta[kk] = dl
    res[kk] = dict(net=b_net.tolist(), target=T.tolist(), scale=scale, delta=dict(zip(RW, dl.tolist())), predicted=pred.tolist(),
                   noise_sd=np.sqrt(noise), R=R.tolist())

# ------------------------------------------------------------------ units and gates (identical rules to v1)
trr = np.flatnonzero(tr)
sub = np.random.default_rng(0).choice(trr, 8000, replace=False)
kl, act = unit_usage(fin, W, d["mask"], sub)
units = np.argsort(kl)[:4].tolist()
P(f"units {units} (mean-ablation KL {[float(f'{kl[u]:.2e}') for u in units]})")
rng = np.random.default_rng(1)
def pairs(rowsel, which):
    rows = np.flatnonzero(rowsel & tr); m = mask[rows].copy()
    if which == "own":
        m[:, 6:] = False
    elif which == "move":
        m[:, :6] = False
    ri, ai = np.nonzero(m)
    if len(ri) > 60000:
        k = rng.choice(len(ri), 60000, replace=False); ri, ai = ri[k], ai[k]
    return np.c_[fin[rows[ri], 1 + ai].astype(np.float64), fin[rows[ri], 0].astype(np.float64)]
V, Fm = ctx_kind["V"], ctx_kind["F"]
nosw = legmv.any(1) & ~legsw.any(1)
gV = gate_direction(pairs(V, "own"), np.r_[pairs(V, "move"), pairs(Fm, "own"), pairs(nosw, "move")], margin=12.0)
gF = gate_direction(pairs(Fm, "own"), np.r_[pairs(V, "all"), pairs(nosw, "move")], margin=12.0)
P(f"gate V {gV[2]}\ngate F {gF[2]}")
rV = lambda fr: fr * sum(delta["V"][i] * w for i, w in enumerate(RW.values()))  # noqa: E731
rF = lambda fr: fr * sum(delta["F"][i] * w for i, w in enumerate(RW.values()))  # noqa: E731


def build(frac, rhoV):
    W2 = make_W(W, units[:2], gV[0], gV[1], rV(frac), mu, rhoV)
    return make_W(W2, units[2:], gF[0], gF[1], rF(frac), mu, 0.0)


vrows = np.random.default_rng(2).choice(np.flatnonzero(V & tr), 8000, replace=False)
ps0 = softmax(head_logits(fin[vrows], W, d["mask"][vrows]))[:, :6].sum(1).mean()
for nm, frac in (("v2s1", 1 / 3), ("v2s2", 2 / 3), ("v2s3", 1.0)):
    lo, hi = -6.0, 6.0
    for _ in range(18):
        mid = (lo + hi) / 2
        ps = softmax(head_logits(fin[vrows], build(frac, mid), d["mask"][vrows]))[:, :6].sum(1).mean()
        lo, hi = (mid, hi) if ps < ps0 else (lo, mid)
    rho = (lo + hi) / 2
    torch.save(build(frac, rho), LOCAL / f"W_{tag}_{nm}.pt")
    log_check(f"design {nm}: ρ_V bisection (18 edited-head passes)", len(vrows), "TRAIN voluntary subsample", rho=rho)
    P(f"{nm}: fraction {frac:.3f}, ρ_V {rho:+.4f} (train voluntary P(switch) kept at {ps0:.4f})")
    res[nm] = dict(frac=frac, rhoV=rho)
res["units"] = units
json.dump(res, open(OUT / f"design2_{tag}.json", "w"), indent=1, default=float)
(OUT / f"design2_{tag}.txt").write_text("\n".join(txt) + "\n")
