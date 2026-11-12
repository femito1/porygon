"""edw step 3: design the edit on TRAIN games only (reading + closed-form solves; no outcome tuning).

1. Targets (from edw_target2's analysis, recomputed here): within-decision candidate ranking regression of
   centred y on centred [dfn, off, spd, hp], separately for voluntary and forced decisions. Target gains for
   (dfn, off) = search coefficients × s, with s = net/search ratio on spd (the best-determined matchup feature the
   edit leaves alone). Oracle-implied targets are reported as a check.
2. Readouts: the 'combined' (observational + swap) ridge readouts of dfn and off from the final latent
   (edw_target2 §3, fit on train).
3. Gains δ: the edited logits are logit + δ_d·d̂ + δ_o·ô (+ρ), so their regression coefficients are
   coef(net) + δ_d·coef(d̂) + δ_o·coef(ô); solve the 2×2 system for the full move to the target (per context).
4. Gates: Fisher LDA on the head input [c; ctx]: path V on = own-mon tokens in voluntary decisions, off = move
   tokens and own tokens in forced / no-switch decisions; path F on = own tokens in forced decisions, off = every
   token of non-forced decisions.
5. Units: the 4 least-used ActionHead hidden units by mean ablation (train rows).
6. ρ_V: the constant that keeps the TRAIN voluntary P(switch) of the full edit equal to the original's (one scalar,
   solved by bisection on train). Strengths s1/s2/s3 = 1/3, 2/3, 1 of (δ, ρ) — ρ re-solved per strength.
    python scripts/wsp/edw_design.py 500M   -> out/edw/design_500M.json, LOCAL/W_500M_{s1,s2,s3}.pt (head weights)
"""
import json
import sys

import numpy as np
import torch

from beh_common import load_local
from edw_common import LOCAL, OUT, head_logits, head_weights, log_check, matchup, softmax, split
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
Q = np.load("/vol/bitbucket/fr1226/porygon-ws-wsp/out/beh/derived_r2-300k.npz", allow_pickle=True)["lab_q"]
pol = load(CKPTS.get(tag, tag))
W = head_weights(pol)
txt, res = [], {"tag": tag}


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


ro = np.load(OUT / f"readouts_{tag}.npz")
mu = ro["mu"]; w_d, w_o = ro["dfn:combined"], ro["off:combined"]
hat_d = (fin[:, 1:7].astype(np.float64) - mu) @ w_d
hat_o = (fin[:, 1:7].astype(np.float64) - mu) @ w_o
legsw, legmv = mask[:, :6], mask[:, 6:]
ctx_kind = {"V": legsw.any(1) & legmv.any(1), "F": legsw.any(1) & ~legmv.any(1)}
FE = ["dfn", "off", "spd", "hp"]


def ranking_coefs(sel, ys):
    rows = np.flatnonzero(sel & tr & f["has_opp"] & (legsw.sum(1) >= 2))
    ri, si = np.nonzero(legsw[rows]); ri = rows[ri]
    cnt = np.bincount(ri, minlength=N).astype(float)
    def cen(v):
        return v - np.bincount(ri, weights=v, minlength=N)[ri] / cnt[ri]
    X = np.c_[[cen(f[k][ri, si]) for k in FE]].T
    out = {}
    for nm, y in ys.items():
        yy = y[ri, si]
        ok = np.isfinite(yy)
        if nm == "oracle":
            bad = np.bincount(ri, weights=~ok, minlength=N)[ri] > 0
            ok = ~bad; yy = np.where(ok, yy, 0)
        yc = cen(yy)
        out[nm] = np.linalg.lstsq(X[ok], yc[ok], rcond=None)[0]
    return out


ys = {"net": lgr, "search": np.log(sh + 1e-2), "oracle": Q,
      "hat_d": np.c_[hat_d, np.zeros((N, 4))], "hat_o": np.c_[hat_o, np.zeros((N, 4))]}
delta = {}
for kk, sel in ctx_kind.items():
    c = ranking_coefs(sel, ys)
    s_srch = c["net"][2] / c["search"][2]; s_or = c["net"][2] / c["oracle"][2]
    T = s_srch * c["search"][:2]; T_or = s_or * c["oracle"][:2]
    A = np.c_[c["hat_d"][:2], c["hat_o"][:2]]
    dl = np.linalg.solve(A, T - c["net"][:2])
    side = c["net"] + dl[0] * c["hat_d"] + dl[1] * c["hat_o"]
    P(f"[{kk}] net coefs {np.round(c['net'], 3)} (dfn, off, spd, hp); search ×{s_srch:.2f} → target dfn {T[0]:+.2f}, off {T[1]:+.2f}; "
      f"oracle ×{s_or:.1f} → dfn {T_or[0]:+.2f}, off {T_or[1]:+.2f}")
    P(f"[{kk}] readout coefs: d̂ {np.round(c['hat_d'], 3)}, ô {np.round(c['hat_o'], 3)}; full-move gains δ_d {dl[0]:+.3f}, δ_o {dl[1]:+.3f}"
      f" → edited coefs {np.round(side, 3)}")
    delta[kk] = dl
    res[f"target_{kk}"] = dict(net=c["net"].tolist(), search=c["search"].tolist(), oracle=c["oracle"].tolist(), scale_search=s_srch,
                                scale_oracle=s_or, T=T.tolist(), T_oracle=T_or.tolist(), delta=dl.tolist(), edited=side.tolist())

# ------------------------------------------------------------------ units
trr = np.flatnonzero(tr)
sub = np.random.default_rng(0).choice(trr, 8000, replace=False)
kl, act = unit_usage(fin, W, d["mask"], sub)
order = np.argsort(kl)
units = order[:4].tolist()
P(f"least-used ActionHead units (mean-ablation KL, train): " + ", ".join(f"#{u} KL {kl[u]:.2e} (active {act[u]:.3f})" for u in order[:8])
  + f"; median unit KL {np.median(kl):.2e}, max {kl.max():.2e}")
res["units"] = units; res["unit_kl"] = {int(u): float(kl[u]) for u in order[:8]}

# ------------------------------------------------------------------ gates
def head_in(rows, toks):
    """[c; ctx] for the given (row, token-index-in-fin) pairs."""
    c = fin[rows, toks].astype(np.float64); x = fin[rows, 0].astype(np.float64)
    return np.c_[c, x]


rng = np.random.default_rng(1)
def pairs(rowsel, which):  # which: 'own' (legal switch tokens) / 'move' (legal move tokens) / 'all' (legal tokens)
    rows = np.flatnonzero(rowsel & tr)
    m = mask[rows]
    if which == "own":
        m = m.copy(); m[:, 6:] = False
    elif which == "move":
        m = m.copy(); m[:, :6] = False
    ri, ai = np.nonzero(m)
    if len(ri) > 60000:
        k = rng.choice(len(ri), 60000, replace=False); ri, ai = ri[k], ai[k]
    return head_in(rows[ri], 1 + ai)


V, Fm = ctx_kind["V"], ctx_kind["F"]
nosw = legmv.any(1) & ~legsw.any(1)
gV = gate_direction(pairs(V, "own"), np.r_[pairs(V, "move"), pairs(Fm, "own"), pairs(nosw, "move")], margin=12.0)
gF = gate_direction(pairs(Fm, "own"), np.r_[pairs(V, "all"), pairs(nosw, "move")], margin=12.0)
P(f"gate V: {gV[2]}")
P(f"gate F: {gF[2]}")
res["gate_V"], res["gate_F"] = gV[2], gF[2]


def build(frac, rhoV):
    W2 = make_W(W, units[:2], gV[0], gV[1], frac * (delta["V"][0] * w_d + delta["V"][1] * w_o), mu, rhoV, a=1.0)
    W2 = make_W(W2, units[2:], gF[0], gF[1], frac * (delta["F"][0] * w_d + delta["F"][1] * w_o), mu, 0.0, a=1.0)
    return W2


vrows = np.random.default_rng(2).choice(np.flatnonzero(V & tr), 8000, replace=False)
p0 = softmax(head_logits(fin[vrows], W, d["mask"][vrows]))
ps0 = p0[:, :6].sum(1).mean()
for nm, frac in (("s1", 1 / 3), ("s2", 2 / 3), ("s3", 1.0)):
    lo, hi = -3.0, 3.0
    for _ in range(18):
        mid = (lo + hi) / 2
        ps = softmax(head_logits(fin[vrows], build(frac, mid), d["mask"][vrows]))[:, :6].sum(1).mean()
        lo, hi = (mid, hi) if ps < ps0 else (lo, mid)
    rho = (lo + hi) / 2
    W2 = build(frac, rho)
    torch.save(W2, LOCAL / f"W_{tag}_{nm}.pt")
    log_check(f"design {nm}: ρ_V bisection (18 edited-head passes)", len(vrows), "TRAIN voluntary subsample", rho=rho)
    P(f"{nm}: fraction {frac:.3f}; ρ_V {rho:+.4f} keeps train voluntary P(switch) at {ps0:.4f}; "
      f"δ_V = ({frac*delta['V'][0]:+.3f}, {frac*delta['V'][1]:+.3f}), δ_F = ({frac*delta['F'][0]:+.3f}, {frac*delta['F'][1]:+.3f})")
    res[nm] = dict(frac=frac, rhoV=rho, dV=(frac * delta["V"]).tolist(), dF=(frac * delta["F"]).tolist())
json.dump(res, open(OUT / f"design_{tag}.json", "w"), indent=1, default=float)
(OUT / f"design_{tag}.txt").write_text("\n".join(txt) + "\n")
