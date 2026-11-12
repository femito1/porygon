"""edw step 2b (TRAIN games only, reading): separate temperature from re-weighting, and add the oracle as a reference.

1. Candidate ranking (decisions with ≥ 2 legal switches; voluntary and forced separately): centred over the legal
   switch candidates, regress the reference on [net logit, dfn, off, spd, hp]. Reference = search log(share+ε),
   oracle Q (r2-300k). A feature coefficient ≠ 0 given the net's logit = the net mis-weights that feature in its
   ranking; (coef / logit coef) = the correction in net-logit units. Also the scale-free ratios dfn/off.
2. Switch vs stay (voluntary): reference log(Σ switch share) − log(Σ move share) on [net's same quantity, dfn_act,
   off_act, spd_act, hp_act, best-candidate dfn/off].
3. Readout fidelity for the edit (train swaps → test swaps): observational ridge probe vs swap-fitted readout vs
   combined; slope and corr of Δreadout on Δdfn, Δoff under the TEST swaps, and R² on test rows.
4. Weight reading of L1 QK: bilinear (candidate type i × opp-active type j) score map of each L1 head from type
   directions at the L1 input (regressed on 2,000 train rows), vs the gen4 log₂ type chart.
    python scripts/wsp/edw_target2.py 500M
"""
import json
import sys

import numpy as np
import torch

from beh_common import NUM_OFF, TYP, load_local
from edw_common import LOCAL, OUT, LOGM, matchup, split
from porygon.encoder import FIELD_DIM, MON_DIM
from wsp_common import CKPTS, Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
d = load_local()
N = len(d["mask"]); mask = d["mask"] > 0
tr = split(d["game"])
f = matchup(d["obs"])
full = np.load(LOCAL / f"{tag}_full.npz")
lgr = np.clip(full["logits"], -80, 80)
sh = d["shares"].astype(np.float64)
Q = np.load("/vol/bitbucket/fr1226/porygon-ws-wsp/out/beh/derived_r2-300k.npz", allow_pickle=True)["lab_q"]
txt, res = [], {}


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


def boot_ols(y, X, g, nb=200):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, g = y[ok], X[ok], g[ok]
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    ug, inv = np.unique(g, return_inverse=True)
    by = [np.flatnonzero(inv == k) for k in range(len(ug))]
    rng = np.random.default_rng(0)
    bs = np.array([np.linalg.lstsq(X[ii], y[ii], rcond=None)[0] for ii in
                   (np.concatenate([by[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(nb))])
    return c, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), bs, ok.sum()


legsw, legmv = mask[:, :6], mask[:, 6:]
FE = ["dfn", "off", "spd", "hp"]
P(f"# edw target2 ({tag}), TRAIN games only")
for kind, sel in (("voluntary", legsw.any(1) & legmv.any(1)), ("forced", legsw.any(1) & ~legmv.any(1))):
    rows = np.flatnonzero(sel & tr & f["has_opp"] & (legsw.sum(1) >= 2))
    ri, si = np.nonzero(legsw[rows]); ri = rows[ri]
    cnt = np.bincount(ri, minlength=N).astype(float)
    def cen(v):
        s = np.bincount(ri, weights=v, minlength=N); return v - s[ri] / cnt[ri]
    X = np.c_[[cen(lgr[ri, si])] + [cen(f[k][ri, si]) for k in FE]].T
    g = d["game"][ri]
    P(f"\n## 1. candidate ranking, {kind}: {len(rows)} decisions, {len(ri)} candidates (centred within decision)")
    Xn = X[:, 1:]
    c, lo, hi, _, n = boot_ols(cen(lgr[ri, si]), Xn, g)
    P(f"net logit ~ features            : " + ", ".join(f"{k} {c[i]:+.2f} [{lo[i]:+.2f},{hi[i]:+.2f}]" for i, k in enumerate(FE))
      + f"   dfn/off {c[0]/c[1]:+.2f}")
    res.setdefault(kind, {})["net~feat"] = dict(zip(FE, map(float, c)))
    refs = {"search log(s+0.01)": np.log(sh[ri, si] + 1e-2), "search log(s+0.001)": np.log(sh[ri, si] + 1e-3),
            "oracle Q": Q[ri, si]}
    for nm, y in refs.items():
        yc = cen(np.nan_to_num(y, nan=np.nan))
        if nm == "oracle Q":  # rows with all candidates labelled
            bad = np.bincount(ri, weights=~np.isfinite(y), minlength=N)[ri] > 0
            yc = np.where(bad, np.nan, cen(np.where(np.isfinite(y), y, 0)))
        c, lo, hi, _, n = boot_ols(yc, Xn, g)
        P(f"{nm:20s} ~ features       : " + ", ".join(f"{k} {c[i]:+.3f} [{lo[i]:+.3f},{hi[i]:+.3f}]" for i, k in enumerate(FE))
          + f"   dfn/off {c[0]/c[1]:+.2f}  (n {n})")
        res[kind][f"{nm}~feat"] = dict(zip(FE, map(float, c)))
        c, lo, hi, bs, n = boot_ols(yc, X, g)
        corr = c[1:] / c[0]; bc = bs[:, 1:] / bs[:, :1]
        P(f"{nm:20s} ~ logit+features : logit {c[0]:+.3f}; " + ", ".join(
            f"{k} {c[i+1]:+.3f} → {corr[i]:+.2f} net-logit units [{np.percentile(bc[:, i], 2.5):+.2f},{np.percentile(bc[:, i], 97.5):+.2f}]"
            for i, k in enumerate(FE)))
        res[kind][f"{nm}~logit+feat"] = dict(logit=float(c[0]), **{k: float(corr[i]) for i, k in enumerate(FE)})

# ------------------------------------------------------------------ 2. switch vs stay
rows = np.flatnonzero(legsw.any(1) & legmv.any(1) & tr & f["has_opp"] & f["has_act"])
def lse(a, m):
    a = np.where(m, a, -np.inf); mx = a.max(1, keepdims=True); return (mx + np.log(np.exp(a - mx).sum(1, keepdims=True)))[:, 0]
lg = np.where(mask, lgr, -np.inf)
net_m = lse(lg[rows, :6], legsw[rows]) - lse(lg[rows, 6:], legmv[rows])
S = sh[rows]
srch_m = np.log(np.where(legsw[rows], S[:, :6], 0).sum(1) + 1e-2) - np.log(np.where(legmv[rows], S[:, 6:], 0).sum(1) + 1e-2)
bestd = np.where(legsw[rows], f["dfn"][rows], np.inf).min(1)
besto = np.where(legsw[rows], f["off"][rows], -np.inf).max(1)
RF = ["dfn_act", "off_act", "spd_act", "hp_act"]
Xr = np.c_[[f[k][rows] for k in RF] + [bestd, besto, np.ones(len(rows))]].T
names = RF + ["best cand dfn", "best cand off", "const"]
g = d["game"][rows]
P(f"\n## 2. switch vs stay (voluntary, {len(rows)} decisions): margin = log P(switch) − log P(move)")
c, lo, hi, _, _ = boot_ols(net_m, Xr, g)
P("net (logits)          : " + ", ".join(f"{k} {c[i]:+.2f} [{lo[i]:+.2f},{hi[i]:+.2f}]" for i, k in enumerate(names)))
c, lo, hi, _, _ = boot_ols(srch_m, Xr, g)
P("search log(Σs+0.01)    : " + ", ".join(f"{k} {c[i]:+.2f} [{lo[i]:+.2f},{hi[i]:+.2f}]" for i, k in enumerate(names)))
c, lo, hi, bs, _ = boot_ols(srch_m, np.c_[net_m, Xr], g)
P("search ~ net margin + : " + f"net margin {c[0]:+.3f}; " + ", ".join(
    f"{k} {c[i+1]:+.3f} → {c[i+1]/c[0]:+.2f} net units" for i, k in enumerate(names)))

# ------------------------------------------------------------------ 3. readout fidelity
P("\n## 3. readouts of dfn / off from the final latent (fit on train, judged on TEST swaps and TEST rows)")
ztr, zte = np.load(LOCAL / f"{tag}_swap_train.npz"), np.load(LOCAL / f"{tag}_swap_test.npz")
pr = np.load(OUT / f"probes_{tag}.npz")
fin = full["fin"]
def swap_parts(z):
    MK = d["mask"][z["rows"]] > 0; leg = MK[:, :6]
    f0, ff = matchup(z["obs0"].astype(np.float32)), matchup(z["obsf"].astype(np.float32))
    dC = (z["fin_f"][:, 1:7] - z["fin_0"][:, 1:7]).astype(np.float64)
    m_ = leg[..., None]
    dCc = dC - (dC * m_).sum(1, keepdims=True) / m_.sum(1, keepdims=True)
    out = {"dC": dC[leg], "dCc": dCc[leg]}
    for k in ("dfn", "off"):
        dv = ff[k] - f0[k]
        out[k] = dv[leg]
        out[k + "_c"] = (dv - (dv * leg).sum(1, keepdims=True) / leg.sum(1, keepdims=True))[leg]
    return out
A, B = swap_parts(ztr), swap_parts(zte)
tok = mask[:, :6] & f["has_opp"][:, None]
tri, tei = np.nonzero(tok & tr[:, None]), np.nonzero(tok & ~tr[:, None])
Ctr = fin[:, 1:7][tri].astype(np.float64); Cte = fin[:, 1:7][tei].astype(np.float64)
mu = Ctr.mean(0)
G_obs = (Ctr - mu).T @ (Ctr - mu) / len(Ctr)
G_sw = A["dCc"].T @ A["dCc"] / len(A["dCc"])
readouts = {}
for k in ("dfn", "off"):
    y = f[k][tri]
    b_obs = (Ctr - mu).T @ (y - y.mean()) / len(Ctr)
    b_sw = A["dCc"].T @ A[k + "_c"] / len(A["dCc"])
    lam_o = 1e-3 * np.trace(G_obs) / 256; lam_s = 1e-3 * np.trace(G_sw) / 256
    for nm, w in (("observational", np.linalg.solve(G_obs + lam_o * np.eye(256), b_obs)),
                  ("swap-fitted", np.linalg.solve(G_sw + lam_s * np.eye(256), b_sw)),
                  ("combined", np.linalg.solve(G_obs + G_sw + (lam_o + lam_s) * np.eye(256), b_obs + b_sw))):
        dr = B["dCc"] @ w; dt = B[k + "_c"]
        slope = float((dr * dt).sum() / (dt * dt).sum()); rc = float(np.corrcoef(dr, dt)[0, 1])
        pred = (Cte - mu) @ w; yt = f[k][tei]
        bb = float(np.polyfit(pred, yt, 1)[0]); r2 = float(np.corrcoef(pred, yt)[0, 1] ** 2)
        # other feature leakage under swap
        ok_ = [kk for kk in ("dfn", "off") if kk != k][0]
        leak = float(np.linalg.lstsq(np.c_[B[k + "_c"], B[ok_ + "_c"]], dr, rcond=None)[0][1])
        P(f"{k} {nm:13s}: test swap slope {slope:+.3f} (corr {rc:.3f}; leak per unit Δ{ok_} {leak:+.3f}); test rows corr² {r2:.3f}")
        readouts[f"{k}:{nm}"] = w
        res.setdefault("readout", {})[f"{k}:{nm}"] = dict(swap_slope=slope, swap_corr=rc, r2=r2, leak=leak)
np.savez(OUT / f"readouts_{tag}.npz", mu=mu, **readouts)

if tag == "500M":
    # ------------------------------------------------------------------ 4. L1 QK type-chart reading
    P("\n## 4. L1 QK: bilinear (candidate type × opp-active type) score map from the weights vs the type chart")
    net = Net(load(CKPTS.get(tag, tag)))
    rr = np.flatnonzero(tr & f["has_opp"])[:2000]
    x = torch.as_tensor(d["obs"][rr].astype(np.float32)); m = torch.as_tensor(d["mask"][rr])
    c = {}
    with torch.no_grad():
        net.forward(x, m, None, c)
        a1 = net.layers[1].norm1(c["L0.out"]).double().numpy()     # L1 attention input [n,17,256]
    nums = d["obs"][rr, FIELD_DIM:].reshape(len(rr), 12, MON_DIM)[..., NUM_OFF:NUM_OFF + 125].astype(np.float64)
    types = nums[..., TYP:TYP + 18]
    oa = f["opp_a"][rr]
    # candidate (own alive bench, legal switch) type directions, controlling for hp/active/fainted flags
    ii, ss = np.nonzero(mask[rr][:, :6])
    Xc = np.c_[types[ii, ss], nums[ii, ss, 0], np.ones(len(ii))]
    Tc = np.linalg.lstsq(Xc, a1[ii, 1 + ss], rcond=None)[0][:18]              # [18,256]
    Xo = np.c_[types[np.arange(len(rr)), oa], nums[np.arange(len(rr)), oa, 0], np.ones(len(rr))]
    To = np.linalg.lstsq(Xo, a1[np.arange(len(rr)), 1 + oa], rcond=None)[0][:18]
    present = types[ii, ss].sum(0) > 20
    present_o = types[np.arange(len(rr)), oa].sum(0) > 20
    L = net.layers[1]
    Wq = L.self_attn.in_proj_weight[:256].detach().double().numpy(); Wk = L.self_attn.in_proj_weight[256:512].detach().double().numpy()
    chart = np.maximum(LOGM, -3.0).T   # [defender type i, attacker type j]: log2 multiplier of attack type j on defender type i
    mm = np.outer(present, present_o)
    for h in range(8):
        sl = slice(32 * h, 32 * h + 32)
        Bm = (Tc @ Wq[sl].T) @ (Wk[sl] @ To.T) / np.sqrt(32)                  # [cand type i, opp type j]
        Bc = Bm - Bm.mean(0, keepdims=True) - Bm.mean(1, keepdims=True) + Bm.mean()   # interaction part only
        Cc = chart - chart.mean(0, keepdims=True) - chart.mean(1, keepdims=True) + chart.mean()
        r = np.corrcoef(Bc[mm], Cc[mm])[0, 1]
        slope = (Bc[mm] * Cc[mm]).sum() / (Cc[mm] ** 2).sum()
        P(f"L1 h{h}: corr(interaction part of the QK type map, log₂ chart) {r:+.3f}; slope {slope:+.3f} nats per doubling; "
          f"interaction sd {Bc[mm].std():.3f} nats")
        res.setdefault("qk", {})[f"h{h}"] = dict(r=float(r), slope=float(slope))
    # measured: within-row centred Δ pre-softmax pattern weight (log attention to opp active) under the swaps vs Δdfn
    z = ztr
    MK = d["mask"][z["rows"]] > 0; leg = MK[:, :6]
    f0, ff = matchup(z["obs0"].astype(np.float32)), matchup(z["obsf"].astype(np.float32))
    dd = ff["dfn"] - f0["dfn"]
    for h in range(8):
        dl = np.log(z["L1pat_f"][:, h] + 1e-9) - np.log(z["L1pat_0"][:, h] + 1e-9)
        def cc(a):
            a = np.where(leg, a, np.nan); return a - np.nanmean(a, 1, keepdims=True)
        a_, b_ = cc(dl), cc(dd); ok = np.isfinite(a_)
        P(f"L1 h{h}: measured centred Δ log attention(candidate → opp active) per doubling of dfn "
          f"{(a_[ok]*b_[ok]).sum()/(b_[ok]**2).sum():+.3f} (corr {np.corrcoef(a_[ok], b_[ok])[0,1]:+.3f})")


json.dump(res, open(OUT / f"target2_{tag}.json", "w"), indent=1, default=float)
(OUT / f"target2_{tag}.txt").write_text("\n".join(txt) + "\n")
