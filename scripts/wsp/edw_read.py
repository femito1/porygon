"""edw step 1: read the switch-matchup circuit (reading only; no edited weights are evaluated here).

A. beh's swap measurement redone with edw's features (train and test swap halves): centred Δ switch logit on
   centred Δ incoming-STAB threat (dfn), Δ best own effectiveness (off), Δ speed ratio (spd).
B. Causal directions: Δ residual at the switch tokens (swap - orig) regressed on Δ(dfn, off, spd) at L1mid, L1out,
   L2out and the final latent; raw and centred within the row (centring removes what is common to all candidates,
   i.e. tells whether the candidate-specific (bilinear) part is present at that depth).
C. Observational linear probes (ridge, fit on TRAIN games, R² on TEST games) of dfn/off/spd/hp/faster at each depth;
   the final-latent probes are saved for the edit; their slope under the swap (does the probe follow the causal change?).
D. ActionHead read: per-token gradient of the logit w.r.t. the candidate latent, applied to the final-latent causal
   directions, at switch tokens and (counterfactually) at move tokens; move tokens' projection on the probes.
E. Value head read of the same directions (autograd on 512 test rows).
F. Layer-1 heads: measured per-head Δ write at the switch tokens projected on the L1out dfn probe, against the
   weight-only OV prediction pattern(s→opp active) · W_O W_V Δnorm1(opp-active token).
    python scripts/wsp/edw_read.py 500M      (600M: A, C, D only)
"""
import json
import sys

import numpy as np
import torch

from beh_common import load_local
from edw_common import LOCAL, OUT, head_weights, matchup, split
from wsp_common import CKPTS, Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
deep = tag == "500M"
OUT.mkdir(parents=True, exist_ok=True)
txt, res = [], {}


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


d = load_local()
N = len(d["mask"]); mask = d["mask"] > 0
tr = split(d["game"])
f = matchup(d["obs"])
pol = load(CKPTS.get(tag, tag)); net = Net(pol)
W = head_weights(pol)
full = np.load(LOCAL / f"{tag}_full.npz")
fin = full["fin"]
FEATS = ["dfn", "off", "spd"]


def gelu_d(x):
    t = torch.as_tensor(x)
    return (torch.special.ndtr(t) + t * torch.exp(-t * t / 2) / np.sqrt(2 * np.pi)).numpy()


def lstsq(A, Y):
    return np.linalg.lstsq(A, Y, rcond=None)[0]


# ------------------------------------------------------------------ A. swap coefficients
P(f"# edw read ({tag})")
P("\n## A. opponent-active swap: centred Δ switch logit on centred Δ features (per doubling)")
sw = {}
for half in ("train", "test"):
    z = np.load(LOCAL / f"{tag}_swap_{half}.npz")
    rows = z["rows"]; MK = d["mask"][rows] > 0
    f0, ff = matchup(z["obs0"].astype(np.float32)), matchup(z["obsf"].astype(np.float32))
    leg = MK[:, :6]
    def cen(a):
        a = np.where(leg, a, np.nan); return a - np.nanmean(a, 1, keepdims=True)
    dl = cen(z["lgf"][:, :6] - z["lg0"][:, :6])
    X = {k: cen(ff[k] - f0[k]) for k in ("dfn", "off", "spd", "off_pow")}
    ok = np.isfinite(dl)
    for spec in (("dfn", "off"), ("dfn", "off_pow"), ("dfn", "off", "spd")):
        A = np.stack([X[k][ok] for k in spec], 1)
        c = lstsq(A, dl[ok])
        r = np.corrcoef(A @ c, dl[ok])[0, 1]
        P(f"{half}: " + ", ".join(f"{k} {v:+.3f}" for k, v in zip(spec, c)) + f"  (r {r:.3f}, {ok.sum()} candidate tokens)")
        res.setdefault("A", {})[f"{half}:{'+'.join(spec)}"] = dict(zip(spec, map(float, c)), r=float(r))
    sw[half] = dict(z=z, f0=f0, ff=ff, leg=leg, rows=rows, MK=MK)

# ------------------------------------------------------------------ C. probes (do first: B and F use them)
P("\n## C. linear probes on switch-candidate tokens (ridge; fit TRAIN, R² TEST)")
tok_ok = mask[:, :6] & f["has_opp"][:, None]
depths = ["L1mid", "L1out", "L2out", "fin"] if deep else ["fin"]
PF = ["dfn", "off", "spd", "hp", "faster"]
probes = {}
for dep in depths:
    R = full[dep][:, :6] if dep != "fin" else fin[:, 1:7]
    out = {}
    tri, tei = np.nonzero(tok_ok & tr[:, None]), np.nonzero(tok_ok & ~tr[:, None])
    Xtr = R[tri].astype(np.float64); Xte = R[tei].astype(np.float64)
    mu = Xtr.mean(0)
    Xtr_c, Xte_c = Xtr - mu, Xte - mu
    G = Xtr_c.T @ Xtr_c
    lam = 1e-3 * np.trace(G) / G.shape[0]
    for k in PF:
        y = f[k][tri].astype(np.float64); yte = f[k][tei]
        wv = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xtr_c.T @ (y - y.mean()))
        pred = Xte_c @ wv + y.mean()
        r2 = 1 - np.mean((yte - pred) ** 2) / np.var(yte)
        out[k] = float(r2)
        probes[(dep, k)] = (wv, float(y.mean() - mu @ wv))
    P(f"{dep}: " + ", ".join(f"{k} R² {v:.3f}" for k, v in out.items()) + f"  (n train {len(tri[0])}, test {len(tei[0])})")
    res.setdefault("C", {})[dep] = out
P("feature sd on test tokens: " + ", ".join(f"{k} {np.std(f[k][np.nonzero(tok_ok & ~tr[:, None])]):.3f}" for k in PF))
np.savez(OUT / f"probes_{tag}.npz", **{f"{dep}:{k}:w": probes[(dep, k)][0] for dep, k in probes},
         **{f"{dep}:{k}:b": probes[(dep, k)][1] for dep, k in probes})

# probe slope under the swap (test half): does the observational read follow the causal change?
z = sw["test"]
fin0, finf = z["z"]["fin_0"].astype(np.float64), z["z"]["fin_f"].astype(np.float64)
leg = z["leg"]
for k in ("dfn", "off", "spd"):
    w_, b_ = probes[("fin", k)]
    dp = (finf[:, 1:7] - fin0[:, 1:7]) @ w_
    dt = z["ff"][k] - z["f0"][k]
    ok = leg & (np.abs(dt) > 0)
    slope = float((dp[ok] * dt[ok]).sum() / (dt[ok] ** 2).sum())
    P(f"fin probe '{k}' under the swap (test): Δprobe = {slope:.3f}·Δ{k} (corr {np.corrcoef(dp[ok], dt[ok])[0,1]:.3f})")
    res.setdefault("C_swap_slope", {})[k] = slope

# ------------------------------------------------------------------ B. causal directions
U_fin = None
for half in (("train",) if deep else ()):
    P("\n## B. causal directions (train swaps): Δ residual at legal switch tokens on Δ(dfn, off, spd)")
    z = sw[half]; zz = z["z"]; leg = z["leg"]
    dF = np.stack([z["ff"][k] - z["f0"][k] for k in FEATS], -1)  # [n,6,3]
    for dep in depths:
        R0 = (zz[f"{dep}_0"][:, :6] if dep != "fin" else zz["fin_0"][:, 1:7]).astype(np.float64)
        Rf = (zz[f"{dep}_f"][:, :6] if dep != "fin" else zz["fin_f"][:, 1:7]).astype(np.float64)
        dR = Rf - R0
        for centred in (False, True):
            if centred:
                m_ = leg[..., None]
                dRc = dR - (dR * m_).sum(1, keepdims=True) / m_.sum(1, keepdims=True)
                dFc = dF - (dF * m_).sum(1, keepdims=True) / m_.sum(1, keepdims=True)
            else:
                dRc, dFc = dR, dF
            Y, X = dRc[leg], dFc[leg]
            A = np.c_[X, np.ones(len(X))] if not centred else X
            U = lstsq(A, Y)
            r2 = 1 - ((Y - A @ U) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
            # how much of the dfn probe change at this depth it explains
            nrm = np.linalg.norm(U[:3], axis=1)
            P(f"{dep} {'centred' if centred else 'raw    '}: ‖u‖ dfn {nrm[0]:.3f} off {nrm[1]:.3f} spd {nrm[2]:.3f}; "
              f"R² of Δresid {r2:.3f}; ‖Δresid‖ rms {np.sqrt((Y**2).sum(1).mean()):.3f}; cos(u_dfn,u_off) "
              f"{U[0]@U[1]/nrm[0]/nrm[1]:+.3f}")
            res.setdefault("B", {})[f"{dep}:{'c' if centred else 'raw'}"] = dict(norm=nrm.tolist(), r2=float(r2))
            if dep == "fin" and centred:
                U_fin = U[:3]
    # cos between the dfn probe readout and the causal direction at fin
    wdf = probes[("fin", "dfn")][0]
    P(f"fin: cos(dfn probe w, centred causal u_dfn) {wdf @ U_fin[0] / np.linalg.norm(wdf) / np.linalg.norm(U_fin[0]):+.3f}; "
      f"probe·u_dfn {wdf @ U_fin[0]:.3f} (1 = probe reads the causal direction at full gain)")

# ------------------------------------------------------------------ D. ActionHead read
P("\n## D. ActionHead read (gradient of the score w.r.t. the candidate latent, per token)")
W1c = W["W1"][:, :256].numpy(); W1x = W["W1"][:, 256:].numpy(); b1 = W["b1"].numpy(); w2 = W["w2"].numpy()
te_rows = np.flatnonzero(~tr & f["has_opp"])[:6000]
Fz = fin[te_rows].astype(np.float64)
ctx = Fz[:, 0]
pre = np.einsum("ntd,kd->ntk", Fz[:, 1:], W1c) + (ctx @ W1x.T)[:, None] + b1
gd = gelu_d(pre) * w2                                     # [n,10,256 units]
J = np.einsum("ntk,kd->ntd", gd, W1c)                     # d logit / d candidate latent
if U_fin is None:  # 600M: causal fin directions from its own train swaps
    z = sw["train"]; zz = z["z"]; leg = z["leg"]
    dF = np.stack([z["ff"][k] - z["f0"][k] for k in FEATS], -1)
    dR = (zz["fin_f"][:, 1:7] - zz["fin_0"][:, 1:7]).astype(np.float64)
    m_ = leg[..., None]
    dRc = dR - (dR * m_).sum(1, keepdims=True) / m_.sum(1, keepdims=True)
    dFc = dF - (dF * m_).sum(1, keepdims=True) / m_.sum(1, keepdims=True)
    U_fin = lstsq(dFc[leg], dRc[leg])
mk = mask[te_rows]
for k, u in zip(FEATS, U_fin):
    rs = J[:, :6] @ u; rm = J[:, 6:] @ u
    P(f"read of centred causal u_{k}: switch tokens mean {rs[mk[:, :6]].mean():+.3f} (sd {rs[mk[:, :6]].std():.3f}); "
      f"if the same vector sat in a move token {rm[mk[:, 6:]].mean():+.3f} (sd {rm[mk[:, 6:]].std():.3f})")
    res.setdefault("D", {})[k] = dict(switch=float(rs[mk[:, :6]].mean()), move=float(rm[mk[:, 6:]].mean()))
for k in ("dfn", "off"):
    w_, b_ = probes[("fin", k)]
    ps, pm = Fz[:, 1:7] @ w_ + b_, Fz[:, 7:] @ w_ + b_
    P(f"probe '{k}' value: switch tokens {ps[mk[:, :6]].mean():+.3f} ± {ps[mk[:, :6]].std():.3f}; "
      f"legal move tokens {pm[mk[:, 6:]].mean():+.3f} ± {pm[mk[:, 6:]].std():.3f}")
# separability of own vs move tokens (gate direction): mean difference, normalised by pooled sd
own_t = Fz[:, 1:7][mk[:, :6]]; mv_t = Fz[:, 7:][mk[:, 6:]]
g = own_t.mean(0) - mv_t.mean(0)
so, sm = own_t @ g, mv_t @ g
P(f"gate direction (mean own − mean move token): own {so.mean():.2f} ± {so.std():.2f} [min {so.min():.2f}], "
  f"move {sm.mean():.2f} ± {sm.std():.2f} [max {sm.max():.2f}]")

# ------------------------------------------------------------------ E. value read
if deep:
    P("\n## E. value-head read of the final-latent causal directions (512 test rows)")
    rr = te_rows[:512]
    x = torch.as_tensor(d["obs"][rr].astype(np.float32)); m = torch.as_tensor(d["mask"][rr])
    c = {}
    with torch.no_grad():
        net.forward(x, m, None, c)
    lat = c["final"].clone().requires_grad_(True)
    v = pol.value_net(pol.mlp_extractor.forward_critic(lat.flatten(1))).squeeze(-1)
    v.sum().backward()
    gv = lat.grad[:, 1:7].double().numpy()
    mk2 = mask[rr][:, :6]
    for k, u in zip(FEATS, U_fin):
        rv = gv @ u
        P(f"dV/d(u_{k}) at a legal switch token: {rv[mk2].mean():+.4f} (sd {rv[mk2].std():.4f}); for scale |V| mean "
          f"{full['value'][rr].__abs__().mean():.3f}")
        res.setdefault("E", {})[k] = float(rv[mk2].mean())

# ------------------------------------------------------------------ F. L1 heads: measured vs OV prediction
if deep:
    P("\n## F. layer-1 heads: per-head direct write onto the L1out dfn probe (train swaps), centred within row")
    z = sw["train"]; zz = z["z"]; leg = z["leg"]
    L = net.layers[1]
    with torch.no_grad():
        a0 = L.norm1(torch.as_tensor(zz["L0oa_0"])).double().numpy()
        af = L.norm1(torch.as_tensor(zz["L0oa_f"])).double().numpy()
    Wv = L.self_attn.in_proj_weight[512:].detach().double().numpy()     # [256(h*dh), 256]
    Wo = L.self_attn.out_proj.weight.detach().double().numpy()          # [256, 256(h*dh)]
    dA = af - a0
    w_l1, _ = probes[("L1out", "dfn")]
    ddfn = z["ff"]["dfn"] - z["f0"]["dfn"]
    def cen(a):
        a = np.where(leg, a, np.nan); return a - np.nanmean(a, 1, keepdims=True)
    cd = cen(ddfn)
    okk = np.isfinite(cd)
    dH = (zz["L1head_f"] - zz["L1head_0"]).astype(np.float64)          # [n,6,8,256]
    pat0 = zz["L1pat_0"]                                                  # [n,8,6]
    tot = 0
    for h in range(8):
        sl = slice(32 * h, 32 * h + 32)
        ov = (dA @ Wv[sl].T) @ Wo[:, sl].T                                # [n,256] write if attention were 1
        pred = pat0[:, h, :, None] * ov[:, None, :]                       # [n,6,256]
        meas_p = cen(dH[:, :, h] @ w_l1); pred_p = cen(pred @ w_l1)
        gm = float((meas_p[okk] * cd[okk]).sum() / (cd[okk] ** 2).sum())
        gp = float((pred_p[okk] * cd[okk]).sum() / (cd[okk] ** 2).sum())
        rr_ = np.corrcoef(meas_p[okk], pred_p[okk])[0, 1]
        tot += gm
        P(f"L1 h{h}: measured {gm:+.3f} dfn-probe units per doubling, OV-only prediction {gp:+.3f}; "
          f"corr(measured, predicted) per token {rr_:+.3f}; mean attention to opp active {pat0[:, h][leg].mean():.3f}")
        res.setdefault("F", {})[f"h{h}"] = dict(measured=gm, ov_pred=gp, corr=float(rr_))
    dmid = cen((zz["L1mid_f"] - zz["L1mid_0"]).astype(np.float64)[:, :6] @ w_l1)
    dout = cen((zz["L1out_f"] - zz["L1out_0"]).astype(np.float64)[:, :6] @ w_l1)
    P(f"sum over heads {tot:+.3f}; whole L1mid {float((dmid[okk]*cd[okk]).sum()/(cd[okk]**2).sum()):+.3f}; "
      f"L1out (after MLP) {float((dout[okk]*cd[okk]).sum()/(cd[okk]**2).sum()):+.3f}")

json.dump(res, open(OUT / f"read_{tag}.json", "w"), indent=1, default=float)
(OUT / f"read_{tag}.txt").write_text("\n".join(txt) + "\n")
