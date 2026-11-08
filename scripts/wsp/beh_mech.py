"""beh analysis 5: which components carry the costly decisions.

A. Opponent-active swap counterfactuals (does the switch choice follow the switch-in's matchup vs the opp active?)
   variants: full (opp-active block from a donor position + all dependent effectiveness numerics recomputed),
   token-only (opp block swapped, effectiveness numerics left stale), numerics-only (only our moves'
   effectiveness numerics recomputed for the donor's types, opp token unchanged).
B. Activation patching of the full swap: residual stream of one token at one layer point copied from the swapped
   run into the original run; recovered fraction of the switch-preference change and of the switch-move margin change.
C. Mean ablation of each head / MLP on the costly disagreement categories vs agreement positions.
D. Input hypothesis tests (opp HP, own HP).
    python scripts/wsp/beh_mech.py 500M   -> out/beh/mech_<tag>.txt / .json
"""
import json
import sys
import time

import numpy as np
import torch

from beh_common import (OUT, FIELD_DIM, MON_DIM, NUM_OFF, MOVENUM_OFF, TYP, HP, FNT, CAT, MTYPE, MULT, IMMUNE, BP,
                        decode, features, load_local, ratio_boot, type_chart, action_type, coarse)
from porygon.encoder import MOVE_DIM
from wsp_common import Intervention, Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
d = load_local()
f = features(d)
M = type_chart()
der = np.load(OUT / "derived_r2-300k.npz", allow_pickle=True)
net = Net(load(tag))
N = len(d["mask"])
mask = d["mask"] > 0
game = d["game"]
rng = np.random.default_rng(1)
out, txt = {}, []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    txt.append(s)


def fwd(x, mk, iv=None, keep=(), bs=1000):
    L, V, C = [], [], {k: [] for k in keep}
    with torch.no_grad():
        for s in range(0, len(x), bs):
            c = {} if keep else None
            ivb = iv(s, s + bs) if callable(iv) else iv
            lg, v = net.forward(torch.as_tensor(x[s:s + bs], dtype=torch.float32), torch.as_tensor(mk[s:s + bs]), ivb, c)
            L.append(lg.double()); V.append(v)
            for k in keep:
                C[k].append(c[k])
    return torch.cat(L).numpy(), torch.cat(V).numpy(), {k: torch.cat(v) for k, v in C.items()}


def mon_view(x):
    return x[:, FIELD_DIM:].reshape(len(x), 12, MON_DIM)


def recompute_mult(mv, def_types):
    """mv [...,4,44] move numerics; def_types [...,1,18] (broadcastable over the 4 moves) -> mv with MULT / IMMUNE recomputed."""
    mv = mv.copy()
    mt = mv[..., MTYPE:MTYPE + 18]
    has = mt.sum(-1) > 0.5
    dmg = (mv[..., CAT + 2] < 0.5) & has
    ti = mt.argmax(-1)
    mult = np.ones(ti.shape)
    for k in range(18):
        mk = def_types[..., k] > 0.5
        mult = mult * np.where(mk, M[ti, k], 1.0)  # def_types broadcastable to [...,4,18]
    mult = np.where(dmg, mult, 1.0)
    mv[..., MULT] = np.where(has, np.where(mult > 0, np.log2(np.maximum(mult, 1e-9)) / 2, -1.5), mv[..., MULT])
    mv[..., IMMUNE] = np.where(has, (mult == 0).astype(np.float32), mv[..., IMMUNE])
    return mv


# ------------------------------------------------------------------ A. opponent swap
P(f"## A. opponent-active swap counterfactuals ({tag})")
own_a, opp_a = f["own_a"], f["opp_a"]
nums_all = decode(d["obs"])["nums"]
alive_opp = f["has_opp"] & (nums_all[np.arange(N), opp_a, FNT] < 0.5)
cand = np.flatnonzero(alive_opp & (mask[:, :6].sum(1) >= 2) & f["has_act"])
rows = np.sort(rng.choice(cand, min(4000, len(cand)), replace=False))
donors = rng.choice(cand, len(rows))
X0 = d["obs"][rows].astype(np.float32)
Xd = d["obs"][donors].astype(np.float32)
MK = d["mask"][rows]
n = len(rows)
ar = np.arange(n)
m0, md = mon_view(X0), mon_view(Xd)
oa, oad, wa = opp_a[rows], opp_a[donors], own_a[rows]
new_opp = md[ar, oad].copy()
new_types = new_opp[:, NUM_OFF + TYP:NUM_OFF + TYP + 18]
our_types = m0[ar, wa, NUM_OFF + TYP:NUM_OFF + TYP + 18]


def own_moves(m):
    return m[:, :6, MOVENUM_OFF:].reshape(n, 6, 4, MOVE_DIM)


# sanity: recomputing our moves' effectiveness vs the ORIGINAL opp types must reproduce the encoder's numerics
_orig_t = m0[ar, oa, NUM_OFF + TYP:NUM_OFF + TYP + 18]
_rc = recompute_mult(own_moves(m0), _orig_t[:, None, None].repeat(6, 1))
_om = own_moves(m0)
_has = _om[..., MTYPE:MTYPE + 18].sum(-1) > .5
P(f"sanity: recomputed MULT matches encoder on {np.mean(np.abs(_rc[..., MULT] - _om[..., MULT])[_has] < 1e-3):.4f} of our move slots "
  f"(differences: abilities/special cases the encoder handles via poke-env)")
# full
Xf = X0.copy(); mf = mon_view(Xf)
mf[ar, oa] = new_opp
mf[ar, oa, MOVENUM_OFF:] = recompute_mult(new_opp[:, MOVENUM_OFF:].reshape(n, 4, MOVE_DIM), our_types[:, None, :]).reshape(n, -1)
mf[:, :6, MOVENUM_OFF:] = recompute_mult(own_moves(mf), new_types[:, None, None].repeat(6, 1)).reshape(n, 6, -1)
# token only
Xt = X0.copy(); mt_ = mon_view(Xt)
mt_[ar, oa] = new_opp
# numerics only
Xn = X0.copy(); mn = mon_view(Xn)
mn[:, :6, MOVENUM_OFF:] = recompute_mult(own_moves(mn), new_types[:, None, None].repeat(6, 1)).reshape(n, 6, -1)

lg0, v0, _ = fwd(X0, MK)


def defmult(types_att, types_def):
    """best multiplier of attacker STAB types on defender types; [n,18],[n,6,18] -> [n,6]"""
    out_ = np.ones((n, types_def.shape[1]))
    for i in range(n):
        ta = np.flatnonzero(types_att[i] > .5)
        for s in range(types_def.shape[1]):
            td = np.flatnonzero(types_def[i, s] > .5)
            out_[i, s] = max((np.prod(M[a, td]) for a in ta), default=1.0)
    return out_


own_types = m0[:, :6, NUM_OFF + TYP:NUM_OFF + TYP + 18]
old_types = m0[ar, oa, NUM_OFF + TYP:NUM_OFF + TYP + 18]
def_old, def_new = defmult(old_types, own_types), defmult(new_types, own_types)


def offmult(mv):  # best damaging-move multiplier x power of each own mon vs current numerics
    mvv = own_moves(mon_view(mv))
    mult = np.where(mvv[..., IMMUNE] > .5, 0, 2.0 ** (2 * mvv[..., MULT]))
    pw = mvv[..., BP] * 150 * np.where(mvv[..., CAT + 2] < .5, 1, 0)
    return np.log2(np.maximum((mult * pw).max(-1), 1.0))


off_old, off_new = offmult(X0), offmult(Xf)
legal_sw = MK[:, :6] > 0
L2 = lambda a: np.log2(np.maximum(a, 0.125))  # noqa: E731  immunity counted as 1/8
ddef = L2(def_new) - L2(def_old)
doff = off_new - off_old


def centre(a):
    a = np.where(legal_sw, a, np.nan)
    return a - np.nanmean(a, 1, keepdims=True)


def regress(y, *xs):
    ok = np.isfinite(y)
    for x_ in xs:
        ok &= np.isfinite(x_)
    A = np.c_[[x_[ok] for x_ in xs]].T
    coef = np.linalg.lstsq(A, y[ok], rcond=None)[0]
    r = np.corrcoef(A @ coef, y[ok])[0, 1]
    return coef, r, ok.sum()


res_A = {}
for name, Xv in (("full", Xf), ("token-only", Xt), ("numerics-only", Xn)):
    lg, v, _ = fwd(Xv, MK)
    dl = centre(lg[:, :6] - lg0[:, :6])
    cdef, cdoff = centre(ddef), centre(doff)
    coef, r, k = regress(dl, cdef, cdoff)
    # switch-move margin vs threat to our active
    lse = lambda a: np.logaddexp.reduce(np.where(MK > 0, a, -np.inf)[:, :6], 1) - np.logaddexp.reduce(np.where(MK > 0, a, -np.inf)[:, 6:], 1)  # noqa
    hasmv = MK[:, 6:].sum(1) > 0
    dmarg = lse(lg) - lse(lg0)
    thr_old = defmult(old_types, our_types[:, None])[:, 0]
    thr_new = defmult(new_types, our_types[:, None])[:, 0]
    dthr = L2(thr_new) - L2(thr_old)
    c2, r2, k2 = regress(np.where(hasmv, dmarg, np.nan), dthr, np.ones(n))
    # probability that the argmax switch-in changes (among positions where the net's argmax is a switch)
    sw0 = np.where(legal_sw, lg0[:, :6], -np.inf).argmax(1)
    sw1 = np.where(legal_sw, lg[:, :6], -np.inf).argmax(1)
    best_def_new = np.where(legal_sw, def_new, np.inf).min(1)
    picks_best = def_new[ar, sw1] <= best_def_new + 1e-9
    picks_best0 = def_old[ar, sw0] <= np.where(legal_sw, def_old, np.inf).min(1) + 1e-9
    res_A[name] = {"coef_def": coef[0], "coef_off": coef[1], "r": r, "n_pairs": int(k), "margin_per_log2threat": c2[0],
                   "r_margin": r2, "mean_abs_dl": float(np.nanmean(np.abs(dl))), "switch_choice_changes": float(np.mean(sw0 != sw1)),
                   "dV_abs": float(np.mean(np.abs(v - v0)))}
    P(f"{name}: centred Δ switch-logit = {coef[0]:+.3f}·Δlog2(defensive mult vs new opp) {coef[1]:+.3f}·Δlog2(best offence), "
      f"r={r:.3f} (n={k} slot-pairs); mean |Δ centred logit| {np.nanmean(np.abs(dl)):.3f}; argmax switch-in changes "
      f"{np.mean(sw0 != sw1):.3f}; Δ(switch−move margin) per log2 threat-to-active {c2[0]:+.3f} (r={r2:.3f}); |ΔV| {np.mean(np.abs(v-v0)):.3f}")
    if name == "full":
        P(f"   net's preferred switch-in is the best defensive type match: original {picks_best0.mean():.3f}, after swap {picks_best.mean():.3f}; "
          f"random legal {np.mean([np.mean(def_new[i, legal_sw[i]] <= best_def_new[i] + 1e-9) for i in range(n)]):.3f}")
        lgF = lg
out["A"] = res_A

# bootstrap the full-variant def coefficient over games
g_rows = game[rows]
dlF = centre(lgF[:, :6] - lg0[:, :6]); cdef, cdoff = centre(ddef), centre(doff)
ug = np.unique(g_rows)
bs = []
for _ in range(300):
    pick = rng.choice(ug, len(ug))
    idx = np.concatenate([np.flatnonzero(g_rows == g) for g in pick])
    bs.append(regress(dlF[idx], cdef[idx], cdoff[idx])[0])
bs = np.array(bs)
P(f"   full: coef_def 95% CI [{np.percentile(bs[:,0],2.5):+.3f},{np.percentile(bs[:,0],97.5):+.3f}], coef_off "
  f"[{np.percentile(bs[:,1],2.5):+.3f},{np.percentile(bs[:,1],97.5):+.3f}]")
# reference scale: logit sd across legal switches
P(f"   reference: sd of centred switch logits in the original positions {np.nanstd(centre(lg0[:, :6])):.3f}")

# ------------------------------------------------------------------ B. activation patching
P("\n## B. activation patching, full swap -> original (recovered fraction of the change)")
sel = np.flatnonzero(np.nanmean(np.abs(dlF), 1) > 0.3)[:1500]
P(f"pairs with mean |Δ centred switch logit| > 0.3: {len(sel)} used")
Xa, Xb, MKs = X0[sel], Xf[sel], MK[sel]
keep = ("L-1.out", "L0.out", "L1.out", "L2.out")
la, va, ca = fwd(Xa, MKs, keep=keep)
lb, vb, cb = fwd(Xb, MKs, keep=keep)
lsw = MKs[:, :6] > 0
def cen(a):
    a = np.where(lsw, a, np.nan); return a - np.nanmean(a, 1, keepdims=True)
D = cen(lb[:, :6] - la[:, :6])
lsef = lambda a: np.logaddexp.reduce(np.where(MKs > 0, a, -np.inf)[:, :6], 1) - np.logaddexp.reduce(np.where(MKs > 0, a, -np.inf)[:, 6:], 1)  # noqa
hm = MKs[:, 6:].sum(1) > 0
Dm = lsef(lb) - lsef(la)
TOK = ["field"] + [f"own{i}" for i in range(6)] + [f"opp{i}" for i in range(6)] + [f"mv{j}" for j in range(4)]
# token roles per row
oa_s, wa_s = oa[sel], wa[sel]
patch_tab = {}
for li, key in zip((-1, 0, 1, 2), keep):
    rec_role = {}
    for t in range(17):
        def iv(s, e, t=t, li=li, key=key):
            delta = (cb[key][s:e, t] - ca[key][s:e, t])
            return Intervention(resid_add={(li, t): delta})
        lp, vp, _ = fwd(Xa, MKs, iv=iv)
        Dp = cen(lp[:, :6] - la[:, :6])
        num = np.nansum(Dp * D, 1); den = np.nansum(D * D, 1)
        rec = num / np.maximum(den, 1e-9)
        recm = (lsef(lp) - lsef(la)) / np.where(np.abs(Dm) > 0.2, Dm, np.nan)
        recv = (vp - va) / np.where(np.abs(vb - va) > 0.05, vb - va, np.nan)
        # map token index to role
        if t == 0:
            role = np.full(len(sel), "field", dtype=object)
        elif t <= 6:
            role = np.where(t - 1 == wa_s, "own active", "own bench")
        elif t <= 12:
            role = np.where(t - 1 == oa_s, "opp active", "opp bench")
        else:
            role = np.full(len(sel), "move tokens", dtype=object)
        for rname in set(role):
            mm = role == rname
            rec_role.setdefault(rname, {"sw": [], "mg": [], "v": []})
            # sum over tokens of that role per row: accumulate per-row sums
            rec_role[rname]["sw"].append(np.where(mm, rec, 0)); rec_role[rname]["mg"].append(np.where(mm, recm, 0))
            rec_role[rname]["v"].append(np.where(mm, recv, 0))
    row = {}
    for rname, dd in rec_role.items():
        row[rname] = [float(np.nanmean(np.sum(dd[k], 0))) for k in ("sw", "mg", "v")]
    patch_tab[key] = row
    P(f"{key}: " + "; ".join(f"{r}: switch-pref {v[0]:.2f}, margin {v[1]:.2f}, V {v[2]:.2f}" for r, v in sorted(row.items())))
out["B"] = patch_tab

# ------------------------------------------------------------------ C. head / MLP ablations on failure categories
P("\n## C. mean ablation of heads / MLPs on costly disagreement categories (Q gaps from r2-300k labels)")
cat = der["cat"]
net_arg, srch_arg = der["net_arg"], der["srch_arg"]
agr = der["agr"]
cats = ["switch->switch", "switch->attack", "status->attack", "switch->status"]
S = {}
for c in cats:
    ii = np.flatnonzero(cat == c)
    S[c] = rng.choice(ii, min(800, len(ii)), replace=False)
S["agree"] = rng.choice(np.flatnonzero(agr), 1500, replace=False)
allrows = np.concatenate(list(S.values()))
XC = d["obs"][allrows].astype(np.float32); MC = d["mask"][allrows]
lgC, vC, cC = fwd(XC, MC, keep=("L0.z", "L1.z", "L2.z", "L0.mlp", "L1.mlp", "L2.mlp"))
# means over a random sample for mean-ablation
samp = rng.choice(N, 4000, replace=False)
_, _, cs = fwd(d["obs"][samp].astype(np.float32), d["mask"][samp], keep=("L0.z", "L1.z", "L2.z", "L0.mlp", "L1.mlp", "L2.mlp"))
zmean = {i: cs[f"L{i}.z"].mean(0) for i in range(3)}      # [T,H,dh]
mlpmean = {i: cs[f"L{i}.mlp"].mean(0) for i in range(3)}  # [T,D]
na, sa = net_arg[allrows], srch_arg[allrows]
lab = np.concatenate([[c] * len(v) for c, v in S.items()])


def err_margin(lg):
    return lg[np.arange(len(lg)), na] - lg[np.arange(len(lg)), sa]


def second_margin(lg):
    l = np.where(MC > 0, lg, -np.inf)
    srt = np.sort(l, 1)
    return srt[:, -1] - srt[:, -2]


e0 = err_margin(lgC); s0 = second_margin(lgC)
comps = [("head", i, h) for i in range(3) for h in range(8)] + [("mlp", i, None) for i in range(3)] + [("attn", i, None) for i in range(3)]
tabC = {}
P("component | " + " | ".join(f"Δ err-margin {c} (flip→search %)" for c in cats) + " | agree: Δ top-2 margin (flip %)")
for kind, i, h in comps:
    def iv(s, e, kind=kind, i=i, h=h):
        if kind == "head":
            return Intervention(head_mean={i: {h: zmean[i][:, h]}})
        if kind == "attn":
            return Intervention(head_mean={i: {hh: zmean[i][:, hh] for hh in range(8)}})
        return Intervention(resid_add={(i, None): (mlpmean[i] - cC[f"L{i}.mlp"][s:e])})
    lg, v, _ = fwd(XC, MC, iv=iv)
    e1 = err_margin(lg)
    am = np.where(MC > 0, lg, -np.inf).argmax(1)
    row = {}
    for c in cats + ["agree"]:
        mm = lab == c
        if c == "agree":
            row[c] = (float(np.mean(second_margin(lg)[mm] - s0[mm])), float(np.mean(am[mm] != na[mm])))
        else:
            row[c] = (float(np.mean(e1[mm] - e0[mm])), float(np.mean(am[mm] == sa[mm])))
    tabC[f"{kind}{i}" + ("" if h is None else f".{h}")] = row
    P(f"{kind} L{i}" + ("" if h is None else f" h{h}") + " | " + " | ".join(f"{row[c][0]:+.2f} ({100*row[c][1]:.0f}%)" for c in cats)
      + f" | {row['agree'][0]:+.2f} ({100*row['agree'][1]:.0f}%)")
out["C"] = tabC
P("baseline error margins: " + ", ".join(f"{c} {np.mean(e0[lab==c]):.2f}" for c in cats))

# ------------------------------------------------------------------ D. input hypothesis tests
P("\n## D. input counterfactuals (voluntary positions)")
vol = np.flatnonzero(f["voluntary"] & f["has_opp"])
vs_ = rng.choice(vol, 4000, replace=False)
XD = d["obs"][vs_].astype(np.float32); MD = d["mask"][vs_]
nd = len(vs_); ad = np.arange(nd)
oaD, waD = opp_a[vs_], own_a[vs_]
def probs_of(X):
    lg, v, _ = fwd(X, MD)
    l = np.where(MD > 0, lg, -np.inf)
    p = np.exp(l - np.logaddexp.reduce(l, 1, keepdims=True))
    return p, v
fD = {k: f[k][vs_] for k in ("attack", "status", "recover", "setup", "maxdmg")}
def summ(p):
    atk = (p[:, 6:] * fD["attack"]).sum(1); sw = p[:, :6].sum(1)
    rec = (p[:, 6:] * fD["recover"]).sum(1); stp = (p[:, 6:] * fD["setup"]).sum(1)
    mx = p[ad, fD["maxdmg"]]
    return np.stack([atk, sw, rec, stp, mx], 1)
res_D = {}
for name, who, val in (("opp HP 10%", "opp", 0.1), ("opp HP 100%", "opp", 1.0), ("own HP 15%", "own", 0.15), ("own HP 100%", "own", 1.0)):
    X = XD.copy(); m = mon_view(X)
    slot = oaD if who == "opp" else waD
    m[ad, slot, NUM_OFF + HP] = val
    p, v = probs_of(X)
    res_D[name] = summ(p).mean(0).tolist() + [float(v.mean())]
for a, b in (("opp HP 10%", "opp HP 100%"), ("own HP 15%", "own HP 100%")):
    dd = np.array(res_D[a]) - np.array(res_D[b])
    P(f"{a} minus {b}: ΔP(attack) {dd[0]:+.3f}, ΔP(switch) {dd[1]:+.3f}, ΔP(recover) {dd[2]:+.3f}, ΔP(setup) {dd[3]:+.3f}, "
      f"ΔP(max-dmg move) {dd[4]:+.3f}, ΔV {dd[5]:+.3f}")
out["D"] = res_D
# observational: net vs search attack share by opp HP bin
P("observational: opp HP bin | net P(attack) | search share(attack) | net P(switch) | search switch")
hb = f["hpB"]
V_all = f["voluntary"] & f["has_opp"]
atk_n = (d["probs"][:, 6:] * f["attack"]).sum(1); atk_s = (d["shares"][:, 6:] * f["attack"]).sum(1)
for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
    s = V_all & (hb >= lo) & (hb < hi)
    P(f"[{lo},{hi}) n={s.sum()} | {atk_n[s].mean():.3f} | {atk_s[s].mean():.3f} | {d['probs'][s,:6].sum(1).mean():.3f} | {d['shares'][s,:6].sum(1).mean():.3f}")
P("observational: own HP bin | net P(recover|legal) | search share(recover)")
ha = f["hpA"]
rec_leg = (f["recover"] & mask[:, 6:]).any(1)
rn = (d["probs"][:, 6:] * f["recover"]).sum(1); rs = (d["shares"][:, 6:] * f["recover"]).sum(1)
for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
    s = V_all & rec_leg & (ha >= lo) & (ha < hi)
    P(f"[{lo},{hi}) n={s.sum()} | {rn[s].mean():.3f} | {rs[s].mean():.3f}")

json.dump(out, open(OUT / f"mech_{tag}.json", "w"), indent=1, default=float)
(OUT / f"mech_{tag}.txt").write_text("\n".join(txt) + "\n")
