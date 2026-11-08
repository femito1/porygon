"""eds step 2a (v2): the gate as AND(Sleep Talk token, asleep ∧ counter ≥ TH), from two read-off directions in
f2 = norm2(L2.mid) (the exact input of layer 2's MLP).

v1 (eds_gate.py) composed three difference-of-means readouts (ST, asleep, counter) and failed: at the ST token the
asleep and counter directions are anti-aligned (cos −0.71: the 'asleep' readout falls as the counter rises), and the
counter readout is large and arbitrary on other tokens, so the sum was no AND (gate_500M_c4.txt).
v2 uses two Fisher (shrunk pooled-covariance LDA) directions — still reading: no outcome enters, only which input
condition holds:
  r_cond : at Sleep Talk tokens: asleep with counter ≥ TH  vs  asleep with counter < TH or awake. Fit on the
           counterfactual sweep (counter := 0..7 and 'not asleep' on the 707 asleep-ST positions) plus the real awake
           ST tokens; classes are counterfactual levels, so the covariance is the row-to-row variation.
  r_ST   : Sleep Talk token vs every other token (all 17 positions).
Each normalised to 0/1 at the two reference means (r_cond: sweep counter TH−1 → 0, TH → 1). The gate:
    pre = K·(r_cond − θ) + M·(r_ST − 1)
Directions fit on even games; tables on odd games (held out); final fit on all games.
    python scripts/wsp/eds_gate2.py TAG TH "K,M,theta;K,M,theta;..." [--final K,M,theta]
"""
import sys
from pathlib import Path

import numpy as np
import torch

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import Net, load

torch.set_num_threads(2)
tag, TH = sys.argv[1], int(sys.argv[2])
grid = [tuple(map(float, g.split(","))) for g in sys.argv[3].split(";")]
final = tuple(map(float, sys.argv[sys.argv.index("--final") + 1].split(","))) if "--final" in sys.argv else None
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds"); LOC = Path("/tmp/wsp-674320/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N); game = d["game"]
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
is_st = mons[r, own_a, 3:7].astype(np.int64) == int(np.flatnonzero(MOVES == "sleeptalk")[0])
has_st = is_st.any(1) & (mons[:, :6, NUM_OFF + 115].max(1) > .5)
jst = is_st.argmax(1)
slp, cnt = ph["slp"], ph["cnt"]
H = slp & ph["st_legal"].any(1)
f2 = np.load(LOC / f"{tag}_f2.npy", mmap_mode="r")
net = Net(load(tag))
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


stt = np.zeros((N, 17), bool); stt[np.flatnonzero(has_st), 13 + jst[has_st]] = True
rows_st = np.flatnonzero(has_st)
fst = np.asarray(f2[rows_st])[np.arange(len(rows_st)), 13 + jst[rows_st]]
# counterfactual sweep at the ST token (reading passes over the 707 asleep-ST positions)
# v2b tried the sweep on all 2,283 ST positions (asleep counterfactuals on awake positions too): worse held-out
# separation (gate2b.log), so the sweep runs on the 707 really-asleep positions
idx = np.flatnonzero(H) if "--all-st" not in sys.argv else rows_st; n = len(idx); ar = np.arange(n)
X0 = d["obs"][idx].astype(np.float32); MK = torch.as_tensor(d["mask"][idx]); oa = own_a[idx]; jj = jst[idx]
SW = np.zeros((9, n, 256), np.float32)   # 0..7 = counter c, 8 = not asleep
for c in range(9):
    X = X0.copy(); m = X[:, FIELD_DIM:].reshape(n, 12, MON_DIM)
    if c < 8:
        m[ar, oa, NUM_OFF + 59:NUM_OFF + 65] = 0.0; m[ar, oa, NUM_OFF + 63] = 1.0; m[ar, oa, NUM_OFF + 65] = c / 10
    else:
        m[ar, oa, NUM_OFF + 63] = 0.0; m[ar, oa, NUM_OFF + 65] = 0.0
    cache = {}
    with torch.no_grad():
        net.forward(torch.as_tensor(X), MK, None, cache)
        SW[c] = net.layers[2].norm2(cache["L2.mid"])[ar, 13 + jj].numpy()
P(f"reading passes: 9 counterfactual passes over the {n} positions whose active has Sleep Talk (asleep with counter := 0..7; not asleep)")
rng = np.random.default_rng(0)
ri_all, ti_all = np.nonzero(~stt)
samp = np.sort(rng.choice(len(ri_all), 200_000, replace=False))
ri_o, ti_o = ri_all[samp], ti_all[samp]


def gather(ri, ti):
    o = np.argsort(ri, kind="stable"); out = np.empty((len(ri), 256), np.float32)
    for s in range(0, len(ri), 20000):
        oo = o[s:s + 20000]
        out[oo] = np.asarray(f2[ri[oo]])[np.arange(len(oo)), ti[oo]]
    return out


f_oth = gather(ri_o, ti_o)


LAM = float(sys.argv[sys.argv.index("--lam") + 1]) if "--lam" in sys.argv else 0.03


def lda(groups_pos, groups_neg, lam=None):
    lam = LAM if lam is None else lam
    allg = groups_pos + groups_neg
    Z = np.concatenate([g - g.mean(0) for g in allg])
    S = np.cov(Z.T); S = S + lam * np.trace(S) / 256 * np.eye(256)
    mp = np.concatenate(groups_pos).mean(0); mn = np.concatenate(groups_neg).mean(0)
    return np.linalg.solve(S, mp - mn)


def auc(neg, pos):
    x = np.r_[neg, pos]; rk = np.empty(len(x)); rk[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return (rk[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def fit(tr):
    gs = np.isin(game[idx], tr); g_st = np.isin(game[rows_st], tr); g_o = np.isin(game[ri_o], tr)
    awake_real = fst[g_st & ~slp[rows_st]]
    w = lda([SW[c, gs] for c in range(TH, 8)], [SW[c, gs] for c in range(0, TH)] + [SW[8, gs], awake_real])
    a0, a1 = SW[TH - 1, gs].mean(0) @ w, SW[TH, gs].mean(0) @ w
    W_c, b_c = w / (a1 - a0), -a0 / (a1 - a0)
    # ST vs other tokens: other tokens grouped by token role so role offsets do not enter the covariance
    neg = [f_oth[g_o & (ti_o == t)] for t in range(17)]
    neg = [g for g in neg if len(g) > 10]
    u = lda([fst[g_st]], neg)
    a0, a1 = f_oth[g_o].mean(0) @ u, fst[g_st].mean(0) @ u
    W_s, b_s = u / (a1 - a0), -a0 / (a1 - a0)
    return dict(W_c=W_c, b_c=b_c, W_s=W_s, b_s=b_s)


CS = {"v": 1.0}   # centre of the ST term; --recenter: the median r_ST over the asleep-ST tokens (so the M term is


def gate(D, K, M, th):  # mean-zero on the target and only the off-target tokens, r_ST ≤ 0.5, are pushed down)
    return (K * D["W_c"] + M * D["W_s"]).astype(np.float32), np.float32(K * (D["b_c"] - th) + M * (D["b_s"] - CS["v"]))


def table(D, games, label, K, M, th, readouts=False):
    w, b = gate(D, K, M, th)
    test = np.isin(game, games)
    pre = np.zeros((N, 17), np.float32)
    for s in range(0, N, 4096):
        pre[s:s + 4096] = np.asarray(f2[s:s + 4096]) @ w + b
    P(f"\n-- {label}: K {K}, M {M}, θ {th}  (GELU(−3) = −0.004, GELU(−5) = −1.4e−6)")
    P("group | n | median | p99 | max | frac pre > −3 | frac > 0 | frac > 1")
    G = {}
    for c in range(0, 8):
        m = test & H & (cnt == c)
        if m.sum():
            G[f"ST tok, asleep, legal, counter {c}"] = (np.flatnonzero(m), 13 + jst[m])
    for s_, nm in ((0, "truth Rest slept 0 (asleep)"), (1, "truth Rest slept 1 (asleep)"), (2, "truth Rest slept 2 (wakes)")):
        m = test & H & (ph["origin"] == "rest") & (ph["slept"] == s_)
        G[nm] = (np.flatnonzero(m), 13 + jst[m])
    m = test & H & np.char.startswith(ph["origin"].astype(str), "other"); G["natural sleep (other origin)"] = (np.flatnonzero(m), 13 + jst[m])
    m = test & has_st & ~slp; G["ST tok, awake"] = (np.flatnonzero(m), 13 + jst[m])
    m = test & has_st & slp & ~ph["st_legal"].any(1); G["ST tok, asleep, not legal"] = (np.flatnonzero(m), 13 + jst[m])
    for nm, sl in (("other move tokens", slice(13, 17)), ("own mon tokens", slice(1, 7)), ("opp mon tokens", slice(7, 13)), ("field", slice(0, 1))):
        msk = np.zeros((N, 17), bool); msk[test, sl] = True; msk &= ~stt
        G[nm] = np.nonzero(msk)
    for nm, (ri, ti) in G.items():
        p = pre[ri, ti]
        if len(p) == 0:
            continue
        P(f"{nm} | {len(p)} | {np.median(p):+.2f} | {np.percentile(p, 99):+.2f} | {p.max():+.2f} | {np.mean(p > -3):.4f} | {np.mean(p > 0):.4f} | {np.mean(p > 1):.4f}")
    if readouts:
        for key, (Wk, bk) in (("r_cond", ("W_c", "b_c")), ("r_ST", ("W_s", "b_s"))):
            P(f"{key} [p1, median, p99]: " + "; ".join(
                f"{nm.split(',')[0] if 'counter' not in nm else nm[-9:]} [{np.percentile(f2v, 1):+.2f},{np.median(f2v):+.2f},{np.percentile(f2v, 99):+.2f}]"
                for nm, f2v in ((nm, np.array([f2[a, t] @ D[Wk] + D[bk] for a, t in zip(ri[:3000], ti[:3000])])) for nm, (ri, ti) in G.items() if len(ri))))
    return pre


ug = np.unique(game)
tr, te = ug[ug % 2 == 0], ug[ug % 2 == 1]
ge = np.isin(game[idx], te)
real3 = H & (cnt == TH - 1) & np.isin(game, te); real4 = H & (cnt >= TH) & np.isin(game, te)
t1 = H & (ph["origin"] == "rest") & (ph["slept"] == 1) & np.isin(game, te); t2 = H & (ph["origin"] == "rest") & (ph["slept"] == 2) & np.isin(game, te)
fr = lambda m: np.asarray(f2[np.flatnonzero(m)])[np.arange(m.sum()), 13 + jst[m]]  # noqa: E731
F3, F4, T1, T2 = fr(real3), fr(real4), fr(t1), fr(t2)
P("held-out AUCs of r_cond by shrinkage λ (fit on even games): sweep counter TH−1 vs TH | real rows counter TH−1 vs ≥ TH | truth slept 1 vs 2")
for lam in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
    LAM = lam
    Dl = fit(tr)
    rc = lambda X: X @ Dl["W_c"] + Dl["b_c"]  # noqa: E731
    P(f"  λ {lam:g}: {auc(rc(SW[TH - 1, ge]), rc(SW[TH, ge])):.3f} | {auc(rc(F3), rc(F4)):.3f} | {auc(rc(T1), rc(T2)):.3f}"
      f"   (sweep sd within level {np.std(rc(SW[TH, ge])):.2f})")
LAM = float(sys.argv[sys.argv.index("--lam") + 1]) if "--lam" in sys.argv else 0.03
D = fit(tr)
P(f"design on even games, held-out tables on odd games; gate counter ≥ {TH}")
for i, (K, M, th) in enumerate(grid):
    table(D, te, f"design pass {i + 1} (held-out odd games)", K, M, th, readouts=(i == 0))
if final:
    K, M, th = final
    D = fit(ug)
    if "--recenter" in sys.argv:
        CS["v"] = float(np.median(fst[H[rows_st]] @ D["W_s"] + D["b_s"]))
        P(f"ST term re-centred at {CS['v']:.3f}")
    P("\nfinal directions refit on all games; sweep r_cond by counter: " + " ".join(
        f"{c}:{np.mean(SW[c] @ D['W_c'] + D['b_c']):+.2f}±{np.std(SW[c] @ D['W_c'] + D['b_c']):.2f}" for c in range(9)))
    table(D, ug, "final gate, all games (in-sample)", K, M, th)
    w, b = gate(D, K, M, th)
    np.savez(OUT / f"gate_{tag}_c{TH}.npz", w=w, b=b, K=K, M=M, theta=th, cs=CS["v"], **D)
(OUT / f"gate2_{tag}_c{TH}.txt").write_text("\n".join(txt) + "\n")
