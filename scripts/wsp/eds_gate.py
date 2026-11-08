"""eds step 2a: build the gate of the repurposed L2 MLP unit from three read-off directions in f2 = norm2(L2.mid)
(the exact input of layer 2's MLP), and measure its pre-activation on every token of every logged row.

  r_ST  : "this move token is Sleep Talk"   difference of means, ST token vs the other move tokens (rows with ST)
  r_slp : "our active is asleep"            difference of means at the ST token, asleep vs awake
  r_c   : "status counter >= 4"             Fisher direction (shrunk within-counter covariance) at the asleep ST token,
                                            from the counterfactual counter sweep (counter := 0..7 on the 707
                                            asleep-ST positions: reading passes) — counter >= 4 vs <= 3
Each readout is affinely normalised (0 = negative class mean, 1 = positive class mean). The gate is
    pre = K·(r_c − θ) + M·(r_ST − 1) + M·(r_slp − 1)
an AND: off the Sleep Talk token or when awake, the M terms hold it far below 0, where GELU ≈ 0.
Directions are found on even-numbered games and the readout tables are reported on odd games (held out), then the
final directions are refit on all games.
    python scripts/wsp/eds_gate.py 500M [THRESH=4] [K] [M] [theta] -> out/eds/gate_<tag>_c<THRESH>.npz / .txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
TH = int(sys.argv[2]) if len(sys.argv) > 2 else 4
K = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
M = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
theta = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds"); LOC = Path("/tmp/wsp-674320/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N); game = d["game"]
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
ST_ID = int(np.flatnonzero(MOVES == "sleeptalk")[0])
is_st = mons[r, own_a, 3:7].astype(np.int64) == ST_ID
has_st = is_st.any(1) & (mons[:, :6, NUM_OFF + 115].max(1) > .5)
jst = is_st.argmax(1)
slp, cnt = ph["slp"], ph["cnt"]
H = slp & ph["st_legal"].any(1)
f2 = np.load(LOC / f"{tag}_f2.npy", mmap_mode="r")
net = Net(load(tag))
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


rows_st = np.flatnonzero(has_st)
F = np.asarray(f2[rows_st])                     # [n,17,256]
fst = F[np.arange(len(rows_st)), 13 + jst[rows_st]]
# counterfactual sweep (reading): f2 at the ST token with counter := c
idx = np.flatnonzero(H); n = len(idx); ar = np.arange(n)
X0 = d["obs"][idx].astype(np.float32); MK = torch.as_tensor(d["mask"][idx]); oa = own_a[idx]; jj = jst[idx]
SW = np.zeros((8, n, 256), np.float32)
for c in range(8):
    X = X0.copy(); X[:, FIELD_DIM:].reshape(n, 12, MON_DIM)[ar, oa, NUM_OFF + 65] = c / 10
    cache = {}
    with torch.no_grad():
        net.forward(torch.as_tensor(X), MK, None, cache)
        SW[c] = net.layers[2].norm2(cache["L2.mid"])[ar, 13 + jj].numpy()


def fit(sel_games):
    """Directions + normalisation from rows of the given games."""
    g_st = np.isin(game[rows_st], sel_games)
    oth = F[g_st][:, 13:17][~is_st[rows_st][g_st]]
    u_st = fst[g_st].mean(0) - oth.mean(0)
    a = slp[rows_st][g_st]
    u_slp = fst[g_st][a].mean(0) - fst[g_st][~a].mean(0)
    gs = np.isin(game[idx], sel_games)
    pos = SW[TH:, gs].reshape(-1, 256); neg = SW[:TH, gs].reshape(-1, 256)
    # within-class covariance: centre each counter level separately (row-level variation is what hurts)
    Z = np.concatenate([SW[c, gs] - SW[c, gs].mean(0) for c in range(8)])
    S = np.cov(Z.T); S = S + 0.1 * np.trace(S) / 256 * np.eye(256)
    w_c = np.linalg.solve(S, pos.mean(0) - neg.mean(0))

    def norm(u, lo, hi):
        a0, a1 = lo @ u, hi @ u
        return u / (a1 - a0), -a0 / (a1 - a0)          # r = f·w + b ; lo-mean -> 0, hi-mean -> 1
    W_st, b_st = norm(u_st, oth.mean(0), fst[g_st].mean(0))
    W_sl, b_sl = norm(u_slp, fst[g_st][~a].mean(0), fst[g_st][a].mean(0))
    W_c, b_c = norm(w_c, SW[TH - 1, gs].mean(0), SW[TH, gs].mean(0))   # counter TH-1 -> 0, TH -> 1
    return dict(W_st=W_st, b_st=b_st, W_sl=W_sl, b_sl=b_sl, W_c=W_c, b_c=b_c)


def gate(D):
    w = K * D["W_c"] + M * D["W_st"] + M * D["W_sl"]
    b = K * (D["b_c"] - theta) + M * (D["b_st"] - 1) + M * (D["b_sl"] - 1)
    return w.astype(np.float32), np.float32(b)


def table(D, test_games, label):
    w, b = gate(D)
    P(f"\n-- gate pre-activation by token group ({label}); GELU(-3) = -0.004, GELU(-5) = -1.4e-6")
    P("group | n tokens | median | p99 | p99.9 | max | frac pre > -3 | frac pre > 0")
    groups = {}
    test = np.isin(game, test_games)
    pre_all = np.zeros((N, 17), np.float32)
    for s in range(0, N, 4096):
        pre_all[s:s + 4096] = np.asarray(f2[s:s + 4096]) @ w + b
    tok = np.zeros((N, 17), bool)
    stt = np.zeros((N, 17), bool); stt[np.flatnonzero(has_st), 13 + jst[has_st]] = True
    for c in range(0, 9):
        m = test & H & ((cnt == c) if c < 8 else (cnt >= 8))
        if m.sum():
            groups[f"ST token, asleep, ST legal, counter {c}{'+' if c == 8 else ''}"] = (np.flatnonzero(m), 13 + jst[m])
    m = test & has_st & ~slp
    groups["ST token, awake"] = (np.flatnonzero(m), 13 + jst[m])
    m = test & has_st & slp & ~ph["st_legal"].any(1)
    groups["ST token, asleep, ST not legal"] = (np.flatnonzero(m), 13 + jst[m])
    for name, sl in (("other move tokens", slice(13, 17)), ("own mon tokens", slice(1, 7)), ("opp mon tokens", slice(7, 13)),
                     ("field token", slice(0, 1))):
        msk = np.zeros((N, 17), bool); msk[test, sl] = True; msk &= ~stt
        groups[name] = np.nonzero(msk)
    for name, (ri, ti) in groups.items():
        p = pre_all[ri, ti]
        P(f"{name} | {len(p)} | {np.median(p):+.2f} | {np.percentile(p, 99):+.2f} | {np.percentile(p, 99.9):+.2f} | "
          f"{p.max():+.2f} | {np.mean(p > -3):.4f} | {np.mean(p > 0):.4f}")
    # ground truth phase on the held-out asleep ST rows
    for s, name in ((0, "Rest slept 0 (asleep)"), (1, "Rest slept 1 (asleep)"), (2, "Rest slept 2 (wakes)")):
        m = test & H & (ph["origin"] == "rest") & (ph["slept"] == s)
        p = pre_all[m, 13 + jst[m]]
        P(f"truth: {name} | n {m.sum()} | median pre {np.median(p):+.2f} | frac pre > 0 {np.mean(p > 0):.3f} | frac pre > 1 {np.mean(p > 1):.3f}")
    return pre_all


ug = np.unique(game)
if "--diag" in sys.argv:
    def roc_auc_score(y, x):
        rk = np.empty(len(x)); rk[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1); n1 = y.sum(); n0 = len(y) - n1
        return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    tr, te = ug[ug % 2 == 0], ug[ug % 2 == 1]
    test = np.isin(game, te)
    stt = np.zeros((N, 17), bool); stt[np.flatnonzero(has_st), 13 + jst[has_st]] = True
    grp = {}
    for nm, m in (("ST asleep c<=%d" % (TH - 1), test & H & (cnt < TH)), ("ST asleep c>=%d" % TH, test & H & (cnt >= TH)),
                  ("ST awake", test & has_st & ~slp)):
        grp[nm] = (np.flatnonzero(m), 13 + jst[m])
    rng = np.random.default_rng(1)
    for nm, sl in (("other move", slice(13, 17)), ("own mon", slice(1, 7)), ("opp mon", slice(7, 13)), ("field", slice(0, 1))):
        msk = np.zeros((N, 17), bool); msk[test, sl] = True; msk &= ~stt
        ri, ti = np.nonzero(msk); k = np.sort(rng.choice(len(ri), min(len(ri), 30000), replace=False))
        grp[nm] = (ri[k], ti[k])
    def gather(ri, ti):
        o = np.argsort(ri, kind="stable"); out = np.empty((len(ri), 256), np.float32)
        out[o] = np.asarray(f2[ri[o]])[np.arange(len(ri)), ti[o]]
        return out
    feats = {nm: gather(np.asarray(ri), np.asarray(ti)) for nm, (ri, ti) in grp.items()}
    for lam in (0.1, 1.0, 10.0, 1e6):
        g_st = np.isin(game[rows_st], tr)
        oth = F[g_st][:, 13:17][~is_st[rows_st][g_st]]
        a = slp[rows_st][g_st]
        gs = np.isin(game[idx], tr)
        Z = np.concatenate([SW[c, gs] - SW[c, gs].mean(0) for c in range(8)])
        S = np.cov(Z.T); S = S + lam * np.trace(S) / 256 * np.eye(256)
        dmu = SW[TH:, gs].reshape(-1, 256).mean(0) - SW[:TH, gs].reshape(-1, 256).mean(0)
        w_c = np.linalg.solve(S, dmu)
        a0, a1 = SW[TH - 1, gs].mean(0) @ w_c, SW[TH, gs].mean(0) @ w_c
        rc = {nm: (X @ w_c - a0) / (a1 - a0) for nm, X in feats.items()}
        y = np.r_[np.zeros(len(rc["ST asleep c<=%d" % (TH - 1)])), np.ones(len(rc["ST asleep c>=%d" % TH]))]
        auc = roc_auc_score(y, np.r_[rc["ST asleep c<=%d" % (TH - 1)], rc["ST asleep c>=%d" % TH]])
        P(f"r_c, shrinkage {lam:g}: held-out AUC (real rows, c>={TH} vs c<{TH}) {auc:.3f}; quantiles [p1, median, p99] by group: " +
          "; ".join(f"{nm} [{np.percentile(v, 1):+.1f},{np.median(v):+.1f},{np.percentile(v, 99):+.1f}]" for nm, v in rc.items()))
    D = fit(tr)
    for key, (Wk_, bk) in (("r_ST", ("W_st", "b_st")), ("r_slp", ("W_sl", "b_sl"))):
        rr = {nm: X @ D[Wk_] + D[bk] for nm, X in feats.items()}
        P(f"{key}: quantiles [p1, median, p99]: " + "; ".join(f"{nm} [{np.percentile(v, 1):+.2f},{np.median(v):+.2f},{np.percentile(v, 99):+.2f}]" for nm, v in rr.items()))
    (OUT / f"gate_diag_{tag}_c{TH}.txt").write_text("\n".join(txt) + "\n")
    sys.exit()
P(f"gate: counter >= {TH}; K {K}, M {M}, theta {theta}")
D = fit(ug[ug % 2 == 0])
table(D, ug[ug % 2 == 1], "directions from even games, table on odd games (held out)")
D = fit(ug)
P("\nnormalised readouts at the ST token on the counter sweep (all games): r_c by counter = " +
  " ".join(f"{c}:{np.mean(SW[c] @ D['W_c'] + D['b_c']):+.2f}±{np.std(SW[c] @ D['W_c'] + D['b_c']):.2f}" for c in range(8)))
pre = table(D, ug, "final directions, all games (in-sample)")
w, b = gate(D)
np.savez(OUT / f"gate_{tag}_c{TH}.npz", w=w, b=b, K=K, M=M, theta=theta, **D)
(OUT / f"gate_{tag}_c{TH}.txt").write_text("\n".join(txt) + "\n")
