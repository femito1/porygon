"""eds: which MLP input separates counter ≥ 4 from ≤ 3 best at the Sleep Talk token, using sweep-fit Fisher
directions (counterfactual counter := 0..7 / not asleep on the 707 asleep-ST positions, fit on even games) and
evaluated on REAL held-out rows (odd games): norm2(L0.mid), norm2(L1.mid), norm2(L2.mid) (= f2).
    python scripts/wsp/eds_where2.py 500M
"""
import sys
from pathlib import Path

import numpy as np
import torch

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import Net, load

torch.set_num_threads(2)
tag = sys.argv[1]
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N); game = d["game"]
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
is_st = mons[r, own_a, 3:7].astype(np.int64) == int(np.flatnonzero(MOVES == "sleeptalk")[0]); jst = is_st.argmax(1)
H = ph["slp"] & ph["st_legal"].any(1); cnt = ph["cnt"]
idx = np.flatnonzero(H); n = len(idx); ar = np.arange(n); oa = own_a[idx]; jj = jst[idx]
X0 = d["obs"][idx].astype(np.float32); MK = torch.as_tensor(d["mask"][idx])
net = Net(load(tag))


def feats(X):
    c = {}
    with torch.no_grad():
        net.forward(torch.as_tensor(X), MK, None, c)
        return {li: net.layers[li].norm2(c[f"L{li}.mid"])[ar, 13 + jj].numpy() for li in range(3)}


real = feats(X0)
SW = []
for c in range(9):
    X = X0.copy(); m = X[:, FIELD_DIM:].reshape(n, 12, MON_DIM)
    if c < 8:
        m[ar, oa, NUM_OFF + 65] = c / 10
    else:
        m[ar, oa, NUM_OFF + 63] = 0.0; m[ar, oa, NUM_OFF + 65] = 0.0
    SW.append(feats(X))


def auc(neg, pos):
    x = np.r_[neg, pos]; rk = np.empty(len(x)); rk[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return (rk[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


tr = game[idx] % 2 == 0; te = ~tr
y = cnt[idx] >= 4; t1 = (ph["origin"][idx] == "rest") & (ph["slept"][idx] == 1); t2 = (ph["origin"][idx] == "rest") & (ph["slept"][idx] == 2)
for li in range(3):
    for lam in (0.03, 0.1, 0.3, 1.0):
        Z = np.concatenate([SW[c][li][tr] - SW[c][li][tr].mean(0) for c in range(9)])
        S = np.cov(Z.T); S = S + lam * np.trace(S) / 256 * np.eye(256)
        mp = np.concatenate([SW[c][li][tr] for c in range(4, 8)]).mean(0); mn = np.concatenate([SW[c][li][tr] for c in range(0, 4)] + [SW[8][li][tr]]).mean(0)
        w = np.linalg.solve(S, mp - mn)
        x = real[li] @ w
        sw3, sw4 = SW[3][li][te] @ w, SW[4][li][te] @ w
        print(f"norm2(L{li}.mid) λ {lam:g}: real held-out AUC c≥4 vs ≤3 {auc(x[te & ~y], x[te & y]):.3f}; truth slept1 vs 2 "
              f"{auc(x[te & t1], x[te & t2]):.3f}; sweep held-out c3 vs c4 {auc(sw3, sw4):.3f}", flush=True)
