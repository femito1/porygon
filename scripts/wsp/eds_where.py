"""eds: where is the status counter encoded precisely enough to threshold at 3|4?
Held-out (odd games) AUC of a shrunk-LDA readout, real asleep-ST rows, counter ≥ 4 vs ≤ 3, fit on even games,
at the own-active token and the Sleep Talk token after each layer; plus the attention weight from the ST token to
the own-active token per layer/head (the counter reaches the ST token only through attention).
    python scripts/wsp/eds_where.py 500M -> out/eds/where_<tag>.txt
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
idx = np.flatnonzero(H); n = len(idx); ar = np.arange(n)
net = Net(load(tag))
c = {}
with torch.no_grad():
    net.forward(torch.as_tensor(d["obs"][idx].astype(np.float32)), torch.as_tensor(d["mask"][idx]), None, c)
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


def auc(neg, pos):
    x = np.r_[neg, pos]; rk = np.empty(len(x)); rk[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return (rk[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


tr = game[idx] % 2 == 0; y = cnt[idx] >= 4
P(f"{tag}: {n} asleep-ST rows; held-out AUC, counter ≥ 4 vs ≤ 3 (shrunk LDA λ=1 / difference of means)")
for key in ("L-1.out", "L0.mid", "L0.out", "L1.mid", "L1.out", "L2.mid", "L2.out"):
    for tok, ti in (("own active", 1 + own_a[idx]), ("Sleep Talk", 13 + jst[idx])):
        X = c[key][ar, ti].numpy()
        res = []
        for lam in (1.0, 1e6):
            S = np.cov(np.concatenate([X[tr & y] - X[tr & y].mean(0), X[tr & ~y] - X[tr & ~y].mean(0)]).T)
            S = S + lam * np.trace(S) / 256 * np.eye(256)
            w = np.linalg.solve(S, X[tr & y].mean(0) - X[tr & ~y].mean(0))
            res.append(auc(X[~tr & ~y] @ w, X[~tr & y] @ w))
        P(f"  {key:7s} {tok:10s}: {res[0]:.3f} / {res[1]:.3f}")
P("raw counter numeric itself: AUC 1.000 by construction; truth slept 1 vs 2 by counter ≥ 4: "
  f"TPR {np.mean(cnt[H & (ph['slept'] == 2) & (ph['origin'] == 'rest')] >= 4):.3f}, FPR {np.mean(cnt[H & (ph['slept'] == 1) & (ph['origin'] == 'rest')] >= 4):.3f}")
P("attention from the ST token to the own-active token (mean ± sd over rows; uniform 1/17 = 0.059):")
for li in range(3):
    A = c[f"L{li}.pattern"][ar, :, 13 + jst[idx], 1 + own_a[idx]].numpy()  # [n, H]
    P(f"  L{li}: " + " ".join(f"h{h} {A[:, h].mean():.2f}±{A[:, h].std():.2f}" for h in range(8)))
(OUT / f"where_{tag}.txt").write_text("\n".join(txt) + "\n")
