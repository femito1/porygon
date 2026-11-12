"""eds: is the counter crisp at the OWN-ACTIVE token? (precondition for a 2-unit path: L0 MLP unit at the own-active
token computes the flag, L1/L2 attention carries it, an L2 unit at the Sleep Talk token acts on it)
- paired counterfactual sweep (counter := 0..7 on the 707 asleep-ST positions); Fisher direction from even games with
  the covariance of ALL own-active tokens (48k rows, real) + the within-level sweep covariance;
  held-out AUC at L-1 (token input) and norm2(L0.mid) (input of L0's MLP), own-active token.
- L0 MLP unit usage: exact mean-ablation screen (heads ... full forward on 8,000 rows) of the 8 lowest var·|W2col|² units.
    python scripts/wsp/eds_where3.py 500M
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import Intervention, Net, load

torch.set_num_threads(2)
tag = sys.argv[1]
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N); game = d["game"]
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
H = ph["slp"] & ph["st_legal"].any(1); cnt = ph["cnt"]
idx = np.flatnonzero(H); n = len(idx); ar = np.arange(n); oa = own_a[idx]
net = Net(load(tag)); L0 = net.layers[0]
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


def feats(X, rows_oa, M):
    c = {}
    with torch.no_grad():
        net.forward(torch.as_tensor(X), torch.as_tensor(M), None, c)
        k = np.arange(len(X))
        return {"L-1": c["L-1.out"][k, 1 + rows_oa].numpy(), "f0": L0.norm2(c["L0.mid"])[k, 1 + rows_oa].numpy(),
                "pre0": c["L0.mlp_pre"][k, 1 + rows_oa].numpy()}


X0 = d["obs"][idx].astype(np.float32); MK = d["mask"][idx]
SW = []
for c in range(8):
    X = X0.copy(); X[:, FIELD_DIM:].reshape(n, 12, MON_DIM)[ar, oa, NUM_OFF + 65] = c / 10
    SW.append(feats(X, oa, MK))
has = d["mask"][:, 6:].any(1) & (mons[:, :6, NUM_OFF + 115].max(1) > .5)
allr = np.flatnonzero(has)[::2]
ALL = {k: [] for k in ("L-1", "f0", "pre0")}
for s in range(0, len(allr), 4096):
    rr = allr[s:s + 4096]
    f = feats(d["obs"][rr].astype(np.float32), own_a[rr], d["mask"][rr])
    for k in ALL:
        ALL[k].append(f[k])
ALL = {k: np.concatenate(v) for k, v in ALL.items()}
P(f"reading passes: 8 sweep passes over {n} asleep-ST positions; 1 pass over {len(allr)} rows (own-active covariance)")


def auc(neg, pos):
    x = np.r_[neg, pos]; rk = np.empty(len(x)); rk[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return (rk[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


tr = game[idx] % 2 == 0; te = ~tr; ga = game[allr] % 2 == 0
real = feats(X0, oa, MK); y = cnt[idx] >= 4
for key in ("L-1", "f0"):
    for wcov in (0.0, 0.1, 1.0):
        Z = np.concatenate([SW[c][key][tr] - SW[c][key][tr].mean(0) for c in range(8)])
        S = np.cov(Z.T) + wcov * np.cov(ALL[key][ga].T); S += 0.01 * np.trace(S) / 256 * np.eye(256)
        dmu = np.concatenate([SW[c][key][tr] for c in range(4, 8)]).mean(0) - np.concatenate([SW[c][key][tr] for c in range(4)]).mean(0)
        w = np.linalg.solve(S, dmu)
        P(f"own-active {key:3s}, weight of all-token covariance {wcov}: sweep held-out AUC c3 vs c4 {auc(SW[3][key][te] @ w, SW[4][key][te] @ w):.4f}; "
          f"real held-out c≤3 vs ≥4 {auc(real[key][te & ~y] @ w, real[key][te & y] @ w):.4f}")
# L0 MLP unit screen
hid = Fn.gelu(torch.as_tensor(ALL["pre0"]))
P("L0 MLP units at the own-active token: least-used screen (mean-ablation everywhere, full forward 8,000 rows)")
rng = np.random.default_rng(0); samp = np.sort(rng.choice(N, 8000, replace=False))
x8 = torch.as_tensor(d["obs"][samp], dtype=torch.float32); m8 = torch.as_tensor(d["mask"][samp])
s1 = np.zeros(512); s2 = np.zeros(512); nt = 0
with torch.no_grad():
    for s in range(0, N, 4096):
        c = {}
        net.forward(torch.as_tensor(d["obs"][s:s + 4096], dtype=torch.float32), torch.as_tensor(d["mask"][s:s + 4096]), None, c)
        h = c["L0.mlp_hid"].double(); s1 += h.sum((0, 1)).numpy(); s2 += (h ** 2).sum((0, 1)).numpy(); nt += h.shape[0] * h.shape[1]
    mean = s1 / nt; var = s2 / nt - mean ** 2
    proxy = var * L0.linear2.weight.norm(dim=0).numpy() ** 2
    l0, v0 = net.forward(x8, m8); p0 = l0.softmax(-1)
    for uu in np.argsort(proxy)[:8]:
        mask = torch.ones(512); mask[uu] = 0
        iv = Intervention(mlp_mask={0: mask}, resid_add={(0, None): (L0.linear2.weight[:, uu] * float(mean[uu]))})
        l1, v1 = net.forward(x8, m8, iv)
        tv = 0.5 * (l1.softmax(-1) - p0).abs().sum(1).numpy()
        P(f"  unit {uu}: proxy {proxy[uu]:.2e}, mean {mean[uu]:+.3f}, TV mean {tv.mean():.2e} p99 {np.percentile(tv, 99):.2e}, |ΔV| {(v1 - v0).abs().mean():.2e}")
np.savez(OUT / f"where3_{tag}.npz", mean0=mean, proxy0=proxy)
(OUT / f"where3_{tag}.txt").write_text("\n".join(txt) + "\n")
