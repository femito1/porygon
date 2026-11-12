"""eds: one reading pass of the stock network over all 48,518 logged decisions.

Saves (node-local /tmp/wsp-674320/eds/<tag>_*.npy, float32 [N,17,256]):
  f2   = norm2(L2.mid)   the exact input of layer 2's MLP (what a repurposed L2 unit reads)
  out2 = L2.out          the residual stream after layer 2 (edits add W2[:,u]·Δhid to it; then final LN + heads)
and in out/eds/<tag>_sub.npz, for the rows whose active has Sleep Talk (plus all asleep rows), the move-token and
own-active-token streams L-1.out, L0.out, L1.out, L2.mid. Also per-unit statistics of layer 2's MLP.
    python scripts/wsp/eds_cache.py 500M
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds"); LOC = Path("/tmp/wsp-674320/eds")
LOC.mkdir(parents=True, exist_ok=True)
d = load_local()
N = len(d["mask"]); r = np.arange(N)
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
names = MOVES[mons[r, own_a, 3:7].astype(np.int64)]
slp = mons[r, own_a, NUM_OFF + 63] > .5
sub = np.flatnonzero((names == "sleeptalk").any(1) | slp)
insub = np.full(N, -1); insub[sub] = np.arange(len(sub))
net = Net(load(tag))
L2 = net.layers[2]
f2 = np.lib.format.open_memmap(LOC / f"{tag}_f2.npy", "w+", np.float32, (N, 17, 256))
o2 = np.lib.format.open_memmap(LOC / f"{tag}_out2.npy", "w+", np.float32, (N, 17, 256))
S = {k: np.zeros((len(sub), 5, 256), np.float32) for k in ("L-1.out", "L0.out", "L1.out", "L2.mid")}
P = np.zeros((N, 10), np.float32); V = np.zeros(N, np.float32)
s1 = np.zeros(512); s2 = np.zeros(512); s1m = np.zeros(512); s2m = np.zeros(512); n_tok = 0; n_mv = 0
t0 = time.time()
with torch.no_grad():
    for s in range(0, N, 1024):
        xb = torch.as_tensor(d["obs"][s:s + 1024], dtype=torch.float32)
        mb = torch.as_tensor(d["mask"][s:s + 1024])
        c = {}
        lg, v = net.forward(xb, mb, None, c)
        P[s:s + len(xb)] = lg.softmax(-1).numpy(); V[s:s + len(xb)] = v.numpy()
        f = L2.norm2(c["L2.mid"])
        f2[s:s + len(xb)] = f.numpy(); o2[s:s + len(xb)] = c["L2.out"].numpy()
        h = c["L2.mlp_hid"].double()
        s1 += h.sum((0, 1)).numpy(); s2 += (h ** 2).sum((0, 1)).numpy(); n_tok += h.shape[0] * h.shape[1]
        s1m += h[:, 13:].sum((0, 1)).numpy(); s2m += (h[:, 13:] ** 2).sum((0, 1)).numpy(); n_mv += h.shape[0] * 4
        idx = np.arange(s, s + len(xb)); m = insub[idx] >= 0
        if m.any():
            toks = np.concatenate([1 + own_a[idx][m, None], np.tile(np.arange(13, 17), (m.sum(), 1))], 1)
            for k in S:
                S[k][insub[idx][m]] = c[k][np.flatnonzero(m)[:, None], toks].numpy()
f2.flush(); o2.flush()
mean = s1 / n_tok; var = s2 / n_tok - mean ** 2
W2n = L2.linear2.weight.norm(dim=0).detach().numpy()
np.savez(OUT / f"{tag}_sub.npz", rows=sub, **{k.replace(".", "_"): v for k, v in S.items()})
np.savez(OUT / f"{tag}_units.npz", mean=mean, var=var, mean_mv=s1m / n_mv, var_mv=s2m / n_mv - (s1m / n_mv) ** 2,
         w2norm=W2n, proxy=var * W2n ** 2)
np.save(LOC / f"{tag}_probs.npy", P); np.save(LOC / f"{tag}_value.npy", V)
print(f"{tag}: {N} rows in {time.time() - t0:.0f}s; sub rows {len(sub)}; max|probs - logged| "
      f"{np.abs(P - d['probs']).max():.2e}" + ("" if tag != "500M" else ""))
order = np.argsort(var * W2n ** 2)
print("10 smallest var*|W2col|^2 units:", [(int(u), f"{var[u]*W2n[u]**2:.2e}", f"mean {mean[u]:+.3f}") for u in order[:10]])
print("median proxy", np.median(var * W2n ** 2))
