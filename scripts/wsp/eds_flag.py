"""eds: design reads for edit v2 — an exact input-level gate in mon_proj.

The one-unit L2 gate cannot threshold 3|4: at the Sleep Talk token the counter is only coarsely represented
(eds_where2.py: paired counter 3→4 held-out AUC 0.80–0.84 at every layer). The counter is exact at the input, so v2
moves the gate there: one hidden unit of mon_proj (573→256 GELU→256) computes
    AND(status = SLP, active, mine, counter ≥ 4, Sleep Talk in the moveset)
from the raw numerics (exact) and a Sleep Talk readout of the 4 move_proj slots, and writes into the own-active token
the change the network itself makes for a higher counter ("exaggerate the counter"): the network's own downstream
circuit (counter sweep: P(ST) 0.05 at counter 6, 0.01 at 7) then switches Sleep Talk off.
Reads here: (1) the Sleep Talk direction in move_proj output space (LDA over every move slot in the data) and its
margins; (2) mon_proj hidden-unit usage + exact mean-ablation screen; (3) the write vector: own-active token input
with counter := 7 minus counter := 4 (counterfactual, target rows).
    python scripts/wsp/eds_flag.py 500M -> out/eds/flag_<tag>.npz / .txt
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB, N_CAT, MOVEPOOL, MOVE_DIM
from wsp_common import Net, load

torch.set_num_threads(2)
tag = sys.argv[1]
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N); game = d["game"]
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"]); ST_ID = int(np.flatnonzero(MOVES == "sleeptalk")[0])
is_st = mons[r, own_a, 3:7].astype(np.int64) == ST_ID; jst = is_st.argmax(1)
H = ph["slp"] & ph["st_legal"].any(1); cnt = ph["cnt"]
pol = load(tag); net = Net(pol); fe = net.fe
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


MV0 = NUM_OFF + 125
# ---- (1) Sleep Talk readout in move_proj output space, over every (mon, slot) in a 20k-row sample
rng = np.random.default_rng(0); samp = np.sort(rng.choice(N, 20000, replace=False))
with torch.no_grad():
    m = torch.as_tensor(mons[samp].astype(np.float32))
    ids = m[..., 3:7].long()
    mvn = m[..., MV0:].reshape(len(samp), 12, 4, MOVE_DIM)
    mp = fe.move_proj(torch.cat([fe.moves(ids), mvn], -1)).numpy()     # [n,12,4,64]
idn = ids.numpy(); real = idn > 0
X = mp[real]; y = idn[real] == ST_ID
S = np.cov(np.concatenate([X[y] - X[y].mean(0), X[~y] - X[~y].mean(0)]).T); S += 0.01 * np.trace(S) / 64 * np.eye(64)
u = np.linalg.solve(S, X[y].mean(0) - X[~y].mean(0))
a0, a1 = X[~y].mean(0) @ u, X[y].mean(0) @ u
u = u / (a1 - a0); b = -a0 / (a1 - a0)
rs = mp @ u + b                       # per slot, ~1 for Sleep Talk, ~0 otherwise
rs = np.where(real, rs, 0.0)          # empty slots: move_proj of padding is not zero; mask in readout below
pad_val = float((fe.move_proj(torch.cat([torch.zeros(64), torch.zeros(MOVE_DIM)])).detach().numpy() @ u) + b)
tot = (mp @ u + b).sum(-1)            # what a mon_proj unit reading all 4 slots sees (pads included)
has = (idn == ST_ID).any(-1)
P(f"(1) Sleep Talk readout per move slot (LDA in move_proj output space, 64-d; {real.sum()} slots, {y.sum()} ST): "
  f"ST slots [min {rs[idn == ST_ID].min():.2f}, median {np.median(rs[idn == ST_ID]):.2f}]; other slots [median {np.median(rs[real & (idn != ST_ID)]):.3f}, "
  f"p99.9 {np.percentile(rs[real & (idn != ST_ID)], 99.9):.2f}, max {rs[real & (idn != ST_ID)].max():.2f}]; pad slot value {pad_val:.2f}")
P(f"    sum over a mon's 4 slots: mons with ST [min {tot[has].min():.2f}, median {np.median(tot[has]):.2f}]; mons without "
  f"[median {np.median(tot[~has]):.2f}, p99.9 {np.percentile(tot[~has], 99.9):.2f}, max {tot[~has].max():.2f}]")
# ---- (2) mon_proj hidden-unit usage over all mon tokens, exact mean-ablation screen
s1 = np.zeros(256); s2 = np.zeros(256); nt = 0
with torch.no_grad():
    for s in range(0, N, 2048):
        c = {}
        net.embed(torch.as_tensor(d["obs"][s:s + 2048], dtype=torch.float32), None, c)
        h = Fn.gelu(fe.mon_proj[0](c["mon_in"])).double()
        s1 += h.sum((0, 1)).numpy(); s2 += (h ** 2).sum((0, 1)).numpy(); nt += h.shape[0] * h.shape[1]
mean = s1 / nt; var = s2 / nt - mean ** 2
W2 = fe.mon_proj[2].weight.detach()
proxy = var * W2.norm(dim=0).numpy() ** 2
order = np.argsort(proxy)[:8]
samp2 = np.sort(rng.choice(N, 8000, replace=False))
x8 = torch.as_tensor(d["obs"][samp2], dtype=torch.float32); m8 = torch.as_tensor(d["mask"][samp2])
with torch.no_grad():
    l0, v0 = net.forward(x8, m8); p0 = l0.softmax(-1)
P("(2) mon_proj hidden units, exact mean-ablation (full forward, 8,000 random rows):")
P("unit | var·|W2 col|² | mean | mean TV | p99 TV | mean |ΔV|")
best, bt = None, 9
for uu in order:
    with torch.no_grad():
        col = W2[:, uu].clone(); bias = fe.mon_proj[2].bias.clone()
        fe.mon_proj[2].bias += col * mean[uu]; fe.mon_proj[2].weight[:, uu] = 0
        l1, v1 = net.forward(x8, m8)
        fe.mon_proj[2].weight[:, uu] = col; fe.mon_proj[2].bias.copy_(bias)
    tv = 0.5 * (l1.softmax(-1) - p0).abs().sum(1).numpy(); dv = (v1 - v0).abs().numpy()
    score = tv.mean() / 0.01 + dv.mean() / 0.005
    if score < bt:
        best, bt = int(uu), score
    P(f"{uu} | {proxy[uu]:.2e} | {mean[uu]:+.3f} | {tv.mean():.2e} | {np.percentile(tv, 99):.2e} | {dv.mean():.2e}")
P(f"chosen unit (smallest TV/0.01 + |ΔV|/0.005): {best}; median proxy over units {np.median(proxy):.2e}")
# ---- (3) write vector: own-active token input, counter := 7 minus counter := 4, on the target rows
T = np.flatnonzero(H & (cnt >= 4)); n = len(T); ar = np.arange(n); oa = own_a[T]


def own_tok(cval):
    X = d["obs"][T].astype(np.float32).copy(); X[:, FIELD_DIM:].reshape(n, 12, MON_DIM)[ar, oa, NUM_OFF + 65] = cval
    c = {}
    with torch.no_grad():
        net.embed(torch.as_tensor(X), None, c)
    return c["tokens"][ar, 1 + oa].numpy()


t4, t7 = own_tok(0.4), own_tok(0.7)
v = (t7 - t4).mean(0)
P(f"(3) write vector v = mean over {n} target rows of token(counter 7) − token(counter 4): |v| {np.linalg.norm(v):.2f}, "
  f"per-row consistency (mean cos) {np.mean([(a @ v) / (np.linalg.norm(a) * np.linalg.norm(v) + 1e-9) for a in (t7 - t4)]):.3f}, "
  f"|own token| {np.linalg.norm(t4, axis=1).mean():.2f}")
np.savez(OUT / f"flag_{tag}.npz", u_st=u, b_st=b, unit=best, mean=mean, v=v, pad_val=pad_val)
(OUT / f"flag_{tag}.txt").write_text("\n".join(txt) + "\n")
