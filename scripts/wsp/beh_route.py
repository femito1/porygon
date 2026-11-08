"""beh analysis 5b: which heads route the opponent-active information into the switch tokens, and what
mean-ablating the candidate heads does to the whole policy (agreement with the search, oracle Q gap r2-300k).

1. attention from each own-bench (switch) token to the opp-active token, per layer/head (mean pattern weight).
2. head-level activation patching of the full opponent swap (beh_mech A): z of head (l,h) at the own-bench
   tokens taken from the swapped run, everything else clean; recovered fraction of the switch-preference change.
3. whole-dataset effect of mean-ablating single heads: argmax agreement with the search, mean oracle Q gap of the
   argmax, P(switch) in voluntary positions.
    python scripts/wsp/beh_route.py 500M -> out/beh/route_<tag>.txt
"""
import sys

import numpy as np
import torch

from beh_common import OUT, FIELD_DIM, MON_DIM, NUM_OFF, MOVENUM_OFF, TYP, FNT, decode, features, load_local, ratio_boot
from beh_mech_lib import build_swaps
from wsp_common import Intervention, Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
d = load_local(); f = features(d)
der = np.load(OUT / "derived_r2-300k.npz", allow_pickle=True)
net = Net(load(tag))
N = len(d["mask"]); mask = d["mask"] > 0
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


rows, X0, Xf, MK = build_swaps(d, f, n=1500, seed=2)
n = len(rows); ar = np.arange(n)
wa, oa = f["own_a"][rows], f["opp_a"][rows]
bench = np.zeros((n, 17), bool)
for s in range(6):
    bench[:, 1 + s] = (MK[:, s] > 0)  # legal switch targets = own bench tokens that can be chosen
tok_opp = 1 + oa


def fwd(X, iv=None, keep=()):
    with torch.no_grad():
        c = {} if keep else None
        lg, v = net.forward(torch.as_tensor(X), torch.as_tensor(MK), iv, c)
    return lg.double().numpy(), v.numpy(), c


la, va, ca = fwd(X0, keep=tuple(f"L{i}.{k}" for i in range(3) for k in ("z", "pattern")))
lb, vb, cb = fwd(Xf, keep=tuple(f"L{i}.z" for i in range(3)))
lsw = MK[:, :6] > 0


def cen(a):
    a = np.where(lsw, a, np.nan); return a - np.nanmean(a, 1, keepdims=True)


D = cen(lb[:, :6] - la[:, :6])
P(f"## 1. attention from switch-candidate tokens to the opp-active token ({tag}); uniform would be {1/17:.3f}")
for i in range(3):
    pat = ca[f"L{i}.pattern"].numpy()  # B,H,Tq,Tk
    w = pat[ar[:, None], :, np.arange(17)[None, :], tok_opp[:, None]]  # B,Tq,H
    wb = np.array([w[:, :, h][bench].mean() for h in range(8)])
    wm = np.array([w[:, 13:, h].mean() for h in range(8)])
    P(f"L{i}: bench→opp-active " + " ".join(f"h{h} {wb[h]:.2f}" for h in range(8)) + " | move tokens→opp-active " + " ".join(f"h{h} {wm[h]:.2f}" for h in range(8)))
P("\n## 2. head-level patching of the opponent swap at the switch-candidate tokens (recovered fraction of the centred switch-logit change)")
bt = torch.as_tensor(bench)
for i in range(3):
    row = []
    for h in range(8):
        za, zb = ca[f"L{i}.z"][:, :, h], cb[f"L{i}.z"][:, :, h]
        m = torch.where(bt.unsqueeze(-1), zb, za)
        lp, _, _ = fwd(X0, Intervention(head_mean={i: {h: m}}))
        Dp = cen(lp[:, :6] - la[:, :6])
        rec = np.nansum(Dp * D, 1) / np.maximum(np.nansum(D * D, 1), 1e-9)
        row.append(f"h{h} {np.mean(rec):.2f}")
    # all heads of the layer together
    m_all = {h: torch.where(bt.unsqueeze(-1), cb[f"L{i}.z"][:, :, h], ca[f"L{i}.z"][:, :, h]) for h in range(8)}
    lp, _, _ = fwd(X0, Intervention(head_mean={i: m_all}))
    Dp = cen(lp[:, :6] - la[:, :6])
    rec = np.nansum(Dp * D, 1) / np.maximum(np.nansum(D * D, 1), 1e-9)
    P(f"L{i}: " + " ".join(row) + f" | all heads {np.mean(rec):.2f}")

P("\n## 3. whole-dataset effect of mean-ablating one head (r2-300k oracle labels, all 48.5k decisions)")
samp = np.random.default_rng(0).choice(N, 4000, replace=False)
with torch.no_grad():
    c = {}
    net.forward(torch.as_tensor(d["obs"][samp].astype(np.float32)), torch.as_tensor(d["mask"][samp]), None, c)
zmean = {i: c[f"L{i}.z"].mean(0) for i in range(3)}
lq = der["lab_q"]; qmax = np.nanmax(np.where(np.isfinite(lq), lq, -1), 1)
sa = der["srch_arg"]; vol = f["voluntary"]


def full_pass(iv):
    L = []
    with torch.no_grad():
        for s in range(0, N, 2000):
            lg, v = net.forward(torch.as_tensor(d["obs"][s:s + 2000].astype(np.float32)), torch.as_tensor(d["mask"][s:s + 2000]), iv)
            L.append(lg.softmax(-1).numpy())
    return np.concatenate(L)


P("ablation | argmax agreement with search | mean Q gap of argmax | vol P(switch) | Q gap on switch->X disagreement rows | agreement on agree rows")
swx = np.isin(der["cat"], ["switch->switch", "switch->attack", "switch->status"])
for name, iv in [("none", None)] + [(f"L{i}h{h}", Intervention(head_mean={i: {h: zmean[i][:, h]}}))
                                     for i, h in ((0, 3), (0, 6), (1, 4), (1, 7), (1, 0), (2, 3), (2, 1))]:
    p = full_pass(iv)
    arg = np.where(mask, p, -1).argmax(1)
    gap = lq[np.arange(N), arg] - qmax
    e = ratio_boot(d["game"], (arg == sa).astype(float), n_boot=300)
    g = ratio_boot(d["game"], np.nan_to_num(gap), w=np.isfinite(gap).astype(float), n_boot=300)
    P(f"{name} | {e[0]:.3f} [{e[1]:.3f},{e[2]:.3f}] | {g[0]:+.4f} [{g[1]:+.4f},{g[2]:+.4f}] | {p[vol,:6].sum(1).mean():.3f} | "
      f"{np.nanmean(gap[swx]):+.4f} | {np.mean(arg[der['agr']] == sa[der['agr']]):.3f}")
(OUT / f"route_{tag}.txt").write_text("\n".join(txt) + "\n")
