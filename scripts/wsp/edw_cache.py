"""edw step 0: cache everything the switch-edit study reads (reading passes, not checks).

  LOCAL/<tag>_full.npz : fin [N,11,256] f16 (final LN'd latents, tokens field/own1-6/moves), logits, value,
                         (500M only) L1mid/L1out/L2out at own tokens [N,6,256] f16
  LOCAL/<tag>_swap_{train,test}.npz : opponent-active swap pairs (beh_mech A 'full' construction, donors from the same
                         half): rows, donors, obs0/obsf, fin0/finf, L1mid/L1out/L2out own tokens (orig, swapped),
                         L1 head_out at own tokens [n,6,8,256], L1 pattern own->opp active [n,8,6], L0.out at the
                         opp-active token (orig, swapped), logits, values.
    python scripts/wsp/edw_cache.py 500M
"""
import sys

import numpy as np
import torch

from beh_common import FIELD_DIM, MON_DIM, NUM_OFF, MOVENUM_OFF, TYP, FNT, load_local
from beh_mech_lib import recompute_mult
from edw_common import FIN_TOK, LOCAL, matchup, split
from porygon.encoder import MOVE_DIM
from wsp_common import Net, load, CKPTS

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
LOCAL.mkdir(parents=True, exist_ok=True)
d = load_local()
N = len(d["mask"])
net = Net(load(CKPTS.get(tag, tag)))
deep = tag == "500M"


def fwd(X, MK, keep, bs=1000):
    acc = {k: [] for k in keep}
    L, V = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            c = {}
            lg, v = net.forward(torch.as_tensor(X[s:s + bs], dtype=torch.float32), torch.as_tensor(MK[s:s + bs]), None, c)
            L.append(lg.double().numpy()); V.append(v.numpy())
            for k, fn in keep.items():
                acc[k].append(fn(c, s))
    return np.concatenate(L), np.concatenate(V), {k: np.concatenate(v) for k, v in acc.items()}


keep = {"fin": lambda c, s: c["final"][:, FIN_TOK].half().numpy()}
if deep:
    for k in ("L1.mid", "L1.out", "L2.out"):
        keep[k.replace(".", "")] = (lambda k: lambda c, s: c[k][:, 1:7].half().numpy())(k)
obs = d["obs"]
lg, v, C = fwd(obs, d["mask"], keep)
np.savez(LOCAL / f"{tag}_full.npz", logits=lg, value=v, **C)
p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
print(f"{tag}: full cache done; max |probs - logged| = {np.abs(p - d['probs']).max():.2e}" if tag == "500M" else f"{tag}: full cache done", flush=True)

# ---------------- swaps
f = matchup(obs)
tr = split(d["game"])
mask = d["mask"] > 0
nums = obs[:, FIELD_DIM:].reshape(N, 12, MON_DIM)[..., NUM_OFF:MOVENUM_OFF]
alive = f["has_opp"] & (nums[np.arange(N), f["opp_a"], FNT] < 0.5)
base = alive & (mask[:, :6].sum(1) >= 2) & f["has_act"]
for half, sel in (("train", tr), ("test", ~tr)):
    rng = np.random.default_rng(7 if half == "train" else 8)
    cand = np.flatnonzero(base & sel)
    rows = np.sort(rng.choice(cand, min(4000, len(cand)), replace=False))
    donors = rng.choice(cand, len(rows))
    k = len(rows); ar = np.arange(k)
    X0 = obs[rows].astype(np.float32); Xd = obs[donors].astype(np.float32)
    m0 = X0[:, FIELD_DIM:].reshape(k, 12, MON_DIM); md = Xd[:, FIELD_DIM:].reshape(k, 12, MON_DIM)
    oa, oad, wa = f["opp_a"][rows], f["opp_a"][donors], f["own_a"][rows]
    new_opp = md[ar, oad].copy()
    new_types = new_opp[:, NUM_OFF + TYP:NUM_OFF + TYP + 18]
    our_types = m0[ar, wa, NUM_OFF + TYP:NUM_OFF + TYP + 18]
    Xf = X0.copy(); mf = Xf[:, FIELD_DIM:].reshape(k, 12, MON_DIM)
    mf[ar, oa] = new_opp
    mf[ar, oa, MOVENUM_OFF:] = recompute_mult(new_opp[:, MOVENUM_OFF:].reshape(k, 4, MOVE_DIM), our_types[:, None, :]).reshape(k, -1)
    own = mf[:, :6, MOVENUM_OFF:].reshape(k, 6, 4, MOVE_DIM)
    mf[:, :6, MOVENUM_OFF:] = recompute_mult(own, new_types[:, None, None].repeat(6, 1)).reshape(k, 6, -1)
    MK = d["mask"][rows]
    tok_oa = 1 + oa
    kp = dict(keep)
    if deep:
        kp["L1head"] = lambda c, s: c["L1.head_out"][:, 1:7].half().numpy()
        kp["L1pat"] = lambda c, s: c["L1.pattern"][np.arange(len(c["final"])), :, 1:7, tok_oa[s:s + len(c["final"])]].numpy()
        kp["L0oa"] = lambda c, s: c["L0.out"][np.arange(len(c["final"])), tok_oa[s:s + len(c["final"])]].numpy()
    lg0, v0, C0 = fwd(X0, MK, kp)
    lgf, vf, Cf = fwd(Xf, MK, kp)
    np.savez(LOCAL / f"{tag}_swap_{half}.npz", rows=rows, donors=donors, obs0=X0.astype(np.float16), obsf=Xf.astype(np.float16),
             lg0=lg0, lgf=lgf, v0=v0, vf=vf, **{k + "_0": v_ for k, v_ in C0.items()}, **{k + "_f": v_ for k, v_ in Cf.items()})
    print(f"{tag}: swap {half} cached ({k} pairs)", flush=True)
print("done")
