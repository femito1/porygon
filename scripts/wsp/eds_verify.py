"""eds step 4: the saved edited policy loads with porygon.model.load_policy and reproduces the offline numbers.
Full stock forward (policy.get_distribution / predict_values, not the instrumented Net) over all rows, compared with
the edit's cached-path probabilities (out/eds/edit_<label>_<tag>.npz) and the stock checkpoint.
    python scripts/wsp/eds_verify.py PT_PATH TAG LABEL
"""
import sys
from pathlib import Path

import numpy as np
import torch

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from porygon.model import load_policy
from wsp_common import CKPTS

torch.set_num_threads(2)
pt, tag, label = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N)
pol = load_policy(pt); ref = load_policy(CKPTS[tag])
n_diff = sum(int((a != b).sum()) for a, b in zip(pol.state_dict().values(), ref.state_dict().values()))
tensors = [k for (k, a), b in zip(pol.state_dict().items(), ref.state_dict().values()) if (a != b).any()]


def fwd(p):
    P, V = [], []
    with torch.no_grad():
        for s in range(0, N, 2048):
            obs = {"observation": torch.as_tensor(d["obs"][s:s + 2048], dtype=torch.float32),
                   "action_mask": torch.as_tensor(d["mask"][s:s + 2048])}
            P.append(p.get_distribution(obs).distribution.probs.double().numpy())
            V.append(p.predict_values(obs).squeeze(-1).numpy())
    return np.concatenate(P), np.concatenate(V)


P1, V1 = fwd(pol)
E = np.load(OUT / f"edit_{label}_{tag}.npz")
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
is_st = MOVES[mons[r, own_a, 3:7].astype(np.int64)] == "sleeptalk"
pst = (P1[:, 6:] * (is_st & (d["mask"][:, 6:] > 0))).sum(1)
H = ph["slp"] & ph["st_legal"].any(1); W = H & (ph["cnt"] >= 4); K = H & (ph["cnt"] < 4)
lines = [f"{pt}: loads with porygon.model.load_policy; {n_diff} weights differ from {tag} in {tensors}",
         f"max |P(saved, stock forward) − P(edit, cached path)| = {np.abs(P1 - E['P1']).max():.2e}; max |ΔV| = {np.abs(V1 - E['V1']).max():.2e}",
         f"P(ST | W) {pst[W].mean():.3f} (offline {E['pst1'][W].mean():.3f}); P(ST | K) {pst[K].mean():.3f} (offline {E['pst1'][K].mean():.3f})"]
print("\n".join(lines))
(OUT / f"verify_{label}_{tag}.txt").write_text("\n".join(lines) + "\n")
