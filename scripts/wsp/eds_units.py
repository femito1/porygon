"""eds: refine the choice of the repurposed L2 MLP unit on ALL rows (heads-only, from the cache): mean-replacement of
each of the 12 lowest var·|W2col|² units; mean / p99 / max TV, max TV on asleep-without-Rest rows, mean |ΔV|.
    python scripts/wsp/eds_units.py 500M
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import load

torch.set_num_threads(2)
tag = sys.argv[1]
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds"); LOC = Path("/tmp/wsp-674320/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N)
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
has_rest = (MOVES[mons[r, own_a, 3:7].astype(np.int64)] == "rest").any(1)
m7 = ph["slp"] & ~has_rest
pol = load(tag); fe = pol.features_extractor; L2 = fe.encoder.layers[2]
U = np.load(OUT / f"{tag}_units.npz"); order = np.argsort(U["proxy"])[:12]
f2 = np.load(LOC / f"{tag}_f2.npy", mmap_mode="r"); o2 = np.load(LOC / f"{tag}_out2.npy", mmap_mode="r")
MK = torch.as_tensor(d["mask"])


def heads(o, mk):
    lat = fe.norm(o); flat = lat.flatten(1)
    lg = pol.action_net(flat).masked_fill(mk == 0, torch.finfo(torch.float32).min)
    return lg.double().softmax(-1), pol.value_net(pol.mlp_extractor.forward_critic(flat)).squeeze(-1)


TV = np.zeros((len(order), N)); DV = np.zeros((len(order), N))
with torch.no_grad():
    for s in range(0, N, 2048):
        fb = torch.as_tensor(np.asarray(f2[s:s + 2048])); ob = torch.as_tensor(np.asarray(o2[s:s + 2048])); mk = MK[s:s + 2048]
        p0, v0 = heads(ob, mk)
        for k, u in enumerate(order):
            hid = Fn.gelu(fb @ L2.linear1.weight[u] + L2.linear1.bias[u])
            p1, v1 = heads(ob + (U["mean"][u] - hid).unsqueeze(-1) * L2.linear2.weight[:, u], mk)
            TV[k, s:s + 2048] = 0.5 * (p1 - p0).abs().sum(1).numpy(); DV[k, s:s + 2048] = (v1 - v0).abs().numpy()
lines = ["unit | mean TV | p99 TV | max TV | max TV asleep-no-Rest | mean |ΔV|"]
for k, u in enumerate(order):
    lines.append(f"{u} | {TV[k].mean():.2e} | {np.percentile(TV[k], 99):.2e} | {TV[k].max():.2e} | {TV[k][m7].max():.2e} | {DV[k].mean():.2e}")
best = order[int(np.argmin([TV[k].mean() / 0.01 + DV[k].mean() / 0.005 for k in range(len(order))]))]
lines.append(f"least used on all rows (TV/0.01 + |ΔV|/0.005): {best}")
print("\n".join(lines)); (OUT / f"units_{tag}.txt").write_text("\n".join(lines) + "\n")
np.save(OUT / f"best_unit_{tag}.npy", np.array(int(best)))
