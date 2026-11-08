"""dyn: the checkpoint list of the gen4 PPO lineage, and raw readers for a .zip (no SB3 construction).

`lineage()` -> [(step, path)] sorted by num_timesteps: near-init copies in $WSP/ckpt, every lineage
checkpoint and final_model of the runs that form the 0 -> 600M chain. `arms()` -> {arm: [(step, path)]}.
`read(path)` -> (state_dict, optimizer_state, num_timesteps).
"""
import glob
import io
import json
import os
import re
import zipfile

import torch

G = "/vol/bitbucket/fr1226/gloria/runs"
WSP = "/vol/bitbucket/fr1226/porygon-ws-wsp"
CHAIN = ["20261020-181327-299545", "20261021-145709-300011", "20261022-120923-300097", "20261023-112550-300355",
         "20261023-125508-300389", "20261024-135245-300664", "20261025-171535-301033", "20261025-195826-301040", "20261026-082356-301120"]
ARMS = {"ctrl": ["arm-ctrl-20261105-010231-371650", "arm-ctrl-20261105-115414-377837"],
        "x30": ["arm-x30-20261105-010656-371651"], "x300": ["arm-x300-20261105-005736-371541"]}


def _step(p):
    m = re.search(r"ppo_(\d+)_steps", p)
    if m:
        return int(m.group(1))
    with zipfile.ZipFile(p) as z:
        return int(json.loads(z.read("data"))["num_timesteps"])


def _collect(runs, extra=()):
    out = {}
    for p in list(extra):
        out[_step(p)] = p
    for r in runs:
        for p in glob.glob(f"{G}/{r}/checkpoints/ppo_*_steps.zip"):
            out[_step(p)] = p
        f = f"{G}/{r}/final_model.zip"
        if os.path.exists(f):
            out[_step(f)] = f
    items, keep = sorted(out.items()), []
    for s, p in items:  # drop a checkpoint < 1M steps before the next one (ppo_400028848 vs final 400029852)
        if keep and s - keep[-1][0] < 1_000_000:
            keep.pop()
        keep.append((s, p))
    return keep


def lineage():
    return _collect(CHAIN, sorted(glob.glob(f"{WSP}/ckpt/ppo_*_steps.zip")))


def arms():
    return {k: _collect(v) for k, v in ARMS.items()}


def read(path, optim=True):
    with zipfile.ZipFile(path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
        opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu",
                         weights_only=False) if optim else None
        n = int(json.loads(z.read("data"))["num_timesteps"])
    return sd, opt, n


if __name__ == "__main__":
    for s, p in lineage():
        print(s, p)
    for k, v in arms().items():
        print(k, [s for s, _ in v])
