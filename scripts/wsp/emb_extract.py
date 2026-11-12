"""Extract parameters and Adam state (exp_avg m, exp_avg_sq v, step) of gen4 PPO checkpoints into compact .pt files.

out/emb/ckpt/<tag>.pt = {"params": {name: tensor}, "m": {name: tensor}, "v": {name: tensor}, "step": {name: int},
                         "hyper": param_groups[0] without params, "src": path}

    python scripts/wsp/emb_extract.py TAG=PATH [TAG=PATH ...]
"""
import io
import sys
import zipfile
from pathlib import Path

import torch

from porygon.model import MaskedPolicy  # noqa: F401  (needed for names via a loaded policy)
from wsp_common import load

OUT = Path("out/emb/ckpt")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        tag, path = arg.split("=", 1)
        dst = OUT / f"{tag}.pt"
        if dst.exists():
            print(tag, "exists")
            continue
        pol = load(path)
        names = [n for n, _ in pol.named_parameters()]
        params = list(pol.parameters())
        with zipfile.ZipFile(path) as z:
            opt = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=False)
        ids = opt["param_groups"][0]["params"]
        assert len(ids) == len(params), (len(ids), len(params))
        m, v, st = {}, {}, {}
        for i, (n, p) in zip(ids, zip(names, params)):
            s = opt["state"].get(i)
            if s is None:
                continue
            assert s["exp_avg"].shape == p.shape, (n, s["exp_avg"].shape, p.shape)
            m[n], v[n], st[n] = s["exp_avg"].float(), s["exp_avg_sq"].float(), float(s["step"])
        hyper = {k: v_ for k, v_ in opt["param_groups"][0].items() if k != "params"}
        torch.save({"params": {n: p.detach().float().clone() for n, p in zip(names, params)}, "m": m, "v": v,
                    "step": st, "hyper": hyper, "src": path}, dst)
        print(tag, path, "adam step", next(iter(st.values()), None), "hyper", hyper)


if __name__ == "__main__":
    main()
