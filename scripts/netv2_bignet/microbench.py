"""Bigger-net design microbench (netv2, 2026-11-17): per Net config, the trainable parameter count, forward FLOPs
per decision (torch FlopCounterMode: matmul / bmm / sdpa only), peak GPU memory, and the time of one BC training step
exactly as `porygon.v2.pretrain` runs it in streaming mode (numpy part slice -> batch_to_tensors -> losses under the
same autocast rule -> backward -> clip -> AdamW step, with its .item() syncs), plus no-grad forward times (eval pass
at batch 2048 with aux, RL-actor style batch 64 / 512 without aux). The data loader thread and validation are NOT
included (they run beside the step in the real run).

    PYTHONPATH=SRC python -u microbench.py --shard SHARD.npz --out OUT.jsonl [--modes tf32,fp32,fp16] [--cfgs NAME,...]

Precision modes: tf32 = fp32 weights, torch.set_float32_matmul_precision("high") (BC-P4's --no-bf16 setting);
fp32 = precision "highest" (no TF32: what a T4, which has no TF32 units, runs for the same code); fp16 = autocast
float16 + GradScaler (the T4's tensor-core path; NOT what the BC code does today).
"""

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from porygon.v2.model import Net, batch_to_tensors
from porygon.v2.pretrain import load_shard, losses
from porygon.v2.train import Meter

CFGS = {
    "base": {"n_layers": 4, "ffn_mult": 1},                       # BC-P4 / G9q (5,809,977 trainable)
    "L4F2": {"n_layers": 4, "ffn_mult": 2},
    "L6F1": {"n_layers": 6, "ffn_mult": 1},
    "L8F1": {"n_layers": 8, "ffn_mult": 1},
    "L6F2": {"n_layers": 6, "ffn_mult": 2},                       # the code default
    "D384L4": {"d_model": 384, "n_layers": 4, "ffn_mult": 1},
    "D384L6": {"d_model": 384, "n_layers": 6, "ffn_mult": 1},
    "D384L8": {"d_model": 384, "n_layers": 8, "ffn_mult": 1},
    "D512L4": {"d_model": 512, "n_layers": 4, "ffn_mult": 1},
    "D512L6": {"d_model": 512, "n_layers": 6, "ffn_mult": 1},
}


def mk_batch(part: dict, sl: slice, device, rng) -> dict:
    """As StreamSource.batches builds a training minibatch (vsel = one value row per ~55 decisions)."""
    mb = batch_to_tensors({k: v[sl] for k, v in part.items() if k not in ("game", "src", "vsel")}, device)
    mb["vsel"] = torch.from_numpy(part["vsel"][sl]).to(device)
    m = len(part["z"][sl])
    mb["opp_cls"] = torch.full((m,), -1, dtype=torch.long, device=device)
    mb["drop_seed"] = torch.from_numpy(rng.integers(1, 2**31 - 1, m)).to(device)
    return mb


def flops_per_decision(net: Net, mb: dict, aux: bool) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        net(mb, aux=aux)
    return fc.get_total_flops() / len(mb["z"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="tf32,fp32,fp16")
    ap.add_argument("--cfgs", default=",".join(CFGS))
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    part = load_shard(a.shard, 9)
    n = len(part["z"])
    rng = np.random.default_rng(0)
    part["vsel"] = (rng.random(n) < 1 / 55).astype(np.float32)
    targs = SimpleNamespace(value_coef=0.5, aux_coef=0.1)
    out = open(a.out, "a", buffering=1)
    for name in a.cfgs.split(","):
        cfg = CFGS[name]
        torch.manual_seed(2)
        net = Net(9, **cfg).to(dev).train()
        n_par = sum(p.numel() for p in net.parameters())
        probe = mk_batch(part, slice(0, 256), dev, rng)
        rec = {"gpu": gpu, "torch": torch.__version__, "cfg_name": name, "cfg": cfg, "params": n_par,
               "fwd_flops_per_dec_aux": flops_per_decision(net, probe, True),
               "fwd_flops_per_dec_noaux": flops_per_decision(net, probe, False)}
        for mode in a.modes.split(","):
            torch.manual_seed(2)
            net = Net(9, **cfg).to(dev).train()
            torch.set_float32_matmul_precision("highest" if mode == "fp32" else "high")
            amp = mode == "fp16"
            groups = net.param_groups(0.01)
            opt = torch.optim.AdamW(groups, lr=3e-4, betas=(0.9, 0.999))
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            torch.cuda.reset_peak_memory_stats()
            m = Meter()
            times = []
            pos = 0
            for i in range(a.warm + a.steps):
                if pos + a.bs > n:
                    pos = 0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                mb = mk_batch(part, slice(pos, pos + a.bs), dev, rng)
                pos += a.bs
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    loss = losses(net, mb, targs, m)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                m.add("grad_norm", gn.item())
                torch.cuda.synchronize()
                if i >= a.warm:
                    times.append(time.perf_counter() - t0)
            st = float(np.median(times))
            rec[f"{mode}/train_step_s"] = round(st, 5)
            rec[f"{mode}/train_samples_per_s"] = round(a.bs / st, 1)
            rec[f"{mode}/peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            rec[f"{mode}/loss_finite"] = bool(torch.isfinite(loss).item())
            # no-grad forwards: validation pass (batch 2048, aux on) and actor-style inference (aux off)
            net.eval()
            for bs, aux in ((2048, True), (512, False), (64, False)):
                mb = mk_batch(part, slice(0, bs), dev, rng)
                tt = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    for j in range(a.warm + 15):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        net(mb, aux=aux)
                        torch.cuda.synchronize()
                        if j >= a.warm:
                            tt.append(time.perf_counter() - t0)
                rec[f"{mode}/fwd_bs{bs}{'_aux' if aux else ''}_decisions_per_s"] = round(bs / float(np.median(tt)), 1)
            del net, opt
            torch.cuda.empty_cache()
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
