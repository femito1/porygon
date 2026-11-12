"""Ablations of attention heads, whole attention layers and MLPs on the logged real-game decisions.

    python scripts/wsp/att_ablate.py CKPT [--heads-only] [--pairs] [--n N]
Pass 1: per-position means of each head's z, each layer's attention output and MLP output (mean ablation).
Pass 2: per batch, the clean forward and every condition, starting from the cached residual of the ablated layer.
Conditions: z{i}.{h} / m{i}.{h} (zero / mean ablation of a head), za{i} / ma{i} (whole attention output),
zm{i} / mm{i} (MLP output), with --pairs mp{i}.{h}.{g} (joint mean ablation of two heads of one layer).
Writes $WSP/out/att/ablate_<CKPT>[_pairs].npz with per-decision metrics per condition:
  kl (KL clean‖ablated over legal actions), agree (argmax equal to clean), agree_search (argmax = search argmax),
  share (search visit share of the ablated argmax), dv (V_abl − V_clean), dsq ((V_abl+Φ−y)² − (V_clean+Φ−y)²,
  y = ±1 outcome, Φ = logged value − 500M V).
"""
import argparse
import time

import numpy as np
import torch

from att_common import CKPT, OUT, XNet, data, device, load, policy_metrics

torch.set_grad_enabled(False)


def conditions(heads_only: bool, pairs: bool):
    c = []
    for i in range(3):
        for h in range(8):
            c.append((f"z{i}.{h}", i, {"head_mask": {i: "zero1:%d" % h}}))
            c.append((f"m{i}.{h}", i, {"head_pmean": {i: [h]}}))
    if not heads_only:
        for i in range(3):
            c += [(f"za{i}", i, {"attn_zero": {i: True}}), (f"ma{i}", i, {"attn_pmean": {i: None}}),
                  (f"zm{i}", i, {"mlp_zero": {i: True}}), (f"mm{i}", i, {"mlp_pmean": {i: None}})]
    if pairs:
        c = []
        for i in range(3):
            for h in range(8):
                for g in range(h + 1, 8):
                    c.append((f"mp{i}.{h}.{g}", i, {"head_pmean": {i: [h, g]}}))
    return c


def materialise(spec, means, dev):
    ix = {}
    for k, v in spec.items():
        if k == "head_mask":
            (i, s), = v.items()
            m = torch.ones(8, device=dev); m[int(s.split(":")[1])] = 0
            ix[k] = {i: m}
        elif k == "head_pmean":
            (i, hs), = v.items()
            ix[k] = {i: {h: means["z"][i][:, h] for h in hs}}
        elif k == "attn_pmean":
            (i, _), = v.items(); ix[k] = {i: means["attn"][i]}
        elif k == "mlp_pmean":
            (i, _), = v.items(); ix[k] = {i: means["mlp"][i]}
        else:
            ix[k] = v
    return ix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--heads-only", action="store_true")
    ap.add_argument("--pairs", action="store_true"); ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    dev = device()
    d = data()
    N = a.n or len(d["obs"])
    obs, mask, shares = d["obs"][:N], d["mask"][:N], d["shares"][:N]
    y = np.where(d["won"][:N] > .5, 1.0, -1.0)
    net = XNet(load(CKPT[a.ckpt], dev))
    net500 = net if a.ckpt == "500M" else XNet(load(CKPT["500M"], dev))
    bs = 2048
    t0 = time.time()
    # pass 1: means (and Φ from the 500M net)
    zs = [torch.zeros(17, 8, 32, device=dev) for _ in range(3)]
    at = [torch.zeros(17, 256, device=dev) for _ in range(3)]
    ml = [torch.zeros(17, 256, device=dev) for _ in range(3)]
    phi = np.zeros(N, np.float32)
    for s in range(0, N, bs):
        xb = torch.as_tensor(obs[s:s + bs], dtype=torch.float32, device=dev)
        mb = torch.as_tensor(mask[s:s + bs], device=dev)
        cache = {}
        _, v = net.fwd(xb, mb, cache=cache)
        if net500 is not net:
            _, v = net500.fwd(xb, mb)
        phi[s:s + bs] = d["value"][s:s + bs] - v.cpu().numpy()
        for i in range(3):
            zs[i] += cache[f"L{i}.z"].sum(0); at[i] += cache[f"L{i}.attn"].sum(0); ml[i] += cache[f"L{i}.mlp"].sum(0)
    means = {"z": [z / N for z in zs], "attn": [x / N for x in at], "mlp": [x / N for x in ml]}
    print(f"pass 1 {time.time() - t0:.0f}s", flush=True)
    conds = conditions(a.heads_only, a.pairs)
    ivs = [(name, i, materialise(spec, means, dev)) for name, i, spec in conds]
    keys = ("kl", "agree", "agree_search", "share", "dv", "dsq")
    res = {name: {k: np.zeros(N, np.float32) for k in keys} for name, _, _ in conds}
    base = {k: np.zeros(N, np.float32) for k in ("v", "sq", "agree_search", "share")}
    base_p = np.zeros((N, 10), np.float32)
    for s in range(0, N, bs):
        xb = torch.as_tensor(obs[s:s + bs], dtype=torch.float32, device=dev)
        mb = torch.as_tensor(mask[s:s + bs], device=dev)
        cache = {}
        lg0, v0 = net.fwd(xb, mb, cache=cache)
        p0 = lg0.softmax(-1).cpu().numpy(); v0 = v0.cpu().numpy()
        sl = slice(s, s + len(xb))
        pm0 = policy_metrics(p0, p0, mask[sl], shares[sl])
        sq0 = (v0 + phi[sl] - y[sl]) ** 2
        base["v"][sl], base["sq"][sl] = v0, sq0
        base["agree_search"][sl], base["share"][sl] = pm0["agree_search"], pm0["share"]
        base_p[sl] = p0
        for name, i, ix in ivs:
            h = None if i == 0 else cache[f"L{i}.in"]
            lg, v = net.fwd(xb, mb, ix=ix, start=i, h=h)
            p1 = lg.softmax(-1).cpu().numpy(); v1 = v.cpu().numpy()
            pm = policy_metrics(p0, p1, mask[sl], shares[sl])
            r = res[name]
            for k in ("kl", "agree", "agree_search", "share"):
                r[k][sl] = pm[k]
            r["dv"][sl] = v1 - v0
            r["dsq"][sl] = (v1 + phi[sl] - y[sl]) ** 2 - sq0
        print(f"{s + len(xb)}/{N} {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"ablate_{a.ckpt}" + ("_pairs" if a.pairs else "") + ("_heads" if a.heads_only else "")
    flat = {f"{name}|{k}": v for name, r in res.items() for k, v in r.items()}
    np.savez(OUT / f"{tag}.npz", game=d["game"][:N], phi=phi, y=y, base_p=base_p,
             **{f"base|{k}": v for k, v in base.items()}, **flat)
    print("done", tag, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
