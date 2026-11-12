"""Gradients on logged data (analysis 6 of the emb brief).

For K independent mini-batches of 1024 logged decisions (>= 2 legal actions) and two losses on the 500M net,
  policy:  cross-entropy of the policy to the search's visit shares
  value:   (V + Phi - y)^2 with y = 1.3 (2 won - 1) (≈ terminal potential), Phi = logged value - V_500M
per parameter tensor (and per column block of the first layers): gradient RMS, the signal-to-noise ratio of one
mini-batch gradient S = |mean|^2 / E|g - mean|^2 (bias-corrected), the mean cosine between independent
mini-batch gradients, the rho^2 = E[m^2]/E[v] Adam would show for that S ((S + 1/19)/(S + 1)), and for the
embedding tables per-row gradient norm and cross-batch cosine vs token frequency.

    python scripts/wsp/emb_grad.py [NET] [K]
"""
import json
import sys
import time

import numpy as np
import torch

from emb_util import OUT, TABLES, load_data, token_counts
from wsp_common import Net, load

torch.set_num_threads(2)
BLOCKS = {"features_extractor.mon_proj.0.weight": {"species": (0, 64), "item": (64, 96), "ability": (96, 128),
                                                   "move_proj_out": (128, 384), "pool_mean": (384, 448),
                                                   "numerics": (448, 573)},
          "features_extractor.move_proj.0.weight": {"move_emb": (0, 64), "move_numerics": (64, 108)}}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "500M"
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    d = load_data()
    counts = token_counts(d)
    pol = load(name)
    net = Net(pol)
    params = dict(pol.named_parameters())
    rows_all = np.flatnonzero(d["mask"].sum(1) >= 2)
    # Phi from the logged value of the 500M net
    X_all = d["obs"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(rows_all)[:K * 1024].reshape(K, 1024)
    grads = {"policy": [], "value": []}
    t0 = time.time()
    for b in range(K):
        r = np.sort(perm[b])
        x = torch.as_tensor(X_all[r].astype(np.float32))
        m = torch.as_tensor(d["mask"][r])
        sh = torch.as_tensor(d["shares"][r])
        won = torch.as_tensor(d["won"][r])
        lv = torch.as_tensor(d["value"][r])
        for loss_name in ("policy", "value"):
            pol.zero_grad()
            lg, v = net.forward(x, m)
            if loss_name == "policy":
                logp = torch.log_softmax(lg, -1)
                loss = -(sh * torch.where(sh > 0, logp, torch.zeros_like(logp))).sum(1).mean()
            else:
                phi = (lv - v).detach()
                ok = ~torch.isnan(won)
                y = 1.3 * (2 * won - 1)
                loss = ((v + phi - y)[ok] ** 2).mean()
            loss.backward()
            grads[loss_name].append({n: p.grad.detach().clone() for n, p in params.items() if p.grad is not None})
        print(f"batch {b} {time.time() - t0:.0f}s", flush=True)
    out = {}
    for loss_name, gl in grads.items():
        res = {}
        gnorm = [float(torch.sqrt(sum((g ** 2).sum() for g in gb.values()))) for gb in gl]
        res["_global_norm_mean"] = float(np.mean(gnorm))

        def stats(G):  # G: K x n
            mu = G.mean(0)
            dev = G - mu
            noise = float((dev ** 2).sum() / (K - 1))         # E|g - mu|^2 per batch
            sig = float((mu ** 2).sum()) - noise / K            # unbiased |E g|^2
            S = sig / noise if noise > 0 else float("nan")
            Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-30)
            cos = Gn @ Gn.T
            mc = float((cos.sum() - cos.diag().sum()) / (K * (K - 1)))
            return {"grad_rms": float(G.pow(2).mean().sqrt()), "S": S, "cos": mc,
                    "rho2_pred": (S + 1 / 19) / (S + 1) if S == S else None}
        for n in gl[0]:
            G = torch.stack([gb[n].flatten() for gb in gl])
            if n.endswith(tuple(f"{t}.weight" for t in TABLES)) and "features_extractor." + n.split(".")[1] + ".weight" == n:
                t = n.split(".")[1]
                f = counts[t]["all"]
                seen = np.flatnonzero((f > 0) & (np.arange(len(f)) >= 2))
                Gt = torch.stack([gb[n][seen] for gb in gl])  # K, rows, D
                G = Gt.reshape(K, -1)
                r = stats(G)
                rown = Gt.norm(dim=2).mean(0).numpy()
                Gn = Gt / Gt.norm(dim=2, keepdim=True).clamp(min=1e-30)
                cs = torch.einsum("ard,brd->rab", Gn, Gn)
                rowcos = ((cs.sum((1, 2)) - cs.diagonal(dim1=1, dim2=2).sum(1)) / (K * (K - 1))).numpy()
                mu = Gt.mean(0)
                noise = ((Gt - mu) ** 2).sum((0, 2)) / (K - 1)
                rowS = (((mu ** 2).sum(1) - noise / K) / noise.clamp(min=1e-30)).numpy()
                fs = f[seen]
                qs = np.quantile(fs, [0, .25, .5, .75, 1])
                bins = []
                for lo, hi in zip(qs[:-1], qs[1:]):
                    s_ = (fs >= lo) & (fs <= hi)
                    bins.append({"freq": [float(lo), float(hi)], "n": int(s_.sum()), "row_grad_norm_median": float(np.median(rown[s_])),
                                 "row_cos_median": float(np.median(rowcos[s_])), "row_S_median": float(np.median(rowS[s_]))})
                r["rows"] = bins
                r["frac_rows_S_gt_0.1"] = float((rowS > 0.1).mean())
                res[n] = r
            else:
                res[n] = stats(G)
            if n in BLOCKS:
                for b_, (lo, hi) in BLOCKS[n].items():
                    Gb = torch.stack([gb[n][:, lo:hi].flatten() for gb in gl])
                    res[f"{n}[{b_}]"] = stats(Gb)
        out[loss_name] = res
        print(f"\n== {loss_name} loss (global grad norm per batch {res['_global_norm_mean']:.3g})")
        print(f"{'tensor':60s} {'grad rms':>9s} {'S':>8s} {'cos':>7s} {'rho2 pred':>9s}")
        for n, r in res.items():
            if n.startswith("_"):
                continue
            print(f"{n:60s} {r['grad_rms']:9.2e} {r['S']:8.4f} {r['cos']:7.4f} {r['rho2_pred'] or 0:9.4f}")
            if "rows" in r:
                for q in r["rows"]:
                    print("     ", q)
                print("      frac rows S>0.1:", r["frac_rows_S_gt_0.1"])
    json.dump(out, open(OUT / f"grad_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
