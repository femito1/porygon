"""How much did the embedding-lr arms change the function, and is the change in the codes or downstream?

On the logged decisions (every 6th with >= 2 legal actions): policies of 600M, ctrl640, x30_640, x300_640, and two
hybrids: the 600M net with the x300 arm's embedding tables (code drift alone) and the x300 net with the 600M
tables (downstream change alone). Pairwise KL, top-1 agreement, agreement with the search's argmax, |dV|;
95% intervals from resampling games.

    python scripts/wsp/emb_arms.py
"""
import copy
import json

import numpy as np
import torch

from emb_func import NETS
from emb_util import OUT, TABLES, fmt, game_boot, load_data
from wsp_common import Net, load

torch.set_num_threads(2)


def run(pol, X, M):
    net = Net(pol)
    P, V = [], []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            lg, v = net.forward(torch.as_tensor(X[i:i + 1024]), torch.as_tensor(M[i:i + 1024]))
            P.append(lg.softmax(-1).numpy())
            V.append(v.numpy())
    return np.concatenate(P), np.concatenate(V)


def swap_tables(dst, src):
    p = copy.deepcopy(dst)
    with torch.no_grad():
        for t in TABLES:
            getattr(p.features_extractor, t).weight.copy_(getattr(src.features_extractor, t).weight)
    return p


def main():
    d = load_data()
    rows = np.flatnonzero(d["mask"].sum(1) >= 2)[::6]
    X, M = d["obs"][rows].astype(np.float32), d["mask"][rows]
    game, tgt = d["game"][rows], d["shares"][rows].argmax(1)
    pols = {k: load(NETS[k]) for k in ("600M", "ctrl640", "x30_640", "x300_640")}
    pols["600M+x300 tables"] = swap_tables(pols["600M"], pols["x300_640"])
    pols["x300 with 600M tables"] = swap_tables(pols["x300_640"], pols["600M"])
    pols["ctrl640+x300 tables"] = swap_tables(pols["ctrl640"], pols["x300_640"])
    # revert the 600M codes to the 2.5M codes (undo the whole training drift of the tables) ...
    early = load("/vol/bitbucket/fr1226/porygon-ws-wsp/ckpt/ppo_2499840_steps.zip")
    pols["600M with 2.5M tables"] = swap_tables(pols["600M"], early)
    # ... vs an isotropic random perturbation of the same per-row size
    p = copy.deepcopy(pols["600M"])
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for t in TABLES:
            W = getattr(p.features_extractor, t).weight
            D = W - getattr(early.features_extractor, t).weight
            n = torch.randn(W.shape, generator=g)
            n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-12) * D.norm(dim=1, keepdim=True)
            W.add_(n)
    pols["600M + random drift-sized noise"] = p
    p = copy.deepcopy(pols["600M"])
    with torch.no_grad():
        for t in TABLES:
            W = getattr(p.features_extractor, t).weight
            D = getattr(pols["x300_640"].features_extractor, t).weight - W
            n = torch.randn(W.shape, generator=g)
            n = n / n.norm(dim=1, keepdim=True).clamp(min=1e-12) * D.norm(dim=1, keepdim=True)
            W.add_(n)
    pols["600M + random x300-drift-sized noise"] = p
    out = {k: run(p, X, M) for k, p in pols.items()}
    res = {}
    for k, (P, V) in out.items():
        res[k] = {"top1_vs_search": game_boot((P.argmax(1) == tgt).astype(float), game)}
        for ref in ("600M", "ctrl640"):
            P0, V0 = out[ref]
            kl = (P0 * (np.log(np.clip(P0, 1e-12, 1)) - np.log(np.clip(P, 1e-12, 1)))).sum(1)
            res[k][f"kl_from_{ref}"] = game_boot(kl, game)
            res[k][f"agree_{ref}"] = game_boot((P.argmax(1) == P0.argmax(1)).astype(float), game)
            res[k][f"abs_dV_{ref}"] = game_boot(np.abs(V - V0), game)
        print(f"{k:24s} top1(search) {fmt(*res[k]['top1_vs_search'])}  KL(600M||.) {fmt(*res[k]['kl_from_600M'])} "
              f"agree600M {fmt(*res[k]['agree_600M'])}  KL(ctrl||.) {fmt(*res[k]['kl_from_ctrl640'])} |dV| vs ctrl {fmt(*res[k]['abs_dV_ctrl640'])}", flush=True)
    json.dump(res, open(OUT / "arms.json", "w"), indent=1)


if __name__ == "__main__":
    main()
