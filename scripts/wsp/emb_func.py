"""Functional role of the categorical embeddings (analyses 3 and 4 of the emb brief).

3. First layer of mon_proj (573 -> 256) and move_proj (108 -> 64) on real positions: per input block
   [species | item | ability | 4 x move_proj out | pool mean | numerics] the norm of W_b x_b and its share of the
   pre-activation variance (share_b = sum_units Cov(z_b, z) / sum_units Var(z); shares sum to 1), per token type.
4. Causal use: per table, replace the seen rows by (mean) their mean, (perm) a permutation of the seen rows,
   (fresh) new N(0,1) rows; reference ablations of numerics that carry species/move identity. Metrics vs the
   unedited net on decisions with >= 2 legal actions: KL(p0||p), top-1 agreement with p0, top-1 agreement with
   the search's argmax share, cross-entropy to the search's shares, |dV|, value MSE vs outcome; also how much of
   the token / final-layer variance the edit removes. 95% intervals from resampling games.

    python scripts/wsp/emb_func.py [NET ...]     (default 500M 600M x300_640)
"""
import copy
import json
import sys
import time

import numpy as np
import torch

from emb_util import OUT, TABLES, data_dir, fmt, game_boot, load_data, token_counts
from porygon.encoder import FIELD_DIM, MON_DIM, MOVE_DIM, N_CAT, MOVEPOOL, MON_NUM_DIM
from wsp_common import GLORIA, Net, load, split_obs

torch.set_num_threads(2)
STRIDE = 6
NETS = {"500M": "500M", "600M": "600M", "x300_640": f"{GLORIA}/runs/arm-x300-20261105-005736-371541/final_model.zip",
        "ctrl640": f"{GLORIA}/runs/arm-ctrl-20261105-115414-377837/final_model.zip",
        "x30_640": f"{GLORIA}/runs/arm-x30-20261105-010656-371651/final_model.zip",
        "s2.5M": "/vol/bitbucket/fr1226/porygon-ws-wsp/ckpt/ppo_2499840_steps.zip",
        "s21M": "/vol/bitbucket/fr1226/porygon-ws-wsp/ckpt/ppo_21261312_steps.zip",
        "s136M": f"{GLORIA}/runs/20261020-181327-299545/checkpoints/ppo_135949332_steps.zip"}
NUM_OFF = N_CAT + MOVEPOOL
MOVENUM_OFF = NUM_OFF + MON_NUM_DIM
SPECIES_NUMS = list(range(2, 15)) + list(range(15, 33)) + [121]          # base stats, stats, level, types, weight
MOVE_STATIC = [0, 1, 2] + list(range(5, 26)) + list(range(29, 44))       # power, acc, prio, category, type, effects
BLOCKS = {"species": (0, 64), "item": (64, 96), "ability": (96, 128), "move_proj_out": (128, 384),
          "pool_mean": (384, 448), "numerics": (448, 573)}


def forward(net, x, mask, keep_tokens=False):
    cache = {} if keep_tokens else None
    lg, v = net.forward(x, mask, None, cache)
    return lg.softmax(-1), v, cache


def edit_table(pol, table, kind, seen, seed=0):
    p = copy.deepcopy(pol)
    W = getattr(p.features_extractor, table).weight
    with torch.no_grad():
        idx = torch.as_tensor(np.flatnonzero(seen))
        g = torch.Generator().manual_seed(seed)
        if kind == "mean":
            mu = W[idx].mean(0)
            W[2:] = mu
        elif kind == "perm":
            perm = idx[torch.randperm(len(idx), generator=g)]
            W[idx] = W[perm].clone()
        elif kind == "fresh":
            W[2:] = torch.randn(W[2:].shape, generator=g)
    return p


def edit_obs(x, kind, stats):
    x = x.clone()
    mons = x[:, FIELD_DIM:].view(len(x), 12, MON_DIM)
    present = mons[..., 0] > 0
    if kind == "species_numerics":
        cols = [NUM_OFF + c for c in SPECIES_NUMS]
        mons[..., cols] = torch.where(present[..., None], stats["species_numerics"], mons[..., cols])
    elif kind == "move_static_numerics":
        mv = mons[..., MOVENUM_OFF:].view(len(x), 12, 4, MOVE_DIM)
        pres = mons[..., 3:N_CAT] > 0
        mv[..., MOVE_STATIC] = torch.where(pres[..., None], stats["move_static"], mv[..., MOVE_STATIC])
    elif kind == "pool_zero":
        mons[..., N_CAT:NUM_OFF] = 0.0
    elif kind == "pool_opp_zero":
        mons[:, 6:, N_CAT:NUM_OFF] = 0.0
    return x


def block_stats(net, X, mask, d, nsub=4096):
    """Analysis 3 on the first nsub rows of X."""
    out = {}
    fe = net.fe
    x = torch.as_tensor(X[:nsub], dtype=torch.float32)
    cache = {}
    with torch.no_grad():
        net.forward(x, torch.as_tensor(mask[:nsub]), None, cache)
    mon_in = cache["mon_in"]                                          # B,12,573
    W = fe.mon_proj[0].weight
    s = split_obs(X[:nsub])
    present = torch.as_tensor(s["cats"][..., 0] > 0)
    for role, sl in (("own", slice(0, 6)), ("opp", slice(6, 12))):
        xi = mon_in[:, sl][present[:, sl]]                            # n,573
        zs = {b: xi[:, lo:hi] @ W[:, lo:hi].T for b, (lo, hi) in BLOCKS.items()}
        z = sum(zs.values())
        zc = z - z.mean(0)
        tot = float((zc ** 2).sum(0).sum())
        r = {}
        for b, zb in zs.items():
            zbc = zb - zb.mean(0)
            r[b] = {"norm_mean": float(zb.norm(dim=1).mean()), "var_own": float((zbc ** 2).sum() / tot),
                    "var_share": float((zbc * zc).sum() / tot),
                    "input_norm_mean": float(xi[:, BLOCKS[b][0]:BLOCKS[b][1]].norm(dim=1).mean())}
        r["n_tokens"] = int(len(xi))
        out[f"mon_proj.0:{role}"] = r
    # move_proj.0 on present moves (all 12 mons)
    cats = torch.as_tensor(s["cats"][..., 3:])
    mvn = torch.as_tensor(s["move_nums"], dtype=torch.float32)
    pres = cats > 0
    e = fe.moves(cats[pres])
    mn = mvn[pres]
    W2 = fe.move_proj[0].weight
    with torch.no_grad():
        ze, zn = e @ W2[:, :64].T, mn @ W2[:, 64:].T
    z = ze + zn
    zc = z - z.mean(0)
    tot = float((zc ** 2).sum())
    out["move_proj.0"] = {b: {"norm_mean": float(zb.norm(dim=1).mean()), "var_own": float(((zb - zb.mean(0)) ** 2).sum() / tot),
                              "var_share": float(((zb - zb.mean(0)) * zc).sum() / tot)} for b, zb in (("move_emb", ze), ("move_numerics", zn))}
    out["move_proj.0"]["n_tokens"] = int(pres.sum())
    return out


def main():
    nets = sys.argv[1:] or ["500M", "600M", "x300_640"]
    d = load_data()
    counts = token_counts(d)
    seen = {t: (counts[t]["all"] > 0) & (np.arange(len(counts[t]["all"])) >= 2) for t in TABLES}
    legal = d["mask"].sum(1) >= 2
    rows = np.flatnonzero(legal)[::STRIDE]                           # every STRIDE-th decision with a choice
    X, M = d["obs"][rows].astype(np.float32), d["mask"][rows]
    shares, game, won, logged_v = d["shares"][rows], d["game"][rows], d["won"][rows], d["value"][rows]
    print(f"data {data_dir()}: {len(rows)} decisions ({legal.sum()} with >=2 legal), {len(np.unique(game))} games")
    s = split_obs(X)
    pres = s["cats"][..., 0] > 0
    stats = {"species_numerics": torch.as_tensor(s["nums"][pres][:, SPECIES_NUMS].mean(0)),
             "move_static": torch.as_tensor(s["move_nums"][s["cats"][..., 3:] > 0][:, MOVE_STATIC].mean(0))}
    res = {}
    for name in nets:
        pol = load(NETS[name])
        net = Net(pol)
        t0 = time.time()
        res[name] = {"blocks": block_stats(net, X, M, d)}

        def run(netx, edit=None, bs=1024):
            P, V, dtok, vtok, dfin, vfin = [], [], 0.0, 0.0, 0.0, 0.0
            for i in range(0, len(X), bs):
                x = torch.as_tensor(X[i:i + bs])
                m = torch.as_tensor(M[i:i + bs])
                with torch.no_grad():
                    if edit:
                        x = edit_obs(x, edit, stats)
                    c = {}
                    lg, v = netx.forward(x, m, None, c)
                P.append(lg.softmax(-1).numpy())
                V.append(v.numpy())
                yield_c = (c["tokens"][:, :13], c["final"])
                if base_cache is not None:
                    bt, bf = base_cache[i // bs]
                    dtok += float(((yield_c[0] - bt) ** 2).sum())
                    vtok += float(((bt - bt.mean(0)) ** 2).sum())
                    dfin += float(((yield_c[1] - bf) ** 2).sum())
                    vfin += float(((bf - bf.mean(0)) ** 2).sum())
                else:
                    store.append((yield_c[0].clone(), yield_c[1].clone()))
            return np.concatenate(P), np.concatenate(V), (dtok / vtok if vtok else 0.0), (dfin / vfin if vfin else 0.0)

        # baseline (store tokens / final per batch: mon tokens only to save memory)
        base_cache, store = None, []
        P0, V0, _, _ = run(net)
        base_cache = [(t, f) for t, f in store]
        # tokens compare: restrict to mon tokens 0..12 (field + mons)
        print(f"{name}: baseline {time.time() - t0:.0f}s")
        phi = logged_v - V0 if name == "500M" else res.get("_phi", logged_v - V0)
        if name == "500M":
            res["_phi"] = phi
            res[name]["max_abs_probs_vs_logged"] = float(np.abs(P0 - d["probs"][rows]).max())
        y = 1.3 * (2 * won - 1)
        ok = ~np.isnan(won)
        tgt = shares.argmax(1)
        base_agree_search = (P0.argmax(1) == tgt).astype(float)
        ce0 = -(shares * np.log(np.clip(P0, 1e-12, 1))).sum(1)
        tv0 = 0.5 * np.abs(P0 - shares).sum(1)
        mse0 = (V0 + phi - y) ** 2
        res[name]["baseline"] = {"top1_vs_search": game_boot(base_agree_search, game),
                                 "ce_to_search": game_boot(ce0, game), "tv_to_search": game_boot(tv0, game),
                                 "value_mse_vs_outcome": game_boot(mse0[ok], game[ok]),
                                 "value_corr_outcome": float(np.corrcoef((V0 + phi)[ok], y[ok])[0, 1])}
        print(name, "baseline", json.dumps(res[name]["baseline"]))
        edits = []
        for t in TABLES:
            for kind in ("mean", "perm", "fresh"):
                edits.append((f"{t}:{kind}", t, kind, None))
        edits.append(("all4:mean", "all", "mean", None))
        edits.append(("all4:perm", "all", "perm", None))
        edits.append(("species:perm2", "species", "perm", None))
        edits.append(("ref:species_numerics", None, None, "species_numerics"))
        edits.append(("ref:move_static_numerics", None, None, "move_static_numerics"))
        edits.append(("ref:pool_zero", None, None, "pool_zero"))
        edits.append(("ref:pool_opp_zero", None, None, "pool_opp_zero"))
        part = OUT / f"func_{name}.partial.json"
        res[name]["ablations"] = json.load(open(part)) if part.exists() else {}
        for label, t, kind, oe in edits:
            if label in res[name]["ablations"]:
                continue
            if t == "all":
                p = pol
                for tt in TABLES:
                    p = edit_table(p, tt, kind, seen[tt], seed=1)
            elif t is not None:
                p = edit_table(pol, t, kind, seen[t], seed=2 if label.endswith("2") else 1)
            else:
                p = pol
            P, V, ftok, ffin = run(Net(p), oe)
            kl = (P0 * (np.log(np.clip(P0, 1e-12, 1)) - np.log(np.clip(P, 1e-12, 1)))).sum(1)
            agree = (P.argmax(1) == P0.argmax(1)).astype(float)
            agree_s = (P.argmax(1) == tgt).astype(float) - base_agree_search
            ce = -(shares * np.log(np.clip(P, 1e-12, 1))).sum(1) - ce0
            dv = np.abs(V - V0)
            dtv = 0.5 * np.abs(P - shares).sum(1) - tv0
            mse = (V + phi - y) ** 2 - mse0
            r = {"kl": game_boot(kl, game), "top1_agree_orig": game_boot(agree, game),
                 "d_top1_vs_search": game_boot(agree_s, game), "d_ce_to_search": game_boot(ce, game), "d_tv_to_search": game_boot(dtv, game),
                 "abs_dV": game_boot(dv, game), "d_value_mse": game_boot(mse[ok], game[ok]),
                 "token_var_removed": ftok, "final_var_removed": ffin}
            res[name]["ablations"][label] = r
            json.dump(res[name]["ablations"], open(part, "w"))
            print(f"{name} {label:28s} KL {fmt(*r['kl'])} agree {fmt(*r['top1_agree_orig'])} dTop1(search) {fmt(*r['d_top1_vs_search'])} "
                  f"dTV(search) {fmt(*r['d_tv_to_search'])} |dV| {fmt(*r['abs_dV'])} dMSE {fmt(*r['d_value_mse'], p=4)} tokVar {ftok:.3f} finVar {ffin:.3f}", flush=True)
        print(f"{name}: done {time.time() - t0:.0f}s", flush=True)
    res.pop("_phi", None)
    json.dump(res, open(OUT / f"func_{'_'.join(nets)}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
