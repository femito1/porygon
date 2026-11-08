"""Positional embedding: what it encodes, whether the policy is slot-biased, and whether slot identity is used.

    python scripts/wsp/att_pos.py
1. fe.pos per checkpoint (2.5M, 10M, 21M, 400M, 500M, 600M): row norms, cosine matrix, drift vs 2.5M, size relative
   to the content tokens (mean ‖token‖ and the across-decision std of each slot's token).
2. Slot bias in the logged data (500M probs vs search shares): for each own slot k, among decisions with ≥2 legal
   switches: mean prob on switch k | legal, and P(switch k | the policy switches, k legal), game bootstrap.
3. Causal (500M, 600M): (a) swap two alive own bench mons (blocks + mask): policy non-equivariance
   KL(perm(p) ‖ p_swapped), flip rate of the preference between the two, |ΔV|; (b) same with the own/opp pos rows
   replaced by their group mean (policy must become exactly equivariant; the remaining ΔV is the value MLP's slot
   dependence, it reads the 17 tokens flattened); (c) the pos-equalising edit itself on all decisions: KL, top-1
   agreement, Δ search share of the argmax, ΔV, Δ value MSE; (d) swap two revealed opp mons.
Output $WSP/out/att/pos.json
"""
import json

import numpy as np
import torch

from att_common import CKPT, OUT, WSP, XNet, boot_games, data, device, load, policy_metrics, roles
from porygon.encoder import FIELD_DIM, MON_DIM

torch.set_grad_enabled(False)
EXTRA = {"10M": f"{WSP}/ckpt/ppo_9999360_steps.zip", "21M": f"{WSP}/ckpt/ppo_21261312_steps.zip"}


def main():
    dev = device()
    d = data()
    out = {}
    # 1. pos per checkpoint
    P = {}
    for name, path in [("2.5M", CKPT["2.5M"]), ("10M", EXTRA["10M"]), ("21M", EXTRA["21M"]),
                       ("400M", CKPT["400M"]), ("500M", CKPT["500M"]), ("600M", CKPT["600M"])]:
        try:
            P[name] = load(path).features_extractor.pos.detach().numpy().copy()
        except Exception as e:  # noqa: BLE001
            print("skip", name, e)
    p0 = P["2.5M"]
    out["pos"] = {}
    for name, p in P.items():
        pn = p / np.linalg.norm(p, axis=1, keepdims=True)
        own = p[1:7]; opp = p[7:13]
        out["pos"][name] = {
            "row_norm": np.linalg.norm(p, axis=1).tolist(), "cos": (pn @ pn.T).tolist(),
            "drift_vs_2.5M": float(np.linalg.norm(p - p0) / np.linalg.norm(p0)),
            "own_spread": float(np.linalg.norm(own - own.mean(0), axis=1).mean()),
            "opp_spread": float(np.linalg.norm(opp - opp.mean(0), axis=1).mean()),
            "own_vs_opp_mean_dist": float(np.linalg.norm(own.mean(0) - opp.mean(0))),
        }
    # token scale (500M, 4096 decisions)
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(d["obs"]), 4096, replace=False))
    net = XNet(load(CKPT["500M"], dev))
    xb = torch.as_tensor(d["obs"][sel], dtype=torch.float32, device=dev)
    tok = (net.embed_x(xb, None) - net.fe.pos).cpu().numpy()
    out["token_norm"] = np.linalg.norm(tok, axis=-1).mean(0).tolist()
    out["token_std"] = np.linalg.norm(tok - tok.mean(0), axis=-1).mean(0).tolist()   # content variation per slot
    # 2. slot bias in logged data
    mask, probs, shares, game = d["mask"], d["probs"], d["shares"], d["game"]
    sw = mask[:, :6] > 0
    multi = sw.sum(1) >= 2
    out["slot_bias"] = {}
    for k in range(6):
        ok = multi & sw[:, k]
        pn = probs[ok, k]; ps = shares[ok, k]
        swm_n = probs[ok, :6].sum(1); swm_s = shares[ok, :6].sum(1)
        cn = pn / np.maximum(swm_n, 1e-9); cs = ps / np.maximum(swm_s, 1e-9)
        nleg = sw[ok].sum(1)
        out["slot_bias"][k] = {
            "n": int(ok.sum()), "net_prob": boot_games(pn, game[ok]), "search_share": boot_games(ps, game[ok]),
            "net_cond": boot_games(cn, game[ok]), "search_cond": boot_games(cs, game[ok]),
            "diff_cond": boot_games(cn - cs, game[ok]), "uniform_cond": float((1 / nleg).mean()),
            "net_argmax": boot_games((probs[ok].argmax(1) == k).astype(float), game[ok]),
            "search_argmax": boot_games((shares[ok].argmax(1) == k).astype(float), game[ok])}
    # 3. causal tests
    R = roles(d["obs"], mask)
    out["causal"] = {}
    X = d["obs"].astype(np.float32)
    for ck in ("500M", "600M"):
        net = XNet(load(CKPT[ck], dev))
        pos = net.fe.pos.detach().clone()
        peq = pos.clone(); peq[1:7] = pos[1:7].mean(0); peq[7:13] = pos[7:13].mean(0)
        res = {}
        # (a,b) own bench swap: decisions with ≥2 alive non-active own mons that are both legal switches
        cand = []
        for b in range(len(R)):
            c = [s for s in range(6) if R[b, 1 + s] == 2 and mask[b, s] > 0]
            if len(c) >= 2:
                i, j = rng.choice(c, 2, replace=False); cand.append((b, i, j))
        cand = cand[:8192]
        bi = np.array([c[0] for c in cand]); si = np.array([c[1] for c in cand]); sj = np.array([c[2] for c in cand])
        x = d["obs"][bi].astype(np.float32); m = mask[bi].copy()
        xs = x.copy(); ms = m.copy()
        for n in range(len(bi)):
            oi, oj = FIELD_DIM + si[n] * MON_DIM, FIELD_DIM + sj[n] * MON_DIM
            xs[n, oi:oi + MON_DIM], xs[n, oj:oj + MON_DIM] = x[n, oj:oj + MON_DIM], x[n, oi:oi + MON_DIM]
            ms[n, si[n]], ms[n, sj[n]] = m[n, sj[n]], m[n, si[n]]

        def fwd(xx, mm, pr=None):
            P_, V_ = [], []
            for s in range(0, len(xx), 2048):
                lg, v = net.fwd(torch.as_tensor(xx[s:s + 2048], device=dev), torch.as_tensor(mm[s:s + 2048], device=dev),
                                ix={"pos_replace": pr} if pr is not None else None)
                P_.append(lg.softmax(-1).cpu().numpy()); V_.append(v.cpu().numpy())
            return np.concatenate(P_), np.concatenate(V_)
        g = game[bi]
        for tag, pr in (("real_pos", None), ("equal_pos", peq)):
            p, v = fwd(x, m, pr); p2, v2 = fwd(xs, ms, pr)
            pp = p.copy(); ar = np.arange(len(bi))
            pp[ar, si], pp[ar, sj] = p[ar, sj], p[ar, si]                  # perm(p)
            kl = (pp * (np.log(pp + 1e-12) - np.log(p2 + 1e-12))).sum(1)
            flip = ((p[ar, si] > p[ar, sj]) != (p2[ar, sj] > p2[ar, si])).astype(float)
            # log-odds shift of the pair: slot advantage
            lo = np.log(p[ar, si] + 1e-12) - np.log(p[ar, sj] + 1e-12)
            lo2 = np.log(p2[ar, sj] + 1e-12) - np.log(p2[ar, si] + 1e-12)
            res[f"own_swap_{tag}"] = {"n": len(bi), "kl": boot_games(kl, g), "flip": boot_games(flip, g),
                                      "abs_dv": boot_games(np.abs(v2 - v), g),
                                      "abs_logodds_shift": boot_games(np.abs(lo2 - lo), g),
                                      "abs_logodds_pair": boot_games(np.abs(lo), g)}
        # (c) pos-equalising edit on all decisions
        y = np.where(d["won"] > .5, 1.0, -1.0)
        p, v = fwd(X, mask); pe, ve = fwd(X, mask, peq)
        if ck == "500M":
            phi = d["value"] - v
        pm0 = policy_metrics(p, p, mask, shares); pm = policy_metrics(p, pe, mask, shares)
        res["pos_equalise"] = {"kl": boot_games(pm["kl"], game), "agree": boot_games(pm["agree"], game),
                               "d_share": boot_games(pm["share"] - pm0["share"], game),
                               "d_agree_search": boot_games(pm["agree_search"] - pm0["agree_search"], game),
                               "abs_dv": boot_games(np.abs(ve - v), game),
                               "d_mse": boot_games((ve + phi - y) ** 2 - (v + phi - y) ** 2, game)}
        # (d) opp swap of two revealed alive non-active opp mons
        cand = []
        for b in range(len(R)):
            c = [s for s in range(6) if R[b, 7 + s] == 5]
            if len(c) >= 2:
                i, j = rng.choice(c, 2, replace=False); cand.append((b, i, j))
        cand = cand[:8192]
        bi = np.array([c[0] for c in cand]); si = np.array([c[1] for c in cand]); sj = np.array([c[2] for c in cand])
        x = d["obs"][bi].astype(np.float32); m = mask[bi]
        xs = x.copy()
        for n in range(len(bi)):
            oi, oj = FIELD_DIM + (6 + si[n]) * MON_DIM, FIELD_DIM + (6 + sj[n]) * MON_DIM
            xs[n, oi:oi + MON_DIM], xs[n, oj:oj + MON_DIM] = x[n, oj:oj + MON_DIM], x[n, oi:oi + MON_DIM]
        g = game[bi]
        for tag, pr in (("real_pos", None), ("equal_pos", peq)):
            p, v = fwd(x, m, pr); p2, v2 = fwd(xs, m, pr)
            kl = (p * (np.log(p + 1e-12) - np.log(p2 + 1e-12))).sum(1)
            res[f"opp_swap_{tag}"] = {"n": len(bi), "kl": boot_games(kl, g), "abs_dv": boot_games(np.abs(v2 - v), g),
                                      "agree": boot_games((p.argmax(1) == p2.argmax(1)).astype(float), g)}
        out["causal"][ck] = res
        print(ck, json.dumps(res), flush=True)
    (OUT / "pos.json").write_text(json.dumps(out))
    print(json.dumps(out["slot_bias"], indent=0))
    print({k: (v["drift_vs_2.5M"], v["own_spread"], v["opp_spread"]) for k, v in out["pos"].items()})


if __name__ == "__main__":
    main()
