"""QK / OV circuits: what each head looks for (keys), what its queries look for, and what it moves (OV),
read from the weights along measured input-feature directions, and checked against the measured effect.

    python scripts/wsp/att_circuits.py CKPT [--n 2048]

Feature directions by finite differences on real positions: for a target token (own active, one random alive own
bench mon, the opp active, one random revealed alive opp bench mon, the field token, move token 0) set one input
numeric to `lo` and to `hi` (or swap the species id, or replace the slot's positional embedding by its group
mean) and record, at every layer i, Δa = norm1_i(x_hi) − norm1_i(x_lo) at the target (the LayerNorm'd input the
head reads; includes the LN gain), plus the measured change of every head's pre-softmax scores and outputs.
Weight predictions (the "read the weights" test):
  key side   pred Δs(q-role → target) = q̄_role · W_K,h Δā / √32   (q̄ = mean clean query of that role)
  OV         pred ‖Δ head_out at q-role‖ ≈ p̄(q-role → target) · ‖W_O,h W_V,h Δā‖
Also per head: ‖W_O,h W_V,h diag(g)‖_F, the positional QK term pos-pos (layer 0), clean score std.
Output $WSP/out/att/circuits_<CKPT>.npz + .json
"""
import argparse
import json
import math

import numpy as np
import torch

from att_common import CKPT, NR, OUT, ROLES, XNet, data, device, load, roles
from porygon.encoder import FIELD_DIM, MON_DIM, MON_NUM_DIM
from wsp_common import MOVENUM_OFF, NUM_OFF

torch.set_grad_enabled(False)
A = 115  # ACTIVE_FLAG
TYPES = ['BUG', 'DARK', 'DRAGON', 'ELECTRIC', 'FAIRY', 'FIGHTING', 'FIRE', 'FLYING', 'GHOST', 'GRASS', 'GROUND',
         'ICE', 'NORMAL', 'POISON', 'PSYCHIC', 'ROCK', 'STEEL', 'WATER']
MON_FEATS = [("hp", 0, .2, 1.), ("fainted", 1, 0, 1), ("base_atk", 3, .3, .6), ("base_spe", 7, .3, .6),
             ("spe_stat", 13, .3, .6), ("level", 14, .75, .95), ("boost_atk", 52, 0, 1 / 3),
             ("boost_spe", 56, 0, 1 / 6), ("brn", 59, 0, 1), ("par", 61, 0, 1), ("slp", 63, 0, 1), ("tox", 64, 0, 1),
             ("substitute", 67, 0, 1), ("active", A, 0, 1), ("mine", A + 1, 0, 1), ("first_turn", A + 2, 0, 1),
             ("trapped", A + 7, 0, 1), ("is_opp_active", A + 8, 0, 1), ("unknown", A + 9, 0, 1)]
MON_FEATS += [(f"type_{t}", 15 + k, 0, 1) for k, t in enumerate(TYPES)]
MOVE_FEATS = [("mv_bp", 0, .3, .8), ("mv_prio", 2, 0, .2), ("mv_avail", 4, 0, 1), ("mv_status_cat", 7, 0, 1),
              ("mv_stab", 26, 0, 1), ("mv_eff", 27, -.5, .5), ("mv_immune", 28, 0, 1)]
FIELD_FEATS = [("our_rocks", 59, 0, 1), ("their_rocks", 77, 0, 1), ("turn", 81, .05, .3), ("unseen_opp", 82, 0, .5),
               ("our_alive", 83, .5, 1), ("their_alive", 84, .5, 1), ("force_switch", 85, 0, 1),
               ("sun", 0, 0, 1), ("n_switches", 93, .2, 1), ("n_moves", 94, .5, 1)]
TARGETS = ["own_act", "own_bench", "opp_act", "opp_bench", "field", "move0"]


def pick_targets(R, rng):
    """token index per decision for each target (-1 = none)."""
    N = len(R)
    t = {}
    for name, role, lo, hi in (("own_act", 1, 1, 7), ("own_bench", 2, 1, 7), ("opp_act", 4, 7, 13),
                               ("opp_bench", 5, 7, 13)):
        out = np.full(N, -1)
        for b in range(N):
            c = np.where(R[b, lo:hi] == role)[0]
            if len(c):
                out[b] = lo + rng.choice(c)
        t[name] = out
    t["field"] = np.zeros(N, int)
    t["move0"] = np.where(R[:, 13] == 8, 13, -1)
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    dev = device()
    d = data()
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(d["obs"]), a.n, replace=False))
    obs = d["obs"][sel].astype(np.float32); mask = d["mask"][sel]
    R = roles(obs, mask)
    T = pick_targets(R, rng)
    net = XNet(load(CKPT[a.ckpt], dev))
    H, dh, D = 8, 32, 256
    species_ids = obs[:, FIELD_DIM:].reshape(len(obs), 12, MON_DIM)[..., 0].ravel()
    species_ids = species_ids[species_ids > 1]

    def run(x, pos_delta=None):
        xb = torch.as_tensor(x, device=dev); mb = torch.as_tensor(mask, device=dev)
        h = net.embed_x(xb, None)
        if pos_delta is not None:
            h = h + torch.as_tensor(pos_delta, device=dev)
        cache = {}
        for i in range(3):
            L = net.layers[i]
            cache[f"a{i}"] = L.norm1(h)
            h = net.layer_x(i, h, None, cache)
        lg, v = net.head(h, mb)
        cache["p"] = lg.softmax(-1); cache["v"] = v
        return cache

    clean = run(obs)
    Rt = torch.as_tensor(R, device=dev)
    oh = torch.nn.functional.one_hot(Rt, NR).float()                               # B,T,NR
    # clean mean query per role and layer, and weight matrices
    Wq, Wk, Wv, Wo = [], [], [], []
    qbar = np.zeros((3, NR, H, dh))
    for i in range(3):
        L = net.layers[i]
        W, bb = L.self_attn.in_proj_weight, L.self_attn.in_proj_bias
        Wq.append(W[:D]); Wk.append(W[D:2 * D]); Wv.append(W[2 * D:]); Wo.append(L.self_attn.out_proj.weight)
        q = (clean[f"a{i}"] @ W[:D].T + bb[:D]).view(len(obs), 17, H, dh)
        qbar[i] = (torch.einsum("btr,bthd->rhd", oh, q) / oh.sum((0, 1)).clamp(min=1)[:, None, None]).cpu().numpy()
    qbar_t = torch.as_tensor(qbar, device=dev, dtype=torch.float32)
    # per-head weight stats (LN gain folded in)
    wstats = {"ov_fro": np.zeros((3, H)), "ov_top": np.zeros((3, H)), "qk_fro": np.zeros((3, H)),
              "score_std": np.zeros((3, H)), "ov_eff_rank": np.zeros((3, H))}
    for i in range(3):
        g = net.layers[i].norm1.weight
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            ov = Wo[i][:, sl] @ Wv[i][sl] * g[None]
            qk = (Wq[i][sl] * g[None]).T @ (Wk[i][sl] * g[None]) / math.sqrt(dh)
            s = torch.linalg.svdvals(ov)
            wstats["ov_fro"][i, h] = float(ov.norm()); wstats["ov_top"][i, h] = float(s[0])
            wstats["ov_eff_rank"][i, h] = float((s.sum() ** 2) / (s ** 2).sum())
            wstats["qk_fro"][i, h] = float(qk.norm())
        sc = clean[f"L{i}.scores"]                                                  # B,H,Tq,Tk
        wstats["score_std"][i] = sc.std(-1).mean((0, 2)).cpu().numpy()
    # positional QK term at layer 0 (pos rows LayerNorm'd as if alone, gain included): 8 x 17 x 17
    L0 = net.layers[0]
    pos = net.fe.pos
    results = {}
    feats = {}
    for tname in TARGETS:
        if tname == "field":
            flist = [(n, ("field", j), lo, hi) for n, j, lo, hi in FIELD_FEATS]
        elif tname == "move0":
            flist = [(n, ("move", j), lo, hi) for n, j, lo, hi in MOVE_FEATS]
        else:
            flist = [(n, ("mon", j), lo, hi) for n, j, lo, hi in MON_FEATS]
            flist += [("species_swap", ("species", 0), 0, 0), ("pos_slot", ("pos", 0), 0, 0)]
        feats[tname] = [f[0] for f in flist]
        tg = T[tname]
        ok = tg >= 0
        if tname == "move0":
            # the move token is built from the active own mon's move block 0
            act = (R[:, 1:7] == 1).argmax(1)
        for fname, (kind, j), lo, hi in flist:
            xlo, xhi = obs.copy(), obs.copy()
            pdlo = pdhi = None
            b_ok = ok.copy()
            for b in np.where(ok)[0]:
                t = tg[b]
                if kind == "mon":
                    off = FIELD_DIM + (t - 1) * MON_DIM + NUM_OFF + j
                    xlo[b, off], xhi[b, off] = lo, hi
                elif kind == "field":
                    xlo[b, j], xhi[b, j] = lo, hi
                elif kind == "move":
                    off = FIELD_DIM + act[b] * MON_DIM + MOVENUM_OFF + j
                    xlo[b, off], xhi[b, off] = lo, hi
                elif kind == "species":
                    off = FIELD_DIM + (t - 1) * MON_DIM
                    xhi[b, off] = rng.choice(species_ids)
            if kind == "pos":
                pdlo = np.zeros((len(obs), 17, D), np.float32)
                P = pos.detach().cpu().numpy()
                for b in np.where(ok)[0]:
                    t = tg[b]
                    grp = slice(1, 7) if t < 7 else slice(7, 13)
                    pdlo[b, t] = P[grp].mean(0) - P[t]                               # lo = group-mean slot
            clo = clean if kind in ("species",) else run(xlo, pdlo)
            chi = run(xhi, pdhi) if kind != "pos" else clean
            okt = torch.as_tensor(b_ok, device=dev)
            tt = torch.as_tensor(np.where(b_ok, tg, 0), device=dev)
            ar = torch.arange(len(obs), device=dev)
            nb = int(b_ok.sum())
            # query-role weights excluding the target token itself
            qmask = oh.clone(); qmask[ar, tt] = 0; qmask = qmask * okt[:, None, None]
            res = {}
            for i in range(3):
                da = (chi[f"a{i}"] - clo[f"a{i}"])[ar, tt] * okt[:, None]          # B,D
                dabar = da.sum(0) / max(nb, 1)
                dS = (chi[f"L{i}.scores"] - clo[f"L{i}.scores"])[ar, :, :, tt]     # B,H,Tq  (target as key)
                cnt = qmask.sum((0, 1)).clamp(min=1)                                # NR
                dS_key = torch.einsum("bhq,bqr->hr", dS, qmask) / cnt
                dS_key_abs = torch.einsum("bhq,bqr->hr", dS.abs(), qmask) / cnt
                # target as query: change of attention mass to key roles
                pq_hi = chi[f"L{i}.pattern"][ar, :, tt]; pq_lo = clo[f"L{i}.pattern"][ar, :, tt]   # B,H,Tk
                dmass_q = torch.einsum("bhk,bkr->hr", (pq_hi - pq_lo) * okt[:, None, None], oh) / max(nb, 1)
                # head output change at other tokens, by query role
                zhi, zlo = chi[f"L{i}.z"], clo[f"L{i}.z"]                           # B,T,H,dh
                Woh = Wo[i].view(D, H, dh)
                dho = torch.einsum("bthd,ehd->bthe", zhi - zlo, Woh).norm(dim=-1)  # B,T,H
                dhead = torch.einsum("bth,btr->hr", dho, qmask) / cnt
                hon = torch.einsum("bthd,ehd->bthe", zlo, Woh).norm(dim=-1)
                headnorm = torch.einsum("bth,btr->hr", hon, qmask) / cnt
                # predictions from weights
                dk = (Wk[i] @ dabar).view(H, dh)
                pred_dS = torch.einsum("rhd,hd->hr", qbar_t[i], dk) / math.sqrt(dh)
                dv = (Wv[i] @ dabar).view(H, dh)
                ovn = torch.einsum("ehd,hd->he", Woh, dv).norm(dim=-1)              # H
                # per-decision version of the OV norm (direction varies per decision for species / pos)
                dvb = (da @ Wv[i].T).view(len(obs), H, dh)
                ovn_b = torch.einsum("ehd,bhd->bhe", Woh, dvb).norm(dim=-1)        # B,H
                pclean = clo[f"L{i}.pattern"][ar, :, :, tt]                         # B,H,Tq  weight on target
                pred_dhead = torch.einsum("bhq,bh,bqr->hr", pclean, ovn_b, qmask) / cnt
                res[i] = {k: v.cpu().numpy() for k, v in dict(
                    dS_key=dS_key, dS_key_abs=dS_key_abs, dmass_q=dmass_q, dhead=dhead, headnorm=headnorm,
                    pred_dS=pred_dS, ov_norm=ovn, pred_dhead=pred_dhead,
                    da_norm=da.norm(dim=-1).sum() / max(nb, 1), dabar_norm=dabar.norm()).items()}
            # behaviour change
            plo, phi_ = clo["p"], chi["p"]
            kl = ((phi_ * ((phi_ + 1e-12).log() - (plo + 1e-12).log())).sum(-1) * okt).sum() / max(nb, 1)
            dv_ = ((chi["v"] - clo["v"]).abs() * okt).sum() / max(nb, 1)
            res["kl"] = float(kl); res["dv"] = float(dv_); res["n"] = nb
            results[(tname, fname)] = res
        print(tname, "done", flush=True)
    # pack
    keys = sorted(results)
    pack = {"keys": np.array([f"{t}|{f}" for t, f in keys]), "qbar": qbar, **{f"w_{k}": v for k, v in wstats.items()}}
    for fld in ("dS_key", "dS_key_abs", "dmass_q", "dhead", "headnorm", "pred_dS", "ov_norm", "pred_dhead", "da_norm",
                "dabar_norm"):
        pack[fld] = np.stack([np.stack([results[k][i][fld] for i in range(3)]) for k in keys])
    pack["kl"] = np.array([results[k]["kl"] for k in keys]); pack["dv"] = np.array([results[k]["dv"] for k in keys])
    pack["n"] = np.array([results[k]["n"] for k in keys])
    # weight-only positional QK at layer 0: mean token + pos for each slot, LN, q·k
    tokmean = (clean["L0.in"] - pos).mean(0)                                         # 17,D content mean per slot
    def qk_pos(P):
        a_ = L0.norm1(tokmean + P)
        W, bb = L0.self_attn.in_proj_weight, L0.self_attn.in_proj_bias
        q = (a_ @ W[:D].T + bb[:D]).view(17, H, dh); k = (a_ @ W[D:2 * D].T + bb[D:2 * D]).view(17, H, dh)
        return torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(dh)
    S_real = qk_pos(pos)
    Pm = pos.clone(); Pm[1:7] = pos[1:7].mean(0); Pm[7:13] = pos[7:13].mean(0)
    S_eq = qk_pos(Pm)
    pack["pos_qk_real"] = S_real.cpu().numpy(); pack["pos_qk_eq"] = S_eq.cpu().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"circuits_{a.ckpt}.npz", **pack)
    print("saved", len(keys), "features")


if __name__ == "__main__":
    main()
