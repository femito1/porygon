"""Attention patterns on the logged real-game decisions, by semantic token role.

    python scripts/wsp/att_patterns.py CKPT   (CKPT in 2.5M|400M|500M|600M)
Writes $WSP/out/att/patterns_<CKPT>.npz:
  mass_sum [3,8,NR,NR], mass_cnt [NR]        Σ over queries of attention mass on each key role, by query role
  perkey_sum [3,8,NR,NR], perkey_cnt [NR,NR] same, divided by the number of keys of that role (only where ≥1)
  ent_sum [3,8,NR]                           Σ attention entropy (nats) by query role
  pos_mean [3,8,17,17]                       mean pattern by slot
  pair_cos [3,8,8]                           mean over decisions of cos(pattern_h, pattern_h') (flattened Tq×Tk)
  cond_* [N,3,8]                             per decision: own_act→opp_act, legal moves→opp_act, field→opp_act,
                                             own_act→own_bench (mass), opp_act→own_act
  feats_* [N]                                threat / matchup features of the position
"""
import sys
import time

import numpy as np
import torch

from att_common import NR, OUT, CKPT, XNet, data, device, load, mon_nums, roles
from porygon.encoder import FIELD_DIM, MON_DIM, MOVE_DIM
from wsp_common import MOVENUM_OFF

torch.set_grad_enabled(False)


def features(obs: np.ndarray, r: np.ndarray) -> dict:
    mons = obs[:, FIELD_DIM:].reshape(len(obs), 12, MON_DIM).astype(np.float32)
    nums = mon_nums(obs)
    mv = mons[..., MOVENUM_OFF:].reshape(len(obs), 12, 4, MOVE_DIM)
    ar = np.arange(len(obs))
    own_i = (r[:, 1:7] == 1).argmax(1)
    opp_i = (r[:, 7:13] == 4).argmax(1) + 6

    def threat(i):  # max over the mon's known moves of bp × effectiveness × STAB (vs the other active)
        m = mv[ar, i]                        # N,4,MOVE_DIM
        bp = m[..., 0] * 150
        mult = np.where(m[..., 28] > .5, 0, 2.0 ** (2 * m[..., 27]))
        stab = np.where(m[..., 26] > .5, 1.5, 1.0)
        return (bp * mult * stab).max(1), (bp > 0).sum(1)
    t_opp, n_opp_moves = threat(opp_i)
    t_own, _ = threat(own_i)
    return {"opp_hp": nums[ar, opp_i, 0], "own_hp": nums[ar, own_i, 0],
            "opp_threat": t_opp, "opp_nmoves": n_opp_moves.astype(np.float32), "own_threat": t_own,
            "opp_faster": (nums[ar, opp_i, 13] > nums[ar, own_i, 13]).astype(np.float32),
            "opp_bst": nums[ar, opp_i, 2:8].sum(1) * 255,
            "n_unrev": (r[:, 7:13] == 7).sum(1).astype(np.float32)}


def main():
    name = sys.argv[1]
    dev = device()
    d = data()
    obs, mask = d["obs"], d["mask"]
    R = roles(obs, mask)
    net = XNet(load(CKPT[name], dev))
    N = len(obs)
    mass_sum = np.zeros((3, 8, NR, NR)); mass_cnt = np.zeros(NR)
    perkey_sum = np.zeros((3, 8, NR, NR)); perkey_cnt = np.zeros((NR, NR))
    ent_sum = np.zeros((3, 8, NR)); pos_sum = np.zeros((3, 8, 17, 17)); pair = np.zeros((3, 8, 8))
    cond = {k: np.zeros((N, 3, 8), np.float32) for k in ("ownact_oppact", "moves_oppact", "field_oppact",
                                                          "ownact_ownbench", "oppact_ownact", "moves_field")}
    t0 = time.time()
    bs = 2048
    for s in range(0, N, bs):
        xb = torch.as_tensor(obs[s:s + bs], dtype=torch.float32, device=dev)
        mb = torch.as_tensor(mask[s:s + bs], device=dev)
        cache = {}
        net.fwd(xb, mb, cache=cache)
        rb = torch.as_tensor(R[s:s + bs], device=dev)
        oh = torch.nn.functional.one_hot(rb, NR).float()                      # B,T,NR
        kc = oh.sum(1)                                                        # B,NR keys per role
        mass_cnt += oh.sum((0, 1)).cpu().numpy()
        has = (kc > 0).float()                                                # B,NR
        perkey_cnt += torch.einsum("btq,bk->qk", oh, has).cpu().numpy()
        B = len(xb)
        for i in range(3):
            P = cache[f"L{i}.pattern"]                                         # B,H,Tq,Tk
            m = torch.einsum("bhqk,bkr->bhqr", P, oh)                          # mass to key role
            mass_sum[i] += torch.einsum("bhqr,bqs->hsr", m, oh).cpu().numpy()
            pk = m / kc.clamp(min=1)[:, None, None, :]
            perkey_sum[i] += torch.einsum("bhqr,bqs->hsr", pk, oh).cpu().numpy()
            ent = -(P * (P + 1e-12).log()).sum(-1)                             # B,H,Tq
            ent_sum[i] += torch.einsum("bhq,bqs->hs", ent, oh).cpu().numpy()
            pos_sum[i] += P.sum(0).cpu().numpy()
            f = P.flatten(2)
            f = f / f.norm(dim=-1, keepdim=True)
            pair[i] += torch.einsum("bhx,bgx->hg", f, f).cpu().numpy()
            own_q = oh[:, :, 1]                                                # B,T (exactly one)
            leg = oh[:, :, 8]
            # mass from a query role (mean over its tokens) to a key role
            def qk(qmask, krole):
                w = (qmask / qmask.sum(1, keepdim=True).clamp(min=1))           # B,T
                return torch.einsum("bq,bhq->bh", w, m[..., krole])
            cond["ownact_oppact"][s:s + B, i] = qk(own_q, 4).cpu().numpy()
            cond["moves_oppact"][s:s + B, i] = qk(leg, 4).cpu().numpy()
            cond["field_oppact"][s:s + B, i] = m[:, :, 0, 4].cpu().numpy()
            cond["ownact_ownbench"][s:s + B, i] = qk(own_q, 2).cpu().numpy()
            cond["oppact_ownact"][s:s + B, i] = qk(oh[:, :, 4], 1).cpu().numpy()
            cond["moves_field"][s:s + B, i] = qk(leg, 0).cpu().numpy()
    print(f"{name}: {N} decisions in {time.time() - t0:.0f}s", flush=True)
    feats = features(obs, R)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"patterns_{name}.npz", mass_sum=mass_sum, mass_cnt=mass_cnt, perkey_sum=perkey_sum,
             perkey_cnt=perkey_cnt, ent_sum=ent_sum, pos_mean=pos_sum / N, pair_cos=pair / N,
             game=d["game"], **{f"cond_{k}": v for k, v in cond.items()}, **{f"feats_{k}": v for k, v in feats.items()})


if __name__ == "__main__":
    main()
