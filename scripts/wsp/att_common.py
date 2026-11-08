"""Helpers for the attention study (prefix att): semantic token roles, an extended instrumented forward
(per-position mean ablation of heads / attention / MLP, start from a cached residual), metrics and a
game-level bootstrap.

Roles (per decision, per token):
  0 field | 1 own_act | 2 own_bench (alive, not active) | 3 own_faint | 4 opp_act | 5 opp_bench (revealed,
  alive, not active) | 6 opp_faint | 7 opp_unrev (never-seen slot: unknown flag) | 8 move_legal | 9 move_illegal
"""
from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from porygon.encoder import ACTIVE_FLAG, FIELD_DIM, MON_DIM, MON_NUM_DIM
from wsp_common import NUM_OFF, Intervention, Net, load, load_data  # noqa: F401

WSP = "/vol/bitbucket/fr1226/porygon-ws-wsp"
GLORIA = "/vol/bitbucket/fr1226/gloria"
OUT = Path(WSP) / "out" / "att"
LOCAL = Path("/tmp/wsp-674320")
CKPT = {
    "2.5M": f"{WSP}/ckpt/ppo_2499840_steps.zip",
    "400M": f"{GLORIA}/runs/20261024-135245-300664/final_model.zip",
    "500M": f"{GLORIA}/runs/20261025-195826-301040/final_model.zip",
    "600M": f"{GLORIA}/runs/20261026-082356-301120/final_model.zip",
}
ROLES = ["field", "own_act", "own_bench", "own_faint", "opp_act", "opp_bench", "opp_faint", "opp_unrev",
         "move_legal", "move_illegal"]
NR = len(ROLES)
# mon numerics offsets (encoder.encode_mon nums order)
N_HP, N_FAINT = 0, 1
N_ACTIVE, N_MINE, N_OPPACT, N_UNK = ACTIVE_FLAG, ACTIVE_FLAG + 1, MON_NUM_DIM - 2, MON_NUM_DIM - 1


def data(local: bool = True) -> dict:
    """decisions.npz, copied once to the node-local /tmp (the shared disk is slow)."""
    src = Path(WSP) / "data" / "decisions.npz"
    if local:
        LOCAL.mkdir(parents=True, exist_ok=True)
        dst = LOCAL / "decisions.npz"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.stat().st_mtime < src.stat().st_mtime:
            tmp = LOCAL / f".decisions.{os.getpid()}"
            shutil.copy(src, tmp)
            os.replace(tmp, dst)
        return load_data(LOCAL)
    return load_data(src.parent)


def mon_nums(obs: np.ndarray) -> np.ndarray:
    """[N,12,MON_NUM_DIM] float32 mon numerics."""
    mons = obs[:, FIELD_DIM:].reshape(len(obs), 12, MON_DIM)
    return mons[..., NUM_OFF:NUM_OFF + MON_NUM_DIM].astype(np.float32)


def roles(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[N,17] int role id per token."""
    n = mon_nums(obs)
    r = np.zeros((len(obs), 17), np.int64)
    own, opp = n[:, :6], n[:, 6:]
    ro = np.where(own[..., N_ACTIVE] > .5, 1, np.where(own[..., N_FAINT] > .5, 3, 2))
    rp = np.where(opp[..., N_UNK] > .5, 7, np.where(opp[..., N_OPPACT] > .5, 4,
                                                     np.where(opp[..., N_FAINT] > .5, 6, 5)))
    r[:, 1:7], r[:, 7:13] = ro, rp
    r[:, 13:17] = np.where(mask[:, 6:10] > 0, 8, 9)
    return r


class XNet(Net):
    """Net with extra interventions (dict `ix`):
    head_pmean[i] : {h: [T,dh]}  per-position mean of head z (mean ablation)
    attn_pmean[i] : [T,D]        replace the whole attention output by its per-position mean
    attn_zero[i]  : True         drop the whole attention output (incl. out_proj bias)
    mlp_pmean[i]  : [T,D]        replace the MLP output by its per-position mean
    mlp_zero[i]   : True         drop the MLP output (incl. bias)
    head_mask[i]  : [H]          multiply head z
    pos_replace   : [T,D]        replace fe.pos
    """

    def embed_x(self, x, ix):
        h = self.embed(x)
        if ix and "pos_replace" in ix:
            h = h - self.fe.pos + ix["pos_replace"].to(h)
        return h

    def layer_x(self, i, x, ix, cache=None):
        L = self.layers[i]
        B, T, D = x.shape
        a = L.norm1(x)
        qkv = F.linear(a, L.self_attn.in_proj_weight, L.self_attn.in_proj_bias).view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.unbind(2)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.dh)
        pat = scores.softmax(-1)
        z = torch.einsum("bhqk,bkhd->bqhd", pat, v)
        ix = ix or {}
        if i in ix.get("head_pmean", {}):
            z = z.clone()
            for hh, m in ix["head_pmean"][i].items():
                z[:, :, hh] = m.to(z)
        if i in ix.get("head_mask", {}):
            z = z * ix["head_mask"][i].view(1, 1, -1, 1).to(z)
        attn = L.self_attn.out_proj(z.reshape(B, T, D))
        if i in ix.get("attn_pmean", {}):
            attn = ix["attn_pmean"][i].to(x).expand(B, T, D)
        if ix.get("attn_zero", {}).get(i):
            attn = torch.zeros_like(attn)
        xm = x + attn
        hid = F.gelu(L.linear1(L.norm2(xm)))
        mlp = L.linear2(hid)
        if i in ix.get("mlp_pmean", {}):
            mlp = ix["mlp_pmean"][i].to(x).expand(B, T, D)
        if ix.get("mlp_zero", {}).get(i):
            mlp = torch.zeros_like(mlp)
        if cache is not None:
            cache[f"L{i}.scores"] = scores
            cache[f"L{i}.pattern"] = pat
            cache[f"L{i}.z"] = z
            cache[f"L{i}.attn"] = attn
            cache[f"L{i}.mlp"] = mlp
            cache[f"L{i}.out"] = xm + mlp
            cache[f"L{i}.in"] = x
        return xm + mlp

    def head(self, h, mask):
        latent = self.fe.norm(h)
        flat = latent.flatten(1)
        logits = self.p.action_net(flat).masked_fill(mask == 0, torch.finfo(torch.float32).min)
        value = self.p.value_net(self.p.mlp_extractor.forward_critic(flat)).squeeze(-1)
        return logits, value

    def fwd(self, x, mask, ix=None, cache=None, start=0, h=None):
        """Forward; if start > 0, `h` is the residual stream entering layer `start`."""
        if start == 0:
            h = self.embed_x(x, ix)
        for i in range(start, len(self.layers)):
            h = self.layer_x(i, h, ix, cache)
        return self.head(h, mask)


def policy_metrics(p0, p1, mask, shares):
    """Per-decision: KL(p0||p1), top1 agree with p0, top1 agree with the search argmax, search share of p1's argmax."""
    eps = 1e-12
    legal = mask > 0
    kl = np.where(legal, p0 * (np.log(p0 + eps) - np.log(p1 + eps)), 0).sum(1)
    a0, a1 = p0.argmax(1), p1.argmax(1)
    sa = shares.argmax(1)
    return {"kl": kl, "agree": (a0 == a1).astype(np.float64), "agree_search": (a1 == sa).astype(np.float64),
            "share": shares[np.arange(len(a1)), a1]}


def boot_games(vals: np.ndarray, game: np.ndarray, n_boot: int = 1000, seed: int = 0, w=None):
    """Mean of per-decision values with a game-level bootstrap 95% interval. Returns (mean, lo, hi)."""
    ok = np.isfinite(vals)
    vals, game = vals[ok], game[ok]
    ug, gi = np.unique(game, return_inverse=True)
    s = np.bincount(gi, vals, len(ug))
    c = np.bincount(gi, None, len(ug))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def device():
    try:
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            return "cuda"
    except Exception:  # noqa: BLE001 (the box's GPU sometimes reports "unknown error")
        pass
    return "cpu"
