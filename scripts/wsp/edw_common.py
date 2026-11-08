"""edw (switch-matchup weight edit) shared helpers: train/test game split, per-candidate matchup features,
an ActionHead re-implementation that runs on cached final latents (with an optional edited weight set),
metrics, and the check-pass log.

Token layout of the cached final latents `fin` [N, 11, 256]: 0 field | 1-6 own mons (switch actions 0-5) |
7-10 move tokens (actions 6-9)  (= model tokens 0..6 and 13..16).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from beh_common import (ACTIVE, BOOST, CAT, FNT, HP, IMMUNE, MULT, BP, MTYPE, OPPACT, STATS, STATUS, TYP,
                        decode, load_local, type_chart)

WSP = Path("/vol/bitbucket/fr1226/porygon-ws-wsp")
OUT = WSP / "out" / "edw"
LOCAL = Path("/tmp/wsp-674320/edw")
FIN_TOK = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16]
SPLIT_SEED = 20261108
M = type_chart()
LOGM = np.log2(np.maximum(M, 1e-9))
IMM = (M == 0).astype(np.float64)


def split(game: np.ndarray, seed: int = SPLIT_SEED) -> np.ndarray:
    """bool [N]: True = TRAIN. Games split 50/50 by game id with a fixed seed."""
    ug = np.unique(game)
    perm = np.random.default_rng(seed).permutation(ug)
    train_games = set(perm[: len(ug) // 2].tolist())
    return np.array([g in train_games for g in game])


def log2_stab(att_types: np.ndarray, def_types: np.ndarray) -> np.ndarray:
    """best log2 multiplier of any attacker type (bool [...,18]) on defender types (bool [...,18]); immune -> -3,
    clamped to [-3, 2]; attacker with no types -> 0."""
    lm = def_types.astype(np.float64) @ LOGM.T          # [..., 18 attack types]
    im = def_types.astype(np.float64) @ IMM.T > 0
    lm = np.where(im, -3.0, np.maximum(lm, -3.0))
    lm = np.where(att_types, lm, -np.inf).max(-1)
    return np.where(np.isfinite(lm), lm, 0.0)


def spe(m: np.ndarray) -> np.ndarray:
    b = np.round(m[..., BOOST + 4] * 6)
    mult = np.where(b >= 0, (2 + b) / 2, 2 / (2 - np.minimum(b, 0)))
    return m[..., STATS + 5] * mult * np.where(m[..., STATUS + 2] > 0.5, 0.25, 1.0)


def matchup(obs: np.ndarray) -> dict:
    """Per (row, own slot) matchup features vs the opponent's active mon. obs [N, OBS_DIM]."""
    o = decode(obs)
    nums, mv = o["nums"], o["mv"]
    N = len(obs)
    r = np.arange(N)
    has_opp = nums[:, 6:, OPPACT].max(1) > 0.5
    opp_a = 6 + nums[:, 6:, OPPACT].argmax(1)
    B = nums[r, opp_a]
    own = nums[:, :6]
    tB = B[:, TYP:TYP + 18] > 0.5
    tO = own[..., TYP:TYP + 18] > 0.5
    dfn = log2_stab(tB[:, None, :].repeat(6, 1), tO)                  # incoming STAB threat on each own mon
    om = mv[:, :6]                                                    # [N,6,4,44]
    dmg = (om[..., CAT + 2] < 0.5) & (om[..., BP] > 0) & (om[..., MTYPE:MTYPE + 18].sum(-1) > 0.5)
    eff = np.where(om[..., IMMUNE] > 0.5, -3.0, np.clip(2 * om[..., MULT], -3, 2))
    off = np.where(dmg, eff, -np.inf).max(-1)
    has_dmg = np.isfinite(off)
    off = np.where(has_dmg, off, 0.0)
    pw = np.where(dmg, om[..., BP] * 150 * 2.0 ** eff * np.where(om[..., 26] > 0.5, 1.5, 1.0), 0).max(-1)
    off_pow = np.log2(np.maximum(pw, 1.0))
    so, sb = spe(own), spe(B)[:, None]
    spd = np.clip(np.log2(np.maximum(so, 1e-3) / np.maximum(sb, 1e-3)), -2, 2)
    own_a = own[..., ACTIVE].argmax(1)
    has_act = own[..., ACTIVE].max(1) > 0.5
    out = dict(dfn=dfn, off=off, off_pow=off_pow, has_dmg=has_dmg, spd=spd, faster=(so > sb * 1.0001).astype(float),
               hp=own[..., HP], fnt=own[..., FNT] > 0.5, has_opp=has_opp, opp_a=opp_a, own_a=own_a, has_act=has_act,
               opp_hp=B[:, HP])
    for k in ("dfn", "off", "spd", "hp", "faster"):
        out[k + "_act"] = np.where(has_act, out[k][r, own_a], 0.0)
    return out


# ------------------------------------------------------------------ ActionHead on cached latents
def head_weights(policy) -> dict:
    s = policy.action_net.score
    return {"W1": s[0].weight.detach().double().clone(), "b1": s[0].bias.detach().double().clone(),
            "w2": s[2].weight.detach().double().clone()[0], "b2": s[2].bias.detach().double().clone()[0]}


def head_logits(fin, W: dict, mask=None, bs: int = 8192, return_hid: bool = False):
    """fin [N,11,256] (np/torch) -> logits [N,10] (float64 numpy, masked to -inf if mask given)."""
    outs, hids = [], []
    for s in range(0, len(fin), bs):
        t = torch.as_tensor(np.asarray(fin[s:s + bs]), dtype=torch.float64)
        c = t[:, 1:]
        x = torch.cat([c, t[:, :1].expand_as(c)], -1)
        h = F.gelu(x @ W["W1"].T + W["b1"])
        outs.append((h @ W["w2"] + W["b2"]).numpy())
        if return_hid:
            hids.append(h.float().numpy())
    lg = np.concatenate(outs)
    if mask is not None:
        lg = np.where(mask > 0, lg, -np.inf)
    return (lg, np.concatenate(hids)) if return_hid else lg


def softmax(lg):
    z = lg - lg.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ bootstrap over games
def gboot(game, fn, rows, n_boot=1000, seed=0):
    """fn(row_idx) -> scalar or array; resample games among `rows`. Returns est, lo, hi."""
    rows = np.asarray(rows)
    g = game[rows]
    ug, inv = np.unique(g, return_inverse=True)
    groups = [rows[inv == k] for k in range(len(ug))]
    rng = np.random.default_rng(seed)
    est = fn(rows)
    bs = [fn(np.concatenate([groups[k] for k in rng.integers(0, len(ug), len(ug))])) for _ in range(n_boot)]
    bs = np.asarray(bs, float)
    return est, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)


def ci(t, p=3):
    e, lo, hi = t
    return f"{e:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


# ------------------------------------------------------------------ check-pass log
def log_check(what: str, n_rows: int, rows_from: str, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "what": what, "n_rows": int(n_rows), "rows": rows_from}
    rec.update(extra)
    with open(OUT / "checks.jsonl", "a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")
