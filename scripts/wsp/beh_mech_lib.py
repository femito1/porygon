"""Opponent-active swap construction shared by beh_route.py (same construction as beh_mech.py section A)."""
import numpy as np

from beh_common import FIELD_DIM, MON_DIM, NUM_OFF, MOVENUM_OFF, TYP, FNT, CAT, MTYPE, MULT, IMMUNE, decode, type_chart
from porygon.encoder import MOVE_DIM

M = type_chart()


def recompute_mult(mv, def_types):
    """mv [...,4,44] move numerics; def_types [...,1,18] -> MULT / IMMUNE recomputed for that defender."""
    mv = mv.copy()
    mt = mv[..., MTYPE:MTYPE + 18]
    has = mt.sum(-1) > 0.5
    dmg = (mv[..., CAT + 2] < 0.5) & has
    ti = mt.argmax(-1)
    mult = np.ones(ti.shape)
    for k in range(18):
        mult = mult * np.where(def_types[..., k] > 0.5, M[ti, k], 1.0)
    mult = np.where(dmg, mult, 1.0)
    mv[..., MULT] = np.where(has, np.where(mult > 0, np.log2(np.maximum(mult, 1e-9)) / 2, -1.5), mv[..., MULT])
    mv[..., IMMUNE] = np.where(has, (mult == 0).astype(np.float32), mv[..., IMMUNE])
    return mv


def build_swaps(d, f, n=1500, seed=2):
    """rows, original obs, fully swapped obs (opp-active block from a donor + effectiveness recomputed), masks."""
    rng = np.random.default_rng(seed)
    N = len(d["mask"])
    mask = d["mask"] > 0
    nums = decode(d["obs"])["nums"]
    alive = f["has_opp"] & (nums[np.arange(N), f["opp_a"], FNT] < 0.5)
    cand = np.flatnonzero(alive & (mask[:, :6].sum(1) >= 2) & f["has_act"])
    rows = np.sort(rng.choice(cand, min(n, len(cand)), replace=False))
    donors = rng.choice(cand, len(rows))
    X0 = d["obs"][rows].astype(np.float32)
    Xd = d["obs"][donors].astype(np.float32)
    k = len(rows); ar = np.arange(k)
    m0 = X0[:, FIELD_DIM:].reshape(k, 12, MON_DIM)
    md = Xd[:, FIELD_DIM:].reshape(k, 12, MON_DIM)
    oa, oad, wa = f["opp_a"][rows], f["opp_a"][donors], f["own_a"][rows]
    new_opp = md[ar, oad].copy()
    new_types = new_opp[:, NUM_OFF + TYP:NUM_OFF + TYP + 18]
    our_types = m0[ar, wa, NUM_OFF + TYP:NUM_OFF + TYP + 18]
    Xf = X0.copy()
    mf = Xf[:, FIELD_DIM:].reshape(k, 12, MON_DIM)
    mf[ar, oa] = new_opp
    mf[ar, oa, MOVENUM_OFF:] = recompute_mult(new_opp[:, MOVENUM_OFF:].reshape(k, 4, MOVE_DIM), our_types[:, None, :]).reshape(k, -1)
    own = mf[:, :6, MOVENUM_OFF:].reshape(k, 6, 4, MOVE_DIM)
    mf[:, :6, MOVENUM_OFF:] = recompute_mult(own, new_types[:, None, None].repeat(6, 1)).reshape(k, 6, -1)
    return rows, X0, Xf, d["mask"][rows]
