"""Shared helpers for the beh (policy/value behaviour) analysis: observation decoding, derived per-decision
features, move classes, type chart, game-level bootstrap.

Layout facts (porygon.encoder):
  mon numerics (125): hp 0, fainted 1, base stats 2-7, stats 8-13, level 14, types 15-32, tera 33-50, is_tera 51,
    boosts 52-58 (atk def spa spd spe acc eva, /6), status 59-64 (BRN FRZ PAR PSN SLP TOX), status counter 65,
    effects 66-114, active 115, mine 116, first_turn 117, protect 118, recharge 119, preparing 120, weight 121,
    trapped 122, is_opp_active 123, unknown 124.
  move numerics (44): bp/150 0, acc 1, prio/5 2, pp 3, available 4, category 5-7 (phys spec status), type 8-25,
    then stab 26, log2(mult)/2 27 (-1.5 immune), immune 28, heal 29, drain 30, recoil 31, self_switch 32,
    force_switch 33, side_condition 34, status 35, volatile 36, boosts 37, self_boost 38, protect 39,
    breaks_protect 40, hits/5 41, weather 42, self_destruct 43.
  field (95): weathers 0-7, ages 8-15, no weather 16, fields 17-30, ages 31-44, our side 45-62 (7 screens, 7 ages,
    4 hazards SR spikes tspikes web), opp side 63-80, tail 81: turn, unseen opp, our alive, their alive,
    force switch, trapped, maybe trapped, can tera, used tera, opp used tera, wait, no active, switches/5, moves/4.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from porygon.encoder import FIELD_DIM, MON_DIM, MON_NUM_DIM, MOVE_DIM, MOVEPOOL, N_CAT, TYPES, VOCAB

WSP = "/vol/bitbucket/fr1226/porygon-ws-wsp"
GLORIA = "/vol/bitbucket/fr1226/gloria"
LOCAL = "/tmp/wsp-674320/data"
OUT = Path(WSP) / "out" / "beh"

NUM_OFF = N_CAT + MOVEPOOL
MOVENUM_OFF = NUM_OFF + MON_NUM_DIM
# mon numerics
HP, FNT, BASE, STATS, LEVEL, TYP, BOOST, STATUS, ACTIVE, OPPACT, UNKNOWN = 0, 1, 2, 8, 14, 15, 52, 59, 115, 123, 124
# move numerics
BP, ACC, PRIO, PP, AVAIL, CAT, MTYPE, STAB, MULT, IMMUNE = 0, 1, 2, 3, 4, 5, 8, 26, 27, 28
HEAL, DRAIN, RECOIL, SELFSW, FORCESW, SIDEC, MSTATUS, VOLAT, BOOSTS_, SELFBOOST, PROTECT = range(29, 40)
# field tail
T0 = 81
F_TURN, F_UNSEEN, F_OURALIVE, F_THEIRALIVE, F_FORCE, F_TRAPPED = T0, T0 + 1, T0 + 2, T0 + 3, T0 + 4, T0 + 5
HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes"}
MOVE_NAMES = VOCAB["moves"]
TYPE_NAMES = [t.name for t in TYPES]


def vname(kind: str, i: int) -> str:
    i = int(i)
    return "-" if i == 0 else "?" if i == 1 else VOCAB[kind][i - 2]


def type_chart() -> np.ndarray:
    """M[attack type, defend type] gen4 multipliers, TYPES order."""
    from poke_env.data import GenData
    tc = GenData.from_gen(4).type_chart  # tc[defender][attacker]
    M = np.ones((len(TYPES), len(TYPES)))
    for i, a in enumerate(TYPE_NAMES):
        for j, d in enumerate(TYPE_NAMES):
            if d in tc and a in tc[d]:
                M[i, j] = tc[d][a]
    return M


def load_local() -> dict:
    os.makedirs(LOCAL, exist_ok=True)
    for f in ("decisions.npz", "games.json"):
        src, dst = Path(WSP, "data", f), Path(LOCAL, f)
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            tmp = dst.with_suffix(f".tmp{os.getpid()}")
            shutil.copy(src, tmp)
            os.replace(tmp, dst)  # atomic: other readers keep their open file
    z = np.load(Path(LOCAL, "decisions.npz"))
    d = {k: z[k] for k in z.files}
    d["games"] = json.loads(Path(LOCAL, "games.json").read_text())
    return d


def decode(obs: np.ndarray) -> dict:
    x = obs.astype(np.float32)
    N = len(x)
    mons = x[:, FIELD_DIM:].reshape(N, 12, MON_DIM)
    return {"field": x[:, :FIELD_DIM], "cats": mons[..., :N_CAT].astype(np.int64),
            "pool": mons[..., N_CAT:NUM_OFF].astype(np.int64), "nums": mons[..., NUM_OFF:MOVENUM_OFF],
            "mv": mons[..., MOVENUM_OFF:].reshape(N, 12, 4, MOVE_DIM)}


BOOST_MULT = lambda b: np.where(b >= 0, (2 + b) / 2, 2 / (2 - np.minimum(b, 0)))  # noqa: E731  stage -> multiplier


def features(d: dict, M: np.ndarray | None = None) -> dict:
    """Derived per-decision features (numpy, vectorised)."""
    M = type_chart() if M is None else M
    o = decode(d["obs"])
    nums, mv, cats, field, mask = o["nums"], o["mv"], o["cats"], o["field"], d["mask"].astype(bool)
    N = len(mask)
    r = np.arange(N)
    has_act = nums[:, :6, ACTIVE].max(1) > 0.5
    own_a = nums[:, :6, ACTIVE].argmax(1)
    has_opp = nums[:, 6:, OPPACT].max(1) > 0.5
    opp_a = 6 + nums[:, 6:, OPPACT].argmax(1)
    A, B = nums[r, own_a], nums[r, opp_a]
    am = mv[r, own_a]  # [N,4,44] our active's moves
    move_ids = cats[r, own_a, 3:]
    legal_mv, legal_sw = mask[:, 6:], mask[:, :6]
    forced = legal_sw.any(1) & ~legal_mv.any(1)
    voluntary = legal_sw.any(1) & legal_mv.any(1)
    # speed: stat * boost multiplier (paralysis x0.25)
    def spe(m):
        s = m[:, STATS + 5] * BOOST_MULT(np.round(m[:, BOOST + 4] * 6))
        return s * np.where(m[:, STATUS + 2] > 0.5, 0.25, 1.0)
    faster = spe(A) > spe(B) * 1.0001
    slower = spe(A) < spe(B) * 0.9999
    # type matchup: best multiplier of B's types on A's types and vice versa (STAB-type proxy)
    tA, tB = A[:, TYP:TYP + 18] > 0.5, B[:, TYP:TYP + 18] > 0.5

    def best_mult(att, dfn):
        out = np.zeros(N)
        for i in range(N):
            ti, tj = np.flatnonzero(att[i]), np.flatnonzero(dfn[i])
            out[i] = max((np.prod(M[a, tj]) for a in ti), default=1.0)
        return out
    opp_se = best_mult(tB, tA)        # opponent's STAB types vs our active
    our_se = best_mult(tA, tB)
    # opponent's revealed moves' effectiveness vs our active (encoded relative to our active)
    opp_rev = (cats[r, opp_a, 3:] > 1)
    opp_rev_mult = np.where(opp_rev & (mv[r, opp_a, :, CAT + 2] < 0.5), mv[r, opp_a, :, MULT], -9).max(1)
    # move classes for our active's 4 moves
    names = np.vectorize(lambda i: vname("moves", i))(move_ids)
    status = am[..., CAT + 2] > 0.5
    attack = ~status & (am[..., BP] > 0)
    # setup: status move raising own stats (boosts on self-targeting moves) or self_boost>0; Swagger/Flatter excluded
    setup = ((status & (am[..., BOOSTS_] > 0)) | (am[..., SELFBOOST] > 0)) & ~np.isin(names, ["swagger", "flatter"])
    protect = am[..., PROTECT] > 0.5
    hazard = np.isin(names, list(HAZARD_MOVES))
    selfsw = am[..., SELFSW] > 0.5
    recover = am[..., HEAL] > 0
    mult = np.where(am[..., IMMUNE] > 0.5, 0.0, 2.0 ** (2 * am[..., MULT]))
    dmg = am[..., BP] * 150 * mult * np.where(am[..., STAB] > 0.5, 1.5, 1.0) * np.where(attack, 1, 0)
    dmg = dmg * np.clip(am[..., ACC], 0, 1) * np.maximum(am[..., 41] * 5, 1)  # accuracy, multi-hit
    dmg_legal = np.where(legal_mv & attack, dmg, -1)
    maxdmg = 6 + dmg_legal.argmax(1)
    has_attack = (legal_mv & attack).any(1)
    # switch-in matchup: for each own slot, best mult of opp active's types vs that mon (defensive) and
    # best mult of the mon's moves vs opp active (offensive, from encoded move numerics)
    own_t = nums[:, :6, TYP:TYP + 18] > 0.5
    def_mult = np.zeros((N, 6))
    for s in range(6):
        def_mult[:, s] = best_mult(tB, own_t[:, s])
    om = mv[:, :6]
    off = np.where((om[..., CAT + 2] < 0.5) & (om[..., BP] > 0),
                   om[..., BP] * 150 * np.where(om[..., IMMUNE] > 0.5, 0, 2.0 ** (2 * om[..., MULT]))
                   * np.where(om[..., STAB] > 0.5, 1.5, 1), 0).max(-1)
    return dict(has_act=has_act, own_a=own_a, has_opp=has_opp, opp_a=opp_a, hpA=A[:, HP], hpB=B[:, HP],
                forced=forced, voluntary=voluntary, faster=faster, slower=slower, opp_se=opp_se, our_se=our_se,
                opp_rev_mult=opp_rev_mult, status=status, attack=attack, setup=setup, protect=protect,
                hazard=hazard, selfsw=selfsw, recover=recover, dmg=dmg, maxdmg=maxdmg, has_attack=has_attack,
                def_mult=def_mult, off=off, move_names=names, move_ids=move_ids, turn=field[:, F_TURN] * 100,
                our_alive=field[:, F_OURALIVE] * 6, their_alive=field[:, F_THEIRALIVE] * 6,
                hp_own=(nums[:, :6, HP] * (nums[:, :6, FNT] < 0.5)).sum(1),
                hp_opp=np.where(nums[:, 6:, UNKNOWN] > 0.5, 1.0, nums[:, 6:, HP] * (nums[:, 6:, FNT] < 0.5)).sum(1))


def action_type(a: np.ndarray, f: dict) -> np.ndarray:
    """'switch' | 'attack' | 'setup' | 'hazard' | 'protect' | 'recover' | 'status' (other status move)."""
    a = np.asarray(a)
    out = np.full(len(a), "switch", dtype=object)
    r = np.arange(len(a))
    j = np.clip(a - 6, 0, 3)
    is_mv = a >= 6
    for name in ("status", "recover", "protect", "hazard", "setup"):  # later wins
        out[is_mv & f[name][r, j]] = name
    out[is_mv & f["attack"][r, j]] = "attack"
    return out


def coarse(t: np.ndarray) -> np.ndarray:
    return np.where(t == "switch", "switch", np.where(t == "attack", "attack", "status"))


def game_boot(game: np.ndarray, fn, n_boot: int = 1000, seed: int = 0):
    """fn(idx) -> scalar or array on a row subset; returns (estimate, lo, hi) resampling GAMES."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(game, return_inverse=True)
    rows = [np.flatnonzero(inv == g) for g in range(len(ug))]
    est = fn(np.arange(len(game)))
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        bs.append(fn(np.concatenate([rows[g] for g in pick])))
    bs = np.asarray(bs, dtype=float)
    return est, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)


def fmt(e, lo, hi, p=3) -> str:
    return f"{e:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def ratio_boot(game: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, n_boot: int = 2000, seed: int = 0):
    """Bootstrap over games of sum(y*w)/sum(w) (w default 1). y may be [N] or [N,K]. Returns est, lo, hi."""
    y = np.asarray(y, float)
    if len(y) == 0:
        nan = np.full(y.shape[1:], np.nan) if y.ndim > 1 else np.nan
        return nan, nan, nan
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ok = np.isfinite(y) if y.ndim == 1 else np.isfinite(y).all(1)
    y, w, game = y[ok], w[ok], game[ok]
    ug, inv = np.unique(game, return_inverse=True)
    G = len(ug)
    Y = np.zeros((G,) + y.shape[1:])
    np.add.at(Y, inv, y * (w if y.ndim == 1 else w[:, None]))
    W = np.bincount(inv, weights=w, minlength=G)
    rng = np.random.default_rng(seed)
    C = rng.multinomial(G, np.full(G, 1 / G), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (C @ Y) / ((C @ W)[:, None] if y.ndim > 1 else (C @ W))
        est = Y.sum(0) / W.sum()
    return est, np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
