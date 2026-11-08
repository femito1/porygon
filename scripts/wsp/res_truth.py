"""Ground-truth battle quantities from the engine for every logged decision with an oracle state.

Oracle state: side_one = us (team order = obs own tokens 1-6), side_two = the opponent's TRUE team (revealed mons
first, in obs opp-token order 7-12, then the unrevealed ones). For each decision:
  our moves j=0..3 (move_ids, actions 6-9) vs their active: max-roll damage D, crit damage C (engine
  calculate_damage returns [max roll, crit]); dmg_frac = min(1, 0.925·D/hp) (mean roll), P(KO) this turn =
  acc · (15/16·P(roll ≥ hp) + 1/16·P(crit roll ≥ hp)) with rolls floor(D·k/100), k = 85..100 (approximation:
  gen4 applies the roll before STAB/type), focus sash at full HP blocks the KO;
  threat (their best move vs our active): max over their true moves of dmg_frac and P(KO); "vis" = max over their
  moves already revealed in our observation (true stats/item otherwise);
  bench[i]: their best move vs our own mon i as if it were in (switch-in safety; hazards not included);
  speed: effective speed (stat × boost × scarf/iron ball × paralysis/quick feet × swift swim/chlorophyll ×
  tailwind × slow start; trick room inverts) -> first ∈ {1, 0.5, 0}; "vis" uses the observation's estimated
  opponent speed and the opponent's item only if revealed;
  type effectiveness of our moves (gen4 chart) and of their STAB types vs our active; true HP fractions; mons left.

    python scripts/wsp/res_truth.py data/decisions.npz data/oracle.jsonl out/res/truth.npz
"""

import json
import math
import sys
from multiprocessing import Pool

import numpy as np

N_PROC = 3


def init():
    global pe, MOVES, CHART, VOCAB, IDX
    import poke_engine as pe_
    from poke_env.data import GenData
    pe = pe_
    g = GenData.from_gen(4)
    MOVES = g.moves
    CHART = g.type_chart
    from porygon.encoder import VOCAB as V
    VOCAB = V


def mult(atk_type: str, def_types) -> float:
    a = atk_type.upper()
    m = 1.0
    for t in def_types:
        t = t.upper()
        if t in ("TYPELESS", "NONE", "") or a in ("TYPELESS", "NONE", ""):
            continue
        m *= CHART.get(t, {}).get(a, 1.0)
    return m


def mdata(mid: str):
    k = mid.lower()
    if k.startswith("hiddenpower"):
        k = k.rstrip("0123456789")
    return MOVES.get(k) or MOVES.get(k.rstrip("0123456789"))


def norm_move(mid: str) -> str:
    k = mid.lower()
    return "hiddenpower" if k.startswith("hiddenpower") else k


def pko(D: int, C: int, hp: int, acc: float, sash: bool) -> float:
    if hp <= 0 or D <= 0:
        return 0.0
    ks = np.arange(85, 101)
    pn = float(np.mean(np.floor(D * ks / 100) >= hp))
    pc = float(np.mean(np.floor(C * ks / 100) >= hp))
    p = acc * (15 / 16 * pn + 1 / 16 * pc)
    return 0.0 if sash else p


def acc_of(mid: str) -> float:
    d = mdata(mid)
    if not d:
        return 1.0
    a = d.get("accuracy", True)
    return 1.0 if a is True else a / 100


def boost_mult(b: int) -> float:
    return (2 + b) / 2 if b >= 0 else 2 / (2 - b)


def eff_speed(state, side, mon, item_known=True, speed_stat=None) -> float:
    s = (speed_stat if speed_stat is not None else mon.speed) * boost_mult(side.speed_boost)
    item = mon.item.upper() if item_known else ""
    if item == "CHOICESCARF":
        s *= 1.5
    if item in ("IRONBALL", "MACHOBRACE", "POWERANKLET", "POWERBAND", "POWERBELT", "POWERBRACER", "POWERLENS",
                "POWERWEIGHT"):
        s *= 0.5
    ab = mon.ability.upper()
    st = mon.status.upper()
    if ab == "QUICKFEET" and st != "NONE":
        s *= 1.5
    elif st == "PARALYZE":
        s *= 0.25
    w = str(state.weather).lower()
    if ab == "SWIFTSWIM" and "rain" in w:
        s *= 2
    if ab == "CHLOROPHYLL" and "sun" in w:
        s *= 2
    if side.side_conditions.tailwind > 0:
        s *= 2
    if any("slowstart" in v.lower() for v in side.volatile_statuses):
        s *= 0.5
    return s


def with_active(state_str: str, side: int, idx: int) -> str:
    parts = state_str.split("/")
    sp = parts[side].split("=")
    sp[6] = str(idx)
    parts[side] = "=".join(sp)
    return "/".join(parts)


def their_best(state, s1_hp: int, s1_mon, s2_moves, sash: bool):
    """Their best move vs side_one's active: (frac, pko) maxima over the given moves."""
    bf, bp = 0.0, 0.0
    for m in s2_moves:
        try:
            r = pe.calculate_damage(state, "none", m, False)[1]
        except BaseException:
            continue
        D, C = (r + [0, 0])[:2]
        bf = max(bf, min(1.0, 0.925 * D / max(s1_hp, 1)))
        bp = max(bp, pko(D, C, s1_hp, acc_of(m), sash))
    return bf, bp


def one(args):
    i, line, cats_opp_moves, opp_item_ids, obs_opp_spe, obs_own_active, obs_opp_active = args
    o = json.loads(line)
    if o is None:
        return i, None
    out = {}
    try:
        s = pe.State.from_string(o["state"])
    except BaseException as e:
        return i, {"err": f"parse {e}"[:80]}
    a, b = s.side_one, s.side_two
    ai, bi = int(a.active_index), int(b.active_index)
    us, them = a.pokemon[ai], b.pokemon[bi]
    out["chk_own"] = int(ai == obs_own_active)
    out["chk_opp"] = int(bi == obs_opp_active)
    out["hp_own"] = us.hp / max(us.maxhp, 1)
    out["hp_opp"] = them.hp / max(them.maxhp, 1)
    out["left_own"] = sum(p.hp > 0 for p in a.pokemon)
    out["left_opp"] = sum(p.hp > 0 for p in b.pokemon)
    n_fail = n_calls = 0
    # our moves
    mids = o["move_ids"]
    dm = np.full(4, np.nan); pk = np.full(4, np.nan); ef = np.full(4, np.nan); dmax = np.full(4, np.nan)
    sash_them = them.item.upper() == "FOCUSSASH" and them.hp == them.maxhp
    for j, m in enumerate(mids[:4]):
        n_calls += 1
        try:
            r = pe.calculate_damage(s, m, "none", True)[0]
        except BaseException:
            n_fail += 1
            continue
        D, C = (r + [0, 0])[:2]
        dmax[j] = D / max(them.hp, 1)
        dm[j] = min(1.0, 0.925 * D / max(them.hp, 1))
        pk[j] = pko(D, C, them.hp, acc_of(m), sash_them)
        d = mdata(m)
        if d and d.get("category") != "Status":
            ef[j] = mult(d["type"], them.types)
    out.update(our_frac=dm, our_pko=pk, our_eff=ef, our_dmax=dmax)
    # their threat vs our active (true moves; revealed-only)
    their_moves = [mv.id for mv in them.moves if mv.id.lower() not in ("none", "")]
    revealed = set(cats_opp_moves) if bi == obs_opp_active else set()
    vis_moves = [m for m in their_moves if norm_move(m) in revealed]
    sash_us = us.item.upper() == "FOCUSSASH" and us.hp == us.maxhp
    out["thr_frac"], out["thr_pko"] = their_best(s, us.hp, us, their_moves, sash_us)
    if vis_moves:
        out["thr_frac_vis"], out["thr_pko_vis"] = their_best(s, us.hp, us, vis_moves, sash_us)
    out["n_their_moves"] = len(their_moves)
    out["n_vis_moves"] = len(vis_moves)
    # STAB effectiveness vs our active
    stab = [mult(t, us.types) for t in them.types if t.upper() not in ("TYPELESS", "NONE")]
    out["stab_eff"] = max(stab) if stab else 1.0
    # bench switch-in safety
    bf = np.full(6, np.nan); bp = np.full(6, np.nan)
    for k, p in enumerate(a.pokemon):
        if k == ai or p.hp <= 0:
            continue
        try:
            s2 = pe.State.from_string(with_active(o["state"], 0, k))
        except BaseException:
            continue
        sash = p.item.upper() == "FOCUSSASH" and p.hp == p.maxhp
        bf[k], bp[k] = their_best(s2, p.hp, p, their_moves, sash)
    out.update(bench_frac=bf, bench_pko=bp)
    # speed
    so = eff_speed(s, a, us)
    st = eff_speed(s, b, them)
    item_known = them.item.lower() in opp_item_ids
    stv = eff_speed(s, b, them, item_known=item_known, speed_stat=obs_opp_spe) if obs_opp_spe > 0 else np.nan
    tr = bool(s.trick_room)

    def first(x, y):
        if not np.isfinite(y):
            return np.nan
        r = 1.0 if x > y else 0.0 if x < y else 0.5
        return 1 - r if tr and r != 0.5 else r
    out["first"] = first(so, st)
    out["first_vis"] = first(so, stv)
    out["spe_ratio"] = math.log2(max(so, 1) / max(st, 1))
    out["n_fail"], out["n_calls"] = n_fail, n_calls
    return i, out


def main():
    npz, orc, dst = sys.argv[1:4]
    from porygon.encoder import ACTIVE_FLAG, FIELD_DIM, MON_DIM, MON_NUM_DIM, MOVEPOOL, N_CAT, VOCAB
    z = np.load(npz)
    obs = z["obs"]
    N = len(obs)
    mons = obs[:, FIELD_DIM:].reshape(N, 12, MON_DIM)
    cats = mons[..., :N_CAT].astype(np.int64)
    nums = mons[..., N_CAT + MOVEPOOL:N_CAT + MOVEPOOL + MON_NUM_DIM].astype(np.float32)
    mv_names = ["", ""] + VOCAB["moves"]
    it_names = ["", ""] + VOCAB["items"]
    own_act = np.where(nums[:, :6, ACTIVE_FLAG].max(1) > 0.5, nums[:, :6, ACTIVE_FLAG].argmax(1), -1)
    opp_flag = nums[:, 6:, MON_NUM_DIM - 2]
    opp_act = np.where(opp_flag.max(1) > 0.5, opp_flag.argmax(1), -1)
    SPE = 2 + 6 + 5  # stats block starts at 8; speed is the 6th stat

    import itertools, os
    lim = int(os.environ.get("RES_LIMIT", N))

    def jobs():
        for i, line in enumerate(itertools.islice(open(orc), lim)):
            k = opp_act[i]
            if k >= 0:
                mvs = [mv_names[c] for c in cats[i, 6 + k, 3:] if c > 1]
                items = {it_names[cats[i, 6 + k, 1]]} if cats[i, 6 + k, 1] > 1 else set()
                spe = float(nums[i, 6 + k, SPE]) * 500
            else:
                mvs, items, spe = [], set(), 0.0
            yield i, line, mvs, items, spe, int(own_act[i]), int(k)

    keys4 = ("our_frac", "our_pko", "our_eff", "our_dmax")
    keys6 = ("bench_frac", "bench_pko")
    res = {k: np.full((N, 4), np.nan, np.float32) for k in keys4}
    res.update({k: np.full((N, 6), np.nan, np.float32) for k in keys6})
    scal = ("chk_own", "chk_opp", "hp_own", "hp_opp", "left_own", "left_opp", "thr_frac", "thr_pko",
            "thr_frac_vis", "thr_pko_vis", "n_their_moves", "n_vis_moves", "stab_eff", "first", "first_vis",
            "spe_ratio", "n_fail", "n_calls")
    res.update({k: np.full(N, np.nan, np.float32) for k in scal})
    valid = np.zeros(N, bool)
    errs = 0
    with Pool(N_PROC, initializer=init) as pool:
        for n, (i, out) in enumerate(pool.imap_unordered(one, jobs(), chunksize=64)):
            if out is None:
                continue
            if "err" in out:
                errs += 1
                continue
            valid[i] = True
            for k, v in out.items():
                res[k][i] = v
            if n % 5000 == 0:
                print(n, flush=True)
    np.savez(dst, valid=valid, **res)
    nc, nf = np.nansum(res["n_calls"]), np.nansum(res["n_fail"])
    print(f"valid {valid.sum()}/{N}, parse errors {errs}, own-move calc failures {nf:.0f}/{nc:.0f}; "
          f"own-active match {np.nanmean(res['chk_own']):.4f}, opp-active match {np.nanmean(res['chk_opp']):.4f}; "
          f"vis threat defined {np.isfinite(res['thr_frac_vis']).mean():.3f}")


if __name__ == "__main__":
    main()
