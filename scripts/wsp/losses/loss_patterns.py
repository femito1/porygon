"""wsp loss-pattern analysis of a raw gen9 net from logged in-process games (spd_h2h.py run --game-log DIR).

Every decision of a side is tested against a list of precisely defined candidate mistakes (PATTERNS below), using
the true-state snapshot spd_host.js took just before the choice was submitted, the omniscient Showdown log before
it (what had been revealed) and the log of that turn (what happened). Side 0 (A) is the net under study in every
game; side 1 (B) is the comparison (the same net in M1 -> loser vs winner contrast; BC-P3, the human-imitation
init, in M2 -> A vs B rates).

    python loss_patterns.py --moves moves9.json --out OUT.json [--examples 3] LABEL=DIR [LABEL=DIR ...]

Output: OUT.json (all counts) and a plain-text table on stdout. Rates come with Wilson 95% intervals over decisions
(clustered by game, so a little narrow); the paired per-game contrasts (loser - winner, A - B) use t-intervals over
games, which are valid under clustering. "Association" slopes regress the game result of A on the per-game
difference in counts (A - B): in self-play this is an association, not a causal cost.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

PROTECT = {"protect", "detect", "kingsshield", "spikyshield", "banefulbunker", "silktrap", "burningbulwark",
           "obstruct", "endure", "maxguard"}
REMOVAL = {"defog", "rapidspin", "mortalspin", "tidyup", "courtchange"}
HAZARD_CAP = {"stealthrock": 1, "spikes": 3, "toxicspikes": 2, "stickyweb": 1}
ABILITY_IMMUNE = {  # opponent ability -> attacking types it absorbs (true ability; may be unrevealed)
    "levitate": {"Ground"}, "eartheater": {"Ground"}, "flashfire": {"Fire"}, "wellbakedbody": {"Fire"},
    "waterabsorb": {"Water"}, "stormdrain": {"Water"}, "dryskin": {"Water"}, "voltabsorb": {"Electric"},
    "lightningrod": {"Electric"}, "motordrive": {"Electric"}, "sapsipper": {"Grass"}, "purifyingsalt": set(),
}
STATUS_BLOCK_AB = {"goodasgold", "magicbounce", "purifyingsalt", "comatose"}  # block / reflect status moves
SLEEP_OK = {"sleeptalk", "snore"}
NOT_IMMUNITY_TYPES_CHECK = {"thousandarrows"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def wilson(k: int, n: int) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def tci(xs: list) -> tuple:
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"), n)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return (m, 1.96 * sd / math.sqrt(n), n)


def ols(xs: list, ys: list) -> tuple:
    """slope of y on x with a 95% half-width (plain OLS SE)."""
    n = len(xs)
    if n < 3:
        return (float("nan"), float("nan"))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return (float("nan"), float("nan"))
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = sum((y - my - b * (x - mx)) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(res / (n - 2) / sxx)
    return (b, 1.96 * se)


class Game:
    """One logged game: names, per-turn log segments and helpers on the omniscient log."""

    def __init__(self, rec: dict):
        self.r = rec
        self.log = rec["log"]
        self.winner = {"p1": 0, "p2": 1}.get(rec["winner"], None)
        # log name -> species per side (team 'name' is what the log and the choices use)
        self.name2sp = [{t["name"]: t["species"] for t in team} for team in rec["teams"]]
        self.sets = [{t["species"]: t for t in team} for team in rec["teams"]]

    def turn_segment(self, li: int) -> list:
        """log lines from index li up to the next |turn| (the resolution of the decision's turn)."""
        out = []
        for j in range(li, len(self.log)):
            ln = self.log[j]
            if ln.startswith("|turn|") and j > li:
                break
            out.append(ln)
        return out

    def revealed_moves(self, side: int, name: str, li: int) -> set:
        p = f"|move|p{side + 1}a: {name}|"
        out = set()
        for ln in self.log[:li]:
            if ln.startswith(p):
                out.add(norm(ln.split("|")[3]))
        return out

    def ability_revealed(self, side: int, name: str, ability: str, li: int) -> bool:
        tag = f"p{side + 1}a: {name}"
        ab = norm(ability)
        for ln in self.log[:li]:
            if tag in ln and "ability" in ln and ab in norm(ln):
                return True
        return False

    def moved_since_switch_in(self, side: int, name: str, li: int) -> bool:
        tag = f"p{side + 1}a: {name}|"
        for j in range(li - 1, -1, -1):
            ln = self.log[j]
            if (ln.startswith("|switch|") or ln.startswith("|drag|")) and tag in ln:
                return False
            if ln.startswith(f"|move|p{side + 1}a: {name}|"):
                return True
        return False


def mon_index_by_name(snap_side: dict, game: Game, side: int, name: str) -> int:
    sp = game.name2sp[side].get(name, name)
    for i, m in enumerate(snap_side["mons"]):
        if m["sp"] == sp:
            return i
    for i, m in enumerate(snap_side["mons"]):  # forme changes: prefix match
        if m["sp"].split("-")[0] == sp.split("-")[0]:
            return i
    return -1


def name_of(game: Game, side: int, sp: str) -> str:
    for n, s in game.name2sp[side].items():
        if s == sp:
            return n
    for n, s in game.name2sp[side].items():
        if s.split("-")[0] == sp.split("-")[0]:
            return n
    return sp.split("-")[0]


def eff_of(eff_entry_list: list, move_id: str, idx: int):
    for mid, cat, ty, bp, effs in eff_entry_list or []:
        if mid == move_id:
            return effs[idx] if 0 <= idx < len(effs) else None
    return None


def patterns_for(game: Game, d: dict, M: dict) -> dict:
    """{pattern: (opportunity: bool, hit: bool, info)} for one decision."""
    me = d["side"]
    op = 1 - me
    S = d["s"]
    ms, os_ = S[me], S[op]
    out = {}
    if ms["act"] < 0 or os_["act"] < 0:
        return out
    a, o = ms["mons"][ms["act"]], os_["mons"][os_["act"]]
    forced = bool(d["rq"]["fs"])
    c = d["c"].split()
    kind = c[0] if c else ""
    tera = "terastallize" in c
    li = d["li"]
    aname, oname = name_of(game, me, a["sp"]), name_of(game, op, o["sp"])
    bench_ok = [i for i, m in enumerate(ms["mons"]) if i != ms["act"] and not m.get("f") and m["hp"] > 0]
    can_switch = bool(bench_ok) and not d["rq"]["tr"]
    seg = None

    def segment():
        nonlocal seg
        if seg is None:
            seg = game.turn_segment(li)
        return seg

    def acted_this_turn() -> bool:
        return any(ln.startswith(f"|move|p{me + 1}a: {aname}|") for ln in segment())

    def opp_switched_first() -> bool:  # the opponent switched before our move resolved this turn
        for ln in segment():
            if ln.startswith(f"|move|p{me + 1}a: {aname}|"):
                return False
            if ln.startswith(f"|switch|p{op + 1}a:"):
                return True
        return False

    def fainted_this_turn() -> bool:
        return any(ln.startswith(f"|faint|p{me + 1}a: {aname}") for ln in segment())

    info = {"turn": d["turn"], "me": a["sp"], "hp": f"{a['hp']}/{a['mh']}", "st": a["st"], "opp": o["sp"],
            "opp_hp": f"{o['hp']}/{o['mh']}", "opp_st": o["st"], "opp_types": o["ty"], "choice": d["c"]}
    if forced or a.get("f"):
        # forced switches: switching into a revealed super-effective attack is still a choice of target
        pass
    mid = c[1] if kind == "move" and len(c) > 1 else None
    mv = M.get(mid) if mid else None
    oty = set(o["ty"])
    voluntary = not forced and not a.get("f")

    # ---- P1 status move wasted on an immune / already-statused / blocked target
    if voluntary and mv and mv["cat"] == "Status" and mv["tg"] in ("normal", "any", "adjacentFoe", "randomNormal"):
        why = []
        if mv["st"] and o["st"]:
            why.append("target already statused")
        if mv["st"] == "par" and ("Electric" in oty or (mid == "thunderwave" and "Ground" in oty)):
            why.append("type immune to paralysis")
        if mv["st"] in ("psn", "tox") and oty & {"Steel", "Poison"} and a.get("ab") != "corrosion":
            why.append("type immune to poison")
        if mv["st"] == "brn" and "Fire" in oty:
            why.append("type immune to burn")
        if "powder" in mv["fl"] and "Grass" in oty:
            why.append("Grass immune to powder")
        if mid == "yawn" and (o["st"] or "yawn" in (o.get("vo") or [])):
            why.append("yawn on statused/yawned target")
        if mid == "leechseed" and ("Grass" in oty or "leechseed" in (o.get("vo") or [])):
            why.append("leech seed on Grass/seeded")
        if mid == "taunt" and "taunt" in (o.get("vo") or []):
            why.append("taunt on taunted")
        if "substitute" in (o.get("vo") or []) and "bypasssub" not in mv["fl"]:
            why.append("status move into a Substitute")
        public = bool(why)
        abl = o.get("ab", "") in STATUS_BLOCK_AB
        if abl and not why:
            why.append(f"blocked by ability {o.get('ab')} (revealed: "
                       f"{game.ability_revealed(op, oname, o.get('ab', ''), li)})")
        out["P1_status_wasted"] = (True, public, info | {"why": why})
        out["P1s_status_wasted_opp_stayed"] = (True, public and not opp_switched_first(), info | {"why": why})
        out["P1b_status_ability_blocked"] = (True, abl and not public, info | {"why": why})

    # ---- P2 damaging move into a type immunity (current types) / ability immunity
    if voluntary and mv and mv["cat"] != "Status" and mv["tg"] in ("normal", "any", "adjacentFoe", "allAdjacentFoes",
                                                                    "randomNormal", "allAdjacent"):
        e = eff_of(ms.get("eff"), mid, os_["act"])
        scrappy = a.get("ab") in ("scrappy", "mindseye") and mv["ty"] in ("Normal", "Fighting")
        type_imm = e == "x" and not mv["ign"] and not scrappy and not mv["dmg"] and mid not in NOT_IMMUNITY_TYPES_CHECK
        oab = o.get("ab", "")
        ab_imm = mv["ty"] in ABILITY_IMMUNE.get(oab, set()) and a.get("ab") not in ("moldbreaker", "teravolt",
                                                                                     "turboblaze")
        if mv["ty"] == "Ground" and o.get("it") == "airballoon":
            ab_imm = True
        out["P2_attack_into_type_immunity"] = (True, type_imm, info | {"eff": e})
        stayed_imm = type_imm and not opp_switched_first()
        out["P2s_type_immunity_opp_stayed"] = (True, stayed_imm, info | {"eff": e})
        if a["hp"] >= 0.5 * a["mh"]:
            out["P2h_type_immunity_opp_stayed_hp50"] = (True, stayed_imm, info | {"eff": e})
        out["P2b_attack_into_ability_immunity"] = (True, ab_imm and not type_imm, info | {
            "ability": oab, "revealed": game.ability_revealed(op, oname, oab, li) if ab_imm else None})
        # ---- P3 resisted attack while a neutral-or-better attack of >= 70% of its base power was available
        if mv["bp"] and not mv["dmg"] and not mv["ss"] and not type_imm and not scrappy and o["hp"] >= 0.3 * o["mh"]:
            if e == "x" or (isinstance(e, (int, float)) and e <= -1):
                usable = {x.rstrip("!") for x in d["rq"]["mv"] if not x.endswith("!")}
                better = []
                for m2, cat2, ty2, bp2, effs in ms.get("eff") or []:
                    if m2 == mid or m2 not in usable or cat2 == "Status" or not bp2:
                        continue
                    e2 = effs[os_["act"]]
                    if e2 != "x" and e2 >= 0 and bp2 >= 0.7 * mv["bp"]:
                        better.append(m2)
                out["P3_resisted_with_better"] = (True, bool(better), info | {"eff": e, "better": better})
                out["P3s_resisted_with_better_opp_stayed"] = (True, bool(better) and not opp_switched_first(),
                                                              info | {"eff": e, "better": better})
            else:
                out["P3_resisted_with_better"] = (True, False, info)
                out["P3s_resisted_with_better_opp_stayed"] = (True, False, info)

    # ---- P4 choice-locked into a resisted/immune move and stayed in although a switch was possible
    lock = a.get("lock")
    if voluntary and lock and can_switch:
        e = eff_of(ms.get("eff"), lock, os_["act"])
        lty = M.get(lock, {}).get("ty")
        if e == "x" and a.get("ab") in ("scrappy", "mindseye") and lty in ("Normal", "Fighting"):
            e = 0
        bad = e == "x" or (isinstance(e, (int, float)) and e <= -1 and o["hp"] >= 0.3 * o["mh"]) or (
            M.get(lock, {}).get("cat") == "Status")
        if bad:
            out["P4_locked_bad_stayed"] = (True, kind == "move", info | {"lock": lock, "eff": e})

    # ---- P5 switch into a revealed super-effective attack of the opponent's active (voluntary or forced)
    if kind == "switch":
        tname = " ".join(c[1:])
        ti = mon_index_by_name(ms, game, me, tname)
        rev = game.revealed_moves(op, oname, li)
        se = []
        for m2, cat2, ty2, bp2, effs in os_.get("eff") or []:
            if m2 in rev and cat2 != "Status" and bp2 and ti >= 0 and effs[ti] != "x" and effs[ti] >= 1:
                se.append(m2)
        # an alternative target with no revealed SE move against it
        alt = False
        for j in bench_ok:
            if j == ti:
                continue
            if not any(m2 in rev and cat2 != "Status" and bp2 and effs[j] != "x" and effs[j] >= 1
                       for m2, cat2, ty2, bp2, effs in os_.get("eff") or []):
                alt = True
        key = "P5_switch_into_revealed_SE" + ("" if voluntary else "_forced")
        hit_se = False
        if se:
            for ln in segment():
                if ln.startswith("|-supereffective|") and f"p{me + 1}a: {tname}" in ln:
                    hit_se = True
        out[key] = (True, bool(se) and alt, info | {"target": tname, "se_moves": se, "took_se_hit": hit_se})

    # ---- P6 stayed in (attacked) while slower, at < 50% HP, facing a revealed SE attack; outcome: KOed first
    if voluntary and kind == "move" and can_switch and a.get("spe") and o.get("spe"):
        rev = game.revealed_moves(op, oname, li)
        se = [m2 for m2, cat2, ty2, bp2, effs in os_.get("eff") or []
              if m2 in rev and cat2 != "Status" and bp2 and effs[ms["act"]] != "x" and effs[ms["act"]] >= 1]
        tr = "trickroom" in d["fld"]["pw"]
        slower = a["spe"] < o["spe"]
        if se and slower and a["hp"] < 0.5 * a["mh"] and mv and mv["pr"] <= 0:
            died = fainted_this_turn() and not acted_this_turn()
            out["P6_stay_slow_low_vs_SE"] = (True, died, info | {"se_moves": se, "trick_room": tr})
    # outcome-only: KOed before moving after choosing to stay and attack, a switch being possible
    if voluntary and kind == "move" and can_switch:
        out["P7_KOed_before_moving"] = (True, fainted_this_turn() and not acted_this_turn(), info)

    # ---- P8 setup at risk: self-boost at < 50% HP while slower; outcome: fainted this turn or next before attacking
    if voluntary and mv and mv["cat"] == "Status" and mv["tg"] == "self" and (
            (mv["bo"] and any(v > 0 for v in mv["bo"].values())) or mid == "bellydrum"):
        risky = a["hp"] < 0.5 * a["mh"] and a.get("spe") and o.get("spe") and a["spe"] < o["spe"]
        capped = mv["bo"] and all((a.get("bo") or {}).get(k, 0) >= 6 for k, v in mv["bo"].items() if v > 0)
        out["P8_setup_low_hp_slower"] = (True, bool(risky), info | {"boosts": a.get("bo")})
        out["P8b_setup_at_cap"] = (True, bool(capped), info | {"boosts": a.get("bo")})
        out["P8c_setup_then_KOed_same_turn"] = (True, fainted_this_turn(), info)

    # ---- P9 hazards
    if voluntary and mv and mv["sc"] in HAZARD_CAP and mv["tg"] == "foeSide":
        layers = os_["sc"].get(mv["sc"], 0)
        out["P9_hazard_wasted"] = (True, layers >= HAZARD_CAP[mv["sc"]], info | {"layers": layers})
    if voluntary and mid == "defog":
        any_h = any(k in ms["sc"] or k in os_["sc"] for k in list(HAZARD_CAP) + ["reflect", "lightscreen", "auroraveil"])
        out["P9b_defog_nothing"] = (True, not any_h, info)
    own_haz = {k: v for k, v in ms["sc"].items() if k in HAZARD_CAP}
    if voluntary and own_haz:
        usable = {x for x in d["rq"]["mv"] if not x.endswith("!")}
        if usable & REMOVAL:
            out["P9c_removal_available_not_used"] = (True, not (mid in REMOVAL), info | {"hazards": own_haz})
    # Stealth Rock available, not up, early game: chosen?
    if voluntary and d["turn"] <= 10 and "stealthrock" not in os_["sc"]:
        usable = {x for x in d["rq"]["mv"] if not x.endswith("!")}
        if "stealthrock" in usable:
            out["P9d_rocks_available_not_set_t10"] = (True, mid != "stealthrock", info)

    # ---- P10 sleep: asleep, stays in with a move that does nothing asleep (no Sleep Talk used), switch possible
    if voluntary and a["st"] == "slp" and a.get("ab") != "comatose":
        wakes = a.get("slp") == 1  # PS: the move attempt with time 1 wakes the mon (hidden from the player)
        if can_switch:
            out["P10_asleep_stayed_no_sleeptalk"] = (True, kind == "move" and mid not in SLEEP_OK,
                                                     info | {"sleep_turns_left": a.get("slp"), "wakes_now": wakes})
            if not wakes:
                out["P10w_asleep_stayed_stays_asleep"] = (True, kind == "move" and mid not in SLEEP_OK,
                                                          info | {"sleep_turns_left": a.get("slp")})
    if voluntary and mid == "rest":
        out["P10b_rest_at_full_hp"] = (True, a["hp"] >= a["mh"] and a["st"] != "slp", info)

    # ---- P11 Substitute that fails (<= 25% HP or already behind one)
    if voluntary and mid == "substitute":
        out["P11_sub_fails"] = (True, a["hp"] <= a["mh"] // 4 or "substitute" in (a.get("vo") or []), info)
    # ---- P12 healing move at full HP (fails)
    if voluntary and mv and "heal" in mv["fl"] and mv["tg"] == "self" and mid not in ("wish", "healingwish",
                                                                                     "lunardance", "revivalblessing"):
        out["P12_heal_at_full"] = (True, a["hp"] >= a["mh"], info)
    # ---- P13 Fake Out / First Impression after the first turn out (fails)
    if voluntary and mid in ("fakeout", "firstimpression"):
        out["P13_fakeout_late"] = (True, game.moved_since_switch_in(me, aname, li), info)
    # ---- P14 consecutive Protect
    if voluntary and mid in PROTECT:
        out["P14_protect_repeat"] = (True, a.get("lm") in PROTECT and game.moved_since_switch_in(me, aname, li), info)
    # ---- P15 Tera: decisions (opportunities = decisions with Tera available)
    if voluntary and d["rq"]["ct"] and kind == "move":
        out["P15_tera_used"] = (True, tera, info | {"tera_type": d["rq"]["ct"]})
    return out


def game_level(game: Game, side: int) -> dict:
    """per-game Tera summary of one side."""
    teras = [d for d in game.r["dec"] if d["side"] == side and "terastallize" in d["c"] and d.get("ok", 1)]
    t = {"tera": bool(teras)}
    if teras:
        d = teras[0]
        a = d["s"][side]["mons"][d["s"][side]["act"]]
        t |= {"tera_turn": d["turn"], "tera_hp": a["hp"] / max(1, a["mh"]), "tera_mon": a["sp"]}
        nm = name_of(game, side, a["sp"])
        # wasted: the Tera'd mon fainted on the Tera turn or the next one
        fain = None
        turn = 0
        for ln in game.log[d["li"]:]:
            if ln.startswith("|turn|"):
                turn += 1
                if turn >= 2:
                    break
            if ln.startswith(f"|faint|p{side + 1}a: {nm}"):
                fain = True
                break
        t["tera_mon_fainted_within_1_turn"] = bool(fain)
    return t


def run(label: str, d: Path, M: dict, n_ex: int) -> dict:
    files = sorted(d.glob("games-*.jsonl*"))
    agg = defaultdict(lambda: {"opp": [0, 0], "hit": [0, 0], "w": {"opp": [0, 0], "hit": [0, 0]},
                               "l": {"opp": [0, 0], "hit": [0, 0]}, "ex": []})
    per_game = defaultdict(list)  # pattern -> [(hitsA, hitsB, scoreA, oppA, oppB)]
    games = 0
    n_short = 0
    dec_n = [0, 0]
    tera_rows = []
    turns_res = []
    for f in files:
        op = gzip.open if f.suffix == ".gz" else open
        with op(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("void") or r.get("truncated"):
                    continue
                g = Game(r)
                if g.winner is None:
                    continue
                games += 1
                scoreA = 1.0 if g.winner == 0 else 0.0
                turns_res.append((r["turns"], scoreA))
                cnt = defaultdict(lambda: [[0, 0], [0, 0]])  # pattern -> [[oppA, hitA], [oppB, hitB]]
                for dd in r["dec"]:
                    if not dd.get("ok", 1):
                        continue
                    side = dd["side"]
                    dec_n[side] += 1
                    for p, (opp, hit, info) in patterns_for(g, dd, M).items():
                        cnt[p][side][0] += opp
                        cnt[p][side][1] += hit
                        a = agg[p]
                        a["opp"][side] += opp
                        a["hit"][side] += hit
                        wl = "w" if g.winner == side else "l"
                        a[wl]["opp"][side] += opp
                        a[wl]["hit"][side] += hit
                        if hit and side == 0 and len(a["ex"]) < 40:
                            a["ex"].append({"pair": r["pair"], "game": r["game"], "won": g.winner == 0, **info,
                                            "turn_log": [ln for ln in g.turn_segment(dd["li"])
                                                         if ln.startswith(("|move|", "|switch|", "|-supereffective",
                                                                           "|-immune", "|-fail", "|faint", "|-miss",
                                                                           "|-status", "|cant", "|-terastallize",
                                                                           "|-activate"))][:12]})
                for p, v in cnt.items():
                    per_game[p].append((v[0][1], v[1][1], scoreA, v[0][0], v[1][0], r["turns"]))
                n_short += r["turns"] < 25
                for side in (0, 1):
                    tera_rows.append({"side": side, "won": g.winner == side, **game_level(g, side)})
    res = {"label": label, "games": games, "decisions": dec_n, "patterns": {}}
    for p, a in sorted(agg.items()):
        row = {}
        for side in (0, 1):
            k, n = a["hit"][side], a["opp"][side]
            row[f"side{side}"] = {"hits": k, "opps": n, "rate": wilson(k, n),
                                  "per100dec": 100 * k / max(1, dec_n[side]),
                                  "in_wins": wilson(a["w"]["hit"][side], a["w"]["opp"][side]) + (a["w"]["hit"][side], a["w"]["opp"][side]),
                                  "in_losses": wilson(a["l"]["hit"][side], a["l"]["opp"][side]) + (a["l"]["hit"][side], a["l"]["opp"][side])}
        pg = per_game[p]
        # paired per-game contrast A - B of hits per game (over all games; 0 where no opportunity)
        allg = pg + [(0, 0, None, 0, 0, 0)] * (games - len(pg))
        row["A_minus_B_hits_per_game"] = tci([x[0] - x[1] for x in allg])
        # loser - winner hits per game (meaningful when A and B are the same net)
        lw = [((x[1] - x[0]) if x[2] == 1.0 else (x[0] - x[1])) for x in pg] + [0] * (games - len(pg))
        row["loser_minus_winner_hits_per_game"] = tci(lw)
        for nm, sel, tot in (("short", lambda t: t < 25, n_short), ("long", lambda t: t >= 25, games - n_short)):
            lws = [((x[1] - x[0]) if x[2] == 1.0 else (x[0] - x[1])) for x in pg if sel(x[5])]
            row[f"loser_minus_winner_{nm}"] = tci(lws + [0] * (tot - len(lws)))
        xs = [x[0] - x[1] for x in pg]
        ys = [x[2] for x in pg]
        row["assoc_slope_scoreA_per_extra_hit"] = ols(xs, ys)
        row["hits_per_game"] = [sum(x[0] for x in pg) / max(1, games), sum(x[1] for x in pg) / max(1, games)]
        row["examples"] = a["ex"][:n_ex] if n_ex else []
        row["examples_more"] = a["ex"][n_ex:n_ex + 5]
        res["patterns"][p] = row
    # Tera summary
    tsum = {}
    for side in (0, 1):
        rows = [t for t in tera_rows if t["side"] == side]
        used = [t for t in rows if t["tera"]]
        tsum[f"side{side}"] = {
            "games": len(rows), "tera_games": len(used),
            "tera_rate": wilson(len(used), len(rows)),
            "win_if_tera": wilson(sum(t["won"] for t in used), len(used)),
            "win_if_no_tera": wilson(sum(t["won"] for t in rows if not t["tera"]), len(rows) - len(used)),
            "tera_turn_median": sorted(t["tera_turn"] for t in used)[len(used) // 2] if used else None,
            "tera_turn_le3": wilson(sum(t["tera_turn"] <= 3 for t in used), len(used)),
            "tera_hp_lt_0.3": wilson(sum(t["tera_hp"] < 0.3 for t in used), len(used)),
            "tera_mon_fainted_within_1_turn": wilson(sum(t["tera_mon_fainted_within_1_turn"] for t in used), len(used)),
            "lost_without_tera": wilson(sum((not t["won"]) and not t["tera"] for t in rows), sum(not t["won"] for t in rows)),
        }
    res["tera"] = tsum
    # game length
    buckets = [(0, 20), (20, 30), (30, 40), (40, 60), (60, 1001)]
    res["by_length"] = {f"{lo}-{hi}": wilson(int(sum(s for t, s in turns_res if lo <= t < hi)),
                                             sum(1 for t, s in turns_res if lo <= t < hi)) +
                        (sum(1 for t, s in turns_res if lo <= t < hi),) for lo, hi in buckets}
    return res


def fmt_rate(r) -> str:
    p, lo, hi = r[:3]
    if p != p:
        return "   -   "
    return f"{100 * p:5.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="LABEL=DIR (DIR holds games-*.jsonl[.gz])")
    ap.add_argument("--moves", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    M = json.load(open(a.moves))
    allres = []
    for spec in a.runs:
        label, d = spec.split("=", 1)
        res = run(label, Path(d), M, a.examples)
        allres.append(res)
        print(f"=== {label}: {res['games']} games, decisions A {res['decisions'][0]} B {res['decisions'][1]}")
        for p, row in res["patterns"].items():
            s0, s1 = row["side0"], row["side1"]
            m, h, n = row["A_minus_B_hits_per_game"]
            lm, lh, _ = row["loser_minus_winner_hits_per_game"]
            b, bh = row["assoc_slope_scoreA_per_extra_hit"]
            ls, lsh, _ = row["loser_minus_winner_short"]
            ll, llh, _ = row["loser_minus_winner_long"]
            print(f"{p:36s} A {s0['hits']:6d}/{s0['opps']:<7d} {fmt_rate(s0['rate'])}  B {s1['hits']:6d}/{s1['opps']:<7d} "
                  f"{fmt_rate(s1['rate'])}  A-B/game {m:+.3f}±{h:.3f}  loser-winner/game {lm:+.3f}±{lh:.3f}  "
                  f"slope {b:+.3f}±{bh:.3f}  l-w short {ls:+.3f}±{lsh:.3f} long {ll:+.3f}±{llh:.3f}")
        for side in ("side0", "side1"):
            t = res["tera"][side]
            print(f"tera {side}: used in {t['tera_games']}/{t['games']} games {fmt_rate(t['tera_rate'])}; win if tera "
                  f"{fmt_rate(t['win_if_tera'])}, no tera {fmt_rate(t['win_if_no_tera'])}; median turn "
                  f"{t['tera_turn_median']}; turn<=3 {fmt_rate(t['tera_turn_le3'])}; HP<30% "
                  f"{fmt_rate(t['tera_hp_lt_0.3'])}; mon fainted <=1 turn {fmt_rate(t['tera_mon_fainted_within_1_turn'])}; "
                  f"losses without tera {fmt_rate(t['lost_without_tera'])}")
        print("A win rate by game length:", {k: (fmt_rate(v), v[3]) for k, v in res["by_length"].items()})
    json.dump(allres, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
