"""The poke-engine `State` of the TRUE position, from fastenv's full simulator state (host.js --dump-states,
fastenv/js/dump.js), for NNUE gate 1 (docs/netsearch/NNUE-GATE1-PREREG.md). Side one = the side to move.

It builds the State the way `porygon.search.convert(battle, rng, truth=, oracle_all=True)` does (`search.make_mon`,
`search.engine_move`, the pe.Side / pe.State construction of `convert.build` / `make_side`), with every field read
from the simulator instead of poke-env, so on everything both know exactly the two agree (scripts/fastenv/
nnue_parity.py). Where the simulator knows more, this module uses the truth:
  - side two's HP (exact, not a percentage) and PP;
  - substitute HP (exact, not `search.substitute_left`'s estimate);
  - Wish's heal (the wisher's max HP / 2, not the current active's);
  - weather / terrain / screens turns left (exact, incl. Damp Rock / Light Clay / Terrain Extender);
  - every simulator volatile whose upper-cased id the engine knows (unknown names are dropped by the engine);
  - side two's disabled moves (Taunt, Disable, choice lock, ... as the simulator applies them).
Elapsed counts the engine takes (taunt / encore turns, confusion attempts) are estimated as poke-env counts them
(dump.js `_start` / `_time0`), so they can be off by one in rare orders; the gate-1 student does not use them.

    from porygon.nnue.convert_full import convert_full, state_dict
    state = convert_full(json.loads(line)["state"])          # a pe.State
"""
from __future__ import annotations

import math
from functools import cache
from typing import Any

import poke_engine as pe

from porygon import search

STATUS = {"brn": "burn", "slp": "sleep", "frz": "freeze", "par": "paralyze", "psn": "poison", "tox": "toxic"}
WEATHER = {"sunnyday": "sun", "raindance": "rain", "sandstorm": "sand", "hail": "hail", "snow": "snow",
           "snowscape": "snow", "desolateland": "harshsun", "primordialsea": "heavyrain"}
PRIMAL = {"desolateland", "primordialsea"}
WEATHER_ABILITY = {"sunnyday": "drought", "raindance": "drizzle", "sandstorm": "sandstream", "hail": "snowwarning",
                   "snow": "snowwarning"}
TERRAIN = {"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}
SCREENS = {"reflect": "reflect", "lightscreen": "light_screen", "auroraveil": "aurora_veil", "safeguard": "safeguard",
           "mist": "mist", "luckychant": "lucky_chant", "tailwind": "tailwind"}
SKIP = {"flinch", "protect", "endure", "stall"}  # per-turn effects (search.SKIP_EFFECTS); stall = protect counter


def engine_move(move_id: str, hp_type: str) -> str:
    """search.engine_move with the simulator's hidden power type (a type name, not a poke-env PokemonType)."""
    if move_id.startswith("hiddenpower"):
        typed = move_id[len("hiddenpower"):].rstrip("0123456789") or hp_type.lower()
        return f"hiddenpower{typed}70" if typed else "hiddenpower"
    return move_id


@cache
def known_volatile(name: str) -> bool:
    """The engine knows this volatile (pe.Side maps unknown names to NONE, or drops them)."""
    st = pe.State(side_one=pe.Side(volatile_statuses={name}))
    return pe.State.from_string(st.to_string()).side_one.volatile_statuses == {name}


def _types(m: dict[str, Any]) -> tuple[str, str]:
    ts = [t.lower() for t in m["types"] if t and t != "???"]
    ts = (ts + ["typeless", "typeless"])[:2]
    return ts[0], ts[1]


def _status(m: dict[str, Any], side: int) -> tuple[str, int, int]:
    """(status, turns asleep so far, rest turns left), as search.status_of: Rest counts down from 3."""
    status = STATUS.get(m["status"], "none")
    if status != "sleep":
        return status, 0, 0
    st = m["status_state"]
    start = st.get("startTime", st.get("_time0"))  # gen5+ stores the drawn length; gen4: the first dump's `time`
    turns = min(max(int(start or 0) - int(st.get("time") or 0), 0), search.MAX_SLEEP_TURNS)
    src = st.get("source")  # only Rest puts a mon to sleep by its own hand
    rest = isinstance(src, dict) and src.get("side") == side and src.get("species") == m["species"]
    return status, turns, max(1, 3 - turns) if rest else 0


def make_mon(m: dict[str, Any], disabled: set[str], side: int) -> pe.Pokemon:
    """One mon; `disabled`: its moves the side's request disables (the active of a side not force-switching)."""
    mv = [(search.norm(engine_move(x["id"], m["hp_type"])), x["pp"], x["id"] in disabled) for x in m["moves"][:4]]
    hp = 0 if m["fainted"] else m["hp"]
    tera = m["tera"] if search.GEN_NUM >= 9 else ""
    return search.make_mon(m["species"], m["level"], _types(m), hp, m["maxhp"] or 1, m["stats"], m["ability"] or "none",
                           m["item"] or "none", mv, *_status(m, side), m["weightkg"], tera or ("typeless" if search.GEN_NUM >= 9
                                                                                         else ""),
                           bool(m["terastallized"]))


def _last_move(m: dict[str, Any] | None) -> int | None:
    if not m or not m["last_move"]:
        return None
    ids = [x["id"] for x in m["moves"][:4]]
    return ids.index(m["last_move"]) if m["last_move"] in ids else None


def _side(s: dict[str, Any], other: dict[str, Any], turn: int, to_move: bool) -> pe.Side:
    req = s["request"] or {}
    force = bool(req.get("forceSwitch")) if to_move else False
    idx = max(s["active"], 0)
    act = s["mons"][idx] if s["active"] >= 0 else None
    off: set[str] = set()
    if not req.get("forceSwitch") and act is not None:
        off = set(req.get("disabled") or [])
        allowed = req.get("moves")
        if allowed is not None:  # a locked mon's request lists only the move it must use (Outrage, Ice Ball, ...)
            off |= {x["id"] for x in act["moves"] if x["id"] not in allowed}
    mons = [make_mon(m, off if i == idx and m["active"] else set(), s["n"]) for i, m in enumerate(s["mons"])]
    mons = (mons + [pe.Pokemon.create_fainted() for _ in range(6)])[:6]
    vol = {k: v for k, v in (act["volatiles"] if act else {}).items() if k not in SKIP}
    names = {k.upper() for k in vol if known_volatile(k.upper())}
    if act is not None and not act["terastallized"] and act["species"] in search.GEN.pokedex \
            and _types(act) != search.dex_types(act["species"]) and known_volatile("TYPECHANGE"):
        names.add("TYPECHANGE")  # the simulator changes types in place (Soak, Burn Up, ...); poke-env's effect
    last = _last_move(act)
    if last is None:
        names.discard("ENCORE")  # the engine panics on encore without a last-used move
    cond = s["conditions"]
    kw: dict[str, int] = {
        "toxic_count": int(act["status_state"].get("stage", 0)) if act and act["status"] == "tox" else 0,
        "protect": round(math.log(act["volatiles"]["stall"].get("counter", 1), 3))
        if act and "stall" in act["volatiles"] else 0,
        "spikes": int(cond.get("spikes", {}).get("layers", 0)),
        "toxic_spikes": int(cond.get("toxicspikes", {}).get("layers", 0)),
        "stealth_rock": int("stealthrock" in cond), "sticky_web": int("stickyweb" in cond)}
    for c, name in SCREENS.items():
        if c in cond:
            kw[name] = max(1, int(cond[c].get("duration") or 1))
    elapsed = {k: min(max(turn - int(vol[k].get("_start", turn)), 0), 9) for k in ("taunt", "encore") if k in vol}
    if "confusion" in vol:
        c = vol["confusion"]
        elapsed["confusion"] = min(max(int(c.get("_time0") or 0) - int(c.get("time") or 0), 0), 9)
    wish = s["slot"].get("wish")
    if wish and not wish.get("duration") and wish.get("startingTurn") is not None:
        # gen8+ Wish keeps the turn it was used (getOverflowedTurnCount = turn - 1) and heals at the end of the next
        wish = dict(wish, duration=max(0, 2 - (turn - (int(wish["startingTurn"]) + 1))))
    if wish and not wish.get("duration"):
        wish = None
    fs = other["slot"].get("futuremove")  # Future Sight / Doom Desire lands on the target's slot
    fs_src = fs.get("source") if fs else None
    fs_idx = next((i for i, m in enumerate(s["mons"]) if isinstance(fs_src, dict) and fs_src.get("side") is not None
                   and m["species"] == fs_src.get("species")), None)
    b = act["boosts"] if act else {}
    return pe.Side(
        pokemon=mons, active_index=str(idx), side_conditions=pe.SideConditions(**kw), volatile_statuses=names,
        volatile_status_durations=pe.VolatileStatusDurations(yawn=int("YAWN" in names), **elapsed),
        # gen5+ heals the wisher's max HP / 2 (stored); gen4 the max HP / 2 of whoever is in the slot then
        wish=(int(wish.get("duration") or 0), int(wish.get("hp") or mons[idx].maxhp // 2)) if wish else (0, 0),
        future_sight=(int(fs.get("duration") or 0), str(fs_idx)) if fs and fs_idx is not None else (0, "0"),
        substitute_health=int(vol["substitute"].get("hp", 0)) if "substitute" in vol else 0,
        attack_boost=b.get("atk", 0), defense_boost=b.get("def", 0), special_attack_boost=b.get("spa", 0),
        special_defense_boost=b.get("spd", 0), speed_boost=b.get("spe", 0), accuracy_boost=b.get("accuracy", 0),
        evasion_boost=b.get("evasion", 0),
        last_used_move=f"switch:{idx}" if last is None else f"move:{last}",
        **({"force_switch": force, "force_trapped": bool(act and act.get("trapped") and not force)}
           if to_move else {}),
    )


def _weather(st: dict[str, Any]) -> tuple[str, int]:
    f = st["field"]
    w = f["weather"]
    if w not in WEATHER:
        return "none", 0
    if w in PRIMAL:
        return WEATHER[w], -1
    dur = f["weather_state"].get("duration")
    if not dur:  # gen <= 5 ability weather is permanent (no duration)
        return WEATHER[w], -1
    return WEATHER[w], max(1, int(dur))


def convert_full(st: dict[str, Any]) -> pe.State:
    """A dump.js battle state -> the poke-engine State of the true position (side one = st['side'])."""
    me, opp = st["sides"][st["side"]], st["sides"][1 - st["side"]]
    turn = st["turn"]
    weather, wturns = _weather(st)
    f = st["field"]
    terrain = f["terrain"] if f["terrain"] in TERRAIN else "none"
    tturns = max(1, int(f["terrain_state"].get("duration") or 1)) if terrain != "none" else 0
    tr = "trickroom" in f["pseudo"]
    trturns = max(1, int(f["pseudo"]["trickroom"].get("duration") or 1)) if tr else 0
    return pe.State(side_one=_side(me, opp, turn, True), side_two=_side(opp, me, turn, False), weather=weather,
                    weather_turns_remaining=wturns, terrain=terrain, terrain_turns_remaining=tturns, trick_room=tr,
                    trick_room_turns_remaining=trturns)


# ---------------------------------------------------------------- comparison form (parity, features, tests)

MON_FIELDS = ("id", "level", "types", "hp", "maxhp", "ability", "item", "attack", "defense", "special_attack",
              "special_defense", "speed", "status", "sleep_turns", "rest_turns", "weight_kg", "tera_type",
              "terastallized")
COND_FIELDS = ("spikes", "toxic_spikes", "stealth_rock", "sticky_web", "reflect", "light_screen", "aurora_veil",
               "safeguard", "mist", "lucky_chant", "tailwind", "toxic_count", "protect")
BOOSTS = ("attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost", "speed_boost",
          "accuracy_boost", "evasion_boost")
DURATIONS = ("confusion", "encore", "taunt", "yawn", "lockedmove", "slowstart")


def mon_dict(p: pe.Pokemon) -> dict[str, Any]:
    d = {k: getattr(p, k) for k in MON_FIELDS}
    d["types"] = tuple(d["types"])
    d["moves"] = [(m.id, m.pp, m.disabled) for m in p.moves if m.id not in ("none", "")]
    return d


def side_dict(s: pe.Side) -> dict[str, Any]:
    d = {"pokemon": [mon_dict(p) for p in s.pokemon], "active_index": s.active_index,
         # NONE: an engine artefact (from_string of names it does not know)
         "volatile_statuses": sorted(v for v in s.volatile_statuses if v != "NONE"),
         "wish": tuple(s.wish), "future_sight": tuple(s.future_sight),
         "substitute_health": s.substitute_health, "last_used_move": s.last_used_move,
         "force_switch": s.force_switch, "force_trapped": s.force_trapped}
    d.update({k: getattr(s.side_conditions, k) for k in COND_FIELDS})
    d.update({k: getattr(s, k) for k in BOOSTS})
    d.update({f"dur_{k}": getattr(s.volatile_status_durations, k) for k in DURATIONS})
    return d


def state_dict(state: pe.State) -> dict[str, Any]:
    """Every field of a pe.State as plain Python values (what the engine holds, read back from it)."""
    return {"side_one": side_dict(state.side_one), "side_two": side_dict(state.side_two), "weather": state.weather,
            "weather_turns_remaining": state.weather_turns_remaining, "terrain": state.terrain,
            "terrain_turns_remaining": state.terrain_turns_remaining, "trick_room": state.trick_room,
            "trick_room_turns_remaining": state.trick_room_turns_remaining}
