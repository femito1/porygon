"""NNUE gate 1 student inputs (docs/netsearch/NNUE-GATE1-PREREG.md, "Student"): a poke-engine `State` -> fixed arrays.
Only the State is read (gate 2 mirrors this in Rust). Side one = the side to move.

    from porygon.nnue.features import features
    f = features(state)                  # dict: ids int32 [12, 8], mon f32 [12, F_MON], side f32 [2, F_SIDE], field f32 [F_FIELD]

Mons: per side the active mon first, then the bench sorted by base species (the student pools them; a canonical
order makes the arrays comparable between converters). A fainted mon is all zeros except its `fainted` bit.
ids per mon: base species, item, ability, 4 moves (sorted by id), tera type (porygon/data/vocab.json positions + 2;
0 = none, 1 = not in the vocabulary; tera: TYPES index + 1).

Train/serve consistency (main's decision 2026-11-16 on the step-3 parity, recorded in $C/nnue/gate1/notes.log): what
must match between training (convert_full's true State) and serving (search.convert's estimated State) is this
student's INPUT, so the fields search.convert can only estimate are made consistent here:
  - formes -> base species (Morpeko-Hangry, Minior-Meteor, Basculin-Blue-Striped, ...); nothing forme-specific
    beyond what the State's stats / types carry;
  - weather / terrain turns left are not inputs (their types are); trick room: on / off only;
  - volatiles poke-env does not track (LOCKEDMOVE, ...; `UNTRACKED`) are not inputs, nor per-turn ones;
  - side two's HP: training States are built with convert_full(seen_hp=True): the HP fraction side one has SEEN
    (Showdown's percentage; a silent Regenerator heal unseen), rebuilt as search.convert does; serving / leaf
    States are used as they are;
  - screens / tailwind / safeguard / mist: on / off only (their turns left are estimates: Light Clay);
  - substitute HP, PP, last move, weight, force_trapped, Wish's heal amount: not inputs.
`scripts/fastenv/nnue_feature_parity.py` checks this column by column (FEATURE-PARITY.txt).
"""
from __future__ import annotations

import hashlib
import json
import math
from functools import cache
from pathlib import Path

import numpy as np
import poke_engine as pe

from porygon import search

DATA = Path(__file__).resolve().parents[1] / "data"
TYPES = ("NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE", "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC",
         "BUG", "ROCK", "GHOST", "DRAGON", "DARK", "STEEL", "FAIRY", "STELLAR", "TYPELESS")
STATUSES = ("NONE", "BURN", "SLEEP", "FREEZE", "PARALYZE", "POISON", "TOXIC")
WEATHERS = ("none", "sun", "rain", "sand", "hail", "snow", "harshsun", "heavyrain")
TERRAINS = ("none", "grassyterrain", "electricterrain", "mistyterrain", "psychicterrain")
BOOSTS = ("attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost", "speed_boost",
          "accuracy_boost", "evasion_boost")
# side conditions (engine names) and their scale; 0 = on / off (screens: turns left are estimates, Light Clay)
CONDS = (("spikes", 3), ("toxic_spikes", 2), ("stealth_rock", 1), ("sticky_web", 1), ("reflect", 0),
         ("light_screen", 0), ("aurora_veil", 0), ("safeguard", 0), ("mist", 0), ("lucky_chant", 0), ("tailwind", 0),
         ("protect", 3))
SKIP = {"FLINCH", "PROTECT", "ENDURE", "NONE"}  # per-turn (search.SKIP_EFFECTS) / engine artefact
# volatiles the simulator holds but poke-env never shows (search.convert cannot know them), or the reverse (gen9's
# Slow Start is an ability counter in the simulator, a volatile in poke-env): not inputs (FEATURE-PARITY.txt)
UNTRACKED = {"LOCKEDMOVE", "SLOWSTART"}
ID_FIELDS = ("species", "item", "ability", "move1", "move2", "move3", "move4", "tera")
MON_COLS = (("level", "hp") + tuple(f"status_{s.lower()}" for s in STATUSES) + ("sleep_turns", "rest_turns")
            + ("atk", "def", "spa", "spd", "spe") + tuple(f"type_{t.lower()}" for t in TYPES) + ("type_other",)
            + ("terastallized", "fainted", "active"))
F_MON = len(MON_COLS)


@cache
def vocab() -> dict[str, dict[str, int]]:
    """Category -> {UPPER id: index}; index = position in porygon/data/vocab.json + 2 (0 none, 1 unknown)."""
    raw = json.loads((DATA / "vocab.json").read_text())
    return {k: {x.upper(): i + 2 for i, x in enumerate(v)} for k, v in raw.items()}


@cache
def base_species(engine_id: str) -> str:
    """The engine's species id -> its base species (UPPER showdown id): formes share their base's embedding."""
    sid = engine_id.lower()
    e = search.GEN.pokedex.get(sid)
    return search.norm(e.get("baseSpecies", sid)).upper() if e else engine_id.upper()


def _id(cat: str, name: str) -> int:
    n = str(name).upper()
    if n in ("", "NONE"):
        return 0
    return vocab()[cat].get(n, 1)


def _move_id(name: str) -> int:
    n = str(name).upper()
    if n.startswith("HIDDENPOWER"):
        n = "HIDDENPOWER"
    return _id("moves", n)


@cache
def volatiles() -> tuple[str, ...]:
    """The volatile bits, fixed order: names search.convert can produce (poke-env effects, MUSTRECHARGE, two-turn
    moves) that the engine knows, minus per-turn and untracked ones."""
    from porygon.nnue.convert_full import known_volatile
    from poke_env.battle import Effect
    cand = {e.name.replace("_", "") for e in Effect} | {"MUSTRECHARGE"} | {m.upper() for m in search.CHARGE_MOVES}
    return tuple(sorted(v for v in cand if v not in SKIP | UNTRACKED and known_volatile(v)))


def side_cols() -> tuple[str, ...]:
    return (tuple(b.replace("_boost", "") for b in BOOSTS) + tuple(f"vol_{v.lower()}" for v in volatiles())
            + tuple(c for c, _ in CONDS) + ("toxic_count", "wish_turns", "future_sight_turns"))


FIELD_COLS = tuple(f"weather_{w}" for w in WEATHERS) + tuple(f"terrain_{t}" for t in TERRAINS) + ("trick_room",)
F_FIELD = len(FIELD_COLS)


def _mon(p, active: bool, gen: int, ids: np.ndarray, num: np.ndarray) -> None:
    hp, maxhp = p.hp, max(p.maxhp, 1)
    if hp <= 0 or str(p.id).upper() in ("", "NONE"):
        num[MON_COLS.index("fainted")] = 1.0
        return
    ids[0] = _id("species", base_species(str(p.id)))
    ids[1] = _id("items", p.item)
    ids[2] = _id("abilities", p.ability)
    mv = sorted(_move_id(m.id) for m in p.moves if str(m.id).upper() not in ("", "NONE"))[:4]
    ids[3:3 + len(mv)] = mv
    tt = str(p.tera_type).upper() if gen >= 9 else ""
    ids[7] = TYPES.index(tt) + 1 if tt in TYPES else (len(TYPES) + 1 if tt not in ("", "NONE") else 0)
    num[0] = p.level / 100
    num[1] = hp / maxhp
    st = str(p.status).upper()
    num[2 + (STATUSES.index(st) if st in STATUSES else 0)] = 1.0
    k = 2 + len(STATUSES)
    num[k] = p.sleep_turns / 3
    num[k + 1] = p.rest_turns / 3
    k += 2
    for j, s in enumerate((p.attack, p.defense, p.special_attack, p.special_defense, p.speed)):
        num[k + j] = s / 100
    k += 5
    for t in p.types:
        t = str(t).upper()
        if t == "TYPELESS":
            continue
        num[k + (TYPES.index(t) if t in TYPES else len(TYPES))] = 1.0
    k += len(TYPES) + 1
    num[k] = float(bool(p.terastallized))
    num[k + 2] = float(active)


def _side(s, gen: int, ids: np.ndarray, num: np.ndarray, sv: np.ndarray) -> None:
    mons = list(s.pokemon)
    ai = int(s.active_index)
    order = [ai] + sorted((i for i in range(len(mons)) if i != ai), key=lambda i: (
        mons[i].hp <= 0, base_species(str(mons[i].id)) if mons[i].hp > 0 else ""))
    for slot, i in enumerate(order[:6]):
        _mon(mons[i], slot == 0, gen, ids[slot], num[slot])
    dead = mons[ai].hp <= 0  # a fainted active: its boosts / volatiles / counters reset at the switch-in
    if not dead:
        for j, b in enumerate(BOOSTS):
            sv[j] = getattr(s, b) / 6
    k = len(BOOSTS)
    vols = volatiles()
    if not dead:
        have = {str(v).upper() for v in s.volatile_statuses}
        for j, v in enumerate(vols):
            if v in have:
                sv[k + j] = 1.0
    k += len(vols)
    sc = s.side_conditions
    for j, (c, scale) in enumerate(CONDS):
        if c == "protect" and dead:
            continue
        v = getattr(sc, c)
        sv[k + j] = v / scale if scale else float(v > 0)
    k += len(CONDS)
    sv[k] = 0.0 if dead else sc.toxic_count / 16
    sv[k + 1] = s.wish[0] / 2
    sv[k + 2] = s.future_sight[0] / 3


def features(state: pe.State) -> dict[str, np.ndarray]:
    gen = search.GEN_NUM
    ids = np.zeros((12, 8), np.int32)
    mon = np.zeros((12, F_MON), np.float32)
    side = np.zeros((2, len(side_cols())), np.float32)
    field = np.zeros(F_FIELD, np.float32)
    _side(state.side_one, gen, ids[:6], mon[:6], side[0])
    _side(state.side_two, gen, ids[6:], mon[6:], side[1])
    w = str(state.weather).lower()
    field[WEATHERS.index(w) if w in WEATHERS else 0] = 1.0
    t = str(state.terrain).lower()
    field[len(WEATHERS) + (TERRAINS.index(t) if t in TERRAINS else 0)] = 1.0
    field[-1] = float(bool(state.trick_room))
    return {"ids": ids, "mon": mon, "side": side, "field": field}


def column_names() -> list[str]:
    """Names of the flattened vector `flat(features(...))` (parity reports)."""
    names = []
    for s in ("s1", "s2"):
        for m in range(6):
            names += [f"{s}.mon{m}.{f}" for f in ID_FIELDS]
    for s in ("s1", "s2"):
        for m in range(6):
            names += [f"{s}.mon{m}.{c}" for c in MON_COLS]
    for s in ("s1", "s2"):
        names += [f"{s}.{c}" for c in side_cols()]
    return names + [f"field.{c}" for c in FIELD_COLS]


def flat(f: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([f["ids"].ravel().astype(np.float64), f["mon"].ravel(), f["side"].ravel(), f["field"]])


def spec() -> dict:
    """What a student checkpoint needs to know about its inputs (and a hash of it)."""
    v = vocab()
    d = {"gen": search.GEN_NUM, "F_MON": F_MON, "F_SIDE": len(side_cols()), "F_FIELD": F_FIELD,
         "n_species": len(v["species"]) + 2, "n_items": len(v["items"]) + 2, "n_abilities": len(v["abilities"]) + 2,
         "n_moves": len(v["moves"]) + 2, "n_tera": len(TYPES) + 2, "volatiles": list(volatiles()),
         "mon_cols": list(MON_COLS), "side_cols": list(side_cols()), "field_cols": list(FIELD_COLS)}
    d["hash"] = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
    return d
