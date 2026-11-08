"""Reference encoder for observation spec v2 (`porygon.v2.spec`): poke-env battle -> dict of fixed-shape arrays.

Generation-parametric (battle.gen: 4 or 9), independent of PORYGON_GEN. What differs from v1 (porygon.encoder):
- structured fields instead of one flat vector; ids are int16, not floats;
- static data (base stats, move power / type / category / flags, ...) is NOT in the observation: the model looks
  it up from the ids (porygon.v2.tables), so implementations only need ids and dynamic state;
- the opponent's unknown set is a posterior over the random-battle set library, filtered by what was revealed
  (moves, item, ability, tera type): candidate moves / items / abilities / tera types with marginal probabilities,
  and stats expected over the candidate sets (v2.0 uses no speed / damage evidence: see CONTRACTS.md);
- raw stats, HP points and boost stages, so the model can compute damage itself;
- `aux_targets` gives training-time targets: the opponent's true set and the map of its actions to our frame.

    obs = encode(battle)                          # dict name -> np.ndarray, shapes in spec.fields(battle.gen)
    aux = aux_targets(battle, opp_battle, obs)    # self-play only: opp_battle = the other side's poke-env battle
"""

import gzip
import json
import math
from functools import cache, lru_cache

import numpy as np
from poke_env.battle import AbstractBattle, Field, Move, Pokemon, SideCondition
from poke_env.data import GenData
from poke_env.environment import SinglesEnv

from porygon.encoder import EFFECTS, HAZARDS, SCREENS, elapsed
from porygon.inference import Set, _spread, matches, norm
from porygon.v2.spec import (
    DATA, INDEX, KA, KI, KM, MON_NUM, N_EFFECTS, N_MONS, N_TYPES, PAD, STATS, STATUSES, TERA_OFFSET, TERRAINS,
    TYPE_ID, UNK, WEATHERS, BOOSTS, aux_fields, fields,
)

assert len(EFFECTS) == N_EFFECTS
COL = {name: i for i, name in enumerate(MON_NUM)}
EFFECT0 = COL["effect_0"]
UNKNOWN_ITEM = GenData.UNKNOWN_ITEM
TERRAIN_FIELDS = {Field.ELECTRIC_TERRAIN: 0, Field.GRASSY_TERRAIN: 1, Field.MISTY_TERRAIN: 2,
                  Field.PSYCHIC_TERRAIN: 3}
WEATHER_INDEX = {w: i for i, w in enumerate(WEATHERS)}


# ---------------------------------------------------------------- ids

def vid(kind: str, key: str | None) -> int:
    """Vocabulary id (v1's rule): None -> UNK, "" -> PAD, hidden power variants -> hiddenpower."""
    if key is None:
        return UNK
    if key == "":
        return PAD
    table = INDEX[kind]
    if key in table:
        return table[key]
    if kind == "moves" and key.startswith("hiddenpower"):
        return table["hiddenpower"]
    return table.get(key.rstrip("0123456789"), UNK)


def type_id(t) -> int:
    """PokemonType or showdown type string -> type id (0 for none / ???)."""
    if t is None:
        return 0
    name = t.name.lower() if hasattr(t, "name") else str(t).lower()
    return TYPE_ID.get(name, 0)


@cache
def gendata(gen: int) -> GenData:
    return GenData.from_gen(gen)


@cache
def move_type_of(gen: int, move_id: str) -> int:
    """Type id of a move as a set lists it: hidden power types from the id (hiddenpowerfire, hiddenpowerfire60)."""
    if move_id.startswith("hiddenpower") and len(move_id) > len("hiddenpower"):
        t = move_id[len("hiddenpower"):].rstrip("0123456789")
        if t in TYPE_ID:
            return TYPE_ID[t]
    entry = gendata(gen).moves.get(move_id) or gendata(gen).moves.get(move_id.rstrip("0123456789"))
    return TYPE_ID.get(entry["type"].lower(), 0) if entry else 0


def species_key(mon: Pokemon) -> str:
    return mon.species if mon.species in INDEX["species"] else mon.base_species


# ---------------------------------------------------------------- set posterior (revealed constraints only)

@cache
def library(gen: int) -> dict[str, tuple[int, list[Set]]]:
    """species -> (level, sets) of gen{gen}randombattle (same file and parsing as porygon.inference)."""
    raw = json.loads(gzip.open(DATA / f"gen{gen}randombattle_library.json.gz").read())["species"]

    def make(e: list) -> Set:
        moves, ability, item, role, evs, ivs, *tera, n, n_lead = e
        return Set(tuple(moves.split(",")), ability, item, role, _spread(evs, 85), _spread(ivs, 31), n, n_lead,
                   tera[0] if tera else "")
    return {sp: (v["level"], [make(e) for e in v["sets"]]) for sp, v in raw.items()}


def set_species(gen: int, species: str) -> str:
    lib, dex = library(gen), gendata(gen).pokedex
    if species not in lib and species in dex:
        return norm(dex[species].get("baseSpecies", species))
    return species


def calc_stats(gen: int, species: str, level: int, evs: dict, ivs: dict) -> np.ndarray:
    base = gendata(gen).pokedex[species]["baseStats"]
    hp = (2 * base["hp"] + ivs["hp"] + evs["hp"] // 4) * level // 100 + level + 10
    rest = [(2 * base[s] + ivs[s] + evs[s] // 4) * level // 100 + 5 for s in STATS[1:]]
    return np.array([hp, *rest], dtype=np.float64)


def estimate_stats(gen: int, mon: Pokemon, level: int) -> np.ndarray:
    """No library entry: 85 EVs, 31 IVs, neutral nature from the mon's (possibly temporary) base stats."""
    b = mon.base_stats
    hp = (2 * b["hp"] + 31 + 21) * level // 100 + level + 10
    return np.array([hp] + [(2 * b[s] + 31 + 21) * level // 100 + 5 for s in STATS[1:]], dtype=np.float64)


def consistent_revealed(sets: list[Set], revealed: tuple[str, ...], ability: str | None, item: str | None,
                        tera: str | None) -> list[Set]:
    """porygon.inference.consistent without battle evidence (no speed / damage / item-behaviour checks): the sets
    with every revealed move, relaxing item, then ability, then tera type, then the moves the library knows."""
    def moves_ok(s: Set, moves) -> bool:
        return all(any(matches(r, m) for m in s.moves) for r in moves)

    for ok in (lambda s: moves_ok(s, revealed) and (tera is None or s.tera == tera)
               and (ability is None or s.ability == ability) and (item is None or s.item == item),
               lambda s: moves_ok(s, revealed) and (tera is None or s.tera == tera)
               and (ability is None or s.ability == ability),
               lambda s: moves_ok(s, revealed) and (tera is None or s.tera == tera),
               lambda s: moves_ok(s, revealed)):
        found = [s for s in sets if ok(s)]
        if found:
            return found
    known = [r for r in revealed if any(matches(r, m) for s in sets for m in s.moves)]
    return [s for s in sets if moves_ok(s, known)] or sets


@lru_cache(maxsize=1 << 16)
def posterior(gen: int, species: str, level: int, revealed: tuple[str, ...], ability: str | None,
              item: str | None, tera: str | None):
    """Marginals of the library sets consistent with the revealed moves / ability / item / tera type
    (`consistent_revealed`), weighted by how often the generator makes them.
    Returns (move probs, item probs, ability probs, tera probs, expected stats) or None without a library entry."""
    sp = set_species(gen, species)
    lvl, sets = library(gen).get(sp, (None, []))
    if not sets:
        return None
    cands = consistent_revealed(sets, revealed, ability, item, tera)
    w = np.array([s.n for s in cands], dtype=np.float64)
    w /= w.sum()
    moves, items, abilities, teras = {}, {}, {}, {}
    st = np.zeros(6)
    dex_ok = sp in gendata(gen).pokedex
    for s, p in zip(cands, w):
        for m in s.moves:
            moves[m] = moves.get(m, 0.0) + p
        items[s.item] = items.get(s.item, 0.0) + p
        abilities[s.ability] = abilities.get(s.ability, 0.0) + p
        if s.tera:
            teras[s.tera] = teras.get(s.tera, 0.0) + p
        if dex_ok:
            st += p * calc_stats(gen, sp, level, s.evs, s.ivs)

    def ranked(d: dict) -> tuple:
        return tuple(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))
    return ranked(moves), ranked(items), ranked(abilities), ranked(teras), (st if dex_ok else None)


# ---------------------------------------------------------------- encoder

def _status_counter(mon: Pokemon) -> float:
    return min(mon.status_counter, 10) / 10


def _mon_common(out: dict, i: int, mon: Pokemon, battle: AbstractBattle, mine: bool) -> None:
    out["species"][i] = vid("species", species_key(mon))
    types = mon.base_types
    out["types"][i, 0] = type_id(types[0]) if types else 0
    out["types"][i, 1] = type_id(types[1]) if len(types) > 1 else 0
    out["boosts"][i] = [mon.boosts.get(b, 0) for b in BOOSTS]
    n = out["mon_num"][i]
    n[COL["present"]] = 1.0
    n[COL["mine"]] = float(mine)
    n[COL["active"]] = float(mon.active)
    n[COL["fainted"]] = float(mon.fainted)
    n[COL["revealed"]] = 1.0
    n[COL["hp_frac"]] = mon.current_hp_fraction
    if mon.status is not None and mon.status.name.lower() in STATUSES:
        n[COL["status_" + mon.status.name.lower()]] = 1.0
    n[COL["status_counter"]] = _status_counter(mon)
    n[COL["terastallized"]] = float(mon.is_terastallized)
    n[COL["first_turn"]] = float(mon.first_turn)
    n[COL["protect_counter"]] = mon.protect_counter / 3
    n[COL["must_recharge"]] = float(mon.must_recharge)
    n[COL["preparing"]] = float(bool(mon.preparing))
    n[COL["trapped"]] = float(mine and mon.active and bool(battle.trapped))
    n[COL["log_weight"]] = math.log10(max(mon.weight, 0.1)) / 3
    for k, e in enumerate(EFFECTS):
        if e in mon.effects:
            n[EFFECT0 + k] = 1.0


def _encode_own(out: dict, i: int, mon: Pokemon, battle: AbstractBattle) -> None:
    gen = battle.gen
    _mon_common(out, i, mon, battle, True)
    n = out["mon_num"][i]
    level = mon.level or 80
    n[COL["level"]] = level / 100
    if gen >= 9 and mon.tera_type is not None:
        t = type_id(mon.tera_type)
        out["tera"][i] = t
        out["tera_p"][i, t] = 1.0
    item = mon.item if mon.item != UNKNOWN_ITEM else None
    out["item"][i, 0] = vid("items", item)
    out["item_p"][i, 0] = 1.0
    n[COL["item_known"]] = 1.0
    n[COL["item_none"]] = float(item == "")
    out["ability"][i, 0] = vid("abilities", mon.ability)
    out["ability_p"][i, 0] = 1.0
    n[COL["ability_known"]] = 1.0
    moves = list(mon.moves.values())[:4]
    avail = {m.id for m in battle.available_moves} if mon.active else set()
    for j, mv in enumerate(moves):
        out["moves"][i, j] = vid("moves", mv.id)
        out["move_p"][i, j] = 1.0
        out["move_type"][i, j] = move_type_of(gen, mv.id) if mv.id.startswith("hiddenpower") else type_id(mv.type)
        out["move_num"][i, j] = [1.0, mv.current_pp / max(mv.max_pp, 1), float(mv.id in avail)]
    n[COL["n_moves_known"]] = len(moves) / 4
    stats = mon.stats or {}
    if all(stats.get(s) for s in STATS[1:]) and mon.max_hp:
        out["stats"][i] = [mon.max_hp] + [stats[s] for s in STATS[1:]]
    else:
        out["stats"][i] = estimate_stats(gen, mon, level)
    out["hp"][i] = [mon.current_hp or 0, mon.max_hp or out["stats"][i, 0]]


def _encode_opp(out: dict, i: int, mon: Pokemon, battle: AbstractBattle) -> None:
    gen = battle.gen
    _mon_common(out, i, mon, battle, False)
    n = out["mon_num"][i]
    sp = set_species(gen, mon.species)
    level = mon.level or library(gen).get(sp, (80, []))[0] or 80
    n[COL["level"]] = level / 100
    revealed = tuple(m for m in mon.moves if norm(m) != "struggle")
    ability = mon.ability
    if ability is None and len(mon.possible_abilities) == 1:
        ability = norm(mon.possible_abilities[0])
    item = None if mon.item in (UNKNOWN_ITEM, None) else mon.item
    tera = None
    if gen >= 9 and mon.is_terastallized and mon.tera_type is not None:
        tera = mon.tera_type.name.lower()
        t = type_id(mon.tera_type)
        out["tera"][i] = t
        out["tera_p"][i, t] = 1.0
    post = posterior(gen, mon.species, level, revealed, ability, item, tera)
    pm, pi, pa, pt, pstats = post if post else ((), (), (), (), None)

    # items / abilities: known -> one slot with p = 1, else the posterior marginals
    if item is not None:
        out["item"][i, 0], out["item_p"][i, 0] = vid("items", item), 1.0
        n[COL["item_known"]] = 1.0
        n[COL["item_none"]] = float(item == "")
    elif pi:
        for k, (it, p) in enumerate(pi[:KI]):
            out["item"][i, k], out["item_p"][i, k] = vid("items", it), p
    else:
        out["item"][i, 0], out["item_p"][i, 0] = UNK, 1.0
    if ability is not None:
        out["ability"][i, 0], out["ability_p"][i, 0] = vid("abilities", ability), 1.0
        n[COL["ability_known"]] = 1.0
    elif pa:
        for k, (ab, p) in enumerate(pa[:KA]):
            out["ability"][i, k], out["ability_p"][i, k] = vid("abilities", ab), p
    else:
        poss = [norm(a) for a in mon.possible_abilities][:KA] or [None]
        for k, ab in enumerate(poss):
            out["ability"][i, k], out["ability_p"][i, k] = vid("abilities", ab), 1.0 / len(poss)
    if tera is None and gen >= 9 and pt:
        for t, p in pt:
            out["tera_p"][i, TYPE_ID.get(t, 0)] += p
        out["tera_p"][i, 0] = 0.0

    # moves: revealed first (order of revelation), then candidates by marginal probability
    j = 0
    used: set[str] = set()
    for mid in revealed[:KM]:
        mv = mon.moves[mid]
        t = type_id(mv.type)
        if mid == "hiddenpower":  # the opponent's hidden power shows without its type: most likely candidate's
            hp = [m for m, _ in pm if m.startswith("hiddenpower")]
            if hp:
                t = move_type_of(gen, hp[0])
                used.update(hp)
        used.add(mid)
        out["moves"][i, j], out["move_p"][i, j], out["move_type"][i, j] = vid("moves", mid), 1.0, t
        out["move_num"][i, j] = [1.0, mv.current_pp / max(mv.max_pp, 1), 0.0]
        j += 1
    for mid, p in pm:
        if j >= KM:
            break
        if mid in used:
            continue
        out["moves"][i, j], out["move_p"][i, j] = vid("moves", mid), p
        out["move_type"][i, j] = move_type_of(gen, mid)
        out["move_num"][i, j] = [0.0, 1.0, 0.0]
        j += 1
    n[COL["n_moves_known"]] = min(len(revealed), 4) / 4

    stats = pstats if pstats is not None else estimate_stats(gen, mon, level)
    out["stats"][i] = stats
    out["hp"][i] = [mon.current_hp_fraction * stats[0], stats[0]]


def _side(conditions: dict, turn: int) -> list[float]:
    screens = [float(c in conditions) for c in SCREENS]
    ages = [elapsed(conditions[c], turn) if c in conditions else 0.0 for c in SCREENS]
    hazards = [min(conditions.get(c, 0), m) / m for c, m in HAZARDS.items()]
    return screens + ages + hazards


def _field(battle: AbstractBattle) -> np.ndarray:
    turn = battle.turn
    f = []
    w_onehot = [0.0] * len(WEATHERS)
    w_elapsed = 0.0
    for w, start in battle.weather.items():
        name = "snow" if w.name == "SNOWSCAPE" else w.name.lower()
        if name in WEATHER_INDEX:
            w_onehot[WEATHER_INDEX[name]] = 1.0
            w_elapsed = elapsed(start, turn)
    t_onehot = [0.0] * len(TERRAINS)
    t_elapsed = 0.0
    for fl, start in battle.fields.items():
        if fl in TERRAIN_FIELDS:
            t_onehot[TERRAIN_FIELDS[fl]] = 1.0
            t_elapsed = elapsed(start, turn)
    tr = Field.TRICK_ROOM in battle.fields
    f = [
        *w_onehot, w_elapsed, *t_onehot, t_elapsed,
        float(tr), elapsed(battle.fields[Field.TRICK_ROOM], turn) if tr else 0.0,
        float(Field.GRAVITY in battle.fields),
        min(turn, 100) / 100,
        (6 - len(battle.opponent_team)) / 6,
        sum(not m.fainted for m in battle.team.values()) / 6,
        (6 - sum(m.fainted for m in battle.opponent_team.values())) / 6,
        float(any(battle.force_switch) if isinstance(battle.force_switch, list) else battle.force_switch),
        float(bool(battle.trapped)), float(bool(battle.maybe_trapped)),
        float(bool(getattr(battle, "can_tera", False))), float(bool(battle.used_tera)),
        float(bool(battle.opponent_used_tera)),
        float(battle.wait), float(battle.active_pokemon is None),
        len(battle.available_switches) / 5, len(battle.available_moves) / 4,
    ]
    return np.asarray(f, dtype=np.float32)


def empty(gen: int, aux: bool = False) -> dict[str, np.ndarray]:
    spec = aux_fields(gen) if aux else fields(gen)
    return {k: np.zeros(shape, dtype) for k, (shape, dtype) in spec.items()}


def encode(battle: AbstractBattle) -> dict[str, np.ndarray]:
    gen = battle.gen
    out = empty(gen)
    out["field_num"][:] = _field(battle)
    out["side_num"][0] = _side(battle.side_conditions, battle.turn)
    out["side_num"][1] = _side(battle.opponent_side_conditions, battle.turn)
    for i, mon in enumerate(list(battle.team.values())[:6]):
        _encode_own(out, i, mon, battle)
    for i, mon in enumerate(list(battle.opponent_team.values())[:6]):
        _encode_opp(out, 6 + i, mon, battle)
    mask = SinglesEnv.get_action_mask(battle)
    out["action_mask"][: len(mask)] = mask
    return out


# ---------------------------------------------------------------- training targets

def _same_move(a: str, b: str) -> bool:
    return a == b or (a.startswith("hiddenpower") and b.startswith("hiddenpower"))


def aux_targets(battle: AbstractBattle, opp_battle: AbstractBattle | None, obs: dict[str, np.ndarray]
                ) -> dict[str, np.ndarray]:
    """Targets from the opponent's own view of the same battle (self-play): its true sets for our opponent slots,
    the species we have not seen, and where each of its actions lands in our frame (spec.N_OPP_CLASSES)."""
    gen = battle.gen
    out = empty(gen, aux=True)
    for k in out:
        if k != "aux_move_in_set":
            out[k][:] = -1
    out["aux_move_in_set"][:] = -1.0
    out["aux_opp_target"][:] = 0.0  # filled by the trainer from the opponent's policy (aux_opp_class)
    if opp_battle is None:
        return out
    theirs = opp_battle.team  # key "p2: Name" -> Pokemon (full info from its requests)
    seen = list(battle.opponent_team)
    for k, key in enumerate(seen[:6]):
        true = theirs.get(key)
        if true is None:  # identifiers differ (e.g. illusion): match by species
            true = next((m for m in theirs.values() if m.species == battle.opponent_team[key].species), None)
        if true is None:
            continue
        out["aux_item"][k] = vid("items", true.item if true.item != UNKNOWN_ITEM else None)
        out["aux_ability"][k] = vid("abilities", true.ability)
        if gen >= 9 and true.tera_type is not None:
            out["aux_tera"][k] = type_id(true.tera_type)
        true_moves = list(true.moves)[:4]
        for j, m in enumerate(true_moves):
            out["aux_moves"][k, j] = vid("moves", m)
        ids = obs["moves"][6 + k]
        for j in range(KM):
            if ids[j] == PAD:
                continue
            # vocab id -> any true move with that id (hidden power types collapse to one id)
            out["aux_move_in_set"][k, j] = float(any(vid("moves", m) == ids[j] for m in true_moves))
    unseen = [m for key, m in theirs.items() if key not in battle.opponent_team]
    for k, m in enumerate(unseen[:6]):
        out["aux_unseen"][k] = vid("species", species_key(m))

    # opponent's action a (its frame) -> class in our frame
    mask = SinglesEnv.get_action_mask(opp_battle)
    team = list(opp_battle.team.items())
    opp_active_slot = next((k for k, key in enumerate(seen[:6]) if battle.opponent_team[key].active), None)
    act = opp_battle.active_pokemon
    known = list(act.moves.values())[:4] if act is not None else []
    for a in ([] if opp_battle.wait else np.flatnonzero(mask)):
        a = int(a)
        if a < 6:
            key = team[a][0] if a < len(team) else None
            out["aux_opp_class"][a] = seen.index(key) if key in seen[:6] else 6
        elif opp_active_slot is not None and (6 <= a < 10 or TERA_OFFSET <= a < TERA_OFFSET + 4):
            j = (a - 6) % 4
            if j >= len(known):
                continue
            mid = vid("moves", known[j].id)
            slots = np.flatnonzero(obs["moves"][6 + opp_active_slot] == mid)
            if len(slots):
                out["aux_opp_class"][a] = 7 + int(slots[0]) + (KM if a >= TERA_OFFSET else 0)
    return out


def opp_action_class(battle: AbstractBattle, obs: dict[str, np.ndarray], kind: str, name: str,
                     tera: bool = False) -> int:
    """Class (spec.N_OPP_CLASSES, our frame) of an observed opponent action: kind "switch" with the species /
    identifier it switched to, or "move" with the move id; -1 when it cannot be placed."""
    seen = list(battle.opponent_team)
    if kind == "switch":
        for k, key in enumerate(seen[:6]):
            mon = battle.opponent_team[key]
            if name in (key, mon.species, species_key(mon), norm(key.split(": ", 1)[-1])):
                return k
        return 6
    slot = next((k for k, key in enumerate(seen[:6]) if battle.opponent_team[key].active), None)
    if slot is None:
        return -1
    hits = np.flatnonzero(obs["moves"][6 + slot] == vid("moves", norm(name)))
    return 7 + int(hits[0]) + (KM if tera else 0) if len(hits) else -1


def aux_targets_from_team(battle: AbstractBattle, true_team: list[dict], obs: dict[str, np.ndarray],
                          opp_action: tuple[str, str, bool] | None = None) -> dict[str, np.ndarray]:
    """`aux_targets` for logged / replayed games without the opponent's battle object (data stream, F2 records):
    `true_team` = the opponent's sets as dicts with species, item, ability, moves, tera_type (F2 `true_team`);
    `opp_action` = the opponent's actual action at this decision (kind "move"/"switch", move id or species,
    tera) -> `aux_opp_target` one-hot. `aux_opp_class` stays -1 (the opponent's legal set is unknown)."""
    gen = battle.gen
    out = empty(gen, aux=True)
    for k in out:
        out[k][:] = -1
    out["aux_move_in_set"][:] = -1.0
    out["aux_opp_target"][:] = 0.0
    sets = {}
    for t in true_team:
        sp = norm(t.get("species", ""))
        sets[sp] = t
        sets.setdefault(set_species(gen, sp), t)
    seen = list(battle.opponent_team)
    matched = set()
    for k, key in enumerate(seen[:6]):
        mon = battle.opponent_team[key]
        t = next((sets[s] for s in (mon.species, species_key(mon), set_species(gen, mon.species)) if s in sets), None)
        if t is None:
            continue
        matched.add(id(t))
        out["aux_item"][k] = vid("items", norm(t.get("item") or ""))
        out["aux_ability"][k] = vid("abilities", norm(t.get("ability") or "") or None)
        if gen >= 9 and t.get("tera_type"):
            out["aux_tera"][k] = TYPE_ID.get(norm(t["tera_type"]), -1)
        true_moves = [norm(m) for m in t.get("moves", [])][:4]
        for j, m in enumerate(true_moves):
            out["aux_moves"][k, j] = vid("moves", m)
        ids = obs["moves"][6 + k]
        for j in range(KM):
            if ids[j] != PAD:
                out["aux_move_in_set"][k, j] = float(any(vid("moves", m) == ids[j] for m in true_moves))
    unseen = [t for t in true_team if id(t) not in matched]
    for k, t in enumerate(unseen[:6]):
        sp = norm(t.get("species", ""))
        out["aux_unseen"][k] = vid("species", sp if sp in INDEX["species"] else set_species(gen, sp))
    if opp_action is not None:
        c = opp_action_class(battle, obs, *opp_action)
        if c >= 0:
            out["aux_opp_target"][c] = 1.0
    return out
