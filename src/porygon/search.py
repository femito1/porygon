"""Decision-time search on top of the PPO network (Wang 2024, ch. 5 style), using poke-engine.

Each decision we
  1. sample `samples` determinizations of the battle: our side is fully known; each opponent mon gets a whole
     gen4 random-battle set (moves, item, ability, EVs / IVs) consistent with everything the battle has shown
     (`porygon.inference`), and unrevealed mons follow the gen4 random team rules (`sample_species`);
  2. run poke-engine's simultaneous-move MCTS on every sample for `ms` milliseconds (in parallel
     worker processes, `porygon.engine_worker`);
  3. score each legal action as  mean visit share over samples + prior_weight * network prob,
     optionally only among the network's top-k actions, and play the argmax.
With value_weight > 0 the visit shares are mixed with a one-turn lookahead scored by the network's value
(`porygon.lookahead`), run on the same determinizations while the engine searches:
  share = (1 - value_weight) * visit share + value_weight * lookahead equilibrium probability.
Any failure (conversion, engine panic, unmappable choice) falls back to the network's argmax.

Tracked beyond what poke-env exposes (`History`, read from the battle log): Wish and Future Sight / Doom
Desire timers, Rest's fixed sleep, both sides' last move (choice lock, encore, fake out), and the evidence about
the opponent's sets (`inference.Evidence`).

Taunt / encore / confusion pass poke-env's elapsed counts; with the patched engine (scripts/build_engine.sh) they
end after their random gen4 durations instead of lasting forever. Substitute HP is 1/4 max HP minus an estimate
of every hit it took (`substitute_left`).

Approximations / ignored (documented, not bugs): opponent PP = what poke-env tracks; hidden power is always
70 BP and its type for the opponent comes from the sampled set; partial-trap / disable timers are not
transferred; weather from abilities is permanent (gen4), other weather lasts 5 turns from when we first saw it.

    python -m porygon.search runs/<run>/best_model.zip --opponent heuristic --n-battles 100
    python -m porygon.search runs/<run>/best_model.zip --opponent search --opp-ms 300 --opp-prior-weight 0.5
    python -m porygon.search runs/new.zip --opponent search --opp-model runs/old.zip   # same search, other prior

Defaults (1000 ms, 4 samples, prior_weight 0.25, no top-k) come from head-to-head tuning: prior 0.25 beat 0.5
(which beat 1.0 and top-k 3), 0.1 and 8 samples were no better, 1000 ms beat 300 ms.
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
import time
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import poke_engine as pe
import torch
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle, Battle, Effect, Field, Pokemon, PokemonType, SideCondition, Status, Weather
from poke_env.data import GenData
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, Player
from stable_baselines3.common.policies import ActorCriticPolicy

from porygon import inference
from porygon.encoder import embed_battle
from porygon.engine_worker import EnginePool
from porygon.env import PolicyPlayer, local_server
from porygon.evaluate import BASELINES
from porygon.model import load_policy

log = logging.getLogger("porygon.search")
GEN = GenData.from_gen(4)

STATUS = {Status.BRN: "burn", Status.SLP: "sleep", Status.FRZ: "freeze", Status.PAR: "paralyze",
          Status.PSN: "poison", Status.TOX: "toxic"}
WEATHER = {Weather.SUNNYDAY: "sun", Weather.RAINDANCE: "rain", Weather.SANDSTORM: "sand", Weather.HAIL: "hail"}
WEATHER_ABILITY = {Weather.SUNNYDAY: "drought", Weather.RAINDANCE: "drizzle",
                   Weather.SANDSTORM: "sandstream", Weather.HAIL: "snowwarning"}
SCREEN_TURNS = {SideCondition.REFLECT: ("reflect", 5), SideCondition.LIGHT_SCREEN: ("light_screen", 5),
                SideCondition.SAFEGUARD: ("safeguard", 5), SideCondition.MIST: ("mist", 5),
                SideCondition.LUCKY_CHANT: ("lucky_chant", 5), SideCondition.TAILWIND: ("tailwind", 3)}
SKIP_EFFECTS = {Effect.FLINCH, Effect.PROTECT, Effect.ENDURE}  # per-turn effects poke-env may still hold
TIMED_EFFECTS = {"taunt": Effect.TAUNT, "encore": Effect.ENCORE, "confusion": Effect.CONFUSION}
# two-turn moves; the engine marks the charging turn with a volatile of the same name
CHARGE_MOVES = {"solarbeam", "fly", "dig", "dive", "bounce", "skyattack", "razorwind", "skullbash", "shadowforce"}
DEFAULT_SET = inference.Set(("tackle",), "none", "none", "", dict.fromkeys(inference.STATS, 85),
                            dict.fromkeys(inference.STATS, 31), 1, 0)  # a species with no random-battle sets

norm, matches = inference.norm, inference.matches


# ---------------------------------------------------------------- random teams


@cache
def levels() -> dict[str, int]:
    return {sp: level for sp, (level, _) in inference.library().items()}


TYPES = ["BUG", "DARK", "DRAGON", "ELECTRIC", "FIGHTING", "FIRE", "FLYING", "GHOST", "GRASS", "GROUND", "ICE",
         "NORMAL", "POISON", "PSYCHIC", "ROCK", "STEEL", "WATER"]
INCOMPATIBLE = [({"blissey"}, {"chansey"}), ({"illumise"}, {"volbeat"}),
                ({"parasect", "jynx", "toxicroak"}, {"ninetales", "groudon"}),
                ({"shedinja"}, {"tyranitar", "hippowdon", "abomasnow"})]


@cache
def species_info(species: str) -> tuple[str, list[str], list[int]]:
    """(base species, types, Showdown's getEffectiveness of every attacking type: >0 weak, >1 doubly weak)."""
    dex = GEN.pokedex[species]
    types = [t.upper() for t in dex["types"]]
    eff = [sum({2: 1, 0.5: -1}.get(GEN.type_chart[d][a], 0) for d in types) for a in TYPES]  # immunity counts 0
    return norm(dex.get("baseSpecies", species)), types, eff


@cache
def base_species_pool() -> dict[str, list[str]]:
    pool: dict[str, list[str]] = {}
    for sp in inference.library():
        if sp in GEN.pokedex:
            pool.setdefault(species_info(sp)[0], []).append(sp)
    return pool


def sample_species(known: list[str], n: int, rng: random.Random) -> list[str]:
    """`n` more team members drawn like gen4 randomTeam() (gen5/teams.ts): uniform base species, then forme;
    species clause; at most 2 per type, 3 weak / 1 doubly weak to any type, one level 100; a few banned pairs.
    The revealed mons count towards the limits (an approximation of the posterior)."""
    team, types, weak, dweak = [], {}, [0] * len(TYPES), [0] * len(TYPES)
    lv100 = 0

    def add(sp: str) -> None:
        nonlocal lv100
        team.append(sp)
        for t in species_info(sp)[1]:
            types[t] = types.get(t, 0) + 1
        for i, e in enumerate(species_info(sp)[2]):
            weak[i] += e > 0
            dweak[i] += e > 1
        lv100 += levels().get(sp, 80) == 100

    def ok(sp: str) -> bool:
        (_, ts, eff), ids = species_info(sp), set(team)
        return (all(types.get(t, 0) < 2 for t in ts) and all(w < 3 for w, e in zip(weak, eff) if e > 0)
                and all(d < 1 for d, e in zip(dweak, eff) if e > 1)
                and (lv100 < 1 or levels().get(sp, 80) != 100)
                and not any((sp in a and ids & b) or (sp in b and ids & a) for a, b in INCOMPATIBLE))

    for sp in known:
        if sp in GEN.pokedex:
            add(sp)
    taken = {species_info(sp)[0] for sp in team}
    bases = [b for b in base_species_pool() if b not in taken]
    rng.shuffle(bases)
    out, skipped = [], []
    for b in bases:
        if len(out) == n:
            break
        sp = rng.choice(base_species_pool()[b])
        if ok(sp):
            add(sp)
            out.append(sp)
        else:
            skipped.append(sp)
    return out + skipped[:n - len(out)]  # constraints unsatisfiable: relax them


# ---------------------------------------------------------------- poke-env -> poke-engine

class History:
    """What poke-env does not track, read from the battle log: per side ('p1' / 'p2') the turn Wish and Future
    Sight / Doom Desire were last used (and by whom), mons asleep from Rest, the hits (attacker, move) each
    standing substitute took, how many moves each confused mon has tried while confused, and the evidence
    about the opponent's sets."""

    def __init__(self, battle: AbstractBattle):
        self.wish: dict[str, int] = {}
        self.future: dict[str, tuple[int, str]] = {}
        self.rest: set[str] = set()
        self.sub_hits: dict[str, list[tuple[str, str]]] = {}
        self.confused: dict[str, int] = {}
        self.evidence = inference.battle_evidence(battle)
        turn, last = 0, ("", "")
        for ev in battle._replay_data:
            if len(ev) < 3:
                continue
            kind, key = ev[1], re.sub(r"^(p\d)[a-c]:", r"\1:", ev[2])  # 'p2a: Blissey' -> 'p2: Blissey'
            if kind == "turn":
                turn = int(ev[2])
            elif kind in ("switch", "drag"):
                self.sub_hits.pop(key, None)
                self.confused.pop(key, None)
            elif kind == "move" and len(ev) > 3:
                move = norm(ev[3])
                last = (key, move)
                if move == "wish":
                    self.wish[key[:2]] = turn
            elif kind == "-start" and len(ev) > 3 and norm(ev[3]).endswith(("futuresight", "doomdesire")):
                self.future[key[:2]] = (turn, key)
            elif kind == "-status" and len(ev) > 3 and ev[3] == "slp":
                (self.rest.add if ev[-1].endswith("move: Rest") else self.rest.discard)(key)
            elif kind in ("-curestatus", "faint"):
                self.rest.discard(key)
                if kind == "faint":
                    self.sub_hits.pop(key, None)
                    self.confused.pop(key, None)
            elif kind == "-start" and len(ev) > 3 and ev[3] == "Substitute":
                self.sub_hits[key] = []
            elif kind == "-activate" and len(ev) > 3 and ev[3] in ("Substitute", "move: Substitute"):
                self.sub_hits.setdefault(key, []).append(last)
            elif kind == "-end" and len(ev) > 3 and ev[3] == "Substitute":
                self.sub_hits.pop(key, None)
            elif kind == "-start" and len(ev) > 3 and ev[3] == "confusion":
                self.confused[key] = 0
            elif kind == "-activate" and len(ev) > 3 and ev[3] == "confusion" and key in self.confused:
                self.confused[key] += 1
            elif kind == "-end" and len(ev) > 3 and ev[3] == "confusion":
                self.confused.pop(key, None)


def engine_move(move_id: str, hp_type: PokemonType | None = None) -> str:
    """poke-engine id: gen4 hidden power is a typed 70 BP move ('hiddenpowerfire70')."""
    if move_id.startswith("hiddenpower"):
        typed = move_id[len("hiddenpower"):].rstrip("0123456789")
        if not typed and hp_type is not None:
            typed = hp_type.name.lower()
        return f"hiddenpower{typed}70" if typed else "hiddenpower"
    return move_id


def engine_types(mon: Pokemon) -> tuple[str, str]:
    ts = [t.name.lower() for t in mon.types if t not in (None, PokemonType.THREE_QUESTION_MARKS)]
    ts = (ts + ["typeless", "typeless"])[:2]
    return ts[0], ts[1]


def calc_stats(species: str, level: int, evs: dict[str, int], ivs: dict[str, int]) -> dict[str, int]:
    """Neutral-nature stats (random battles have no natures)."""
    base = GEN.pokedex[species]["baseStats"]
    out = {"hp": (2 * base["hp"] + ivs["hp"] + evs["hp"] // 4) * level // 100 + level + 10}
    for st in ("atk", "def", "spa", "spd", "spe"):
        out[st] = (2 * base[st] + ivs[st] + evs[st] // 4) * level // 100 + 5
    return out


def make_mon(species: str, level: int, types: tuple[str, str], hp: int, maxhp: int, stats: dict[str, int],
             ability: str, item: str, moves: list[tuple[str, int, bool]], status: str, sleep_turns: int,
             rest_turns: int, weight: float) -> pe.Pokemon:
    return pe.Pokemon(
        id=species, level=level, types=types, base_types=types, hp=hp, maxhp=maxhp,
        ability=ability or "none", base_ability=ability or "none", item=item or "none",
        attack=stats["atk"], defense=stats["def"], special_attack=stats["spa"],
        special_defense=stats["spd"], speed=stats["spe"], status=status, sleep_turns=sleep_turns,
        rest_turns=rest_turns, weight_kg=float(weight),
        moves=[pe.Move(id=m, pp=max(pp, 0), disabled=d) for m, pp, d in moves],
    )


def status_of(mon: Pokemon, rest: bool) -> tuple[str, int, int]:
    """(status, sleep turns so far, rest turns left: Rest sleeps exactly 2 turns, the engine counts down from 3)."""
    status = STATUS.get(mon.status, "none") if mon.status is not None else "none"
    if status != "sleep":
        return status, 0, 0
    return status, mon.status_counter, max(1, 3 - mon.status_counter) if rest else 0


def last_used(mon: Pokemon | None, moves: list[str]) -> int | None:
    """Index in `moves` of the move `mon` used last since it switched in (poke-env resets it on switch-out)."""
    last = next((m for m, mv in mon.moves.items() if mv.is_last_used), None) if mon else None
    return next((i for i, m in enumerate(moves) if last and matches(last, m)), None)


def own_mon(mon: Pokemon, disabled: set[str], rest: bool) -> tuple[pe.Pokemon, list[str]]:
    """Our mon exactly as the request describes it; also returns its engine move ids (action order)."""
    moves = list(mon.moves.items())[:4]  # keys, not Move.id: they differ for hidden power
    ids = [norm(engine_move(k, m.type)) for k, m in moves]
    hp = 0 if mon.fainted else mon.current_hp
    item = mon.item if mon.item and mon.item != GEN.UNKNOWN_ITEM else "none"
    stats = {k: mon.stats.get(k) or 100 for k in ("atk", "def", "spa", "spd", "spe")}
    mv = [(i, m.current_pp, k in disabled) for i, (k, m) in zip(ids, moves)]
    return make_mon(mon.species, mon.level, engine_types(mon), hp, mon.max_hp or 1, stats, mon.ability or "none",
                    item, mv, *status_of(mon, rest), mon.weight), ids


def true_set(truth: dict[str, Any]) -> inference.Set:
    """A true set (Showdown's PokemonSet, oracle runs) as a library set."""
    return inference.Set(tuple(norm(m) for m in truth["moves"]), norm(truth["ability"]), norm(truth["item"]),
                         truth.get("role", ""), truth["evs"], truth["ivs"], 1, 0)


def opp_mon(mon: Pokemon | None, species: str, rng: random.Random, hist: History | None = None,
            key: str = "", truth: dict[str, Any] | None = None) -> tuple[pe.Pokemon, list[str]]:
    """Opponent mon (None = unrevealed, sampled from `species`) with its set drawn by `inference.sample`; also
    returns its poke-env move ids. A known item is kept (None = it has none), and an active choice holder is
    locked into its last move. `truth`: its true set (oracle runs), used instead of sampling."""
    revealed = [m for m in (mon.moves if mon else {}) if norm(m) != "struggle"]
    if truth is not None:
        level, s = truth["level"], true_set(truth)
    else:
        level, s = inference.sample(species, mon, rng, hist.evidence if hist else None, key)
    s = s or DEFAULT_SET
    if mon is not None:
        level = mon.level
    moves = [next((m for m in s.moves if matches(r, m)), r) for r in revealed][:4]
    moves += [m for m in s.moves if m not in moves][:4 - len(moves)]
    ability = norm(mon.ability) if mon and mon.ability else s.ability
    item = (mon.item or "none") if mon is not None and mon.item != GEN.UNKNOWN_ITEM else s.item or "none"
    stats = calc_stats(species, level, s.evs, s.ivs)
    pp = {m: mv.current_pp for m, mv in mon.moves.items()} if mon else {}
    lock = last_used(mon, moves) if mon and mon.active and item.startswith("choice") else None
    mv = [(norm(engine_move(m)), pp.get("hiddenpower" if m.startswith("hiddenpower") else m, 16),
           lock is not None and i != lock) for i, m in enumerate(moves)]
    status, sleep, rest = status_of(mon, key in hist.rest if hist else False) if mon else ("none", 0, 0)
    frac = 0.0 if mon and mon.fainted else mon.current_hp_fraction if mon else 1.0
    hp = 0 if frac <= 0 else max(1, round(frac * stats["hp"]))
    types = engine_types(mon) if mon else tuple((t.lower() for t in GEN.pokedex[species]["types"] + ["typeless"]))[:2]
    weight = mon.weight if mon else GEN.pokedex[species].get("weightkg", 50.0)
    return make_mon(species, level, types, hp, stats["hp"], stats, ability, item, mv, status, sleep, rest,
                    weight), moves


def side_conditions(conds: dict[SideCondition, int], turn: int, active: Pokemon | None) -> pe.SideConditions:
    kw: dict[str, int] = {"toxic_count": active.status_counter if active and active.status is Status.TOX else 0,
                          "protect": active.protect_counter if active else 0}  # consecutive protects
    kw["spikes"] = conds.get(SideCondition.SPIKES, 0)
    kw["toxic_spikes"] = conds.get(SideCondition.TOXIC_SPIKES, 0)
    kw["stealth_rock"] = int(SideCondition.STEALTH_ROCK in conds)
    for cond, (name, length) in SCREEN_TURNS.items():
        if cond in conds:
            kw[name] = max(1, length - (turn - conds[cond]))
    return pe.SideConditions(**kw)


def make_side(mons: list[pe.Pokemon], active: Pokemon | None, active_idx: int, conds: pe.SideConditions,
              last_move: int | None, battle: Battle, hist: History, role: str, team: list[str],
              sub_hp: int | None = None, **kw: Any) -> pe.Side:
    """`last_move`: index of the active mon's last move this stint (None: it just switched in); `sub_hp`: HP of
    its substitute if it has one (default 1/4 max HP)."""
    mons = (mons + [pe.Pokemon.create_fainted() for _ in range(6)])[:6]
    boosts = active.boosts if active else {}
    effects = {e for e in (active.effects if active else {}) if e not in SKIP_EFFECTS}
    volatiles = {e.name.replace("_", "") for e in effects}
    if last_move is None:
        volatiles.discard("ENCORE")  # the engine panics on encore without a last-used move
    recharge = active is not None and active.must_recharge
    if role == (battle.player_role or "p1"):  # poke-env's flag can be stale; our request is not
        recharge = recharge and all(m.id == "recharge" for m in battle.available_moves)
    if recharge:
        volatiles.add("MUSTRECHARGE")
    preparing = active.preparing_move if active is not None else None
    if preparing is not None and preparing.id in CHARGE_MOVES:
        volatiles.add(preparing.id.upper())
    sub = (mons[active_idx].maxhp // 4 if sub_hp is None else sub_hp) if Effect.SUBSTITUTE in effects else 0
    counts = dict(active.effects) if active else {}  # poke-env counts the turns taunt / encore have lasted
    counts[Effect.CONFUSION] = hist.confused.get(team[active_idx], 0) if active and team else 0
    timed = {k: min(counts.get(e, 0), 9) for k, e in TIMED_EFFECTS.items() if e.name in volatiles}
    # Wish heals at the end of the turn after it was used, Future Sight hits at the end of the 2nd turn after
    wish = 2 - (battle.turn - hist.wish[role]) if role in hist.wish else 0
    fs_turn, fs_user = hist.future.get(role, (0, ""))
    fs = 3 - (battle.turn - fs_turn) if fs_user in team else 0
    return pe.Side(
        pokemon=mons, active_index=str(active_idx), side_conditions=conds, volatile_statuses=volatiles,
        volatile_status_durations=pe.VolatileStatusDurations(yawn=int("YAWN" in volatiles), **timed),
        wish=(wish, mons[active_idx].maxhp // 2) if wish > 0 else (0, 0),
        future_sight=(fs, str(team.index(fs_user))) if fs > 0 else (0, "0"),
        substitute_health=sub, attack_boost=boosts.get("atk", 0), defense_boost=boosts.get("def", 0),
        special_attack_boost=boosts.get("spa", 0), special_defense_boost=boosts.get("spd", 0),
        speed_boost=boosts.get("spe", 0), accuracy_boost=boosts.get("accuracy", 0),
        evasion_boost=boosts.get("evasion", 0),
        last_used_move=f"switch:{active_idx}" if last_move is None else f"move:{last_move}", **kw,
    )


def substitute_left(state: pe.State, ours: bool, hits: list[tuple[str, str]], facing: str,
                    facing_moves: list[tuple[str, str]]) -> int:
    """HP left on side one's (ours) or side two's substitute, made with 1/4 max HP, after `hits` [(attacker, move)].
    A hit by the mon now facing it takes off the engine's damage for that move (its first roll), any other hit
    half of what is left; it is never below 1 (it is still up)."""
    side = state.side_one if ours else state.side_two
    left = side.pokemon[int(side.active_index)].maxhp // 4
    for attacker, move in hits:
        damage = left // 2
        engine_id = next((e for m, e in facing_moves if matches(move, m)), None) if attacker == facing else None
        if engine_id:
            try:  # side one's move first or second only matters for a few moves
                ones, twos = pe.calculate_damage(state, "splash" if ours else engine_id,
                                                 engine_id if ours else "splash", not ours)
                damage = (twos if ours else ones)[0]
            except BaseException as e:  # unknown move: Rust panics are pyo3 PanicExceptions
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
        left = max(1, left - damage)
    return left


def weather_turns(battle: AbstractBattle, weather: Weather, first_seen: int) -> int:
    """Turns left; -1 (never decremented by the engine) for gen4's permanent ability weather."""
    mons = list(battle.team.values()) + list(battle.opponent_team.values())
    ability = WEATHER_ABILITY.get(weather)
    if any(m.ability == ability or (m.ability is None and ability in m.possible_abilities) for m in mons):
        return -1
    return max(1, 5 - (battle.turn - first_seen))


class Converted:
    """One determinization plus the maps from engine choices back to poke-env action indices."""

    def __init__(self, state: pe.State, switch_ids: list[str], move_ids: list[str]):
        self.state_str = state.to_string()
        self.switch_ids, self.move_ids = switch_ids, move_ids

    def action(self, choice: str) -> int | None:
        if choice.startswith("switch "):
            name = choice[len("switch "):]
            return self.switch_ids.index(name) if name in self.switch_ids else None
        return 6 + self.move_ids.index(choice) if choice in self.move_ids else None


def convert(battle: Battle, rng: random.Random, weather_seen: int | None = None,
            hist: History | None = None, truth: list[dict[str, Any]] | None = None,
            oracle_all: bool = False) -> Converted:
    """`truth`: the opponent's true team (oracle runs): revealed mons get their true sets, and with `oracle_all`
    the unrevealed ones are the true remaining mons instead of samples."""
    hist = hist or History(battle)
    true_sets = {norm(t["speciesId"]): t for t in truth or []}
    role, opp_role = battle.player_role or "p1", battle.opponent_role or "p2"
    keys = list(battle.team)
    team = list(battle.team.values())
    active_idx = next((i for i, m in enumerate(team) if m.active), 0)
    disabled = set()
    if battle.active_pokemon is not None and not battle.force_switch:
        avail = {id(m) for m in battle.available_moves}  # Move objects: ids and keys differ for hidden power
        disabled = {k for k, m in battle.active_pokemon.moves.items() if id(m) not in avail}
    ours, move_ids = [], []
    for i, (key, mon) in enumerate(zip(keys, team)):
        m, ids = own_mon(mon, disabled if i == active_idx else set(), key in hist.rest)
        ours.append(m)
        if i == active_idx:
            move_ids = ids
    active = team[active_idx]
    own_moves = list(active.moves)[:4]
    last = last_used(active, own_moves)
    avail = [k for k, m in active.moves.items() if any(m is a for a in battle.available_moves)]
    if last is None and len(avail) == 1 and avail[0] in own_moves:
        last = own_moves.index(avail[0])  # locked (encore, choice, outrage) without poke-env knowing the move
    opp_keys = list(battle.opponent_team)
    opp = list(battle.opponent_team.values())
    n_unrevealed = max(0, min(6, battle.max_team_size or 6) - len(opp))
    if oracle_all and true_sets:
        unrevealed = [s for s in true_sets if s not in {inference.set_species(m.species) for m in opp}][:n_unrevealed]
    else:
        unrevealed = sample_species([m.species for m in opp], n_unrevealed, rng)
    theirs, opp_moves = [], []
    for key, m in zip(opp_keys, opp):
        mon, moves = opp_mon(m, m.species, rng, hist, key, true_sets.get(inference.set_species(m.species)))
        theirs.append(mon)
        opp_moves.append(moves)
    theirs += [opp_mon(None, s, rng, truth=true_sets.get(s) if oracle_all else None)[0] for s in unrevealed]
    opp_active = battle.opponent_active_pokemon
    opp_idx = next((i for i, m in enumerate(opp) if m is opp_active), 0)
    opp_last = last_used(opp_active, opp_moves[opp_idx]) if opp else None

    weather, turns = "none", 0
    for w in battle.weather:
        if w in WEATHER:
            weather, turns = WEATHER[w], weather_turns(battle, w, battle.turn if weather_seen is None else weather_seen)
    tr = Field.TRICK_ROOM in battle.fields
    tr_turns = max(1, 5 - (battle.turn - battle.fields[Field.TRICK_ROOM])) if tr else 0

    def build(sub_one: int | None = None, sub_two: int | None = None) -> pe.State:
        side_one = make_side(ours, active, active_idx, side_conditions(battle.side_conditions, battle.turn, active),
                             last, battle, hist, role, keys, sub_one, force_switch=bool(battle.force_switch),
                             force_trapped=bool(battle.trapped))
        side_two = make_side(theirs, opp_active, opp_idx,
                             side_conditions(battle.opponent_side_conditions, battle.turn, opp_active),
                             opp_last, battle, hist, opp_role, opp_keys, sub_two)
        return pe.State(side_one=side_one, side_two=side_two, weather=weather, weather_turns_remaining=turns,
                        trick_room=tr, trick_room_turns_remaining=tr_turns)

    state = build()
    opp_key = opp_keys[opp_idx] if opp else ""
    our_hits, their_hits = hist.sub_hits.get(keys[active_idx]), hist.sub_hits.get(opp_key)
    if (our_hits and state.side_one.substitute_health) or (their_hits and state.side_two.substitute_health):
        theirs_ids = [(m, norm(engine_move(m))) for m in opp_moves[opp_idx]] if opp else []
        state = build(substitute_left(state, True, our_hits, opp_key, theirs_ids) if our_hits else None,
                      substitute_left(state, False, their_hits, keys[active_idx], list(zip(own_moves, move_ids)))
                      if their_hits else None)
    return Converted(state, [norm(m.species) for m in team], move_ids)


# ---------------------------------------------------------------- player

_POOLS: dict[tuple[int, str | None], EnginePool] = {}


def engine_pool(workers: int | None = None, python: str | None = None) -> EnginePool:
    """Process-wide pool of poke-engine worker subprocesses, shared by all SearchPlayers."""
    n = workers or max(1, (os.cpu_count() or 2) - 1)
    return _POOLS.setdefault((n, python), EnginePool(n, python=python))


class SearchPlayer(Player):
    """Network-guided determinized MCTS. prior_weight=0 and top_k=None is pure engine search; value_weight=1
    replaces the engine search by the value lookahead (value_top_k: only the network's top-k actions,
    value_min_prob: chance outcomes below it are dropped). engine_python: interpreter for the engine workers.
    oracle ("revealed" / "all", diagnosis only): the opponent's true sets from `oracle_dir`, written by the
    duplicate server (PORYGON_DUPLICATE_TEAMS), replace the samples for revealed mons, or for all of them."""

    def __init__(self, policy: ActorCriticPolicy, *args: Any, ms: int = 1000, samples: int = 4,
                 prior_weight: float = 0.25, top_k: int | None = None, value_weight: float = 0.0,
                 value_top_k: int | None = None, value_min_prob: float = 0.0, workers: int | None = None,
                 engine_python: str | None = None, oracle: str | None = None, oracle_dir: str | None = None,
                 seed: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.policy, self.ms, self.samples = policy, ms, samples
        self.prior_weight, self.top_k, self.value_weight = prior_weight, top_k, value_weight
        self.pool = engine_pool(workers, engine_python)
        self.lookahead = None
        if value_weight > 0:
            from porygon.lookahead import Lookahead  # it imports this module
            self.lookahead = Lookahead(policy, value_top_k, value_min_prob)
        self.rng = random.Random(seed)
        self.weather_seen: dict[str, tuple[Weather, int]] = {}
        self.stats = {"decisions": 0, "searched": 0, "fallbacks": 0, "agree": 0, "failed_samples": 0,
                      "value_failed": 0}
        self.last_convs: dict[str, list[Converted]] = {}  # battle tag -> its last determinizations
        assert oracle in (None, "revealed", "all") and (oracle is None or oracle_dir), "oracle needs oracle_dir"
        self.oracle, self.oracle_dir = oracle, oracle_dir
        self.truth: dict[str, list[dict[str, Any]] | None] = {}

    def network_probs(self, battle: Battle) -> tuple[np.ndarray, np.ndarray]:
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.float32)
        obs = {"observation": torch.as_tensor(embed_battle(battle)).unsqueeze(0),
               "action_mask": torch.as_tensor(mask).unsqueeze(0)}
        obs = {k: v.to(self.policy.device) for k, v in obs.items()}
        with torch.no_grad():
            probs = self.policy.get_distribution(obs).distribution.probs[0].cpu().numpy()
        return probs, mask

    def _weather_first_seen(self, battle: Battle) -> int | None:
        current = next(iter(battle.weather), None)
        seen = self.weather_seen.get(battle.battle_tag)
        if current is None:
            self.weather_seen.pop(battle.battle_tag, None)
            return None
        if seen is None or seen[0] != current:
            # weather started during the previous turn (or this one, on a switch-in)
            seen = self.weather_seen[battle.battle_tag] = (current, max(battle.turn - 1, 0))
        return seen[1]

    async def visit_shares(self, battle: Battle, probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Mean (over determinizations) MCTS visit share of every poke-env action index, mixed with the value
        lookahead's (value_weight). If one of the two fails, the other is used alone."""
        first_seen = self._weather_first_seen(battle)
        hist = History(battle)
        truth = self.true_team(battle)
        convs = [convert(battle, self.rng, first_seen, hist, truth, self.oracle == "all") for _ in range(self.samples)]
        self.last_convs[battle.battle_tag] = convs
        w = self.value_weight
        value = (asyncio.create_task(asyncio.to_thread(self.lookahead.shares, battle, convs, probs, mask))
                 if self.lookahead else None)
        try:
            mcts = await self.mcts_shares(convs) if w < 1 else None
        except Exception:
            if value is None:
                raise
            mcts = None
        if value is not None:
            try:
                v = await value
            except Exception as e:
                if mcts is None:
                    raise
                self.stats["value_failed"] += 1
                log.warning("value lookahead failed in %s turn %d: %s", battle.battle_tag, battle.turn, e)
            else:
                return v if mcts is None else (1 - w) * mcts + w * v
        return mcts

    def true_team(self, battle: Battle) -> list[dict[str, Any]] | None:
        if not self.oracle:
            return None
        if battle.battle_tag not in self.truth:
            try:
                teams = json.loads((Path(self.oracle_dir) / f"{battle.battle_tag}.json").read_text())
                self.truth[battle.battle_tag] = teams["p2team"] if teams["p1"] == self.username else teams["p1team"]
            except OSError as e:
                log.warning("no true team for %s: %s", battle.battle_tag, e)
                return None
        return self.truth[battle.battle_tag]

    async def mcts_shares(self, convs: list[Converted]) -> np.ndarray:
        results = await asyncio.gather(*(self.pool.search(c.state_str, self.ms) for c in convs),
                                       return_exceptions=True)
        shares, ok, errors = np.zeros(10), 0, []
        for conv, options in zip(convs, results):  # a failed sample (engine panic) is just skipped
            try:
                shares += self._sample_shares(conv, options)
                ok += 1
            except Exception as e:
                errors.append(e)
        self.stats["failed_samples"] += len(errors)
        if not ok:
            raise errors[0]
        return shares / ok

    @staticmethod
    def _sample_shares(conv: Converted, options: list | BaseException) -> np.ndarray:
        if isinstance(options, BaseException):
            raise options
        shares, total = np.zeros(10), sum(v for _, v, _ in options) or 1
        for choice, visits, _ in options:
            if choice == "No Move":  # the engine thinks we cannot act (truant, recharge): no information
                continue
            a = conv.action(choice)
            if a is None:
                raise ValueError(f"unmappable engine choice {choice!r} ({conv.switch_ids}, {conv.move_ids})")
            shares[a] += visits / total
        return shares

    def combine(self, shares: np.ndarray, probs: np.ndarray, mask: np.ndarray) -> int:
        legal = mask > 0
        if self.top_k:
            legal &= probs >= np.sort(np.where(legal, probs, -1))[::-1][min(self.top_k, int(legal.sum())) - 1]
        score = np.where(legal, shares + self.prior_weight * probs, -np.inf)
        return int(np.argmax(score))

    async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        assert isinstance(battle, Battle)
        probs, mask = self.network_probs(battle)
        net_action = int(np.argmax(np.where(mask > 0, probs, -1)))
        action = net_action
        self.stats["decisions"] += 1
        if mask.sum() > 1:
            try:
                shares = await self.visit_shares(battle, probs, mask)
                if not shares.any():
                    return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)
                if shares[mask > 0].sum() < 0.5:  # the engine searched mostly illegal options -> desync
                    raise ValueError(f"engine options disagree with the request: {shares} vs mask {mask}")
                action = self.combine(shares, probs, mask)
                self.on_decision(battle, probs, mask, shares)
                self.stats["searched"] += 1
                self.stats["agree"] += action == net_action
            except Exception as e:
                self.stats["fallbacks"] += 1
                log.warning("search failed in %s turn %d, using the network: %s", battle.battle_tag, battle.turn, e)
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)

    def on_decision(self, battle: Battle, probs: np.ndarray, mask: np.ndarray, shares: np.ndarray) -> None:
        """Called after every searched decision; `porygon.expert` records them as training targets."""

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self.weather_seen.pop(battle.battle_tag, None)
        self.last_convs.pop(battle.battle_tag, None)
        self.truth.pop(battle.battle_tag, None)


# ---------------------------------------------------------------- CLI

CONFIG = ("ms", "samples", "prior_weight", "top_k", "value_weight", "value_top_k", "value_min_prob", "engine_python")


def config(args: argparse.Namespace, prefix: str = "") -> dict[str, Any]:
    return {k: getattr(args, prefix + k) for k in CONFIG}


async def run(args: argparse.Namespace) -> None:
    policy = load_policy(args.model, args.device)
    tag = random.randint(0, 10**6)
    common = dict(battle_format=args.format, server_configuration=local_server(args.port),
                  max_concurrent_battles=args.concurrency, log_level=40)
    cfg = config(args)
    if args.log_dir:
        from porygon.decisions import LoggedSearchPlayer  # it imports this module
        player = LoggedSearchPlayer(policy, **cfg, workers=args.workers, out=args.log_dir, oracle=args.oracle,
                                    oracle_dir=args.oracle_dir,
                                    account_configuration=AccountConfiguration(f"search{tag}", None), **common)
    else:
        player = SearchPlayer(policy, **cfg, workers=args.workers, oracle=args.oracle, oracle_dir=args.oracle_dir,
                              account_configuration=AccountConfiguration(f"search{tag}", None), **common)
    opp_conf = AccountConfiguration(f"opp{args.opponent[:8]}{tag}", None)
    opp_policy = load_policy(args.opp_model, args.device) if args.opp_model else policy
    if args.opponent == "search":
        opp_cfg = config(args, "opp_")
        opponent = SearchPlayer(opp_policy, **opp_cfg, workers=args.workers, account_configuration=opp_conf, **common)
        desc = f"SearchPlayer({', '.join(f'{k}={v}' for k, v in opp_cfg.items())})"
    else:
        opponent = (PolicyPlayer(opp_policy, account_configuration=opp_conf, **common) if args.opponent == "policy"
                    else BASELINES[args.opponent](account_configuration=opp_conf, **common))
        desc = args.opponent
    if args.opp_model:
        desc += f" [{args.opp_model}]"
    start = time.time()
    half = args.n_battles // 2  # alternate who challenges (p1 / p2)
    await player.battle_against(opponent, n_battles=args.n_battles - half)
    await opponent.battle_against(player, n_battles=half)
    n, won = player.n_finished_battles, player.n_won_battles
    s = player.stats
    print(f"SearchPlayer({', '.join(f'{k}={v}' for k, v in cfg.items())}) vs {desc}: won {won}/{n} = "
          f"{won / max(n, 1):.1%}  (±{1.96 * np.sqrt(max(won, 1) * max(n - won, 1) / max(n, 1) ** 3):.1%})")
    print(f"decisions {s['decisions']}, searched {s['searched']}, fallbacks {s['fallbacks']}, "
          f"failed samples {s['failed_samples']}, value lookahead failed {s['value_failed']}, "
          f"same as network argmax {s['agree'] / max(s['searched'], 1):.1%}, {time.time() - start:.0f}s")


CONFIG_TYPES = {"prior_weight": float, "value_weight": float, "value_min_prob": float, "engine_python": str}


def add_value_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--value-weight", type=float, default=0.0,
                   help="weight of the network-valued one-turn lookahead (1 = no engine search)")
    p.add_argument("--value-top-k", type=int, default=None, help="lookahead: only the network's top-k actions")
    p.add_argument("--value-min-prob", type=float, default=0.0, help="lookahead: drop rarer chance outcomes")
    p.add_argument("--device", default="cpu", help="for the network (cuda: the lookahead's leaf values)")
    p.add_argument("--engine-python", help="interpreter for the engine workers (default: this one)")


def add_oracle_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--oracle", choices=["revealed", "all"], help="diagnosis: search with the opponent's true sets")
    p.add_argument("--oracle-dir", default=os.environ.get("PORYGON_DUPLICATE_TEAMS"),
                   help="true teams from the duplicate server (default $PORYGON_DUPLICATE_TEAMS)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--opponent", default="policy", choices=["policy", "search", *BASELINES])
    p.add_argument("--opp-model", help="network for the policy/search opponent (default: MODEL)")
    p.add_argument("--n-battles", type=int, default=50)
    p.add_argument("--ms", type=int, default=1000, help="MCTS time per determinization per decision")
    p.add_argument("--samples", type=int, default=4, help="determinizations per decision")
    p.add_argument("--prior-weight", type=float, default=0.25, help="0 = pure engine search")
    p.add_argument("--top-k", type=int, default=None, help="only consider the network's top-k actions")
    add_value_args(p)
    for k in CONFIG:  # the --opponent search configuration
        p.add_argument(f"--opp-{k.replace('_', '-')}", type=CONFIG_TYPES.get(k, int), default=p.get_default(k))
    p.add_argument("--concurrency", type=int, default=3, help="simultaneous battles")
    p.add_argument("--log-dir", help="record MODEL's decisions and replays here (porygon.decisions)")
    add_oracle_args(p)
    p.add_argument("--workers", type=int, default=None, help="engine processes (default: cores - 1)")
    p.add_argument("--format", default="gen4randombattle")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
