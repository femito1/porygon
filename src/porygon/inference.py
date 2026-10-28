"""Opponent set inference for the search's determinizations.

Revealed opponent mons get a whole set (moves, ability, item, EVs / IVs) drawn from a library of gen4
random-battle sets made by Showdown's own generator (`scripts/build_set_library.js`), in proportion to how
often the generator makes it, among the sets consistent with everything the battle has shown:
  - the moves it used, its ability and its original item when revealed;
  - the lead (slot 0, the only one the generator may give a Focus Sash) or not;
  - what the log rules out (`Evidence`): no choice item after two different moves in one stint, no Leftovers /
    Black Sludge after an end of turn spent damaged without a heal, no Life Orb after a hit without recoil.
If nothing is consistent the item evidence is dropped first, then the ability, then the moves that no set has.
Unrevealed mons get a set drawn from the same library unconditioned.

    python scripts/score_inference.py runs/foulplay-<date>-<job> ...   # accuracy against the true teams
"""

import gzip
import json
import random
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from poke_env.battle import AbstractBattle, Pokemon
from poke_env.data import GenData

GEN = GenData.from_gen(4)
LIBRARY_PATH = Path(__file__).parent / "data/gen4randombattle_library.json.gz"
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
LEFTOVERS = {"leftovers", "blacksludge"}
LEAD_SMOOTHING = 0.01  # a set never seen as a lead in the library is not impossible as one


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def matches(revealed: str, move: str) -> bool:
    # poke-env reports the opponent's hidden power without its type
    return move == revealed or (revealed == "hiddenpower" and move.startswith("hiddenpower"))


@dataclass(frozen=True)
class Set:
    moves: tuple[str, ...]
    ability: str
    item: str
    role: str
    evs: dict[str, int]
    ivs: dict[str, int]
    n: int  # times the generator made it
    n_lead: int  # ... as the team's lead


def _spread(s: str, default: int) -> dict[str, int]:
    return dict(zip(STATS, map(int, s.split(",")))) if s else dict.fromkeys(STATS, default)


@cache
def library() -> dict[str, tuple[int, list[Set]]]:
    """species -> (level, its sets, most frequent first)."""
    raw = json.loads(gzip.open(LIBRARY_PATH).read())["species"]
    return {sp: (v["level"], [Set(tuple(moves.split(",")), ability, item, role, _spread(evs, 85), _spread(ivs, 31),
                                  n, n_lead) for moves, ability, item, role, evs, ivs, n, n_lead in v["sets"]])
            for sp, v in raw.items()}


def set_species(species: str) -> str:
    """The generator's species id: poke-env names cosmetic formes (gastrodoneast), Showdown's sets do not."""
    if species not in library() and species in GEN.pokedex:
        return norm(GEN.pokedex[species].get("baseSpecies", species))
    return species


# ---------------------------------------------------------------- evidence from the log

@dataclass
class Evidence:
    """What the battle log shows about the opponent's (`role`) items, per mon ('p2: Name'): its original item
    when some event names it (`item`), whether it has lost or swapped it since (`changed`), and what the log rules
    out while it still held its original item."""
    lead: str | None
    item: dict[str, str]
    changed: set[str]
    unlocked: set[str]  # used two different moves in one stint: no choice item
    no_leftovers: set[str]  # ended a turn damaged without a Leftovers / Black Sludge heal
    no_lifeorb: set[str]  # hit with a base-power move without Life Orb recoil (U-turn never shows it)
    races: dict[str, list["Race"]] = field(default_factory=dict)  # turns both actives moved (`races`)


@dataclass(frozen=True)
class Race:
    """One turn the opponent's mon and ours both moved with the same priority: who went first, our mon's speed,
    and the opponent's speed modifiers at the start of that turn (gen4 orders moves then)."""
    first: bool  # the opponent moved first
    ours: float  # our mon's effective speed
    stage: int
    paralyzed: bool
    statused: bool
    weather: str
    tailwind: bool
    item_held: bool  # still its original item


SPEED_ABILITIES = {"swiftswim", "chlorophyll", "quickfeet", "unburden", "slowstart", "stall", "truant"}
SPEED_ITEMS = {"ironball", "laggingtail", "quickclaw", "fullincense", "machobrace", "custapberry", "quickpowder"}


def _key(ident: str) -> str:
    return re.sub(r"^(p\d)[a-c]:", r"\1:", ident)  # 'p2a: Blissey' -> 'p2: Blissey'


def _hp(status: str) -> float:
    hp = status.split()[0]
    if hp == "0" or status.endswith("fnt"):
        return 0.0
    cur, _, mx = hp.partition("/")
    return float(cur) / float(mx or 100)


def evidence(log: list[list[str]], role: str) -> Evidence:
    """Read `role`'s (the opponent's) item evidence from battle events (poke-env's `_replay_data`)."""
    ev = Evidence(None, {}, set(), set(), set(), set())
    stint: dict[str, set[str]] = {}
    hp: dict[str, float] = {}
    active: str | None = None
    hp_at_residual: float | None = None  # the active mon's HP when the end-of-turn effects started
    healed = full = False  # a Leftovers heal / the active mon at full HP during the end-of-turn effects
    swapped: set[str] = set()  # mons whose item a Trick in the current move took (the messages come in either order)
    attack: tuple[str, bool, bool] | None = None  # (attacker, hit us, recoil seen) of the current move

    def original(key: str, item: str) -> None:
        if key not in ev.changed or key in swapped:
            ev.item.setdefault(key, norm(item))

    def end_attack() -> None:
        nonlocal attack
        if attack and attack[1] and not attack[2] and hp.get(attack[0], 1) > 0 and attack[0] not in ev.changed:
            ev.no_lifeorb.add(attack[0])
        attack = None

    for e in log:
        if len(e) < 2:
            continue
        kind = e[1]
        if kind in ("move", "switch", "drag", "upkeep", "turn", "", "replace"):  # gen4 recoil comes after faints
            end_attack()
            swapped = set()
        if kind == "":  # the last break before upkeep starts the end-of-turn effects
            hp_at_residual = hp.get(active) if active else None
            healed = full = False
            continue
        if kind == "upkeep":
            if (active and hp_at_residual is not None and 0 < hp_at_residual < 1 and 0 < hp.get(active, 0) < 1
                    and not healed and not full and active not in ev.changed):  # Wish can top it up first
                ev.no_leftovers.add(active)
            hp_at_residual = None
            continue
        if len(e) < 3:
            continue
        key = _key(e[2])
        mine = key.startswith(role)
        if kind in ("switch", "drag", "replace") and mine:
            if ev.lead is None and kind == "switch":
                ev.lead = key
            active, stint[key] = key, set()
            if len(e) > 4:
                hp[key] = _hp(e[4])
        elif kind in ("-damage", "-heal", "-sethp") and len(e) > 3:
            if mine:
                hp[key] = _hp(e[3])
                full |= hp[key] >= 1
                if kind == "-heal" and any(x in ("[from] item: Leftovers", "[from] item: Black Sludge") for x in e[4:]):
                    healed = True
            elif attack and kind == "-damage" and len(e) == 4:
                attack = (attack[0], True, attack[2])
        elif kind == "faint" and mine:
            hp[key] = 0.0
        elif kind == "move" and len(e) > 3 and mine:
            move = norm(e[3])
            if move != "struggle" and not any(x.startswith("[from]") for x in e[4:]):
                stint.setdefault(key, set()).add(move)
                if len(stint[key]) > 1 and key not in ev.changed:
                    ev.unlocked.add(key)
            data = GEN.moves.get(move, {})
            if data.get("basePower", 0) > 0 and not data.get("selfSwitch") and move != "struggle":
                attack = (key, False, False)
        if kind in ("-item", "-enditem") and len(e) > 3:
            swap = any(x.startswith("[from] move:") for x in e[4:])  # Trick / Switcheroo / Thief / Covet
            if mine:
                if not (kind == "-item" and swap):
                    original(key, e[3])
                if kind == "-enditem" or swap:
                    if swap and key not in ev.changed:
                        swapped.add(key)
                    ev.changed.add(key)
            elif kind == "-item" and swap and active:
                original(active, e[3])  # we received their item
                if active not in ev.changed:
                    swapped.add(active)
                ev.changed.add(active)
        elif mine:
            for x in e[3:]:
                if x.startswith("[from] item: "):
                    original(key, x[len("[from] item: "):])
                    if x == "[from] item: Life Orb" and attack and attack[0] == key:
                        attack = (key, attack[1], True)
    return ev


def _stage(n: int) -> float:
    return (2 + n) / 2 if n >= 0 else 2 / (2 - n)


def races(log: list[list[str]], role: str, ours: dict[str, tuple[int, str, str]]) -> dict[str, list[Race]]:
    """Per opponent (`role`) mon, the turns it and our active mon both moved with the same priority and no Trick
    Room. `ours`: our mons' (speed stat, ability, item) by key; turns where our speed is uncertain (a speed
    ability or item, an item that changed during the battle) are left out."""
    out: dict[str, list[Race]] = {}
    active: dict[str, str] = {}
    stage: dict[str, int] = {}
    status: dict[str, str] = {}
    changed: set[str] = set()
    tailwind: set[str] = set()
    weather, trick_room = "", False
    start: dict | None = None  # the state when the current turn started
    moved: list[tuple[str, str]] = []  # (mon, move) in order, the first move of each side this turn
    spoiled = False  # a mon could not move this turn
    pending: list[tuple[str, str, Race]] = []  # (our mon, theirs, race)

    def finish() -> None:
        if not start or spoiled or len(moved) != 2 or start["trick_room"]:
            return
        (a, ma), (b, mb) = moved
        opp, mine = (a, b) if a.startswith(role) else (b, a)
        if not opp.startswith(role) or mine.startswith(role):
            return
        if opp != start["active"].get(role) or mine != start["active"].get(mine[:2]):
            return
        prio = [GEN.moves.get(m, {}).get("priority", 0) for m in (ma, mb)]
        if prio[0] != prio[1] or "pursuit" in (ma, mb) or mine not in ours:
            return
        spe, ability, item = ours[mine]
        if ability in SPEED_ABILITIES or item in SPEED_ITEMS or start["stage"].get(mine, 0) == 99:
            return
        mult = _stage(start["stage"].get(mine, 0)) * (0.25 if start["status"].get(mine) == "par" else 1)
        mult *= (1.5 if item == "choicescarf" else 1) * (2 if mine[:2] in start["tailwind"] else 1)
        weather_now = "" if ability in ("airlock", "cloudnine") else start["weather"]
        pending.append((mine, opp, Race(a == opp, spe * mult, start["stage"].get(opp, 0),
                                        start["status"].get(opp) == "par", bool(start["status"].get(opp)),
                                        weather_now, role in start["tailwind"], opp not in start["changed"])))

    for e in log:
        if len(e) < 2:
            continue
        kind = e[1]
        if kind == "turn":
            finish()
            moved, spoiled = [], False
            start = {"active": dict(active), "stage": dict(stage), "status": dict(status), "changed": set(changed),
                     "tailwind": set(tailwind), "weather": weather, "trick_room": trick_room}
            continue
        if len(e) < 3:
            continue
        key = _key(e[2])
        side = key[:2]
        if kind in ("switch", "drag", "replace"):
            if active.get(side):
                stage.pop(active[side], None)
            if any(k[:2] == side and m == "batonpass" for k, m in moved):
                stage[key] = 99  # boosts passed on: not tracked
            active[side] = key
            if len(e) > 4:
                status[key] = e[4].split()[1] if len(e[4].split()) > 1 else ""
        elif kind == "move" and not any(x.startswith("[from]") for x in e[4:]):
            if not any(k[:2] == side for k, _ in moved):
                moved.append((key, norm(e[3])))
        elif kind == "cant" or (kind == "-start" and e[3:4] == ["Encore"]):
            spoiled = True  # it did not move / Encore replaced the move it chose
        elif kind == "-transform":
            stage[key] = 99  # its speed is the target's: not tracked until it switches out
        elif kind in ("-boost", "-unboost") and len(e) > 4 and e[3] == "spe":
            stage[key] = max(-6, min(6, stage.get(key, 0) + (1 if kind == "-boost" else -1) * int(e[4])))
        elif kind in ("-clearallboost",):
            stage.clear()
        elif (kind in ("-clearboost", "-swapboost", "-copyboost", "-invertboost")
              or (kind == "-setboost" and e[3:4] == ["spe"]) or (kind == "-clearnegativeboost" and stage.get(key, 0) < 0)):
            stage[key] = 99  # not tracked: no race until it switches out
        elif kind == "-status" and len(e) > 3:
            status[key] = e[3]
        elif kind == "-curestatus":
            status[key] = ""
        elif kind in ("-item", "-enditem"):
            if kind == "-enditem" or any(x.startswith("[from] move:") for x in e[4:]):
                changed.add(key)
        elif kind == "-weather" and len(e) > 2 and "[upkeep]" not in e:
            weather = {"RainDance": "rain", "SunnyDay": "sun"}.get(e[2], "")
        elif kind in ("-fieldstart", "-fieldend") and "Trick Room" in e[2]:
            trick_room = kind == "-fieldstart"
        elif kind in ("-sidestart", "-sideend") and len(e) > 3 and "Tailwind" in e[3]:
            (tailwind.add if kind == "-sidestart" else tailwind.discard)(side)
    for mine, opp, race in pending:
        if mine not in changed and race.ours > 0:
            out.setdefault(opp, []).append(race)
    return out


def battle_evidence(battle: AbstractBattle) -> Evidence:
    ev = evidence(battle._replay_data, battle.opponent_role or "p2")
    ours = {k: (m.stats.get("spe") or 0, norm(m.ability or ""), norm(m.item or "")) for k, m in battle.team.items()
            if m.stats.get("spe")}
    ev.races = races(battle._replay_data, battle.opponent_role or "p2", ours)
    return ev


def speed(species: str, level: int, s: Set) -> int:
    base = GEN.pokedex[species]["baseStats"]["spe"]
    return (2 * base + s.ivs["spe"] + s.evs["spe"] // 4) * level // 100 + 5


def outran(s: Set, spe: int, race: Race) -> bool | None:
    """Whether set `s` (speed stat `spe`) moves before our mon in `race`; None: too close to call (a speed tie,
    rounding) or a speed effect not modelled."""
    if race.stage == 99 or s.ability in ("slowstart", "stall", "truant", "unburden") or s.item in SPEED_ITEMS:
        return None
    mult = _stage(race.stage)
    if race.paralyzed and s.ability != "quickfeet":
        mult *= 0.25
    if race.statused and s.ability == "quickfeet":
        mult *= 1.5
    if race.item_held and s.item == "choicescarf":
        mult *= 1.5
    if (s.ability, race.weather) in (("swiftswim", "rain"), ("chlorophyll", "sun")):
        mult *= 2
    if race.tailwind:
        mult *= 2
    theirs = spe * mult
    if abs(theirs - race.ours) <= max(1.5, 0.01 * max(theirs, race.ours)):  # Showdown rounds each modifier
        return None
    return theirs > race.ours


# ---------------------------------------------------------------- sampling

def consistent(species: str, level: int, sets: list[Set], revealed: list[str], ability: str | None,
               item: str | None, key: str, ev: Evidence | None) -> list[Set]:
    """The sets consistent with what was revealed, relaxing the speed order, then the item evidence, then the
    ability, then moves."""
    def speed_ok(s: Set) -> bool:
        spe = speed(species, level, s)
        return all(outran(s, spe, r) in (None, r.first) for r in (ev.races.get(key, []) if ev else []))

    def item_ok(s: Set) -> bool:
        if item is not None:
            return s.item == item
        if ev is None:
            return True
        return not ((key in ev.unlocked and s.item.startswith("choice"))
                    or (key in ev.no_leftovers and s.item in LEFTOVERS)
                    or (key in ev.no_lifeorb and s.item == "lifeorb" and s.ability != "magicguard"))

    def moves_ok(s: Set, moves: list[str]) -> bool:
        return all(any(matches(r, m) for m in s.moves) for r in moves)

    def ability_ok(s: Set) -> bool:
        return ability is None or s.ability == ability

    for ok in (lambda s: moves_ok(s, revealed) and ability_ok(s) and item_ok(s) and speed_ok(s),
               lambda s: moves_ok(s, revealed) and ability_ok(s) and item_ok(s),
               lambda s: moves_ok(s, revealed) and ability_ok(s),
               lambda s: moves_ok(s, revealed)):
        found = [s for s in sets if ok(s)]
        if found:
            return found
    known = [r for r in revealed if any(matches(r, m) for s in sets for m in s.moves)]
    return [s for s in sets if moves_ok(s, known)] or sets


def posterior(species: str, mon: Pokemon | None, ev: Evidence | None = None,
              key: str = "") -> tuple[int, list[Set], list[float]]:
    """(level, candidate sets, their weights) for the opponent's `species` (`mon`: what poke-env shows of it,
    None = unrevealed)."""
    species = set_species(species)
    level, sets = library().get(species, (None, []))
    if not sets:
        return level or 80, [], []
    revealed = [m for m in (mon.moves if mon else {}) if norm(m) != "struggle"]
    ability = norm(mon.ability) if mon and mon.ability else None
    item = ev.item.get(key) if ev else None
    if item is None and mon is not None and mon.item not in (GEN.UNKNOWN_ITEM, None, "") and (
            ev is None or key not in ev.changed):
        item = mon.item
    level = mon.level if mon else level
    cands = consistent(species, level, sets, revealed, ability, item, key, ev)
    lead = ev is not None and mon is not None and ev.lead == key
    weights = [s.n_lead + LEAD_SMOOTHING * s.n if lead else s.n - s.n_lead + LEAD_SMOOTHING * s.n for s in cands]
    return level, cands, weights


def sample(species: str, mon: Pokemon | None, rng: random.Random, ev: Evidence | None = None,
           key: str = "") -> tuple[int, Set | None]:
    level, cands, weights = posterior(species, mon, ev, key)
    return level, (rng.choices(cands, weights)[0] if cands else None)
