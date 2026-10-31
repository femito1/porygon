"""Opponent set inference for the search's determinizations.

Revealed opponent mons get a whole set (moves, ability, item, EVs / IVs) drawn from a library of gen4
random-battle sets made by Showdown's own generator (`scripts/build_set_library.js`), in proportion to how
often the generator makes it, among the sets consistent with everything the battle has shown:
  - the moves it used, its ability and its original item when revealed;
  - the lead (slot 0, the only one the generator may give a Focus Sash) or not;
  - what the log rules out (`Evidence`): no choice item after two different moves in one stint, no Leftovers /
    Black Sludge after an end of turn spent damaged without a heal, no Life Orb after a hit without recoil,
    the speed order of each turn both mons moved (`races`; Choice Scarf);
  - the damage each of its hits did to our mon (exact HP), against poke-engine's damage range for each set
    (`hits`), which tells Choice Band from Life Orb / Silk Scarf / no boosting item.
If nothing is consistent the damage evidence is dropped first, then the speed order, then the other item
evidence, then the ability, then the moves that no set has.
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
    hits: dict[str, list["Hit"]] = field(default_factory=dict)  # its attacks on our mons (`hits`)


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


@dataclass(frozen=True)
class Ours:
    """One of our mons as our requests describe it."""
    species: str
    level: int
    types: tuple[str, str]
    maxhp: int
    stats: tuple[int, int, int, int, int]  # atk, def, spa, spd, spe
    ability: str
    item: str
    weight: float


@dataclass(frozen=True)
class Hit:
    """One of the opponent's attacks that hit our mon directly, under conditions the damage check models: no
    crit, no stat stages on either mon, no screens, a fixed-power single-hit move, both items unchanged."""
    move: str
    dealt: int  # HP our mon lost
    ko: bool  # it fainted: `dealt` is a lower bound
    ours: Ours
    our_hp: int  # before the hit
    their_hp: float  # the attacker's HP fraction (Blaze & co.)
    their_status: str
    our_status: str  # Marvel Scale
    weather: str


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


def races(log: list[list[str]], role: str, ours: dict[str, Ours]) -> dict[str, list[Race]]:
    """Per opponent (`role`) mon, the turns it and our active mon both moved with the same priority and no Trick
    Room. `ours`: our mons by key; turns where our speed is uncertain (a speed ability or item, an item that
    changed during the battle) are left out."""
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
        spe, ability, item = ours[mine].stats[4], ours[mine].ability, ours[mine].item
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


def our_mons(battle: AbstractBattle) -> dict[str, Ours]:
    out = {}
    for key, m in battle.team.items():
        stats = tuple(m.stats.get(k) or 0 for k in ("atk", "def", "spa", "spd", "spe"))
        if all(stats) and m.max_hp:
            types = ([t.name.lower() for t in m.types if t is not None] + ["typeless", "typeless"])[:2]
            out[key] = Ours(m.species, m.level, (types[0], types[1]), m.max_hp, stats, norm(m.ability or ""),
                            norm(m.item or ""), float(m.weight))
    return out


def battle_evidence(battle: AbstractBattle) -> Evidence:
    role = battle.opponent_role or "p2"
    ev = evidence(battle._replay_data, role)
    ours = our_mons(battle)
    ev.races = races(battle._replay_data, role, ours)
    ev.hits = hits(battle._replay_data, role, ours)
    return ev


def stats(species: str, level: int, evs: dict[str, int], ivs: dict[str, int]) -> dict[str, int]:
    """Neutral-nature stats (random battles have no natures)."""
    base = GEN.pokedex[species]["baseStats"]
    out = {"hp": (2 * base["hp"] + ivs["hp"] + evs["hp"] // 4) * level // 100 + level + 10}
    for st in ("atk", "def", "spa", "spd", "spe"):
        out[st] = (2 * base[st] + ivs[st] + evs[st] // 4) * level // 100 + 5
    return out


def speed(species: str, level: int, s: Set) -> int:
    return stats(species, level, s.evs, s.ivs)["spe"]


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


# moves whose power depends on more than the two mons' sets and HP (or hit more than once / hit the user)
VARIABLE_POWER = {"eruption", "waterspout", "flail", "reversal", "gyroball", "payback", "avalanche", "revenge",
                  "pursuit", "facade", "wakeupslap", "brine", "punishment", "crushgrip", "wringout", "assurance",
                  "magnitude", "present", "naturalgift", "trumpcard", "spitup", "beatup", "fling", "explosion",
                  "selfdestruct", "futuresight", "doomdesire", "struggle", "rollout", "iceball", "furycutter",
                  "metronome", "smellingsalt", "weatherball", "return", "frustration", "lastresort", "focuspunch",
                  "chatter"}  # poke-engine's Chatter has the wrong gen4 power
PINCH = {"blaze", "torrent", "overgrow", "swarm"}  # x1.5 at <= 1/3 HP, which a percentage only roughly shows
DAMAGE_ABILITIES = {"rivalry", "flashfire", "plus", "minus", "normalize", "sniper", "superluck", "download", "trace"}
TYPE_CHANGING = {"colorchange", "forecast"}  # our mon's type may not be its species' any more
DAMAGE_SLACK = 2, 0.03  # HP, fraction: rounding differences between poke-engine and Showdown


def hits(log: list[list[str]], role: str, ours: dict[str, Ours]) -> dict[str, list[Hit]]:
    """Per opponent (`role`) mon, its clean hits (`Hit`) on our mons."""
    out: dict[str, list[Hit]] = {}
    active: dict[str, str] = {}
    hp: dict[str, float] = {}
    our_hp: dict[str, int] = {}
    status: dict[str, str] = {}
    staged: set[str] = set()  # mons with any stat stage change this stint
    changed: set[str] = set()
    screens: set[str] = set()  # sides with Reflect / Light Screen up
    roosted: set[str] = set()  # mons that used Roost this turn (gen4: no Flying type until the turn ends)
    weather = ""
    move: tuple[str, str, bool] | None = None  # (attacker, move, crit) of the opponent's move being resolved

    for e in log:
        if len(e) < 3:
            if len(e) == 2:
                move = None
            continue
        kind, key = e[1], _key(e[2])
        side = key[:2]
        if kind == "turn":
            roosted.clear()
        elif kind == "-singleturn" and e[3:4] == ["move: Roost"]:
            roosted.add(key)
        if kind in ("switch", "drag", "replace"):
            staged.discard(active.get(side, ""))
            active[side] = key
            if len(e) > 4:
                parts = e[4].split()
                status[key] = parts[1] if len(parts) > 1 else ""
                if side == role:
                    hp[key] = _hp(e[4])
                elif "/" in parts[0]:
                    our_hp[key] = int(parts[0].split("/")[0])
            move = None
        elif kind == "move":
            move = None
            data = GEN.moves.get(norm(e[3]) if len(e) > 3 else "", {})
            if (side == role and not any(x.startswith("[from]") for x in e[4:]) and data.get("basePower", 0) > 0
                    and "multihit" not in data and norm(e[3]) not in VARIABLE_POWER):
                move = (key, norm(e[3]), False)
        elif kind == "-crit" and move:
            move = (move[0], move[1], True)
        elif kind in ("-damage", "-heal", "-sethp") and len(e) > 3:
            if side == role:
                hp[key] = _hp(e[3])
                continue
            before = our_hp.get(key)
            parts = e[3].split()
            now = 0 if parts[0] == "0" or e[3].endswith("fnt") else int(parts[0].split("/")[0])
            our_hp[key] = now
            if kind == "-damage" and len(e) == 4 and move and before and key == active.get(side):
                attacker, mv, crit = move
                move = None  # only the first damage message is the hit
                clean = not (crit or attacker in staged or key in staged or side in screens or attacker in changed
                             or key in changed or key in roosted or attacker != active.get(role))
                mine = ours.get(key)
                if clean and mine and mine.ability not in TYPE_CHANGING and mine.species != "ditto" and now != 1:
                    out.setdefault(attacker, []).append(Hit(mv, before - now, now == 0, mine, before,  # 1 HP: Sash?
                                                            hp.get(attacker, 1.0), status.get(attacker, ""),
                                                            status.get(key, ""), weather))
        elif kind in ("-boost", "-unboost", "-setboost", "-clearboost", "-swapboost", "-copyboost", "-invertboost",
                      "-clearnegativeboost", "-transform"):
            staged.add(key)
        elif kind == "-clearallboost":
            staged.update(active.values())
        elif kind == "-status" and len(e) > 3:
            status[key] = e[3]
        elif kind == "-curestatus":
            status[key] = ""
        elif kind in ("-item", "-enditem"):
            if kind == "-enditem" or any(x.startswith("[from] move:") for x in e[4:]):
                changed.add(key)
        elif kind == "-weather":
            if "[upkeep]" not in e:
                weather = {"RainDance": "rain", "SunnyDay": "sun", "Sandstorm": "sand", "Hail": "hail"}.get(e[2], "")
        elif kind in ("-sidestart", "-sideend") and len(e) > 3 and ("Reflect" in e[3] or "Light Screen" in e[3]):
            (screens.add if kind == "-sidestart" else screens.discard)(side)
    return out


_DAMAGE: dict[tuple, tuple[int, int] | None] = {}


def damage_range(species: str, level: int, s: Set, hit: Hit) -> tuple[int, int] | None:
    """(min, max) non-crit damage poke-engine gives set `s` for `hit`; None: not computable."""
    move = hit.move
    if move == "hiddenpower":
        move = next((m for m in s.moves if m.startswith("hiddenpower")), "hiddenpower")
    if move.startswith("hiddenpower") and move != "hiddenpower":
        move = move + "70"
    key = (species, level, move, s.ability, s.item, tuple(s.evs.values()), tuple(s.ivs.values()), hit)
    if key not in _DAMAGE:
        if len(_DAMAGE) > 200_000:  # a long benchmark process: hits of finished battles never come back
            _DAMAGE.clear()
        _DAMAGE[key] = _engine_damage(species, level, move, s, hit)
    return _DAMAGE[key]


def _engine_damage(species: str, level: int, move: str, s: Set, hit: Hit) -> tuple[int, int] | None:
    import poke_engine as pe  # poke_engine is only needed for the damage evidence

    dex = GEN.pokedex[species]
    stat = stats(species, level, s.evs, s.ivs)
    maxhp = stat["hp"]
    types = tuple((t.lower() for t in dex["types"] + ["typeless"]))[:2]
    statuses = {"brn": "burn", "par": "paralyze", "psn": "poison", "tox": "toxic", "slp": "sleep", "frz": "freeze"}
    status = statuses.get(hit.their_status, "none")
    o = hit.ours
    try:
        theirs = pe.Pokemon(id=species, level=level, types=types, base_types=types,
                            hp=max(1, round(hit.their_hp * maxhp)), maxhp=maxhp, ability=s.ability,
                            base_ability=s.ability, item=s.item or "none", attack=stat["atk"], defense=stat["def"],
                            special_attack=stat["spa"], special_defense=stat["spd"], speed=stat["spe"],
                            status=status, weight_kg=float(dex.get("weightkg", 50.0)),
                            moves=[pe.Move(id=move, pp=16)])
        mine = pe.Pokemon(id=o.species, level=o.level, types=o.types, base_types=o.types, hp=hit.our_hp,
                          maxhp=o.maxhp, ability=o.ability or "none", base_ability=o.ability or "none",
                          item=o.item or "none", attack=o.stats[0], defense=o.stats[1], special_attack=o.stats[2],
                          special_defense=o.stats[3], speed=o.stats[4], weight_kg=o.weight,
                          status=statuses.get(hit.our_status, "none"),
                          moves=[pe.Move(id="splash", pp=16)])
        rest = [pe.Pokemon.create_fainted()] * 5
        state = pe.State(side_one=pe.Side(pokemon=[mine] + rest), side_two=pe.Side(pokemon=[theirs] + rest),
                         weather=hit.weather or "none", weather_turns_remaining=-1 if hit.weather else 0)
        _, (high, _crit) = pe.calculate_damage(state, "splash", move, False)
    except BaseException as e:  # unknown move / pyo3 PanicException
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return None
    return (int(high * 0.85), int(high)) if high > 0 else None


def damage_fits(species: str, level: int, s: Set, hit: Hit) -> bool | None:
    """Whether set `s` explains `hit`; None: not decidable (an ability whose effect is not modelled, no range)."""
    if s.ability in DAMAGE_ABILITIES or (s.ability in PINCH and 0.28 < hit.their_hp < 0.4):
        return None
    rng = damage_range(species, level, s, hit)
    if rng is None:
        return None
    hp, frac = DAMAGE_SLACK
    lo, hi = rng[0] * (1 - frac) - hp, rng[1] * (1 + frac) + hp
    return hit.dealt <= hi if hit.ko else lo <= hit.dealt <= hi  # a KO: it took at least what was left


# ---------------------------------------------------------------- sampling

def consistent(species: str, level: int, sets: list[Set], revealed: list[str], ability: str | None,
               item: str | None, key: str, ev: Evidence | None) -> list[Set]:
    """The sets consistent with what was revealed, relaxing the damage evidence, then the speed order, then the
    item evidence, then the ability, then moves."""
    def speed_ok(s: Set) -> bool:
        spe = speed(species, level, s)
        return all(outran(s, spe, r) in (None, r.first) for r in (ev.races.get(key, []) if ev else []))

    def damage_ok(s: Set) -> bool:
        return all(damage_fits(species, level, s, h) is not False for h in (ev.hits.get(key, []) if ev else []))

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

    for ok in (lambda s: moves_ok(s, revealed) and ability_ok(s) and item_ok(s) and speed_ok(s) and damage_ok(s),
               lambda s: moves_ok(s, revealed) and ability_ok(s) and item_ok(s) and speed_ok(s),
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
