"""V3 gates (`.orchestra/fastenv/V3-GATES-PREREG.md`): a v2 observation for a poke-engine state reached from a
decision's root, by patching the root observation (porygon.v2.encoder.encode) with the state's dynamic fields.

The port of v1's `porygon.lookahead.Leaves.encode` to observation spec v2. What it writes is the set D of the
pre-registration (HP, boosts, status and its counter, active / fainted / first turn, protect, recharge, charging,
trapped, volatiles, terastallization, PP, available moves, weather / terrain / trick room and their ages, turn,
counts, force switch / wait, tera flags, screens / hazards, the action mask); species, types, items / abilities /
tera candidates, move ids, stats and levels stay as at the root. An unrevealed opponent mon that is in (active or
damaged) takes the next free opponent slot as the real encoder shows a mon that just came in (species, level,
posterior with nothing revealed).

Counters the conversion cannot carry exactly (sleep / toxic counts, protect count, timed screens and weather) are
the root's value plus the engine's change since the root, so a state equal to the root reproduces the root.

Gen9 only (the gates' scope); needs PORYGON_GEN=9, gloria's .engine-gen9 and netv2's porygon.v2 on the path.

    root = LeafRoot(battle, first_seen)          # once per decision
    enc = LeafEncoder(root, conv)                # once per determinization (search.convert)
    obs = enc.encode(state, turns=1)             # any engine state reached from conv's state
"""

from functools import cache

import numpy as np
import poke_engine as pe
from poke_env.battle import Battle, Effect, Pokemon

from porygon import search as S
from porygon.v2 import encoder as E
from porygon.v2.spec import BOOSTS, N_TYPES, STATUSES, TERA_OFFSET, TERRAINS, WEATHERS
from porygon.v2.spec import FIELD_NUM, MOVE_NUM

COL = E.COL
FCOL = {n: i for i, n in enumerate(FIELD_NUM)}
PP, AVAIL = MOVE_NUM.index("pp_frac"), MOVE_NUM.index("available")
EFFECT0 = E.EFFECT0
ENGINE_STATUS = {"burn": "brn", "freeze": "frz", "paralyze": "par", "poison": "psn", "sleep": "slp", "toxic": "tox"}
ENGINE_WEATHER = {"sun": "sunnyday", "rain": "raindance", "sand": "sandstorm", "hail": "hail", "snow": "snow",
                  "harshsun": "desolateland", "heavyrain": "primordialsea"}
VOLATILE_EFFECT = {e.name.replace("_", ""): k for k, e in enumerate(E.EFFECTS) if e not in S.SKIP_EFFECTS}
SKIP_COLS = [k for k, e in enumerate(E.EFFECTS) if e in S.SKIP_EFFECTS]
SCREEN_ATTR = ["reflect", "light_screen", "aurora_veil", "safeguard", "mist", "tailwind", "lucky_chant"]
SCREEN_LEN = [5, 5, 5, 5, 5, 3, 5]  # search.SCREEN_TURNS, in the encoder's SCREENS order
HAZARD_ATTR = [("stealth_rock", 1), ("spikes", 3), ("toxic_spikes", 2), ("sticky_web", 1)]
BOOST_ATTR = ["attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost", "speed_boost",
              "accuracy_boost", "evasion_boost"]
CHARGING = {m.upper() for m in S.CHARGE_MOVES}
assert len(BOOST_ATTR) == len(BOOSTS) and len(SCREEN_ATTR) == 7


@cache
def is_status_move(move_id: str) -> bool:
    entry = E.gendata(9).moves.get(move_id)
    return bool(entry) and entry.get("category") == "Status"


# ---------------------------------------------------------------- legal options of an engine state

def needs_switch(side: pe.Side) -> bool:
    mons, a = side.pokemon, int(side.active_index)
    return (side.force_switch or mons[a].hp <= 0) and any(m.hp > 0 for i, m in enumerate(mons) if i != a)


def side_options(side: pe.Side, other: pe.Side, tera_ok: bool) -> tuple[list[str], list[int]]:
    """(engine choices, the poke-env action index of each) for one side, in the engine's own frame: switches
    'switch <id>' -> slot index, moves '<id>' -> 6 + j, '<id>-tera' -> 22 + j; ['none'] when the side waits."""
    mons, a = side.pokemon, int(side.active_index)
    switches = [(f"switch {m.id}", i) for i, m in enumerate(mons) if i != a and m.hp > 0]
    if needs_switch(side):
        return [c for c, _ in switches], [i for _, i in switches]
    if needs_switch(other):
        return ["none"], [-1]
    vol = side.volatile_statuses
    active = mons[a]
    if "MUSTRECHARGE" in vol:
        return ["none"], [-1]
    moves = active.moves[:4]
    avail = [j for j, mv in enumerate(moves) if mv.pp > 0 and not mv.disabled and mv.id.lower() != "none"]
    charging = CHARGING & vol
    if charging:
        avail = [j for j in avail if moves[j].id.upper() in charging]
    if "TAUNT" in vol:
        avail = [j for j in avail if not is_status_move(moves[j].id)]
    choices, actions = [moves[j].id for j in avail], [6 + j for j in avail]
    if not avail:
        choices, actions = ["struggle"], [-1]
    if tera_ok and avail and not active.terastallized and not any(m.terastallized for m in mons):
        choices += [f"{moves[j].id}-tera" for j in avail]
        actions += [TERA_OFFSET + j for j in avail]
    if not side.force_trapped and not charging:
        choices += [c for c, _ in switches]
        actions += [i for _, i in switches]
    return choices, actions


def engine_choice(choice: str) -> str:
    """As generate_instructions takes it (lookahead.engine_choice)."""
    return choice.removeprefix("switch ")


def terminal(state: pe.State) -> float | None:
    """1 / 0 (side one wins / loses) when a side has nothing left, else None."""
    one = any(m.hp > 0 for m in state.side_one.pokemon)
    two = any(m.hp > 0 for m in state.side_two.pokemon)
    if one and two:
        return None
    return 1.0 if one else 0.0 if two else 0.5


# ---------------------------------------------------------------- root and leaves

class LeafRoot:
    """One decision: its real v2 observation and the plain data patching it needs (nothing that the live battle
    object changes later is kept)."""

    def __init__(self, battle: Battle, first_seen: int | None = None):
        self.battle_turn = battle.turn
        self.first_seen = first_seen
        self.obs = E.encode(battle)
        own = list(battle.team.values())[:6]
        opp = list(battle.opponent_team.values())[:6]
        self.n_own, self.n_opp = len(own), len(opp)
        self.own_active = next((i for i, m in enumerate(own) if m.active), 0)
        self.opp_active = next((i for i, m in enumerate(opp) if m is battle.opponent_active_pokemon), 0)
        self.own_max_pp = [[max(mv.max_pp, 1) for mv in list(m.moves.values())[:4]] for m in own]
        self.opp_max_pp = []  # per revealed opponent mon: max PP of its revealed moves in engine move order
        for m in opp:
            rev = [k for k in m.moves if E.norm(k) != "struggle"][:4]
            self.opp_max_pp.append([max(m.moves[k].max_pp, 1) for k in rev])
        f = self.obs["field_num"]
        self.trapped, self.maybe_trapped = f[FCOL["trapped"]], f[FCOL["maybe_trapped"]]
        self.used_tera = bool(battle.used_tera)
        self.gen = battle.gen


@cache
def fresh_opp(species: str, level: int) -> dict[str, np.ndarray]:
    """The real encoder's slot for an opponent mon that just came in with nothing revealed (one-slot arrays)."""
    mon = Pokemon(gen=9, species=species)
    mon._level = level
    mon._current_hp, mon._max_hp = 100, 100

    class _B:  # what _encode_opp reads from the battle
        gen, trapped = 9, False
    out = E.empty(9)
    E._encode_opp(out, 6, mon, _B())
    return {k: v[6].copy() for k, v in out.items() if v.ndim >= 1 and v.shape[0] == 12}


def _rem(x: int) -> int:
    return x if x > 0 else 0


class LeafEncoder:
    """Leaves of one determinization of a LeafRoot; `conv`: its search.Converted."""

    def __init__(self, root: LeafRoot, conv: S.Converted):
        self.root = root
        self.state = pe.State.from_string(conv.state_str)
        st = self.state
        self.s1, self.s2 = st.side_one, st.side_two
        o1, o2 = self.s1.pokemon, self.s2.pokemon
        self.r_sleep = ([m.sleep_turns for m in o1], [m.sleep_turns for m in o2])
        self.r_status = ([m.status.lower() for m in o1], [m.status.lower() for m in o2])
        self.r_tox = (self.s1.side_conditions.toxic_count, self.s2.side_conditions.toxic_count)
        self.r_protect = (self.s1.side_conditions.protect, self.s2.side_conditions.protect)
        self.r_screens = ([getattr(self.s1.side_conditions, a) for a in SCREEN_ATTR],
                          [getattr(self.s2.side_conditions, a) for a in SCREEN_ATTR])
        self.r_weather = (st.weather.lower(), st.weather_turns_remaining)
        self.r_terrain = (st.terrain.lower(), st.terrain_turns_remaining)
        self.r_tr = (st.trick_room, st.trick_room_turns_remaining)
        self.r_active = (int(self.s1.active_index), int(self.s2.active_index))
        # volatiles poke-env shows on the root actives that the engine state does not carry: kept while they stay
        self.keep = []
        for side, slot in ((self.s1, self.r_active[0]), (self.s2, 6 + self.r_active[1])):
            vol = side.volatile_statuses
            row = root.obs["mon_num"][slot]
            self.keep.append([k for k in range(len(E.EFFECTS)) if row[EFFECT0 + k] > 0
                              and E.EFFECTS[k].name.replace("_", "") not in vol])
        self.unseen = [(m.id, m.level) for m in o2[root.n_opp:]]

    def encode(self, state: pe.State, turns: int = 0) -> dict[str, np.ndarray]:
        """The leaf's observation; `turns`: full turns since the root (the turn counter and per-turn flags)."""
        root = self.root
        obs = {k: v.copy() for k, v in root.obs.items()}
        s1, s2 = state.side_one, state.side_two
        a1, a2 = int(s1.active_index), int(s2.active_index)
        own, opp = s1.pokemon, s2.pokemon
        hp, boosts, num, mnum = obs["hp"], obs["boosts"], obs["mon_num"], obs["move_num"]

        # opponent engine index -> observation slot; sampled mons that are in take the next free slots
        slots: list[int | None] = list(range(6, 6 + root.n_opp))
        free = 6 + root.n_opp
        for k in range(root.n_opp, 6):
            m = opp[k]
            if k == a2 or m.hp < m.maxhp or m.status.lower() != "none":
                tpl = fresh_opp(*self.unseen[k - root.n_opp])
                for name, v in tpl.items():
                    obs[name][free] = v
                slots.append(free)
                free += 1
            else:
                slots.append(None)

        force1, force2 = needs_switch(s1), needs_switch(s2)
        tera_ok = root.gen >= 9 and not root.used_tera
        choices, actions = side_options(s1, s2, tera_ok)
        sides = ((0, s1, own, list(range(root.n_own)), a1, self.r_active[0]),
                 (1, s2, opp, slots, a2, self.r_active[1]))
        for si, side, mons, toks, a, ra in sides:
            sc = side.side_conditions
            vol = side.volatile_statuses
            stayed = a == ra
            for i, (m, t) in enumerate(zip(mons, toks)):
                if t is None:
                    continue
                row = num[t]
                active = i == a
                was_active = i == ra
                h = m.hp
                row[COL["active"]] = float(active)
                row[COL["fainted"]] = float(h <= 0)
                row[COL["hp_frac"]] = h / max(m.maxhp, 1)
                if si == 0:
                    hp[t] = [max(h, 0), m.maxhp]
                else:
                    hp[t] = [h / max(m.maxhp, 1) * obs["stats"][t, 0], obs["stats"][t, 0]]
                st = m.status.lower() if h > 0 else "none"
                code = ENGINE_STATUS.get(st)
                prev = row[[COL["status_" + s] for s in STATUSES]].copy()
                for s in STATUSES:
                    row[COL["status_" + s]] = float(code == s)
                same_status = st == self.r_status[si][i] and not (t >= 6 + root.n_opp)
                if code == "slp":
                    row[COL["status_counter"]] = (min(max(row[COL["status_counter"]] * 10 + m.sleep_turns
                                                          - self.r_sleep[si][i], 0), 10) / 10
                                                  if same_status else min(m.sleep_turns, 10) / 10)
                elif code == "tox":
                    if active and stayed and same_status:
                        row[COL["status_counter"]] = min(max(row[COL["status_counter"]] * 10 + sc.toxic_count
                                                             - self.r_tox[si], 0), 10) / 10
                    elif active:
                        row[COL["status_counter"]] = min(sc.toxic_count, 10) / 10
                    elif not (was_active and not stayed):
                        pass  # benched since the root: unchanged
                    else:
                        row[COL["status_counter"]] = 0.0
                elif not same_status or not np.array_equal(prev, row[[COL["status_" + s] for s in STATUSES]]):
                    row[COL["status_counter"]] = 0.0
                if m.terastallized and row[COL["terastallized"]] == 0:
                    row[COL["terastallized"]] = 1.0
                    tt = E.type_id(m.tera_type)
                    obs["tera"][t] = tt
                    obs["tera_p"][t] = 0.0
                    obs["tera_p"][t, tt] = 1.0
                if active:
                    boosts[t] = [getattr(side, b) for b in BOOST_ATTR]
                    if not (stayed and turns == 0):
                        row[COL["first_turn"]] = float(not stayed)
                    row[COL["protect_counter"]] = (row[COL["protect_counter"]] * 3 + sc.protect
                                                   - self.r_protect[si]) / 3 if stayed else sc.protect / 3
                    row[COL["must_recharge"]] = float("MUSTRECHARGE" in vol)
                    row[COL["preparing"]] = float(not CHARGING.isdisjoint(vol))
                    keep = {k: row[EFFECT0 + k] for k in self.keep[si]} if stayed else {}
                    eff = row[EFFECT0:EFFECT0 + len(E.EFFECTS)]
                    skip = eff[SKIP_COLS].copy() if stayed and turns == 0 else 0.0  # per-turn (flinch)
                    eff[:] = 0.0
                    eff[SKIP_COLS] = skip
                    for v in vol:
                        if v in VOLATILE_EFFECT:
                            eff[VOLATILE_EFFECT[v]] = 1.0
                    for k, x in keep.items():
                        eff[k] = x
                    if si == 0:
                        row[COL["trapped"]] = float(stayed and root.trapped > 0)
                elif was_active and not stayed:  # switched out since the root: volatiles and boosts reset
                    boosts[t] = 0.0
                    row[EFFECT0:EFFECT0 + len(E.EFFECTS)] = 0.0
                    row[COL["first_turn"]] = 0.0
                    row[COL["protect_counter"]] = 0.0
                    row[COL["must_recharge"]] = row[COL["preparing"]] = row[COL["trapped"]] = 0.0
                # PP
                em = m.moves
                if si == 0 and i < root.n_own:
                    for j, mx in enumerate(root.own_max_pp[i]):
                        if j < len(em):
                            mnum[t, j, PP] = em[j].pp / mx
                        mnum[t, j, AVAIL] = 0.0
                elif si == 1 and i < root.n_opp:
                    for j, mx in enumerate(root.opp_max_pp[i]):
                        if j < len(em):
                            mnum[t, j, PP] = em[j].pp / mx

        # our available moves and the action mask
        mask = obs["action_mask"]
        mask[:] = 0.0
        n_moves = 0
        for c, act in zip(choices, actions):
            if act < 0:
                continue
            mask[act] = 1.0
            if 6 <= act < 10:
                mnum[a1, act - 6, AVAIL] = 1.0
                n_moves += 1
        n_switches = sum(1 for act in actions if 0 <= act < 6)
        if not force1 and not force2 and n_moves == 0:
            n_moves = 1  # recharge / struggle: one pseudo-move, as poke-env lists it

        # field
        f = obs["field_num"]
        w = ENGINE_WEATHER.get(state.weather.lower())
        el = f[FCOL["weather_elapsed"]]
        for x in WEATHERS:
            f[FCOL["weather_" + x]] = float(w == x)
        f[FCOL["weather_elapsed"]] = self._age(w is not None, state.weather.lower() == self.r_weather[0], el,
                                               self.r_weather[1], state.weather_turns_remaining, 5, turns)
        tname = state.terrain.lower()
        t_on = tname in TERRAINS
        for x in TERRAINS:
            f[FCOL["terrain_" + x]] = float(tname == x)
        f[FCOL["terrain_elapsed"]] = self._age(t_on, tname == self.r_terrain[0], f[FCOL["terrain_elapsed"]],
                                               self.r_terrain[1], state.terrain_turns_remaining, 5, turns)
        f[FCOL["trickroom"]] = float(state.trick_room)
        f[FCOL["trickroom_elapsed"]] = self._age(state.trick_room, state.trick_room and self.r_tr[0],
                                                 f[FCOL["trickroom_elapsed"]], self.r_tr[1],
                                                 state.trick_room_turns_remaining, 5, turns)
        f[FCOL["turn"]] = min(root.battle_turn + turns, 100) / 100
        f[FCOL["opp_unseen"]] = (6 - (free - 6)) / 6
        f[FCOL["own_alive"]] = sum(m.hp > 0 for m in own[:root.n_own]) / 6
        f[FCOL["opp_alive"]] = (6 - sum(opp[k].hp <= 0 for k, t in enumerate(slots) if t is not None)) / 6
        f[FCOL["force_switch"]] = float(force1)
        stayed1 = a1 == self.r_active[0] and own[a1].hp > 0 and opp[a2].hp > 0
        f[FCOL["trapped"]] = float(stayed1 and root.trapped > 0)
        f[FCOL["maybe_trapped"]] = float(stayed1 and root.maybe_trapped > 0)
        used = any(m.terastallized for m in own)
        f[FCOL["used_tera"]] = float(used or root.used_tera)
        f[FCOL["opp_used_tera"]] = float(any(m.terastallized for m in opp))
        f[FCOL["can_tera"]] = float(any(TERA_OFFSET <= act for act in actions))
        f[FCOL["wait"]] = float(force2 and not force1)
        f[FCOL["no_active"]] = 0.0
        f[FCOL["n_switches"]] = n_switches / 5
        f[FCOL["n_moves"]] = n_moves / 4

        # screens and hazards
        for si, side in ((0, s1), (1, s2)):
            sc, row = side.side_conditions, obs["side_num"][si]
            for k, (attr, ln) in enumerate(zip(SCREEN_ATTR, SCREEN_LEN)):
                rem = getattr(sc, attr)
                row[k] = float(rem > 0)
                row[7 + k] = self._age(rem > 0, self.r_screens[si][k] > 0, row[7 + k], self.r_screens[si][k],
                                       rem, ln, turns)
            for k, (attr, n) in enumerate(HAZARD_ATTR):
                row[14 + k] = min(getattr(sc, attr), n) / n
        return obs

    @staticmethod
    def _age(on: bool, same: bool, root_age: float, root_rem: int, rem: int, length: int, turns: int) -> float:
        """An *_elapsed column: off -> 0; on since the root -> its root age plus the engine's countdown since (or
        the turns passed, for a permanent one); new -> the turns since it was set, from the engine's countdown."""
        if not on:
            return 0.0
        if same:
            step = (root_rem - rem) if root_rem > 0 and rem > 0 else turns
            return min(root_age * 8 + max(step, 0), 8) / 8
        return min(max(length - rem, 0) if rem > 0 else turns, 8) / 8
