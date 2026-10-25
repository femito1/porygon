"""One-turn lookahead scored by the network's value, instead of poke-engine's evaluation.

For each determinization (`search.convert`), every (our action, their action) pair is simulated for one turn
with poke-engine's `generate_instructions`, every chance outcome is scored by the PPO value head, and the
expected scores form a zero-sum matrix game; our equilibrium strategy is the lookahead's distribution over
actions (`SearchPlayer(value_weight=...)` mixes it into the MCTS visit shares). A turn that ends with only
the opponent to switch (a faint, their U-turn) continues with the switch-in that is worst for us; positions
where we switch are scored directly, since the network makes those decisions in training.

The value head was trained on shaped rewards (Φ = `potential`, the HP / faint balance of
`expert.state_value`, ±1 once the battle is over), so with γ ≈ 1  V(s) ≈ E[Φ(s_end)] − Φ(s): a leaf scores
V(s) + Φ(s), a finished battle Φ(s).

Leaves are encoded by patching the root observation (`embed_battle`) instead of rebuilding a poke-env
Battle: species, revealed moves / items / abilities and the other static features stay, the per-turn ones
come from the engine state (HP, faints, status and its counter, the actives, boosts, volatiles, protect
count, recharge, two-turn move charging, PP, available moves, type matchups against the new actives, screens,
hazards, weather, trick room, counts, turn). An unrevealed opponent mon that comes in is shown with its species
only. Approximations: moves, items and abilities revealed during the turn stay hidden; type changes are ignored;
the trapped flags are kept only while both actives stay in; available moves follow the engine's disabled flags
(choice lock) and taunt, not encore (the gen4 engine does not track the last move).
"""

from functools import cache

import numpy as np
import poke_engine as pe
import torch
from poke_env.battle import Battle, Field, MoveCategory, Pokemon, SideCondition
from stable_baselines3.common.policies import ActorCriticPolicy

from porygon.encoder import (
    ACTIVE_FLAG, BOOSTS, EFFECTS, FIELD_DIM, FIELDS, HAZARDS, MON_DIM, MON_NUM_DIM, MOVE_DIM, MOVEPOOL, N_CAT,
    N_MOVES, SCREENS, STATUSES, TYPES, WEATHERS, elapsed, embed_battle, encode_mon, type_multiplier,
)
from porygon.env import SHAPING
from porygon.search import CHARGE_MOVES, SKIP_EFFECTS, STATUS, WEATHER, Converted, levels

# offsets into a mon token (encoder.encode_mon)
NUM = N_CAT + MOVEPOOL
HP, FAINTED = NUM, NUM + 1
BOOST = NUM + 15 + 2 * len(TYPES) + 1
STATUS0 = BOOST + len(BOOSTS)
COUNTER = STATUS0 + len(STATUSES)
EFFECT0 = COUNTER + 1
ACTIVE = EFFECT0 + len(EFFECTS)
FIRST_TURN, PROTECT, RECHARGE, PREPARING, TRAPPED, OPP_ACTIVE = (ACTIVE + i for i in (2, 3, 4, 5, 7, 8))
MOVES = NUM + MON_NUM_DIM
PP, AVAILABLE, MULT = 3, 4, 5 + 3 + len(TYPES) + 1  # in a move block: pp ratio, available, log2 multiplier, immune
assert ACTIVE == NUM + ACTIVE_FLAG and OPP_ACTIVE == NUM + MON_NUM_DIM - 2

# offsets into the field token (encoder.encode_field)
NW, NF, NS = len(WEATHERS), len(FIELDS), len(SCREENS)
SIDE0 = 2 * NW + 1 + 2 * NF  # our screens, screen ages, hazards; then the opponent's
SIDE_DIM = 2 * NS + len(HAZARDS)
TAIL = SIDE0 + 2 * SIDE_DIM  # turn, unseen opponents, our alive, their alive, force switch, trapped, maybe trapped,
assert TAIL + 14 == FIELD_DIM  # 3 x tera, wait, no active, switches, moves

ENGINE_WEATHER = {v: k for k, v in WEATHER.items()}
ENGINE_STATUS = {v: k for k, v in STATUS.items()}
VOLATILE_EFFECT = {e.name.replace("_", ""): i for i, e in enumerate(EFFECTS) if e not in SKIP_EFFECTS}
SCREEN_ATTR = {SideCondition.REFLECT: "reflect", SideCondition.LIGHT_SCREEN: "light_screen",
               SideCondition.AURORA_VEIL: "aurora_veil", SideCondition.SAFEGUARD: "safeguard",
               SideCondition.MIST: "mist", SideCondition.TAILWIND: "tailwind", SideCondition.LUCKY_CHANT: "lucky_chant"}
HAZARD_ATTR = {SideCondition.STEALTH_ROCK: "stealth_rock", SideCondition.SPIKES: "spikes",
               SideCondition.TOXIC_SPIKES: "toxic_spikes", SideCondition.STICKY_WEB: "sticky_web"}
TRICK_ROOM = FIELDS.index(Field.TRICK_ROOM)
CHARGING = {m.upper() for m in CHARGE_MOVES}
BOOST_ATTR = ["attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost", "speed_boost",
              "accuracy_boost", "evasion_boost"]
assert len(BOOST_ATTR) == len(BOOSTS)


@cache
def unseen_mon(species: str) -> Pokemon:
    """A sampled opponent mon as poke-env shows one that just switched in (read-only)."""
    mon = Pokemon(gen=4, species=species)
    mon._level = levels().get(species, 80)
    return mon


def matchup(move, defender: Pokemon | None) -> tuple[float, float]:
    """The encoder's (log2 multiplier, immune) features of `move` against `defender`."""
    mult = type_multiplier(move, defender, 4)
    return (np.log2(mult) / 2 if mult > 0 else -1.5), float(mult == 0)


def engine_slots(mon: Pokemon) -> list[int | None]:
    """Engine move index of each of an opponent mon's revealed move slots (`search.opp_mon` drops struggle)."""
    revealed = [k for k in mon.moves if k != "struggle"]
    return [revealed.index(k) if k != "struggle" else None for k in list(mon.moves)[:N_MOVES]]


def alive(side: pe.Side) -> list[bool]:
    return [m.hp > 0 for m in side.pokemon]


def potential(state: pe.State, shaping: tuple[float, float, float] = SHAPING) -> tuple[float, bool]:
    """(Φ of `state` from side one's view, battle over): `expert.state_value` on an engine state."""
    fainted, hp, status = shaping
    value, left = 0.0, []
    for sign, side in ((1, state.side_one), (-1, state.side_two)):
        n = 0
        for m in side.pokemon:
            if m.hp <= 0:
                value -= sign * fainted
            else:
                n += 1
                value += sign * hp * m.hp / max(m.maxhp, 1)
                value -= sign * status * (m.status.lower() != "none")
        left.append(n)
    if left[0] and left[1]:
        return value, False
    return value + (left[0] > 0) - (left[1] > 0), True


def needs_switch(side: pe.Side) -> bool:
    """The side has to pick a switch-in before the next turn (its active fainted, or it used U-turn & co.)."""
    mons = side.pokemon
    a = int(side.active_index)
    return (side.force_switch or mons[a].hp <= 0) and any(m.hp > 0 for i, m in enumerate(mons) if i != a)


def solve(m: np.ndarray, iters: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Approximate equilibrium of the zero-sum game `m` (rows maximise): alternating regret matching+ with
    linearly weighted average strategies."""
    def strategy(r: np.ndarray) -> np.ndarray:
        s = r.sum()
        return r / s if s > 0 else np.full(len(r), 1 / len(r))

    rx, ry = np.zeros(m.shape[0]), np.zeros(m.shape[1])
    sx, sy = np.zeros_like(rx), np.zeros_like(ry)
    for t in range(1, iters + 1):
        u = m @ strategy(ry)
        x = strategy(rx)
        rx = np.maximum(rx + u - x @ u, 0)
        x = strategy(rx)
        v = -(x @ m)
        y = strategy(ry)
        ry = np.maximum(ry + v - y @ v, 0)
        sx += t * x
        sy += t * strategy(ry)
    return sx / sx.sum(), sy / sy.sum()


class Root:
    """The root observation of one decision and the static data that patching it for a leaf needs."""

    def __init__(self, battle: Battle):
        self.battle, self.turn = battle, battle.turn
        self.obs = embed_battle(battle)
        own, opp = list(battle.team.values())[:6], list(battle.opponent_team.values())[:6]
        self.own, self.opp = own, opp
        self.own_active = next((i for i, m in enumerate(own) if m.active), 0)
        self.opp_active = next((i for i, m in enumerate(opp) if m is battle.opponent_active_pokemon), 0)
        self.trapped = float(battle.trapped), float(battle.maybe_trapped)
        self.own_moves = [list(m.moves.values())[:N_MOVES] for m in own]
        self.opp_moves = [list(m.moves.values())[:N_MOVES] for m in opp]
        self.opp_slots = [engine_slots(m) for m in opp]
        # matchup features of the opponent's revealed moves against each of our mons: [mon, move, defender, 2]
        self.opp_mult = np.zeros((6, N_MOVES, 6, 2), dtype=np.float32)
        for i, moves in enumerate(self.opp_moves):
            for j, mv in enumerate(moves):
                for k, d in enumerate(own):
                    self.opp_mult[i, j, k] = matchup(mv, d)


class Leaves:
    """Leaf encoder for one determinization of a `Root`."""

    def __init__(self, root: Root, conv: Converted):
        self.root, self.conv = root, conv
        self.state = pe.State.from_string(conv.state_str)
        n_opp = len(root.opp)
        self.unseen = [unseen_mon(m.id.lower()) for m in self.state.side_two.pokemon[n_opp:]]
        defenders = root.opp + self.unseen
        # matchup features of our moves against each of their mons: [mon, move, defender, 2]
        self.own_mult = np.zeros((6, N_MOVES, 6, 2), dtype=np.float32)
        for i, moves in enumerate(root.own_moves):
            for j, mv in enumerate(moves):
                for k, d in enumerate(defenders[:6]):
                    self.own_mult[i, j, k] = matchup(mv, d)
        # token of a sampled opponent mon once it is revealed (moves, item, ability unknown)
        self.unseen_tokens = [encode_mon(m, root.battle, mine=False) for m in self.unseen]

    def encode(self, state: pe.State) -> np.ndarray:
        root, battle = self.root, self.root.battle
        obs = root.obs.copy()
        field, mons = obs[:FIELD_DIM], obs[FIELD_DIM:].reshape(12, MON_DIM)
        s1, s2 = state.side_one, state.side_two
        a1, a2 = int(s1.active_index), int(s2.active_index)
        own, opp = s1.pokemon, s2.pokemon
        turn = root.turn + 1
        stayed = a1 == root.own_active and a2 == root.opp_active and own[a1].hp > 0 and opp[a2].hp > 0

        # opponent engine slot -> token; sampled mons that came in take the next free tokens
        n_opp = len(root.opp)
        tokens: list[int | None] = list(range(6, 6 + n_opp))
        free = 6 + n_opp
        for i in range(n_opp, 6):
            if i == a2 or opp[i].hp < opp[i].maxhp:
                mons[free] = self.unseen_tokens[i - n_opp]
                tokens.append(free)
                free += 1
            else:
                tokens.append(None)

        force = needs_switch(s1)
        for side, engine, toks, a, root_a, mine in ((s1, own, range(len(root.own)), a1, root.own_active, True),
                                                   (s2, opp, tokens, a2, root.opp_active, False)):
            sc = side.side_conditions
            for i, (m, t) in enumerate(zip(engine, toks)):
                if t is None:
                    continue
                row, active, hp = mons[t], i == a, m.hp
                row[HP], row[FAINTED] = hp / max(m.maxhp, 1), hp <= 0
                status = ENGINE_STATUS.get(m.status.lower()) if hp > 0 else None
                was = row[STATUS0:COUNTER].copy()
                row[STATUS0:COUNTER] = 0
                if status is not None:
                    row[STATUS0 + STATUSES.index(status)] = 1
                if status is not None and status.name == "SLP":
                    row[COUNTER] = min(m.sleep_turns, 10) / 10
                elif status is not None and status.name == "TOX":  # poke-env resets it on switch-out
                    row[COUNTER] = min(sc.toxic_count, 10) / 10 if active else 0
                elif not (row[STATUS0:COUNTER] == was).all():
                    row[COUNTER] = 0
                row[ACTIVE], row[FIRST_TURN] = active, active and i != root_a
                row[BOOST:STATUS0] = [getattr(side, b) / 6 for b in BOOST_ATTR] if active else 0
                row[EFFECT0:ACTIVE] = 0
                row[PROTECT] = sc.protect / 3 if active else 0
                row[RECHARGE] = active and "MUSTRECHARGE" in side.volatile_statuses
                row[PREPARING] = active and not CHARGING.isdisjoint(side.volatile_statuses)
                if active:
                    for v in side.volatile_statuses:
                        if v in VOLATILE_EFFECT:
                            row[EFFECT0 + VOLATILE_EFFECT[v]] = 1
                if mine:
                    row[TRAPPED] = active and stayed and root.trapped[0]
                else:
                    row[OPP_ACTIVE] = active
                if i in (a, root_a) and i < (len(root.own) if mine else n_opp):  # only actives spend PP
                    moves = root.own_moves[i] if mine else root.opp_moves[i]
                    em = m.moves
                    for j, mv in enumerate(moves):
                        e = j if mine else root.opp_slots[i][j]
                        if e is not None and e < len(em):
                            row[MOVES + j * MOVE_DIM + PP] = em[e].pp / max(mv.max_pp, 1)

        # our moves against their active, their moves against ours
        for i in range(len(root.own)):
            for j in range(len(root.own_moves[i])):
                mons[i, MOVES + j * MOVE_DIM + MULT:MOVES + j * MOVE_DIM + MULT + 2] = self.own_mult[i, j, a2]
        for i in range(n_opp):
            for j in range(len(root.opp_moves[i])):
                mons[6 + i, MOVES + j * MOVE_DIM + MULT:MOVES + j * MOVE_DIM + MULT + 2] = root.opp_mult[i, j, a1]

        # our available moves at the next decision
        recharge = "MUSTRECHARGE" in s1.volatile_statuses
        avail = self.available(s1, a1) if not (force or recharge) else []
        for i in range(len(root.own)):
            for j in range(len(root.own_moves[i])):
                mons[i, MOVES + j * MOVE_DIM + AVAILABLE] = i == a1 and j in avail

        # field
        field[:2 * NW + 1] = 0
        w = ENGINE_WEATHER.get(state.weather.lower())
        if w is not None:
            k = WEATHERS.index(w)
            field[k], field[NW + k] = 1, elapsed(battle.weather.get(w, root.turn), turn)
        else:
            field[2 * NW] = 1
        field[2 * NW + 1 + TRICK_ROOM] = state.trick_room
        field[2 * NW + 1 + NF + TRICK_ROOM] = (
            elapsed(battle.fields.get(Field.TRICK_ROOM, root.turn), turn) if state.trick_room else 0)
        for base, side, conds in ((SIDE0, s1, battle.side_conditions),
                                  (SIDE0 + SIDE_DIM, s2, battle.opponent_side_conditions)):
            sc = side.side_conditions
            for k, c in enumerate(SCREENS):
                on = getattr(sc, SCREEN_ATTR[c]) > 0
                field[base + k], field[base + NS + k] = on, elapsed(conds.get(c, root.turn), turn) if on else 0
            for k, (c, n) in enumerate(HAZARDS.items()):
                field[base + 2 * NS + k] = min(getattr(sc, HAZARD_ATTR[c]), n) / n
        field[TAIL] = min(turn, 100) / 100
        field[TAIL + 1] = (6 - (free - 6)) / 6
        field[TAIL + 2] = sum(alive(s1)) / 6
        field[TAIL + 3] = sum(alive(s2)) / 6
        field[TAIL + 4] = force
        field[TAIL + 5] = stayed and root.trapped[0]
        field[TAIL + 6] = stayed and root.trapped[1]
        field[TAIL + 10] = field[TAIL + 11] = 0
        bench = sum(m.hp > 0 for i, m in enumerate(own) if i != a1)
        field[TAIL + 12] = 0 if field[TAIL + 5] and not force else bench / 5
        field[TAIL + 13] = len(avail) / 4 if avail or force else 1 / 4  # recharge / struggle: one pseudo-move
        return obs

    def available(self, side: pe.Side, a: int) -> list[int]:
        """Our active's usable move slots: PP left, not disabled (the engine disables a choice user's other
        moves), only the charging move mid-charge, no status moves under taunt."""
        moves = side.pokemon[a].moves[:N_MOVES]
        avail = [j for j, mv in enumerate(moves) if mv.pp > 0 and not mv.disabled and mv.id.lower() != "none"]
        charging = CHARGING & side.volatile_statuses
        if charging:
            avail = [j for j in avail if moves[j].id in charging]
        if "TAUNT" in side.volatile_statuses:
            cats = self.root.own_moves[a]
            avail = [j for j in avail if j < len(cats) and cats[j].category is not MoveCategory.STATUS]
        return avail


def engine_choice(choice: str) -> str:
    """An MCTS option ('switch xatu', 'No Move') as `generate_instructions` takes it."""
    return "none" if choice == "No Move" else choice.removeprefix("switch ")


def options(state: pe.State) -> tuple[list[str], list[str]]:
    r = pe.monte_carlo_tree_search(state, iterations=1)
    return [o.move_choice for o in r.side_one], [o.move_choice for o in r.side_two]


def outcomes(state: pe.State, ours: str, theirs: str, min_prob: float) -> list[tuple[float, pe.State]]:
    """Chance outcomes of one turn; branches below `min_prob` are dropped and the rest renormalised."""
    branches = pe.generate_instructions(state, ours, theirs)
    top = max(b.percentage for b in branches)
    kept = [b for b in branches if b.percentage >= min(min_prob * 100, top)]
    total = sum(b.percentage for b in kept)
    return [(b.percentage / total, state.apply_instructions(b)) for b in kept]


class Lookahead:
    """Network-valued one-turn lookahead over a decision's determinizations."""

    def __init__(self, policy: ActorCriticPolicy, top_k: int | None = None, min_prob: float = 0.0,
                 batch: int = 512):
        self.policy, self.top_k, self.min_prob, self.batch = policy, top_k, min_prob, batch
        self.failed = 0  # determinizations skipped

    def values(self, obs: list[np.ndarray]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(obs), self.batch):
                x = torch.as_tensor(np.stack(obs[i:i + self.batch])).to(self.policy.device)
                o = {"observation": x, "action_mask": torch.ones(len(x), 10, device=x.device)}
                out.append(self.policy.predict_values(o)[:, 0].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def shares(self, battle: Battle, convs: list[Converted], probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Mean (over determinizations) equilibrium probability of every poke-env action index."""
        root = Root(battle)
        legal = mask > 0
        if self.top_k:
            legal &= probs >= np.sort(np.where(legal, probs, -1))[::-1][min(self.top_k, int(legal.sum())) - 1]
        obs: list[np.ndarray] = []
        games = []  # per determinization: (row actions, payoff tree)
        for conv in convs:
            n = len(obs)
            try:
                games.append(self._tree(Leaves(root, conv), legal, obs))
            except BaseException as e:  # a failed sample is skipped; Rust panics are pyo3 PanicExceptions
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                del obs[n:]
                self.failed += 1
        if not games:
            raise RuntimeError("lookahead failed on every determinization")
        leaf = self.values(obs)

        def value(node) -> float:
            if isinstance(node, tuple):  # (leaf index, Φ)
                return float(leaf[node[0]]) + node[1]
            if isinstance(node, list):  # opponent's switch-in: the one worst for us
                return min(sum(p * value(n) for p, n in branch) for branch in node)
            return node  # finished battle

        out = np.zeros(10)
        for actions, tree in games:
            m = np.array([[sum(p * value(n) for p, n in cell) for cell in row] for row in tree])
            x, _ = solve(m)
            for a, p in zip(actions, x):
                out[a] += p
        return out / len(games)

    def _tree(self, leaves: Leaves, legal: np.ndarray, obs: list[np.ndarray]):
        state, conv = leaves.state, leaves.conv
        ours, theirs = options(state)
        rows = [(conv.action(c), engine_choice(c)) for c in ours if c != "No Move"]
        rows = [(a, c) for a, c in rows if a is not None and legal[a]]
        if not rows:
            raise ValueError(f"no legal engine option among {ours}")
        cols = [engine_choice(c) for c in theirs]

        def node(s: pe.State, expand: bool = True):
            phi, done = potential(s)
            if done:
                return phi
            if expand and needs_switch(s.side_two) and not needs_switch(s.side_one):
                a = int(s.side_two.active_index)
                return [[(p, node(c, False)) for p, c in outcomes(s, "none", m.id.lower(), self.min_prob)]
                        for i, m in enumerate(s.side_two.pokemon) if i != a and m.hp > 0]
            obs.append(leaves.encode(s))
            return len(obs) - 1, phi

        tree = [[[(p, node(s)) for p, s in outcomes(state, c1, c2, self.min_prob)] for c2 in cols]
                for _, c1 in rows]
        return [a for a, _ in rows], tree
