"""poke-engine side of the engine-fidelity spike: whole random battles inside poke-engine on the same team pairs and
with the same policies as ps_games.js (random: uniform over the engine's legal options; greedy: the usable move with
the highest basePower x type effectiveness against the foe's current types from Showdown's dex tables, else uniform).
States come from porygon.selfplay.initial_state (the conversion engine self-play uses); chance is sampled from
generate_instructions' branch probabilities. One JSON line per game, as ps_games.js:
    {pair, rep, winner 1/2/0 tie/-2 too long/-1 engine panic, turns, decisions, first_faint}
Timing (stderr, last line): Python-driven us per decision with its split (options / policy / generate / apply).

Options need the spike build's `spike_options` (build_spike_engine.sh); without it they come from a 1000-iteration
MCTS (correct but slow; laptop smoke tests only).

    PORYGON_GEN=9 PYTHONPATH=eng9:src python engine_games.py --teams pairs.jsonl --tables tables-gen9.json \
        --policy random --reps 4 > out.jsonl
    ... engine_games.py --native --teams pairs.jsonl --reps 4      # pure-Rust random games (spike_random_games)
"""

import argparse
import json
import random
import re
import sys
import time

import poke_engine as pe

from porygon.selfplay import initial_state

MAX_DECISIONS = 2000

if hasattr(pe, "spike_options"):
    def options(state):
        return pe.spike_options(state)
else:
    def options(state):
        r = pe.monte_carlo_tree_search(state, iterations=1)
        return [o.move_choice for o in r.side_one], [o.move_choice for o in r.side_two]


def engine_choice(opt: str) -> str:
    """An option string (movechoice_to_string) in generate_instructions' format."""
    if opt == "No Move":
        return "none"
    return opt.removeprefix("switch ")


def move_key(engine_id: str) -> str:
    return re.sub(r"\d+$", "", engine_id) if engine_id.startswith("hiddenpower") else engine_id


class Greedy:
    def __init__(self, tables: dict):
        self.moves, self.types = tables["moves"], tables["types"]

    def eff(self, mtype: str, def_types: list[str]) -> float:
        row = self.types.get(mtype)
        m = 1.0
        if row:
            for d in def_types:
                m *= row.get(d, 1.0)
        return m

    def __call__(self, rng: random.Random, me, foe, opts: list[str]) -> str:
        active = me.pokemon[int(me.active_index)]
        target = foe.pokemon[int(foe.active_index)]
        tera = str(getattr(target, "tera_type", "typeless")).lower()
        if getattr(target, "terastallized", False) and tera not in ("typeless", "stellar"):
            def_types = [tera]
        else:
            def_types = [t.lower() for t in target.types if t.lower() != "typeless"]
        usable = set(opts)
        best, best_moves = 0.0, []
        for mv in active.moves:
            mid = mv.id.lower()  # states rebuilt from Rust spell ids in upper case
            if mid not in usable:
                continue
            d = self.moves.get(move_key(mid))
            if not d or d[2] == "Status" or not d[0]:
                continue
            s = d[0] * self.eff(d[1], def_types)
            if s > best:
                best, best_moves = s, [mid]
            elif s == best and s > 0:
                best_moves.append(mid)
        if best > 0:
            return rng.choice(best_moves)
        return rng.choice(opts)


def all_fainted(side) -> bool:
    return all(p.hp <= 0 for p in side.pokemon)


def n_fainted(state) -> int:
    return sum(p.hp <= 0 for p in state.side_one.pokemon) + sum(p.hp <= 0 for p in state.side_two.pokemon)


def play(teams, policy, rng: random.Random, t: list[float]) -> dict:
    state = initial_state(*teams)
    res = {"winner": -1, "turns": 0, "decisions": 0, "first_faint": -1}
    while True:
        s1, s2 = state.side_one, state.side_two
        d1, d2 = all_fainted(s1), all_fainted(s2)
        if d1 or d2:
            res["winner"] = 0 if d1 and d2 else 2 if d1 else 1
            return res
        if res["decisions"] >= MAX_DECISIONS:
            res["winner"] = -2
            return res
        a1, a2 = s1.pokemon[int(s1.active_index)], s2.pokemon[int(s2.active_index)]
        if not s1.force_switch and not s2.force_switch and a1.hp > 0 and a2.hp > 0:
            res["turns"] += 1
        before = n_fainted(state)
        x0 = time.perf_counter()
        o1, o2 = options(state)
        x1 = time.perf_counter()
        if policy is None:
            c1, c2 = rng.choice(o1), rng.choice(o2)
        else:
            c1, c2 = policy(rng, s1, s2, o1), policy(rng, s2, s1, o2)
        x2 = time.perf_counter()
        branches = pe.generate_instructions(state, engine_choice(c1), engine_choice(c2))
        x3 = time.perf_counter()
        i = rng.choices(range(len(branches)), weights=[b.percentage for b in branches])[0]
        state = state.apply_instructions(branches[i])
        x4 = time.perf_counter()
        t[0] += x1 - x0; t[1] += x2 - x1; t[2] += x3 - x2; t[3] += x4 - x3
        res["decisions"] += 1
        if res["first_faint"] < 0 and n_fainted(state) > before:
            res["first_faint"] = res["turns"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", required=True)
    ap.add_argument("--tables", help="dex_tables.js output (greedy policy)")
    ap.add_argument("--policy", default="random", choices=["random", "greedy"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=10**9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--native", action="store_true", help="pure-Rust random games (spike_random_games)")
    a = ap.parse_args()
    lines = [l for l in open(a.teams) if l.strip()][a.start:a.start + a.count]
    pairs = [json.loads(l) for l in lines]
    if a.native:
        conv0 = time.perf_counter()
        states = [initial_state(*p) for p in pairs]
        conv = time.perf_counter() - conv0
        secs, phases, games = pe.spike_random_games(states, a.reps, MAX_DECISIONS, a.seed)
        reps: dict[int, int] = {}
        for (i, w, turns, dec, ff) in games:
            r = reps.get(i, 0)
            reps[i] = r + 1
            print(json.dumps({"pair": a.start + i, "rep": r, "winner": w, "turns": turns, "decisions": dec,
                              "first_faint": ff}))
        n = sum(g[3] for g in games)
        print(json.dumps({"sim": "engine-native", "seconds": secs, "decisions": n, "games": len(games),
                          "us_per_decision": 1e6 * secs / n,
                          "phase_us": dict(zip(["options", "generate", "sample_apply", "clone"],
                                               [1e6 * p / n for p in phases])),
                          "state_conversion_s": conv}), file=sys.stderr)
        return
    policy = Greedy(json.load(open(a.tables))) if a.policy == "greedy" else None
    t = [0.0] * 4
    total_dec = 0
    t0 = time.perf_counter()
    for k, pair in enumerate(pairs):
        for rep in range(a.reps):
            rng = random.Random(f"{a.seed}-{a.start + k}-{rep}-{a.policy}")
            try:
                res = play(pair, policy, rng, t)
            except BaseException as e:  # pyo3 PanicException is a BaseException
                if isinstance(e, KeyboardInterrupt):
                    raise
                res = {"winner": -1, "turns": 0, "decisions": 0, "first_faint": -1,
                       "error": f"{type(e).__name__}: {e}"[:200]}
            total_dec += res["decisions"]
            print(json.dumps({"pair": a.start + k, "rep": rep, **res}), flush=False)
    secs = time.perf_counter() - t0
    print(json.dumps({"sim": "engine-python", "policy": a.policy, "seconds": secs, "decisions": total_dec,
                      "us_per_decision": 1e6 * secs / max(total_dec, 1),
                      "phase_us": dict(zip(["options", "policy", "generate", "apply"],
                                           [1e6 * x / max(total_dec, 1) for x in t])),
                      "native_options": hasattr(pe, "spike_options")}), file=sys.stderr)


if __name__ == "__main__":
    main()
