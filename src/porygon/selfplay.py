"""Engine-only self-play: whole gen4 random battles played inside poke-engine, for tuning its leaf evaluation.

Both players see the full state (perfect information) and search it with poke-engine's MCTS for a fixed number
of iterations, each from its own side (the state is mirrored for side two, so both search as side one, like
`porygon.search` does). Chance is sampled from the engine's own branch probabilities (generate_instructions).
Teams come from Showdown's gen4 random-battle generator (scripts/gen_teams.js); every team pair is played twice
with the teams swapped between the players (duplicate pairing), so a team pair's strength cancels in comparisons.

Why the engine's own world: during a search every leaf is a determinized, perfect-information engine state, so
the value evaluate() should approximate is exactly the win probability of such a state under engine play.

Two modes:
  data  both players use the same evaluation; every position (the state before each decision) is recorded with
        its eval features (pe.eval_features), and every transition with its *luck*: V(state reached) minus the
        expectation of V over the engine's chance branches of that transition, V(s) = sigmoid(value logit) of a
        fixed value model (`--value`) and 1 / 0 at the end of the battle. outcome - (sum of later luck) is an
        unbiased, lower-variance target for a position's win probability (a control variate in the style of
        AIVAT / backgammon luck adjustment): every luck term has expectation zero given the position before it.
  h2h   player A (--a) vs player B (--b); only results are kept.

Evaluation configs are JSON files {"weights": [64 floats] or null, "scale": s, "root_weight": r} (`pe.set_eval`;
null = poke-engine's stock evaluate(), "stock" = the stock config {null, 0.0125, 1}).

    node scripts/gen_teams.js 5000 > teams.jsonl
    python -m porygon.selfplay data --teams teams.jsonl --out runs/selfplay/r1 --workers 24 --iterations 20000
    python -m porygon.selfplay h2h --teams teams.jsonl --out runs/h2h/x --a fit.json --b stock --workers 24
"""

import argparse
import gzip
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import poke_engine as pe

from porygon.inference import GEN, norm, stats as calc_stats  # light: no torch (workers restart often)

MAX_DECISIONS = 500  # a battle longer than this is dropped (engine loops, endless stall)
GAME_TIMEOUT = 900  # seconds; SIGALRM's default action kills a worker stuck inside the engine
STOCK = {"weights": None, "scale": 0.0125, "root_weight": 1.0}


@dataclass(frozen=True)
class EvalConfig:
    weights: tuple[float, ...] | None = None
    scale: float = 0.0125
    root_weight: float = 1.0

    @classmethod
    def load(cls, spec: str | dict | None) -> "EvalConfig":
        if spec is None or spec == "stock":
            spec = STOCK
        elif isinstance(spec, str):
            spec = json.loads(Path(spec).read_text())
        w = spec.get("weights")
        return cls(tuple(float(x) for x in w) if w is not None else None, float(spec.get("scale", 0.0125)),
                   float(spec.get("root_weight", 1.0)))

    def apply(self) -> None:
        pe.set_eval(list(self.weights) if self.weights is not None else None, self.scale, self.root_weight)

    def logit(self, f: np.ndarray) -> np.ndarray:
        """The leaf sigmoid's argument for an absolute evaluation (root weight 0): scale * eval."""
        if self.weights is None:
            return self.scale * 100.0 * f[..., 0]
        return self.scale * (f @ np.asarray(self.weights, dtype=np.float64))


# ---------------------------------------------------------------- states


def engine_move(move: str) -> str:
    """poke-engine id (`porygon.search.engine_move`): gen4 hidden power is a typed 70 BP move."""
    move = norm(move)
    return move + "70" if move.startswith("hiddenpower") and move != "hiddenpower" else move


def engine_mon(s: dict[str, Any]) -> pe.Pokemon:
    """A Showdown PokemonSet at full health (as `porygon.search.make_mon` builds mons)."""
    species, level = s["speciesId"], s["level"]
    st = calc_stats(species, level, s["evs"], s["ivs"])
    if species == "shedinja":
        st["hp"] = 1
    dex = GEN.pokedex[species]
    types = tuple((t.lower() for t in dex["types"] + ["typeless"]))[:2]
    ability, item = norm(s["ability"]) or "none", norm(s["item"]) or "none"
    return pe.Pokemon(
        id=species, level=level, types=types, base_types=types, hp=st["hp"], maxhp=st["hp"], ability=ability,
        base_ability=ability, item=item, attack=st["atk"], defense=st["def"], special_attack=st["spa"],
        special_defense=st["spd"], speed=st["spe"], weight_kg=float(dex.get("weightkg", 50.0)),
        moves=[pe.Move(id=engine_move(m), pp=16, disabled=False) for m in s["moves"][:4]],
    )


def initial_state(team_one: list[dict], team_two: list[dict]) -> pe.State:
    def side(team: list[dict]) -> pe.Side:
        return pe.Side(pokemon=[engine_mon(s) for s in team], side_conditions=pe.SideConditions(),
                       last_used_move="switch:0")
    return pe.State(side_one=side(team_one), side_two=side(team_two))


def mirrored(state: pe.State) -> pe.State:
    return pe.State(side_one=state.side_two, side_two=state.side_one, weather=state.weather,
                    weather_turns_remaining=state.weather_turns_remaining, terrain=state.terrain,
                    terrain_turns_remaining=state.terrain_turns_remaining, trick_room=state.trick_room,
                    trick_room_turns_remaining=state.trick_room_turns_remaining, team_preview=state.team_preview)


def winner(state: pe.State) -> int | None:
    """1 / 2 once one side has no mon left, else None."""
    if all(p.hp <= 0 for p in state.side_one.pokemon):
        return 2
    if all(p.hp <= 0 for p in state.side_two.pokemon):
        return 1
    return None


# ---------------------------------------------------------------- games


def choose(cfg: EvalConfig, state: pe.State, iterations: int, rng: random.Random, explore: float) -> str:
    """Side one's choice (generate_instructions' format: a move id or the species to switch to): the most visited
    root option; with probability `explore` one drawn by visit share."""
    cfg.apply()
    opts = pe.monte_carlo_tree_search(state, iterations=iterations).side_one
    if len(opts) > 1 and rng.random() < explore:
        choice = rng.choices(opts, weights=[o.visits + 1e-9 for o in opts])[0].move_choice
    else:
        choice = max(opts, key=lambda o: o.visits).move_choice
    return "none" if choice == "No Move" else choice.removeprefix("switch ")


def value(model: EvalConfig, state: pe.State) -> float:
    w = winner(state)
    if w is not None:
        return 1.0 if w == 1 else 0.0
    return 1.0 / (1.0 + math.exp(-float(model.logit(np.asarray(pe.eval_features(state))))))


def play(teams: list[list[dict]], one: EvalConfig, two: EvalConfig, iterations: int, rng: random.Random,
         record: bool = False, model: EvalConfig | None = None, explore: float = 0.0) -> dict[str, Any] | None:
    """One battle, `one` (with teams[0]) as side one; None if it ran too long. With `record`: the features of
    every position and the luck of every transition under `model`."""
    state = initial_state(*teams)
    feats: list[list[float]] = []
    luck: list[float] = []
    n = 0
    while (w := winner(state)) is None:
        if n >= MAX_DECISIONS:
            return None
        if record:
            feats.append(pe.eval_features(state))
        a1 = choose(one, state, iterations, rng, explore)
        a2 = choose(two, mirrored(state), iterations, rng, explore)
        branches = pe.generate_instructions(state, a1, a2)
        p = np.array([b.percentage for b in branches], dtype=np.float64)
        i = rng.choices(range(len(branches)), weights=p)[0]
        if record:
            nxt = [state.apply_instructions(b) for b in branches]
            v = np.array([value(model, s) for s in nxt])
            luck.append(float(v[i] - (p @ v) / p.sum()))
            state = nxt[i]
        else:
            state = state.apply_instructions(branches[i])
        n += 1
    out: dict[str, Any] = {"winner": w, "decisions": n}
    if record:
        out["features"] = [{j: round(x, 5) for j, x in enumerate(f) if x != 0.0} for f in feats]
        out["luck"] = [round(x, 6) for x in luck]
    return out


# ---------------------------------------------------------------- workers


def worker(args: argparse.Namespace) -> None:
    out = Path(args.out)
    started_path = out / f"started-{args.shard}.txt"
    started = set(started_path.read_text().split()) if started_path.exists() else set()
    rng = random.Random(f"{args.seed}-{args.shard}-{len(started)}")
    a, b = EvalConfig.load(args.a), EvalConfig.load(args.b)
    model = EvalConfig.load(args.value) if args.mode == "data" else None
    pairs = [json.loads(line) for i, line in enumerate(open(args.teams)) if i % args.workers == args.shard]
    pair_ids = [i for i in range(args.shard, args.shard + args.workers * len(pairs), args.workers)]
    # a new file per (re)start: a worker killed mid-game leaves its gzip stream unterminated
    sink_path = out / f"games-{args.shard}-{len(started)}.jsonl.gz"
    with gzip.open(sink_path, "at") as sink, open(started_path, "a") as log:
        for pid, teams in zip(pair_ids, pairs):
            for swap in (0, 1):
                key = f"{pid}:{swap}"
                if key in started:
                    continue
                log.write(key + "\n")
                log.flush()
                signal.alarm(GAME_TIMEOUT)
                t0 = time.time()
                try:
                    # swap 0: A plays teams[0] as side one; swap 1: A plays teams[1], still as side one
                    g = play(teams if not swap else teams[::-1], a, b, args.iterations, rng,
                             record=args.mode == "data", model=model, explore=args.explore)
                except BaseException as e:  # Rust panics are pyo3 PanicExceptions (BaseException)
                    g = {"error": f"{type(e).__name__}: {e}"[:300]}
                signal.alarm(0)
                if g is None:
                    g = {"error": "too long"}
                g |= {"pair": pid, "swap": swap, "seconds": round(time.time() - t0, 1)}
                sink.write(json.dumps(g) + "\n")
                sink.flush()


def launch(args: argparse.Namespace, argv: list[str]) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"argv": argv}, indent=1))
    cmd = [sys.executable, "-m", "porygon.selfplay", "_worker", *argv]
    procs = {i: subprocess.Popen(cmd + ["--shard", str(i)]) for i in range(args.workers)}
    restarts = 0
    while procs:
        time.sleep(5)
        for i, p in list(procs.items()):
            if p.poll() is None:
                continue
            del procs[i]
            if p.returncode != 0 and restarts < 20 * args.workers:  # killed by the game timeout: carry on
                restarts += 1
                print(f"worker {i} exited {p.returncode}; restarting", flush=True)
                procs[i] = subprocess.Popen(cmd + ["--shard", str(i)])
    games = list(read_games(out))
    print(f"{len(games)} games, {sum('error' in g for g in games)} errors, {restarts} restarts", flush=True)
    if args.mode == "h2h":
        print(json.dumps(h2h_summary(games)), flush=True)


def read_games(out: Path):
    for f in sorted(out.glob("games-*.jsonl.gz")):
        try:
            with gzip.open(f, "rt") as fh:
                for line in fh:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:  # a worker killed mid-write
                        continue
        except (EOFError, gzip.BadGzipFile, OSError):  # the stream of a killed worker ends early
            continue


def h2h_summary(games: list[dict]) -> dict[str, Any]:
    """A's score, with its standard error over team pairs (the two games of a pair are correlated); engine panics
    are dropped."""
    by_pair: dict[int, list[float]] = {}
    for g in games:  # a battle cut off at MAX_DECISIONS (a repetition loop) counts as a draw
        if "winner" in g or g.get("error") == "too long":
            by_pair.setdefault(g["pair"], []).append(0.5 if "winner" not in g else float(g["winner"] == 1))
    full = [np.mean(v) for v in by_pair.values() if len(v) == 2]
    allg = [x for v in by_pair.values() for x in v]
    if not full:
        return {"games": len(allg)}
    m, se = float(np.mean(full)), float(np.std(full, ddof=1) / math.sqrt(len(full))) if len(full) > 1 else float("nan")
    return {"games": len(allg), "pairs": len(full), "a_score": round(m, 4), "se": round(se, 4),
            "split_pairs": int(sum(x == 0.5 for x in full)), "a_score_all_games": round(float(np.mean(allg)), 4)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["data", "h2h", "_worker", "summary"])
    p.add_argument("--teams")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--iterations", type=int, default=20000, help="MCTS iterations per decision")
    p.add_argument("--a", default="stock", help="data: both players' evaluation; h2h: player A's")
    p.add_argument("--b", default=None, help="h2h: player B's evaluation (data: = --a)")
    p.add_argument("--value", default="stock", help="data: the value model the luck terms use")
    p.add_argument("--explore", type=float, default=0.0, help="data: P(a decision is drawn by visit share)")
    p.add_argument("--seed", default="0")
    p.add_argument("--shard", type=int, default=0)
    argv = sys.argv[2:]
    args = p.parse_args()
    if args.mode == "summary":
        print(json.dumps(h2h_summary(list(read_games(Path(args.out))))))
        return
    if args.mode == "_worker":
        args.mode = json.loads((Path(args.out) / "config.json").read_text())["mode"]
        args.b = args.b if args.b is not None else args.a
        worker(args)
        return
    if args.mode == "data":
        args.b = args.a
    elif args.b is None:
        p.error("h2h needs --b")
    launch(args, argv)


if __name__ == "__main__":
    main()
