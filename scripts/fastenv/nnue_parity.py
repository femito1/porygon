"""NNUE gate 1 converter parity (docs/netsearch/NNUE-GATE1-PREREG.md, "Converter parity"): porygon.nnue.convert_full
on fastenv's dumped simulator states vs porygon.search.convert(battle, rng, truth=, oracle_all=True) on the same
decisions replayed through poke-env, field by field.

1. `--play N`: N fastenv games (uniform random legal actions, both sides) dumped with host.js --dump-states
   --dump-streams (one file per worker).
2. Each finished game becomes two F2-style records (one per side: `stream` = the side's client stream, `true_team`
   = the opponent's true sets, decisions at the dumped rows' message index `e`) and is replayed with
   netsearch_study.replay_targets (data's StreamReplayer, vendored), which yields the live-identical poke-env
   battle and the weather first-seen turn at each decision.
3. At each decision: A = convert_full(dumped state), B = search.convert(battle, ..., truth, oracle_all=True); both
   go through the engine's string form (pe.State.from_string(to_string())), i.e. what the engine would search, and
   are compared field by field.

Normalisation: a side whose active mon has fainted (forced switch pending) is not compared on its active mon's
fields (boosts, volatiles, last move, substitute, toxic / protect counters, durations; ACTIVE_FIELDS): the engine
resets them on the switch-in, the simulator has already cleared them, poke-env still holds the old values. Likewise
a fainted mon's `terastallized` (the simulator deletes it at the faint, poke-env keeps it).
Rule fields (PREREG): all of side one; side two's species / items / abilities / moves / stats / status / boosts /
active mon; hazards and screens of both sides; weather, terrain, trick room. Mismatch rate > 0.5% of decisions
on any rule field => FAIL (fix before generating). Other fields (side two HP / PP / volatiles / ...) are reported
as information: convert estimates them from what poke-env knows.

    cd $C && PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$C/src $PY scripts/fastenv/nnue_parity.py \\
        --play 60 --workers 6 --dump-dir nnue/gate1/parity --out nnue/gate1/PARITY.txt
    ... nnue_parity.py --dump 'nnue/gate1/parity/dump-*.jsonl' --out PARITY.txt      # compare existing dumps
"""
import argparse
import asyncio
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import netsearch_study as study  # noqa: E402

THRESHOLD = 0.005
SCREEN = ("reflect", "light_screen", "aurora_veil", "safeguard", "mist", "lucky_chant", "tailwind")
HAZARD = ("spikes", "toxic_spikes", "stealth_rock", "sticky_web")
BOOSTS = ("attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost", "speed_boost",
          "accuracy_boost", "evasion_boost")
STATS = ("maxhp", "attack", "defense", "special_attack", "special_defense", "speed")

# Fields known not to match exactly (laptop gen4 30 games + a 3-game gen9 smoke, 2026-11-16): in each case
# convert_full holds the simulator's truth and search.convert an estimate from what poke-env shows. How the gate-1
# student copes is given with each (features.py, step 4).
KNOWN = {
    "field.weather_turns_remaining": "convert counts 5 turns from when it first saw the weather (no Damp Rock / Heat "
        "Rock, and +1 when the weather started after the residual phase, e.g. Drought on a forced switch-in); "
        "convert_full = the simulator's duration. Student: weather turns as a capped scalar from the State "
        "(train = truth); on set H it sees convert's estimate, within 1-3 turns.",
    "field.terrain_turns_remaining": "as weather (Terrain Extender, terrain started after the residual phase). "
        "Student: as weather.",
    "side_one.toxic_count": "convert = poke-env's end-of-turn count, +1 when the poison came after the residual "
        "phase (Toxic Spikes on a forced switch-in, Toxic Orb); convert_full = the simulator's toxic stage. "
        "Student: status one-hot + toxic stage scalar; on set H it sees convert's (+1) value.",
    "side_one.pokemon.sleep_turns": "gen4 only: the simulator draws the sleep length without storing it; convert_full "
        "counts from the first dump that saw the status (1 short if the mon tried to act in the turn it fell asleep). "
        "gen9 stores startTime: exact. Student (gen9): sleep turns scalar from the State.",
    "side_one.pokemon.rest_turns": "follows sleep_turns (rest = the mon put itself to sleep).",
    "side_one.pokemon.pp": "poke-env ignores Pressure (2 PP per use); convert_full = the simulator's PP. "
        "Student: PP is not an input.",
    "side_one.volatile_statuses": "the simulator has volatiles poke-env does not show but the engine knows "
        "(e.g. UNBURDEN); convert_full keeps every one the engine knows. Student: volatile bits from the State "
        "(set H: convert's subset).",
    "side_one.wish": "rare: convert's Wish timer from the log (a failed Wish still counts); convert_full = the "
        "simulator's slot condition (gen5+: the wisher's max HP / 2).",
    "side_one.last_used_move": "poke-env's last-used flag differs from the simulator's lastMove in rare orders "
        "(Pursuit on a switch, moves called by other moves). Student: not an input.",
    "side_one.force_trapped": "convert uses poke-env's `trapped` (the request's); convert_full the simulator's, "
        "which also traps by a hidden ability (request maybeTrapped). Student: not an input (engine-only flag).",
    "side_two.pokemon.maxhp": "gen4: convert recomputes stats from the true set's EVs/IVs with its own formula; "
        "convert_full = the simulator's stats. Student: stats / 100 from the State.",
    "side_two.pokemon.status": "rare: an end-of-turn status (Flame Orb / Toxic Orb) poke-env has not applied yet.",
}

def play(n_games: int, fmt: str, workers: int, out_dir: Path, seed: int) -> list[str]:
    import numpy as np
    from porygon.fastenv import FastEnv, random_actions
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("dump-*.jsonl"):
        f.unlink()
    rng = np.random.default_rng(seed)
    pattern = str(out_dir / "dump-{w}.jsonl")
    done = 0
    with FastEnv(fmt, n_workers=workers, battles_per_worker=4, seed=seed, encoder="stub", dump_states=pattern,
                 dump_streams=True) as env:
        b = env.reset()
        while done < n_games:
            done += int(b.done.sum()) // 2
            b = env.step(random_actions(b.mask[b.needs_action], rng))
    return sorted(str(f) for f in out_dir.glob("dump-*.jsonl"))


def load(paths: list[str]):
    """(games {id: G line}, states {(game, side, e): S line}) of finished, untruncated, uncrashed games."""
    games, states = {}, {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                x = json.loads(line)
                if x["t"] == "G":
                    if not x["crashed"] and not x["truncated"] and "streams" in x:
                        games[x["game"]] = x
                else:
                    states[(x["game"], x["side"], x["e"])] = x
    return games, states


def records(game: dict, states: dict, fmt: str):
    for side in (0, 1):
        dec = [{"idx": e} for (g, s, e) in states if g == game["game"] and s == side]
        yield side, {"id": f"{game['game']}-p{side + 1}", "battle": game["game"].replace(":", "-"), "format": fmt,
                     "side": side, "player": f"p{side + 1}", "stream": game["streams"][side],
                     "true_team": game["teams"][1 - side], "decisions": sorted(dec, key=lambda d: d["idx"])}


# fields of a side that belong to its active mon: when that mon has fainted (a forced switch is pending) the engine
# resets them on the switch-in, and the simulator has already cleared them while poke-env keeps the old values
ACTIVE_FIELDS = set(BOOSTS) | {"volatile_statuses", "last_used_move", "substitute_health", "toxic_count", "protect",
                               "dur_confusion", "dur_encore", "dur_taunt", "dur_yawn", "dur_lockedmove", "dur_slowstart"}


def dead_active(side: dict) -> bool:
    return side["pokemon"][int(side["active_index"])]["hp"] == 0


def flat_side(d: dict, prefix: str, out: dict) -> None:
    dead = dead_active(d)
    for k, v in d.items():
        if k == "pokemon" or (dead and k in ACTIVE_FIELDS):
            continue
        out[f"{prefix}.{k}"] = v


def compare(a: dict, b: dict) -> tuple[dict, dict]:
    """(rule mismatches, info mismatches): field -> (a value, b value) for fields that differ."""
    rule, info = {}, {}

    def diff(bucket, key, x, y):
        if x != y and key not in bucket:
            bucket[key] = (x, y)

    s1a, s1b = a["side_one"], b["side_one"]
    fa, fb = {}, {}
    flat_side(s1a, "side_one", fa)
    flat_side(s1b, "side_one", fb)
    for k in fa:
        diff(rule, k, fa[k], fb.get(k))
    for i, (ma, mb) in enumerate(zip(s1a["pokemon"], s1b["pokemon"])):
        for k in ma:
            if k == "terastallized" and ma["hp"] == 0 and mb["hp"] == 0:
                continue  # the simulator clears a fainted mon's tera (battle.faintMessages), poke-env keeps it
            if k == "moves":  # ids + disabled flags; PP separately (poke-env ignores Pressure)
                diff(rule, "side_one.pokemon.moves", (i, [(m[0], m[2]) for m in ma[k]]), (i, [(m[0], m[2]) for m in mb[k]]))
                diff(rule, "side_one.pokemon.pp", (i, [m[1] for m in ma[k]]), (i, [m[1] for m in mb[k]]))
            else:
                diff(rule, f"side_one.pokemon.{k}", (i, ma[k]), (i, mb[k]))
    s2a, s2b = a["side_two"], b["side_two"]
    by_b = {m["id"]: m for m in s2b["pokemon"] if m["id"] != "none"}
    ids_a = sorted(m["id"] for m in s2a["pokemon"] if m["id"] != "none")
    diff(rule, "side_two.pokemon.id", ids_a, sorted(by_b))
    for ma in s2a["pokemon"]:
        mb = by_b.get(ma["id"])
        if mb is None:
            continue
        for k in ("item", "ability", "status") + STATS:
            diff(rule, f"side_two.pokemon.{k}", (ma["id"], ma[k]), (ma["id"], mb[k]))
        diff(rule, "side_two.pokemon.moves", (ma["id"], sorted(m[0] for m in ma["moves"])),
             (ma["id"], sorted(m[0] for m in mb["moves"])))
        for k in ("hp", "sleep_turns", "rest_turns", "types", "tera_type", "terastallized", "level", "weight_kg"):
            if not (k == "terastallized" and ma["hp"] == 0):
                diff(info, f"side_two.pokemon.{k}", (ma["id"], ma[k]), (ma["id"], mb[k]))
        diff(info, "side_two.pokemon.moves_pp_disabled", (ma["id"], sorted(ma["moves"])), (ma["id"], sorted(mb["moves"])))
    act = lambda s: s["pokemon"][int(s["active_index"])]["id"]  # noqa: E731
    diff(rule, "side_two.active", act(s2a), act(s2b))
    dead2 = dead_active(s2a)
    for k in BOOSTS:
        if not dead2:
            diff(rule, f"side_two.{k}", s2a[k], s2b[k])
    for k in HAZARD + SCREEN:
        diff(rule, f"side_two.{k}", s2a[k], s2b[k])
    for k in ("volatile_statuses", "substitute_health", "wish", "future_sight", "last_used_move", "toxic_count",
              "protect", "dur_confusion", "dur_encore", "dur_taunt", "dur_yawn"):
        if not (dead2 and k in ACTIVE_FIELDS):
            diff(info, f"side_two.{k}", s2a[k], s2b[k])
    for k in ("weather", "weather_turns_remaining", "terrain", "terrain_turns_remaining", "trick_room",
              "trick_room_turns_remaining"):
        diff(rule, f"field.{k}", a[k], b[k])
    return rule, info


async def run(a) -> int:
    netsearch, S, R = study.setup(a.format)
    import poke_engine as pe
    from porygon.nnue.convert_full import convert_full, state_dict
    gen = int(re.match(r"gen(\d+)", a.format).group(1))
    t0 = time.time()
    paths = play(a.play, a.format, a.workers, Path(a.dump_dir), a.seed) if a.play else sorted(
        f for p in a.dump for f in glob.glob(p))
    t_play = time.time() - t0
    games, states = load(paths)
    rule_n, info_n, examples = Counter(), Counter(), defaultdict(list)
    n_dec, n_errors, n_replay_missing, errs = 0, 0, 0, Counter()
    for gid in sorted(games):
        for side, rec in records(games[gid], states, a.format):
            chosen = {d["idx"] for d in rec["decisions"]}
            got = set()
            async for d, battle, first, _ in study.replay_targets(rec, R, netsearch, gen, chosen):
                got.add(d.msg)
                x = states[(gid, side, d.msg)]
                try:
                    sa = pe.State.from_string(convert_full(x["state"]).to_string())
                    conv = S.convert(battle, random.Random(0), first, None, rec["true_team"], True)
                    sb = pe.State.from_string(conv.state_str)
                    da, db = state_dict(sa), state_dict(sb)
                except BaseException as e:  # engine panics are pyo3 PanicExceptions
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    n_errors += 1
                    errs[repr(e)[:160]] += 1
                    continue
                n_dec += 1
                rule, info = compare(da, db)
                for bucket, counter in ((rule, rule_n), (info, info_n)):
                    for k, (va, vb) in bucket.items():
                        counter[k] += 1
                        if len(examples[k]) < a.examples:
                            examples[k].append(f"{gid} p{side + 1} e{d.msg} turn {x['turn']}: full {va!r} vs convert {vb!r}")
            n_replay_missing += len(chosen - got)
    rate = lambda c: c / max(n_dec, 1)  # noqa: E731
    bad = sorted(k for k, c in rule_n.items() if rate(c) > THRESHOLD)
    enough = len(games) >= a.min_games and n_dec >= a.min_decisions
    verdict = "PASS" if enough and not bad and n_errors <= THRESHOLD * max(n_dec, 1) else "FAIL"
    why = ([] if enough else [f"too few games/decisions ({len(games)} / {n_dec}; need {a.min_games} / {a.min_decisions})"]) \
        + ([f"rule fields > {THRESHOLD:.1%}: {', '.join(bad)}"] if bad else []) \
        + ([f"{n_errors} conversion errors"] if n_errors > THRESHOLD * max(n_dec, 1) else [])
    lines = [f"{verdict}" + (f": {'; '.join(why)}" if why else ""),
             f"NNUE gate 1 converter parity: convert_full vs search.convert(truth=, oracle_all=True); format {a.format}",
             f"{len(games)} games, {n_dec} decisions compared, {n_replay_missing} dumped rows "
             f"the replay did not reach, {n_errors} conversion errors; play {t_play:.0f} s, total {time.time() - t0:.0f} s",
             f"inputs: {len(paths)} dump files ({', '.join(paths[:3])}{' ...' if len(paths) > 3 else ''})",
             "", f"RULE fields (mismatch rate per decision; threshold {THRESHOLD:.1%}):"]
    all_rule = sorted(set(rule_n) | {"side_one.pokemon.id", "side_two.pokemon.id", "side_two.active", "field.weather"})
    for k in all_rule:
        lines.append(f"  {k:44s} {rate(rule_n[k]):8.4%}  ({rule_n[k]})")
    lines.append("  (every other rule field: 0 mismatches)")
    lines += ["", "INFO fields (not in the rule; convert estimates them):"]
    for k in sorted(info_n):
        lines.append(f"  {k:44s} {rate(info_n[k]):8.4%}  ({info_n[k]})")
    seen = [k for k in sorted(rule_n) if k in KNOWN]
    if seen:
        lines += ["", "Known differences (convert_full = the simulator's truth; convert = its estimate):"]
        lines += [f"  {k}: {KNOWN[k]}" for k in seen]
    unexplained = [k for k in bad if k not in KNOWN]
    lines.insert(1, f"rule fields > {THRESHOLD:.1%} without a known cause: {', '.join(unexplained) or 'none'}")
    lines += ["", "Examples (full = convert_full, convert = search.convert):"]
    for k in sorted(examples):
        lines.append(f"  {k}:")
        lines += [f"    {e}" for e in examples[k]]
    if errs:
        lines += ["", "Conversion errors:"] + [f"  {c} x {e}" for e, c in errs.most_common(10)]
    text = "\n".join(lines) + "\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    (out.parent / (out.stem + ".json")).write_text(json.dumps(
        {"verdict": verdict, "games": len(games), "decisions": n_dec, "errors": n_errors,
         "rates": {k: rate(c) for k, c in sorted(rule_n.items())} | {k: 0.0 for k in all_rule if k not in rule_n},
         "info_rates": {k: rate(c) for k, c in sorted(info_n.items())}}, indent=1))
    print(text)
    return 0 if a.no_verdict or verdict == "PASS" else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--play", type=int, default=0, help="play this many games first (else --dump)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dump-dir", default="nnue/gate1/parity")
    p.add_argument("--dump", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--min-games", type=int, default=50)
    p.add_argument("--min-decisions", type=int, default=2000)
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--no-verdict", action="store_true", help="exit 0 whatever the verdict (tests)")
    a = p.parse_args()
    os.environ.setdefault("PORYGON_GEN", re.match(r"gen(\d+)", a.format).group(1))
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
