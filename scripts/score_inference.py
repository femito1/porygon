"""Score opponent set inference (`porygon.inference`) offline against the true teams of logged benchmarks.

    python scripts/score_inference.py runs/foulplay-<date>-<job> [...] [--samples 4] [--battles N]

Each benchmark directory (slurm_foulplay / slurm_search with the duplicate server) holds the true teams
(battles.jsonl), the searched decisions (decisions.jsonl) and our side's replays. Every decision's battle
state is rebuilt by replaying the log up to its turn, and the opponent's revealed mons are sampled as the
search would; the rates are those of analyze_benchmark.py's inference report (hidden item / ability / moves,
whole set), so they compare with the logged samples' numbers. `P(true set)` is the posterior mass on the
exact true set (moves, ability, item).
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from poke_env.battle import Battle

sys.path.insert(0, str(Path(__file__).parent))
from analyze_benchmark import Rate, norm, same_move  # noqa: E402
from porygon import inference  # noqa: E402

SKIP = {"", "request", "win", "tie", "inactive", "inactiveoff", "t:", "j", "l", "c", "raw", "html", "uhtml"}


def true_set(t: dict) -> inference.Set:
    return inference.Set(tuple(norm(m) for m in t["moves"]), norm(t["ability"]), norm(t["item"]), t.get("role", ""),
                         t["evs"], t["ivs"], 0, 0)


def fill_ours(battle: Battle, team: list[dict]) -> None:
    """What our requests would have told poke-env: our mons' speed, ability and (original) item."""
    by_species = {t["speciesId"]: t for t in team}
    for mon in battle.team.values():
        t = by_species.get(inference.set_species(mon.species))
        if t is not None and not mon.stats.get("spe"):
            mon._stats = {"spe": inference.speed(norm(mon.species), t["level"], true_set(t))}
            mon._ability = norm(t["ability"])
            if mon.item == inference.GEN.UNKNOWN_ITEM:
                mon._item = norm(t["item"])


def states(log: list[list[str]], me: str, turns: set[int]):
    """(turn, battle) at the start of each of `turns`, replaying `log` (our side's view)."""
    role = next(e[2] for e in log if len(e) > 3 and e[1] == "player" and e[3] == me)
    battle = Battle("replay", me, logging.getLogger("replay"), gen=4)
    battle._player_role = role
    for e in log:
        if len(e) < 2 or e[1] in SKIP:
            if len(e) == 2 and e[1] == "":
                battle._replay_data.append(e)  # the evidence reads the section breaks
            continue
        battle.parse_message(e)
        if e[1] == "turn" and int(e[2]) in turns:
            yield int(e[2]), battle


def score(dirs: list[Path], samples: int, max_battles: int | None, seed: int) -> dict[str, Rate]:
    rng = random.Random(seed)
    rates: dict[str, Rate] = defaultdict(Rate)
    mass: dict[str, list[float]] = defaultdict(list)
    for d in dirs:
        decisions = [json.loads(line) for line in (d / "decisions.jsonl").open()]
        ours = {r["player"] for r in decisions}
        by_battle: dict[str, list[dict]] = defaultdict(list)
        for r in decisions:
            by_battle[r["battle"]].append(r)
        battles = [json.loads(line) for line in (d / "battles.jsonl").open()]
        for b in battles[:max_battles]:
            tag, recs = b["roomid"], by_battle.get(b["roomid"], [])
            replay = d / "replays" / f"{tag}.log"
            if not recs or not replay.exists():
                continue
            me = recs[0]["player"]
            truth = {t["speciesId"]: t for t in (b["p2team"] if b["p1"] in ours else b["p1team"])}
            our_team = b["p1team"] if b["p1"] in ours else b["p2team"]
            per_turn = defaultdict(int)
            for r in recs:
                per_turn[r["turn"]] += 1
            log = [line.rstrip("\n").split("|") for line in replay.open()]
            for turn, battle in states(log, me, set(per_turn)):
                fill_ours(battle, our_team)
                ev = inference.battle_evidence(battle)
                for key, mon in battle.opponent_team.items():
                    t = truth.get(inference.set_species(mon.species))
                    if t is None:
                        rates["species not in the true team"].add(True)
                        continue
                    where = "active" if mon.active else "bench"
                    true_moves = [norm(m) for m in t["moves"]]
                    level, cands, weights = inference.posterior(mon.species, mon, ev, key)
                    if not cands:
                        rates["species not in the library"].add(True)
                        continue
                    total = sum(weights)
                    exact = sum(w for s, w in zip(cands, weights) if s.item == norm(t["item"])
                                and s.ability == norm(t["ability"]) and sorted(s.moves) == sorted(true_moves))
                    for _ in range(per_turn[turn]):
                        mass[where].append(exact / total)
                        for s in rng.choices(cands, weights, k=samples):
                            if mon.item == inference.GEN.UNKNOWN_ITEM:
                                rates[f"{where}: hidden item"].add(s.item == norm(t["item"]))
                            if mon.ability is None:
                                rates[f"{where}: hidden ability"].add(s.ability == norm(t["ability"]))
                            for m in s.moves:
                                if not any(same_move(m, k) for k in mon.moves):
                                    rates[f"{where}: hidden move"].add(any(same_move(m, k) for k in true_moves))
                            rates[f"{where}: whole set right"].add(
                                s.item == norm(t["item"]) and s.ability == norm(t["ability"])
                                and all(any(same_move(m, k) for k in true_moves) for m in s.moves))
                    for name, flagged in (("no leftovers", ev.no_leftovers), ("no life orb", ev.no_lifeorb),
                                          ("unlocked", ev.unlocked)):
                        if key in flagged:  # a false flag is a true item it ruled out
                            bad = {"no leftovers": norm(t["item"]) in inference.LEFTOVERS,
                                   "no life orb": norm(t["item"]) == "lifeorb" and norm(t["ability"]) != "magicguard",
                                   "unlocked": norm(t["item"]).startswith("choice")}[name]
                            rates[f"evidence '{name}' correct"].add(not bad)
                    if ev.races.get(key):
                        ts = true_set(t)
                        spe = inference.speed(norm(mon.species), t["level"], ts)
                        for r in ev.races[key]:
                            call = inference.outran(ts, spe, r)
                            rates["evidence 'speed' decisive"].add(call is not None)
                            if call is not None:
                                rates["evidence 'speed' correct"].add(call == r.first)
                    if key in ev.item:
                        rates["evidence: original item correct"].add(ev.item[key] == norm(t["item"]))
    for where, xs in mass.items():
        rates[f"{where}: P(true set)"].hit, rates[f"{where}: P(true set)"].n = sum(xs), len(xs)
    return rates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--samples", type=int, default=4, help="determinizations per decision, as in the search")
    p.add_argument("--battles", type=int, default=None, help="only the first N battles of each directory")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    for k, v in sorted(score(a.dirs, a.samples, a.battles, a.seed).items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
