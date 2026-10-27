"""Analyse a benchmark directory written with the duplicate server and --log-dir (slurm_foulplay / slurm_search).

    python scripts/analyze_benchmark.py runs/foulplay-<date>-<job> [--top 15]

1. Win rate scored by duplicate pairs (same two teams, swapped): its standard error next to the naive one per
   game, and how the pairs split (won both / split / lost both). A split pair is the teams deciding; won or
   lost both is the play deciding.
2. Opponent inference: what the determinizations assumed about the opponent against the true team, for what
   was still hidden at each decision: items, abilities, moves of revealed mons, and the unrevealed species.
3. In games of lost-both pairs, the decisions after which the network's value of our position fell most,
   with the replay to read.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def move_id(m: str) -> str:
    """'hiddenpowerfire70' (engine) / 'hiddenpowerfire' (sets) / 'hiddenpower' (poke-env) -> comparable id."""
    m = norm(m)
    return m.rstrip("0123456789") if m.startswith("hiddenpower") else m


def same_move(a: str, b: str) -> bool:
    a, b = move_id(a), move_id(b)
    return a == b or ("hiddenpower" in (a, b) and a.startswith("hiddenpower") and b.startswith("hiddenpower"))


POOL = 295  # gen4 random-battle species (pokemon-showdown data/random-battles/gen4/sets.json)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def pairs_report(battles: list[dict], ours: set[str]) -> tuple[list[str], set[str]]:
    by_pair = defaultdict(dict)
    for b in battles:
        by_pair[b["pair"]][b["game"]] = b

    def score(b: dict) -> float:
        return 0.5 if not b["winner"] else float(b["winner"] in ours)

    games = [score(b) for b in battles]
    full = [p for p in by_pair.values() if len(p) == 2]
    means = [(score(p[0]) + score(p[1])) / 2 for p in full]
    lines = []
    if games:
        w = sum(games) / len(games)
        lines.append(f"all games: {w:.1%} ± {1.96 * math.sqrt(w * (1 - w) / len(games)):.1%} (n={len(games)})")
    if len(means) > 1:
        m = sum(means) / len(means)
        sd = math.sqrt(sum((x - m) ** 2 for x in means) / (len(means) - 1))
        se_dup, se_naive = sd / math.sqrt(len(means)), math.sqrt(m * (1 - m) / (2 * len(means)))
        lines.append(f"duplicate pairs: {m:.1%} ± {1.96 * se_dup:.1%} ({len(means)} pairs; naive ± "
                     f"{1.96 * se_naive:.1%}, so pairing is worth {se_naive ** 2 / max(se_dup ** 2, 1e-12):.2f}x the games)")
        won = sum(x == 1 for x in means)
        lost = sum(x == 0 for x in means)
        lines.append(f"pairs won both {won}, split {len(means) - won - lost}, lost both {lost} "
                     f"(incomplete pairs dropped: {len(by_pair) - len(full)})")
    lost_both = {b["roomid"] for p in full for b in p.values()
                 if (score(p[0]) + score(p[1])) == 0}
    return lines, lost_both


class Rate:
    def __init__(self):
        self.hit, self.n = 0, 0

    def add(self, ok: bool) -> None:
        self.hit += ok
        self.n += 1

    def __str__(self) -> str:
        return f"{self.hit / self.n:.1%} (n={self.n})" if self.n else "n/a"



def inference_report(decisions: list[dict], teams: dict[str, list[dict]]) -> list[str]:
    rates = defaultdict(Rate)
    for d in decisions:
        truth = teams.get(d["battle"])
        if not truth:
            continue
        true = {t["speciesId"]: t for t in truth}
        shown = d["revealed"]
        shown_species = {norm(r["species"]) for r in shown}
        hidden_species = set(true) - shown_species
        for sample in d["samples"]:
            for i, mon in enumerate(sample):
                if i < len(shown):
                    r, t = shown[i], true.get(norm(shown[i]["species"]))
                    if t is None:
                        continue
                    where = "active" if r["active"] else "bench"
                    if not r["item_known"]:
                        rates[f"{where}: hidden item"].add(norm(mon["item"]) == norm(t["item"]))
                    if not r["ability_known"]:
                        rates[f"{where}: hidden ability"].add(norm(mon["ability"]) == norm(t["ability"]))
                    for m in mon["moves"]:
                        if not any(same_move(m, k) for k in r["moves"]):
                            rates[f"{where}: hidden move"].add(any(same_move(m, k) for k in t["moves"]))
                    rates[f"{where}: whole set right"].add(
                        norm(mon["item"]) == norm(t["item"]) and norm(mon["ability"]) == norm(t["ability"])
                        and all(any(same_move(m, k) for k in t["moves"]) for m in mon["moves"]))
                elif norm(mon["species"]):
                    rates["unrevealed species in their team"].add(norm(mon["species"]) in hidden_species)
                    rates["unrevealed species, chance level"].hit += len(hidden_species) / POOL
                    rates["unrevealed species, chance level"].n += 1
    return [f"{k}: {v}" for k, v in sorted(rates.items())]


def swing_report(decisions: list[dict], lost: set[str], out: Path, top: int) -> list[str]:
    by_battle = defaultdict(list)
    for d in decisions:
        if d["battle"] in lost:
            by_battle[d["battle"]].append(d)
    drops = []
    for tag, ds in by_battle.items():
        ds.sort(key=lambda d: d["turn"])
        for a, b in zip(ds, ds[1:]):
            drops.append((a["value"] - b["value"], tag, a))
    drops.sort(key=lambda x: -x[0])
    lines = []
    for drop, tag, d in drops[:top]:
        legal = [i for i, m in enumerate(d["mask"]) if m]
        net = max(legal, key=lambda i: d["probs"][i])
        lines.append(f"{tag} turn {d['turn']}: value {d['value']:+.2f} fell {drop:.2f}; played {d['action']} "
                     f"(net argmax {net}, p={d['probs'][d['action']]:.2f}, share={d['shares'][d['action']]:.2f}) "
                     f"{out / 'replays' / (tag + '.log')}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--top", type=int, default=15, help="value drops to list")
    a = p.parse_args()
    battles = load(a.dir / "battles.jsonl")
    decisions = load(a.dir / "decisions.jsonl")
    ours = {d["player"] for d in decisions} | {r["player"] for r in load(a.dir / "results.jsonl")}
    teams = {b["roomid"]: (b["p2team"] if b["p1"] in ours else b["p1team"]) for b in battles}
    print("== results")
    lines, lost_both = pairs_report(battles, ours)
    print("\n".join(lines) or "no battles.jsonl (run without the duplicate server?)")
    print(f"\n== opponent inference ({len(decisions)} decisions, every determinization)")
    print("\n".join(inference_report(decisions, teams)) or "no decisions")
    print(f"\n== largest value drops in lost-both games ({len(lost_both)} games)")
    print("\n".join(swing_report(decisions, lost_both, a.dir, a.top)) or "none")


if __name__ == "__main__":
    main()
