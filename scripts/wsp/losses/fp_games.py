"""Real games of our raw gen9 nets vs Foul Play (duplicate-server battles.jsonl: winner, turns, both true teams):
(1) our win rate by game length; (2) where our edge shrinks: the pair score (our mean over the two games of a
duplicate pair; both sides play both teams) for pairs whose teams contain a feature vs pairs without it, with a
95% interval of the difference. A feature present in either team is played once by each side, so a lower pair
score means our net handles battles with that feature (piloting or facing it) worse than Foul Play does.

    python fp_games.py RUN_DIR [RUN_DIR ...]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HAZ = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
SETUP = {"swordsdance", "nastyplot", "calmmind", "dragondance", "bulkup", "quiverdance", "shellsmash", "bellydrum",
         "agility", "irondefense", "coil", "shiftgear", "victorydance", "tidyup", "filletaway", "curse", "growth",
         "workup", "tailglow", "geomancy", "clangoroussoul", "noretreat", "honeclaws", "rockpolish", "autotomize"}
STATUS = {"toxic", "willowisp", "thunderwave", "spore", "sleeppowder", "yawn", "glare", "stunspore", "hypnosis",
          "toxicspikes", "nuzzle"}
FEATURES = {
    "hazard setter": lambda m: bool(set(m["moves"]) & HAZ),
    "hazard remover": lambda m: bool(set(m["moves"]) & {"defog", "rapidspin", "mortalspin", "tidyup", "courtchange"}),
    "setup move": lambda m: bool(set(m["moves"]) & SETUP),
    "status move": lambda m: bool(set(m["moves"]) & STATUS),
    "Rest": lambda m: "rest" in m["moves"],
    "Substitute": lambda m: "substitute" in m["moves"],
    "Protect": lambda m: "protect" in m["moves"],
    "Trick/Switcheroo": lambda m: bool(set(m["moves"]) & {"trick", "switcheroo"}),
    "Choice item": lambda m: m.get("item", "").startswith("Choice"),
    "Tera Blast": lambda m: "terablast" in m["moves"],
    "pivot move": lambda m: bool(set(m["moves"]) & {"uturn", "voltswitch", "flipturn", "partingshot", "teleport",
                                                     "chillyreception", "shedtail"}),
    "Illusion/Ditto": lambda m: m.get("ability") == "Illusion" or m["species"] == "Ditto",
    "weather setter": lambda m: m.get("ability") in ("Drought", "Drizzle", "Sand Stream", "Snow Warning",
                                                       "Orichalcum Pulse"),
    "Encore/Taunt": lambda m: bool(set(m["moves"]) & {"encore", "taunt"}),
    "phazer/Haze": lambda m: bool(set(m["moves"]) & {"roar", "whirlwind", "dragontail", "haze", "clearsmog",
                                                      "circlethrow"}),
    "priority": lambda m: bool(set(m["moves"]) & {"extremespeed", "suckerpunch", "aquajet", "bulletpunch",
                                                  "machpunch", "iceshard", "shadowsneak", "quickattack", "jetpunch",
                                                  "vacuumwave", "grassyglide", "accelerock", "firstimpression",
                                                  "fakeout"}),
}


def wilson(k, n):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def mean_ci(xs):
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), n
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(v / n), n


def main():
    games = []
    pairs = defaultdict(list)
    for d in sys.argv[1:]:
        for line in (Path(d) / "battles.jsonl").open():
            r = json.loads(line)
            ours_is_p1 = not r["p1"].startswith("fp")
            us = r["p1"] if ours_is_p1 else r["p2"]
            s = 0.5 if not r.get("winner") else float(r["winner"] == us)
            games.append((r["turns"], s))
            our_team = r["p1team"] if ours_is_p1 else r["p2team"]
            fp_team = r["p2team"] if ours_is_p1 else r["p1team"]
            pairs[(d, r["pair"])].append((s, our_team, fp_team, r["turns"]))
    print(f"{len(games)} games, {len(pairs)} pairs from {len(sys.argv) - 1} runs")
    print("our win rate by game length (turns):")
    for lo, hi in [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 1000)]:
        sel = [s for t, s in games if lo <= t < hi]
        p, a, b = wilson(sum(sel), len(sel))
        print(f"  {lo:3d}-{hi:<4d} n {len(sel):5d}  {100 * p:5.1f}% [{100 * a:.1f}, {100 * b:.1f}]")
    full = {k: v for k, v in pairs.items() if len(v) == 2}
    base = mean_ci([(v[0][0] + v[1][0]) / 2 for v in full.values()])
    print(f"pair score overall {100 * base[0]:.1f}% ± {100 * base[1]:.1f} ({base[2]} pairs)")
    # pair-level mean game length vs pair score
    print("pair score by the pair's mean game length:")
    for lo, hi in [(0, 20), (20, 25), (25, 30), (30, 40), (40, 1000)]:
        sel = [(v[0][0] + v[1][0]) / 2 for v in full.values() if lo <= (v[0][3] + v[1][3]) / 2 < hi]
        m, h, n = mean_ci(sel)
        print(f"  {lo:3d}-{hi:<4d} pairs {n:4d}  {100 * m:5.1f}% ± {100 * h:.1f}")
    print("pair score with vs without a feature in either team (count of mons with it across both teams >= k):")
    for name, f in FEATURES.items():
        cnt = {k: sum(f(m) for m in v[0][1] + v[0][2]) for k, v in full.items()}
        med = sorted(cnt.values())[len(cnt) // 2]
        k_thr = max(1, med + 1) if med >= 1 else 1
        w = [(v[0][0] + v[1][0]) / 2 for k, v in full.items() if cnt[k] >= k_thr]
        wo = [(v[0][0] + v[1][0]) / 2 for k, v in full.items() if cnt[k] < k_thr]
        if len(w) < 20 or len(wo) < 20:
            continue
        mw, hw, nw = mean_ci(w)
        mo, ho, no = mean_ci(wo)
        se = math.sqrt((hw / 1.96) ** 2 + (ho / 1.96) ** 2)
        print(f"  {name:18s} >= {k_thr}: {100 * mw:5.1f}% ({nw:4d} pairs) vs {100 * mo:5.1f}% ({no:4d}); "
              f"diff {100 * (mw - mo):+5.1f} [{100 * (mw - mo - 1.96 * se):+.1f}, {100 * (mw - mo + 1.96 * se):+.1f}]")


if __name__ == "__main__":
    main()
