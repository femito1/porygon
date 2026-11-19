"""Post-hoc (informational) read of LA2-FP by game length and of lookahead2's overrides.

LA2-FP (`docs/netsearch/LA2-FP-PREREG.md`): arm 1 = lookahead2(cd@805, delta 0.005, K 8), arm 2 = raw cd@805 at T 0.5,
both vs Foul Play gen9 on the same team pairs (block j of every arm is seeded TAG b<j>, pair id i within the block).

Length classes are set per team pair from the mean length of ALL FOUR games of that pair (both arms, both sides). That
is a symmetric function of the two arms, so under "no difference between the arms" the paired difference inside every
class has mean 0 (classifying by one arm's lengths would not: short games are mostly our losses, so regression to the
mean alone would favour the other arm in that arm's short pairs). Game length is still endogenous (a better player may
change it), so the per-class differences are associations, not effects.

Usage: python la2_fp_by_length.py RUN_DIR [--cut 25] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def interval(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else math.nan), math.nan
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def wilson(k: int, n: int) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def kind(a: int) -> str:
    if a < 6:
        return "switch"
    if 22 <= a <= 25:
        return "tera"
    return "move"


def arm_games(run: Path, arm: int) -> tuple[dict, list]:
    """{(block, pair): {game: row}} with row['won'] and row['turns']; plus a list of (block, roomid, row)."""
    pairs: dict = defaultdict(dict)
    rooms = []
    for b in sorted((run / f"arm-{arm}").glob("block-*")):
        j = int(b.name.split("-")[1])
        ours = {r.get("player") for r in load(b / "results.jsonl")} - {None}
        for row in load(b / "battles.jsonl"):
            if not ({row.get("p1"), row.get("p2")} & ours):
                continue
            row["won"] = 1.0 if row.get("winner") in ours else (0.5 if not row.get("winner") else 0.0)
            pairs[(j, int(row["pair"]))][int(row["game"])] = row
            rooms.append((j, row["roomid"], row))
    return pairs, rooms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--cut", type=int, default=25)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    out: dict = {"cut": a.cut}
    p1, rooms1 = arm_games(a.run, 1)
    p2, rooms2 = arm_games(a.run, 2)
    keys = sorted(k for k in set(p1) & set(p2) if len(p1[k]) == 2 and len(p2[k]) == 2)
    out["pairs"] = len(keys)
    print(f"common complete pairs: {len(keys)}")

    # --- paired difference by the pair's symmetric mean length (all four games)
    def pm(p, k):
        return (p[k][0]["won"] + p[k][1]["won"]) / 2

    def mlen(k):
        return statistics.mean([p1[k][g]["turns"] for g in (0, 1)] + [p2[k][g]["turns"] for g in (0, 1)])

    classes = {"short": [k for k in keys if mlen(k) < a.cut], "long": [k for k in keys if mlen(k) >= a.cut]}
    rows = {}
    for name, ks in [("all", keys)] + list(classes.items()):
        d = [pm(p1, k) - pm(p2, k) for k in ks]
        m, hw = interval(d)
        m1, h1 = interval([pm(p1, k) for k in ks])
        m2, h2 = interval([pm(p2, k) for k in ks])
        rows[name] = {"pairs": len(ks), "arm1": m1, "arm1_hw": h1, "arm2": m2, "arm2_hw": h2, "diff": m, "hw": hw,
                      "sd": statistics.stdev(d) if len(d) > 1 else math.nan}
        print(f"[sym mean len {'<' if name == 'short' else '>='}{a.cut}] {name:5s} pairs {len(ks):4d}  "
              f"LA2 {100*m1:5.1f}±{100*h1:4.1f}  raw {100*m2:5.1f}±{100*h2:4.1f}  diff {100*m:+5.1f}±{100*hw:4.1f}")
    s, l = rows["short"], rows["long"]
    inter = s["diff"] - l["diff"]
    ihw = 1.96 * math.sqrt(s["sd"] ** 2 / s["pairs"] + l["sd"] ** 2 / l["pairs"])
    rows["short_minus_long"] = {"diff": inter, "hw": ihw}
    print(f"interaction short - long: {100*inter:+.1f} ± {100*ihw:.1f}")
    out["by_pair_length"] = rows

    # finer bins of the symmetric mean length
    bins = [(0, 20), (20, 25), (25, 30), (30, 40), (40, 1000)]
    fine = []
    for lo, hi in bins:
        ks = [k for k in keys if lo <= mlen(k) < hi]
        d = [pm(p1, k) - pm(p2, k) for k in ks]
        m, hw = interval(d)
        m1, _ = interval([pm(p1, k) for k in ks])
        m2, _ = interval([pm(p2, k) for k in ks])
        fine.append({"lo": lo, "hi": hi, "pairs": len(ks), "arm1": m1, "arm2": m2, "diff": m, "hw": hw})
        print(f"  bin [{lo},{hi}) pairs {len(ks):4d}  LA2 {100*m1:5.1f}  raw {100*m2:5.1f}  diff {100*m:+5.1f}±{100*hw:4.1f}")
    out["by_pair_length_fine"] = fine

    # --- per game, per arm (descriptive; length is that arm's own game length)
    per_game = {}
    for arm, p in ((1, p1), (2, p2)):
        g = [p[k][x] for k in keys for x in (0, 1)]
        sh = [r for r in g if r["turns"] < a.cut]
        lg = [r for r in g if r["turns"] >= a.cut]
        ent = {}
        for name, grp in (("short", sh), ("long", lg)):
            w = sum(r["won"] for r in grp)
            lo, hi = wilson(int(round(w)), len(grp))
            ent[name] = {"games": len(grp), "win": w / len(grp) if grp else math.nan, "lo": lo, "hi": hi}
        ent["median_turns"] = statistics.median(r["turns"] for r in g)
        ent["mean_turns"] = statistics.mean(r["turns"] for r in g)
        ent["losses_short"] = sum(1 for r in sh if r["won"] == 0)
        ent["losses"] = sum(1 for r in g if r["won"] == 0)
        per_game[arm] = ent
        print(f"arm {arm} per game: <{a.cut}: {ent['short']['games']} g {100*ent['short']['win']:.1f}% "
              f"[{100*ent['short']['lo']:.1f}, {100*ent['short']['hi']:.1f}]; >= : {ent['long']['games']} g "
              f"{100*ent['long']['win']:.1f}% [{100*ent['long']['lo']:.1f}, {100*ent['long']['hi']:.1f}]; "
              f"median turns {ent['median_turns']}, mean {ent['mean_turns']:.1f}; losses {ent['losses']} "
              f"({ent['losses_short']} short)")
    # paired difference in the share of short games and in game length (same team pairs)
    dshort = [sum(p1[k][x]["turns"] < a.cut for x in (0, 1)) / 2 - sum(p2[k][x]["turns"] < a.cut for x in (0, 1)) / 2
              for k in keys]
    dlen = [statistics.mean(p1[k][x]["turns"] for x in (0, 1)) - statistics.mean(p2[k][x]["turns"] for x in (0, 1))
            for k in keys]
    ms, hs = interval(dshort)
    ml, hl = interval(dlen)
    per_game["paired_short_share_diff"] = {"diff": ms, "hw": hs}
    per_game["paired_len_diff"] = {"diff": ml, "hw": hl}
    print(f"paired LA2 - raw: share of games < {a.cut} {100*ms:+.1f} ± {100*hs:.1f} pts; mean length {ml:+.2f} ± {hl:.2f} turns")
    out["per_game"] = per_game

    # --- lookahead2's decisions (arm 1 gate rows), joined to their game within the block
    gl = {(j, room): row for j, room, row in rooms1}
    turns_bins = [(1, 2), (2, 6), (6, 11), (11, 21), (21, 31), (31, 1000)]
    tb = defaultdict(lambda: [0, 0])
    trans = Counter()
    by_short = defaultdict(lambda: [0, 0])
    by_won = defaultdict(lambda: [0, 0])
    t_all, leaves, cpu, gaps = [], [], [], []
    per_game_searched = Counter()
    n = ov = 0
    for b in sorted((a.run / "arm-1").glob("block-*")):
        j = int(b.name.split("-")[1])
        for r in load(b / "gate.jsonl"):
            n += 1
            o = bool(r.get("override"))
            ov += o
            t = int(r.get("turn") or 0)
            for lo, hi in turns_bins:
                if lo <= t < hi:
                    tb[(lo, hi)][0] += 1
                    tb[(lo, hi)][1] += o
            if r.get("t") is not None:
                t_all.append(float(r["t"]))
            if r.get("leaves") is not None:
                leaves.append(int(r["leaves"]))
            if r.get("cpu_expand") is not None:
                cpu.append(float(r["cpu_expand"]))
            if o:
                trans[(kind(int(r["net"])), kind(int(r["action"])))] += 1
                gaps.append(float(r.get("gap") or 0))
            g = gl.get((j, r.get("battle")))
            if g is not None:
                per_game_searched[(j, r.get("battle"))] += 1
                c = "short" if g["turns"] < a.cut else "long"
                by_short[c][0] += 1
                by_short[c][1] += o
                w = "won" if g["won"] == 1 else "lost"
                by_won[w][0] += 1
                by_won[w][1] += o
    print(f"arm 1 searched decisions {n}, overrides {ov} ({100*ov/n:.1f}%)")
    ovr = {"searched": n, "overrides": ov, "rate": ov / n}
    ovr["by_turn"] = []
    for (lo, hi), (c, o) in sorted(tb.items()):
        l_, h_ = wilson(o, c)
        ovr["by_turn"].append({"lo": lo, "hi": hi, "searched": c, "overrides": o, "rate": o / c, "ci": [l_, h_]})
        print(f"  turn [{lo},{hi}) searched {c:6d} overrides {o:5d} = {100*o/c:5.1f}% [{100*l_:.1f}, {100*h_:.1f}]")
    tot = sum(trans.values())
    ovr["transitions"] = {f"{x}->{y}": v for (x, y), v in sorted(trans.items())}
    print("  override net-kind -> played-kind: " + ", ".join(f"{x}->{y} {v} ({100*v/tot:.1f}%)"
                                                            for (x, y), v in sorted(trans.items(), key=lambda z: -z[1])))
    for name, dd in (("game length", by_short), ("result", by_won)):
        for c, (s_, o_) in sorted(dd.items()):
            l_, h_ = wilson(o_, s_)
            print(f"  by {name}: {c:5s} searched {s_:6d} override {100*o_/s_:5.1f}% [{100*l_:.1f}, {100*h_:.1f}]")
    ovr["by_game_length"] = {c: {"searched": s_, "rate": o_ / s_} for c, (s_, o_) in by_short.items()}
    ovr["by_result"] = {c: {"searched": s_, "rate": o_ / s_} for c, (s_, o_) in by_won.items()}
    ovr["gap_median"] = statistics.median(gaps) if gaps else math.nan

    def q(xs, f):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else math.nan

    ovr["t"] = {"median": statistics.median(t_all), "mean": statistics.mean(t_all), "q90": q(t_all, 0.9),
                "max": max(t_all)}
    ovr["leaves"] = {"median": statistics.median(leaves), "mean": statistics.mean(leaves), "q90": q(leaves, 0.9)}
    ovr["cpu_expand"] = {"median": statistics.median(cpu), "mean": statistics.mean(cpu)} if cpu else {}
    ovr["searched_per_game"] = statistics.mean(per_game_searched.values())
    print(f"  wall t per decision median {ovr['t']['median']:.2f} s mean {ovr['t']['mean']:.2f} q90 {ovr['t']['q90']:.2f} "
          f"max {ovr['t']['max']:.1f}; leaves median {ovr['leaves']['median']} mean {ovr['leaves']['mean']:.0f} q90 "
          f"{ovr['leaves']['q90']}; cpu_expand median {ovr['cpu_expand'].get('median', math.nan):.2f} mean "
          f"{ovr['cpu_expand'].get('mean', math.nan):.2f} core-s; searched decisions per game "
          f"{ovr['searched_per_game']:.1f}; median override gap {ovr['gap_median']:.4f}")
    out["overrides"] = ovr
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
