"""Results of a network + search slot run (netsearch_slot.sh; test plan NETSEARCH-DESIGN.md §6): the looks after
each block, the futility stops, and the per-arm report with the Stage 2 selection. Pure functions of the run
directory, so every number can be recomputed after the run.

Run layout (netsearch_slot.sh): RUN/plan.env, RUN/manifest.txt, RUN/looks.jsonl, RUN/arm-K/block-J/ with the
server's battles.jsonl, the client's results.jsonl / summary.json / evaluate.log and, for netsearch arms, gate.jsonl
(one line per searched decision); RUN/arm-K/STOPPED after a futility stop.

    python netsearch_slot.py look RUN K --blocks-max B [--futility 0.5]   # exit 20: arm K stops (futility)
    python netsearch_slot.py report RUN [--json OUT] [--go 0.515] [--tie 0.01] [--void-fallbacks 0.02]
    python netsearch_slot.py paired RUN K1 K2 [--json OUT]   # Stage 3 Q3: arm K1 - arm K2 per team pair
    python netsearch_slot.py fake DIR --games N --p P --a NAME --seed TAG  # dry runs: synthetic block logs

Scoring is netv2's (porygon.v2.evaluate.pair_means / interval, checked equal in tests/test_netsearch_slot.py): a pair
is the two games of one pair id in one block's server log, scored (s0 + s1) / 2 with a tie = 1/2; the interval is
mean +- 1.96 sd / sqrt(pairs). An arm's pairs are taken block by block in block order, so the data behind look j stay
fixed as later blocks arrive (nested looks).
"""
import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

FUTILITY_EXIT = 20


# ---------------------------------------------------------------- scoring (= porygon.v2.evaluate's)

def norm_id(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def game_score(winner: str | None, ours: set[str]) -> float:
    return 0.5 if not winner else float(norm_id(winner) in ours)


def pair_means(battles: list[dict], ours: set[str]) -> tuple[list[float], int, int]:
    """(pair means in pair-id order, games of ours, incomplete pairs) of one server log."""
    by_pair: dict[int, dict[int, dict]] = defaultdict(dict)
    games = 0
    for b in battles:
        if not ({norm_id(b.get("p1")), norm_id(b.get("p2"))} & ours):
            continue
        by_pair[int(b["pair"])][int(b["game"])] = b
        games += 1
    means = [(game_score(p[0].get("winner"), ours) + game_score(p[1].get("winner"), ours)) / 2
             for _, p in sorted(by_pair.items()) if len(p) == 2 and 0 in p and 1 in p]
    return means, games, sum(len(p) != 2 for p in by_pair.values())


def interval(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return math.nan, math.nan
    m = sum(xs) / n
    if n < 2:
        return m, math.nan
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.open():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # a line cut by a killed process
                    pass
    return rows


# ---------------------------------------------------------------- run directory

def blocks_of(run: Path, arm: int) -> list[Path]:
    """arm-K/block-J directories in block order."""
    d = run / f"arm-{arm}"
    found = [p for p in d.glob("block-*") if p.is_dir() and re.fullmatch(r"block-\d+", p.name)]
    return sorted(found, key=lambda p: int(p.name.split("-")[1]))


def arm_stats(run: Path, arm: int, upto: int | None = None) -> dict:
    """Pooled pairs of arm K over its blocks (the first `upto` blocks, in order), with the client-side counts,
    the gate log and both players' search statistics."""
    means, games, incomplete, blocks = [], 0, 0, []
    stats_a: dict[str, float] = defaultdict(float)
    stats_b: dict[str, float] = defaultdict(float)
    gate = {"decisions": 0, "overrides": 0, "undefined_gap": 0}
    seconds, statuses = 0.0, defaultdict(int)
    for b in blocks_of(run, arm)[:upto]:
        results = load_jsonl(b / "results.jsonl")
        ours = {norm_id(r.get("a")) for r in results} - {""}
        m, g, inc = pair_means(load_jsonl(b / "battles.jsonl"), ours) if ours else ([], 0, 0)
        means += m
        games += g
        incomplete += inc
        for r in results:
            statuses[r.get("status", "?")] += 1
        summ = json.loads((b / "summary.json").read_text()) if (b / "summary.json").exists() else {}
        seconds += summ.get("seconds", 0) or 0
        for key, acc in (("search_stats_a", stats_a), ("search_stats_b", stats_b)):
            for k, v in (summ.get(key) or {}).items():
                acc[k] += v
        for row in load_jsonl(b / "gate.jsonl"):
            gate["decisions"] += 1
            gate["overrides"] += bool(row.get("override"))
            gate["undefined_gap"] += row.get("gap") is None
        blocks.append({"block": b.name, "pairs": len(m), "games": g, "incomplete": inc,
                       "seconds": summ.get("seconds"), "done": bool(summ)})
    mean, hw = interval(means)
    return {"arm": arm, "blocks": len(blocks), "pairs": len(means), "games": games, "incomplete_pairs": incomplete,
            "mean": mean, "hw": hw, "lo": mean - hw, "hi": mean + hw,
            "won_both": sum(x == 1 for x in means), "lost_both": sum(x == 0 for x in means),
            "split": sum(0 < x < 1 for x in means), "client_status": dict(statuses), "seconds": seconds,
            "stats_a": dict(stats_a), "stats_b": dict(stats_b), "gate": gate, "per_block": blocks}


def read_plan(run: Path) -> dict:
    """The plan's scalars and its ARMS array (bash syntax as netsearch_slot.sh writes it: plan.env plus
    arms.txt with one spec per line)."""
    plan: dict = {}
    for line in (run / "plan.env").read_text().splitlines():
        m = re.match(r"\s*([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if m and not m.group(2).startswith("("):
            plan[m.group(1)] = m.group(2).strip().strip("'\"")
    arms = run / "arms.txt"
    plan["ARMS"] = [ln for ln in arms.read_text().splitlines() if ln.strip()] if arms.exists() else []
    return plan


def spec_margin(spec: str) -> float:
    """The arm's margin: netsearch specs carry margin=M; any other player (a raw net) is margin +inf."""
    if not spec.startswith("netsearch:"):
        return math.inf
    m = re.search(r",margin=([^,]+)", spec)
    return float(m.group(1)) if m else 0.03  # NetSearchPlayer's default


# ---------------------------------------------------------------- look / report

def look(run: Path, arm: int, blocks_max: int, futility: float | None) -> tuple[dict, int]:
    """The look after arm K's latest block. FUTILITY when its upper 95% bound is below `futility` before its last
    block; DONE after its last block; else CONTINUE."""
    s = arm_stats(run, arm)
    decision = "continue"
    if s["blocks"] >= blocks_max:
        decision = "done"
    elif futility is not None and s["pairs"] >= 2 and math.isfinite(s["hw"]) and s["hi"] < futility:
        decision = "futility"
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arm": arm, "look": s["blocks"],
           "pairs": s["pairs"], "games": s["games"], "mean": round(s["mean"], 4) if s["pairs"] else None,
           "hw": round(s["hw"], 4) if math.isfinite(s["hw"]) else None,
           "hi": round(s["hi"], 4) if math.isfinite(s["hi"]) else None, "decision": decision}
    with open(run / "looks.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row, FUTILITY_EXIT if decision == "futility" else 0


def select(arms: list[dict], go: float, tie: float) -> dict:
    """Stage 2 selection (NETSEARCH-DESIGN.md §6): among arms not stopped for futility, the best point estimate;
    arms within `tie` of it go to the larger margin (closer to the network). Go iff the selected arm's estimate >=
    `go`."""
    live = [a for a in arms if not a["stopped"] and a["pairs"] > 0]
    if not live:
        return {"selected": None, "go": False, "why": "every arm stopped for futility or has no pairs"}
    best = max(a["mean"] for a in live)
    near = [a for a in live if a["mean"] >= best - tie - 1e-12]
    pick = max(near, key=lambda a: (a["margin"], a["mean"]))
    return {"selected": pick["arm"], "margin": pick["margin"], "mean": pick["mean"], "best_mean": best,
            "go": pick["mean"] >= go, "rule": f"best estimate, ties within {tie:.3f} -> larger margin; go iff >= {go}"}


def report(run: Path, go: float, tie: float, void_fallbacks: float) -> dict:
    plan = read_plan(run)
    arms = []
    for k, spec in enumerate(plan["ARMS"], 1):
        s = arm_stats(run, k)
        s |= {"spec": spec, "margin": spec_margin(spec), "stopped": (run / f"arm-{k}" / "STOPPED").exists()}
        dec_a, dec_b = s["stats_a"].get("decisions", 0), s["stats_b"].get("decisions", 0)
        s["fallback_rate_a"] = s["stats_a"].get("fallbacks", 0) / dec_a if dec_a else None
        s["fallback_rate_b"] = s["stats_b"].get("fallbacks", 0) / dec_b if dec_b else None
        s["override_rate"] = s["gate"]["overrides"] / s["gate"]["decisions"] if s["gate"]["decisions"] else None
        s["void"] = any(r is not None and r > void_fallbacks for r in (s["fallback_rate_a"], s["fallback_rate_b"]))
        arms.append(s)
    out = {"run": str(run), "name": plan.get("NAME"), "tag": plan.get("TAG"), "opponent": plan.get("OPP"),
           "arms": arms}
    if plan.get("FUTILITY"):
        out["selection"] = select(arms, go, tie)
    return out


def format_report(r: dict) -> str:
    lines = [f"{r['name']} (seed tag {r['tag']}) vs {r['opponent']}: {r['run']}"]
    for a in r["arms"]:
        if a["pairs"]:
            wr = f"{a['mean']:.1%} ± {a['hw']:.1%}" if math.isfinite(a["hw"]) else f"{a['mean']:.1%}"
        else:
            wr = "no pairs"
        ov = f"{a['override_rate']:.1%}" if a["override_rate"] is not None else "-"
        fb = "/".join("-" if x is None else f"{x:.2%}" for x in (a["fallback_rate_a"], a["fallback_rate_b"]))
        flags = (" STOPPED (futility)" if a["stopped"] else "") + (" VOID (fallbacks)" if a["void"] else "")
        lines.append(f"  arm {a['arm']} margin {a['margin']}: {wr} ({a['pairs']} pairs, {a['games']} games, "
                     f"{a['blocks']} blocks; won both {a['won_both']}, split {a['split']}, lost both "
                     f"{a['lost_both']}; incomplete {a['incomplete_pairs']}); overrides {ov} of "
                     f"{a['gate']['decisions']} searched decisions; fallbacks ours/theirs {fb}; "
                     f"{a['seconds'] / 3600:.2f} h{flags}")
        lines.append(f"      {a['spec']}")
    sel = r.get("selection")
    if sel:
        if sel["selected"] is None:
            lines.append(f"  selection: none ({sel['why']}) -> NO-GO")
        else:
            lines.append(f"  selection: arm {sel['selected']} (margin {sel['margin']}, {sel['mean']:.1%}; best "
                         f"{sel['best_mean']:.1%}) -> {'GO' if sel['go'] else 'NO-GO'} [{sel['rule']}]")
    return "\n".join(lines)


def pair_table(run: Path, arm: int) -> dict[tuple[int, int], float]:
    """{(block number, pair id): pair mean} of one arm: block j of every arm is seeded TAG b<j>, so the same key is
    the same team pair in every arm."""
    out = {}
    for b in blocks_of(run, arm):
        results = load_jsonl(b / "results.jsonl")
        ours = {norm_id(r.get("a")) for r in results} - {""}
        by_pair: dict[int, dict[int, dict]] = defaultdict(dict)
        for row in load_jsonl(b / "battles.jsonl"):
            if {norm_id(row.get("p1")), norm_id(row.get("p2"))} & ours:
                by_pair[int(row["pair"])][int(row["game"])] = row
        j = int(b.name.split("-")[1])
        for i, pr in by_pair.items():
            if len(pr) == 2 and 0 in pr and 1 in pr:
                out[(j, i)] = (game_score(pr[0].get("winner"), ours) + game_score(pr[1].get("winner"), ours)) / 2
    return out


def paired(run: Path, a: int, b: int) -> dict:
    """Arm a minus arm b on the team pairs both completed: mean difference of pair means +- 1.96 sd / sqrt(n)."""
    ta, tb = pair_table(run, a), pair_table(run, b)
    keys = sorted(set(ta) & set(tb))
    d = [ta[k] - tb[k] for k in keys]
    m, hw = interval(d)
    ma, _ = interval([ta[k] for k in keys])
    mb, _ = interval([tb[k] for k in keys])
    return {"arms": [a, b], "pairs": len(keys), "only_a": len(set(ta) - set(tb)), "only_b": len(set(tb) - set(ta)),
            "mean_a": ma, "mean_b": mb, "diff": m, "hw": hw, "lo": m - hw, "hi": m + hw}


# ---------------------------------------------------------------- dry runs

def fake(out: Path, games: int, p: float, name: str, seed: str, opp: str = "pbfake") -> None:
    """Synthetic block logs as a duplicate server + porygon.v2.evaluate would leave them (no server, no players)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{seed}:{name}:{p}")
    n = games + games % 2
    with open(out / "battles.jsonl", "a") as bf, open(out / "results.jsonl", "a") as rf:
        for k in range(n):
            won = rng.random() < p
            winner = name if won else opp
            bf.write(json.dumps({"roomid": f"battle-fake-{k}", "pair": k >> 1, "game": k & 1, "winner": winner,
                                 "p1": name, "p2": opp}) + "\n")
            rf.write(json.dumps({"battle": f"battle-fake-{k}", "a": name, "b": opp, "status": "win" if won else "loss",
                                 "seed_tag": seed}) + "\n")
    stats = {"decisions": 25 * n, "searched": 24 * n, "fallbacks": 0, "overrides": 5 * n}
    (out / "summary.json").write_text(json.dumps({"seconds": 1, "search_stats_a": stats,
                                                  "search_stats_b": {"decisions": 25 * n, "fallbacks": 0}}) + "\n")
    with open(out / "gate.jsonl", "a") as gf:
        for i in range(24 * n):
            gf.write(json.dumps({"override": i % 5 == 0, "gap": 0.01}) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("look")
    lk.add_argument("run", type=Path)
    lk.add_argument("arm", type=int)
    lk.add_argument("--blocks-max", type=int, required=True)
    lk.add_argument("--futility", type=float, default=None, help="stop below this upper 95%% bound (none: fixed n)")
    rp = sub.add_parser("report")
    rp.add_argument("run", type=Path)
    rp.add_argument("--json", type=Path)
    rp.add_argument("--go", type=float, default=0.515)
    rp.add_argument("--tie", type=float, default=0.01)
    rp.add_argument("--void-fallbacks", type=float, default=0.02)
    pd = sub.add_parser("paired")
    pd.add_argument("run", type=Path)
    pd.add_argument("a", type=int)
    pd.add_argument("b", type=int)
    pd.add_argument("--json", type=Path)
    fk = sub.add_parser("fake")
    fk.add_argument("out", type=Path)
    fk.add_argument("--games", type=int, required=True)
    fk.add_argument("--p", type=float, required=True)
    fk.add_argument("--a", required=True)
    fk.add_argument("--seed", required=True)
    a = p.parse_args()
    if a.cmd == "look":
        row, rc = look(a.run, a.arm, a.blocks_max, a.futility)
        print(json.dumps(row), flush=True)
        return rc
    if a.cmd == "report":
        r = report(a.run, a.go, a.tie, a.void_fallbacks)
        print(format_report(r))
        if a.json:
            a.json.write_text(json.dumps(r, indent=1, default=str) + "\n")
        return 0
    if a.cmd == "paired":
        r = paired(a.run, a.a, a.b)
        print(f"arm {a.a} - arm {a.b} on {r['pairs']} common team pairs: {r['diff']:+.1%} ± {r['hw']:.1%} "
              f"(arm {a.a} {r['mean_a']:.1%}, arm {a.b} {r['mean_b']:.1%}; unmatched {r['only_a']} / {r['only_b']})")
        if a.json:
            a.json.write_text(json.dumps(r, indent=1) + "\n")
        return 0
    fake(a.out, a.games, a.p, norm_id(a.a), a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
