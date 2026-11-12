"""Pooled win rate over benchmark runs, 95% interval from duplicate-pair means (pairs never span logs).

    python scripts/netv2_pool_ci.py runs/h2h-*/ runs/vs-search-20261109-*/ [--json]

A run directory holds one or more duplicate-server logs (battles.jsonl, or shard-*/battles.jsonl from the
netv2 sbatch scripts) and, next to each, the client's results.jsonl that names our side:
- porygon.v2.evaluate (netv2_h2h / netv2_vs_search): field "a" (we score side A);
- scripts/vs_foulplay.py (netv2_foulplay / slurm_foulplay): field "player".
A pair = the two games of one pair id in one log (same seeds, teams swapped); it scores (s₀ + s₁)/2 with a tie = ½.
Pooled interval: mean of all pair means ± 1.96·sd/√pairs. Runs whose players differ (spec_a / spec_b in results,
or manifest.txt) are flagged, since pooling them answers no single question. Logs without results.jsonl, and
runs on a plain server (no battles.jsonl), fall back to per-game binomial counts from results.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from porygon.v2.evaluate import format_summary, load_jsonl, norm_id, pair_means, summarize  # noqa: E402

STATUS = {"win": 1.0, "loss": 0.0, "tie": 0.5}  # timeouts / no_accept are not games


def logs_of(run: Path) -> list[Path]:
    """Directories in `run` (itself or shards below it) holding a battles.jsonl or a results.jsonl."""
    found = {p.parent for name in ("battles.jsonl", "results.jsonl") for p in run.rglob(name)}
    return sorted(found)


def matchup(results: list[dict]) -> str:
    specs = {(r.get("spec_a"), r.get("spec_b")) for r in results if "spec_a" in r}
    return "; ".join(f"{a} vs {b}" for a, b in sorted(specs, key=str)) if specs else "(vs_foulplay run)"


def run_stats(run: Path) -> tuple[list[float], list[float], int, str]:
    means, games, incomplete, kinds = [], [], 0, set()
    for d in logs_of(run):
        results = load_jsonl(d / "results.jsonl")
        battles = load_jsonl(d / "battles.jsonl")
        ours = {norm_id(r.get("a") or r.get("player")) for r in results} - {""}
        kinds.add(matchup(results))
        if battles and ours:
            m, g, inc = pair_means(battles, ours, str(d))
            means, games, incomplete = means + m, games + g, incomplete + inc
        else:
            games += [STATUS[r["status"]] for r in results if r.get("status") in STATUS]
    return means, games, incomplete, "; ".join(sorted(kinds))


def manifest_line(run: Path) -> str:
    f = run / "manifest.txt"
    if not f.exists():
        return ""
    keep = [ln.strip() for ln in f.read_text().splitlines() if ln.split(":")[0].strip() in ("kind", "a", "b", "args")]
    return " | ".join(keep)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--json", action="store_true", help="also print the pooled summary as JSON")
    a = p.parse_args()
    all_means, all_games, labels = [], [], set()
    a.dirs = list(dict.fromkeys(d.resolve() for d in a.dirs))  # a run listed twice counts once
    for run in a.dirs:
        means, games, incomplete, label = run_stats(run)
        labels.add(label)
        if not games:
            print(f"{run}: no games")
            continue
        s = summarize(means, games)
        extra = f", {incomplete} incomplete pairs dropped" if incomplete else ""
        print(f"{run}: {format_summary(s)}{extra}\n    {label}" + (f"\n    {manifest_line(run)}" if manifest_line(run) else ""))
        all_means += means
        all_games += games
    if len(a.dirs) > 1 and all_games:
        if len(labels) > 1:
            print("WARNING: the runs have different matchups:\n    " + "\n    ".join(sorted(labels)))
        s = summarize(all_means, all_games)
        print(f"POOLED ({len(a.dirs)} runs): {format_summary(s)}")
        if a.json:
            print(json.dumps(s))
    elif a.json and all_games:
        print(json.dumps(summarize(all_means, all_games)))


if __name__ == "__main__":
    main()
