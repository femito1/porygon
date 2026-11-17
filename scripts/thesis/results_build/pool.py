"""Pooled duplicate-pair win rate over gloria benchmark dirs (same logic as runs/wsp/pool_ci.py, no porygon import)."""
import json, math, sys, glob
name, jobs = sys.argv[1], sys.argv[2:]
wins = n = 0; means = []
for j in jobs:
    d = glob.glob(f"runs/*-{j}")[0]
    res = [json.loads(l) for l in open(f"{d}/results.jsonl")]
    ours = {r["player"] for r in res}
    w = sum(r["status"] == "win" for r in res); l = sum(r["status"] == "loss" for r in res); wins += w; n += w + l
    by = {}
    for line in open(f"{d}/battles.jsonl"):
        b = json.loads(line); by.setdefault(b["pair"], {})[b["game"]] = b
    for p in by.values():
        if len(p) == 2:
            means.append(sum(0.5 if not b["winner"] else float(b["winner"] in ours) for b in p.values()) / 2)
p = wins / n
m = sum(means) / len(means); sd = math.sqrt(sum((x - m) ** 2 for x in means) / (len(means) - 1)); h = 1.96 * sd / math.sqrt(len(means))
print(f"{name}|jobs {' '.join(jobs)}|games {n}|per-game {p:.1%} ±{1.96*math.sqrt(p*(1-p)/n):.1%}|pairs {len(means)}|pair-mean {m:.2%} ±{h:.2%} [{m-h:.2%}, {m+h:.2%}]")
