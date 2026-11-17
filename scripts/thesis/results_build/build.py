"""Build docs/thesis/data/results.csv and docs/thesis/results.md from rows.py + the gloria per-run scan."""
import csv, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from rows import ROWS, G, FPG4, PM, BIN

HERE = Path(__file__).resolve().parent.parent
REPO = Path("/home/nand/Documents/personal-projects/ShowdownAgent")
FIELDS = ["id", "date", "theme", "question", "arm_a", "arm_b", "harness", "games", "pairs", "estimate_pct", "ci_low",
          "ci_high", "ci_type", "prereg", "prereg_md5", "verdict", "outcome", "provenance", "source"]

# ---- gloria per-run Foul Play / search benchmark rows (analysis.txt + vs.log config)
SINGLE = {  # distinct configurations that were single tests (README 'tried, no gain' / paper)
    "value_weight=0.5": ("one-turn v1 value lookahead in the search (0.5)", "no gain", "null"),
    "prior_weight=1.0": ("network prior weight 1.0", "no gain", "null"),
    "prior_weight=0.5 oracle=None": ("network prior weight 0.5", "no gain", "null"),
    "samples=8 oracle=None": ("8 determinizations instead of 4 (2x compute)", "no gain", "null"),
    "prior_weight=0.0 c_puct=8.0 oracle=None": ("root PUCT c 8 on the prior", "no gain", "null"),
    "root_game=sample oracle=None": ("Bayesian root game, sampled", "-13 points: harmful", "negative"),
    "root_game=max oracle=None": ("Bayesian root game, argmax", "no gain", "null"),
    "rm_gamma=0.1 oracle=None": ("regret matching in the tree (rm_gamma 0.1)", "no gain", "null"),
    "root_game=sample rm_gamma=0.1 oracle=None": ("regret matching + sampled root game", "harmful", "negative"),
}
gruns = json.load(open(HERE / "gloria-runs.json"))
scan = {}
for l in open(HERE / "gloria-foulplay-scan.txt"):
    m = re.match(r"(\d+) (\d{8}) (\d+)-(\d+) ([\d.]+)% n=(\d+) \| ?(.*)", l.strip())
    if m:
        scan[m[1]] = m.groups()
have = {g["job"] for g in gruns}
for j, s in scan.items():  # early runs with results.jsonl but no analysis.txt (before the duplicate server, 10-27)
    if j not in have:
        gruns.append(dict(job=j, dir=None, kind="foulplay", date=s[1], pm=None, ag=None, cfg=s[6], wl=(s[2], s[3])))
gen_rows = []
for g in sorted(gruns, key=lambda x: (x["date"], x["job"])):
    if g["kind"] != "foulplay":
        continue
    j, cfg = g["job"], (g["cfg"] or "").strip()
    d = f"{g['date'][:4]}-{g['date'][4:6]}-{g['date'][6:]}"
    gen9 = j.startswith(("515", "492"))
    if g.get("pm"):
        est, h, pairs = float(g["pm"][0]), float(g["pm"][1]), int(g["pm"][2]); games = int(g["ag"][2]); ct = PM; prov = "M"
    else:
        w, l = map(int, g["wl"]); games = w + l; est = 100 * w / games; h = 196 * (est / 100 * (1 - est / 100) / games) ** .5
        pairs = ""; ct = BIN; prov = "D"; est = round(est, 1); h = round(h, 1)
    q, verdict, outcome = SINGLE.get(cfg, (None, "", "info"))
    if q is None:
        if "oracle=revealed" in cfg or "oracle=all" in cfg:
            q = f"oracle diagnosis ({cfg})"
        elif gen9:
            q = f"gen9 pure search vs Foul Play gen9 ({cfg}); part of a pooled row"
        else:
            q = f"gen4 search vs Foul Play, config [{cfg or 'default'}]; part of a pooled row" if cfg not in ("", "oracle=None") or True else ""
    arm = ("search:none (gen9), " if gen9 else "SearchPlayer 4x1 s, v1 prior (net not logged), ") + (cfg or "default")
    src = f"{G}/{g['dir']}/analysis.txt" if g["dir"] else f"{G}/foulplay-*-{j}/results.jsonl (scan_runs.py; no analysis.txt)"
    gen_rows.append((f"fpr-{j}", d, "gen9-port" if gen9 else "bench-fp", q, arm, "Foul Play " + ("gen9" if gen9 else "gen4"),
                     "gloria slurm_foulplay.sbatch" + (" --gen 9" if gen9 else ""), games, pairs, est,
                     round(est - h, 1), round(est + h, 1), ct, "", "", verdict, outcome, prov, src))

rows = [list(x) for x in ROWS] + [list(x) for x in gen_rows]
rows.sort(key=lambda x: (x[1].replace("2x", "25"), x[0]))
for x in rows:
    assert len(x) == len(FIELDS), x[0]
(REPO / "docs/thesis/data").mkdir(parents=True, exist_ok=True)
with open(REPO / "docs/thesis/data/results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(FIELDS); w.writerows(rows)

n = len(rows); npre = sum(1 for x in rows if x[13]); nneg = sum(1 for x in rows if x[16] in ("negative", "null", "inconclusive"))
print(f"rows {n}, pre-registered {npre}, negative/null/inconclusive {nneg}")
print("by outcome", {k: sum(1 for x in rows if x[16] == k) for k in sorted({x[16] for x in rows})})
print("dates", min(x[1] for x in rows), max(x[1] for x in rows))
json.dump(dict(n=n, npre=npre, nneg=nneg), open(HERE / "counts.json", "w"))

def fmt(x):
    est, lo, hi = x[9], x[10], x[11]
    if est == "":
        e = x[15] if x[15] else "-"
    else:
        e = f"{est}" + (f" [{lo}, {hi}]" if lo != "" else "")
    return e

THEMES = [("bench-fp", "Benchmarks against Foul Play (all generations)"),
          ("search4", "Gen4 search and the fitted leaf evaluation (engine and search-vs-search)"),
          ("v1-ppo", "v1 PPO network (gen4, SB3, Oct 2026)"),
          ("gen9-port", "Gen9 port: pure engine search"),
          ("v2-gen4", "v2 network, gen4 (R1b / R1b-F)"),
          ("v2-gen9", "v2 network, gen9 (G9s, G9p, G9q milestones)"),
          ("recipe", "v2 PPO recipe arms (one change each)"),
          ("human-bc", "Human behaviour cloning and the data-quality lever (P1-P4, U7)"),
          ("net+search", "Network + search: hybrid, V3, lookahead2, NNUE, deepen"),
          ("calibration", "Calibration (harnesses, opponent strength, value)"),
          ("diagnostics", "Diagnostics")]
out = []
for th, title in THEMES:
    sel = [x for x in rows if x[2] == th]
    if th in ("bench-fp", "gen9-port"):
        main = [x for x in sel if not x[0].startswith("fpr-")]; per = [x for x in sel if x[0].startswith("fpr-")]
    else:
        main, per = sel, []
    out.append(f"\n### {title} ({len(sel)} rows)\n")
    out.append("| id | date | question | A vs B | harness | games / pairs | A's win % [95%] | prereg (md5) | verdict | src |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in main:
        pre = (f"{x[13]} ({x[14]})" if x[14] else x[13]) if x[13] else "-"
        gp = f"{x[7]}" + (f" / {x[8]}" if x[8] != "" else "")
        out.append(f"| {x[0]} | {x[1]} | {x[3]} | {x[4]} vs {x[5]} | {x[6]} | {gp} | {fmt(x)} [{x[17]}] | {pre} | {x[15] or x[16]} | `{x[18]}` |")
    if per:
        out.append(f"\nPer-run constituents ({len(per)} gloria runs; each [M] from its `analysis.txt` unless marked [D]; the arm's "
                   "network is not logged, see the note below):\n")
        out.append("| job | date | config (non-default `SearchPlayer` options) | games / pairs | win % [95%] | outcome |")
        out.append("|---|---|---|---|---|---|")
        for x in per:
            cfg = x[4].split(", ", 1)[1] if ", " in x[4] else x[4]
            out.append(f"| {x[0][4:]} | {x[1]} | {cfg} | {x[7]}" + (f" / {x[8]}" if x[8] != "" else "") + f" | {fmt(x)} [{x[17]}] | {x[16]}{': ' + x[15] if x[15] else ''} |")
neg = [x for x in rows if x[16] in ("negative", "null", "inconclusive")]
negl = ["\n| id | what | A's win % [95%] | outcome | verdict |", "|---|---|---|---|---|"]
for x in neg:
    negl.append(f"| {x[0]} | {x[3]} ({x[4]} vs {x[5]}) | {fmt(x)} | {x[16]} | {x[15]} |")
tmpl = open(HERE / "scripts/results_template.md").read()
md = tmpl.replace("@@COUNTS@@", f"{n} rows, {npre} pre-registered, {nneg} negative / null / inconclusive "
                  f"({sum(1 for x in rows if x[16]=='negative')} negative, {sum(1 for x in rows if x[16]=='null')} null, "
                  f"{sum(1 for x in rows if x[16]=='inconclusive')} inconclusive)")
md = md.replace("@@TABLES@@", "\n".join(out)).replace("@@NEG@@", "\n".join(negl))
open(REPO / "docs/thesis/results.md", "w").write(md)
