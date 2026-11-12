#!/bin/bash
# Pre-registered throughput A/B for the label pipeline (throughput change only; the labels' settings are identical):
# FAST=0 (glibc malloc in the engine workers, default torch threads) vs FAST=1 (tcmalloc via LD_PRELOAD, 1 torch
# thread for the B=1 prior forward). Arms A B B A, each a fixed number of gen4 full-search label games (fitted eval,
# 500M prior 0.25, 4 x 500 ms, concurrency 9) on the shared long slot (claim it first). Metrics per arm: games/hour,
# searched decisions/s, mean root visits per determinization (iterations per search). The runs are normal label runs
# (their games are kept as labels).
#   sbatch scripts/slurm_labels_speed_ab.sbatch [GAMES_PER_ARM]
#SBATCH --job-name=ws-data-labels-ab
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-data/logs/slurm-%j.out
set -uo pipefail
G=${1:-54}
cd "$SLURM_SUBMIT_DIR"
NET=/vol/bitbucket/fr1226/gloria/runs/20261025-195826-301040/final_model.zip
SUM=/vol/bitbucket/fr1226/porygon-ws-data/logs/labels-speed-ab-$SLURM_JOB_ID.txt
for arm in A B B A; do
  fast=(); [ "$arm" = A ] && fast=(FAST=0)
  t0=$(date +%s)
  bash scripts/slurm_searchlabels.sbatch 4 1 "$G" "${fast[@]}" --model "$NET" --device cuda --ms 500 --samples 4 \
    --concurrency 9 > /dev/null 2>&1
  t1=$(date +%s)
  run=$(ls -dt /vol/bitbucket/fr1226/porygon-ws-data/runs/searchlabels-gen4-*-"$SLURM_JOB_ID" | head -1)
  .venv/bin/python -I - "$arm" "$run" "$((t1 - t0))" <<'PY' | tee -a "$SUM"
import json, sys
from pathlib import Path
arm, run, wall = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
games = sum(1 for _ in open(run / "battles.jsonl"))
recs = [json.loads(l) for s in ("a", "b") for l in open(run / s / "decisions.jsonl")]
vis = [r["visits"] for r in recs if r.get("visits")]
print(json.dumps({"arm": arm, "run": run.name, "wall_s": wall, "games": games, "games_per_h": round(3600 * games / wall, 1),
                  "decisions": len(recs), "decisions_per_s": round(len(recs) / wall, 2),
                  "mean_visits_per_sample": round(sum(vis) / max(1, len(vis)))}))
PY
done
