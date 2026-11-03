#!/bin/bash
# Several engine-only evaluation head-to-heads (porygon.selfplay h2h) in one job, on the same team pairs:
# every config A against config B. The node's cold start is paid once.
#   sbatch scripts/slurm_h2h_arms.sbatch B PAIRS ITERATIONS A1 [A2 ...]
#   e.g. ... stock 3000 20000 runs/evalfit/r1/full.json runs/evalfit/r1/stock_k.json
#SBATCH --job-name=porygon-h2h
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
B=$1; PAIRS=$2; ITER=$3; shift 3
export PATH=$ROOT/.node/bin:$PATH
OUT=runs/h2h-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT"
node scripts/gen_teams.js "$PAIRS" > "$OUT/teams.jsonl"
for A in "$@"; do
  name=$(echo "$A" | sed 's#^runs/##; s#/#_#g; s#\.json$##')
  echo "== $A vs $B ($PAIRS pairs, $ITER iterations)"
  .venv/bin/python -m porygon.selfplay h2h --teams "$OUT/teams.jsonl" --out "$OUT/$name" --workers "$(nproc)" \
    --iterations "$ITER" --a "$A" --b "$B" 2>&1 | tail -2 | tee -a "$OUT/summary.txt"
done
