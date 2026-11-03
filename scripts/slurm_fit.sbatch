#!/bin/bash
# Fit the leaf evaluation (scripts/fit_eval.py) on a cluster node.
#   sbatch scripts/slurm_fit.sbatch OUT DATA_DIR [DATA_DIR ...] [fit_eval.py args]
#SBATCH --job-name=porygon-fit
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
OUT=$1; shift; mkdir -p "$(dirname "$OUT")"
export OMP_NUM_THREADS=$(nproc)
.venv/bin/python scripts/fit_eval.py --out "$OUT" "$@" 2>&1 | tee "$OUT.log"
