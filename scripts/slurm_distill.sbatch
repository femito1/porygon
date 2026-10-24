#!/bin/bash
# Re-distil existing expert-iteration shards with several settings (porygon.expert distill), one
# after another, into OUT/<name>.zip. Needs the repo's .venv (set up by slurm_train.sbatch).
#   sbatch scripts/slurm_distill.sbatch DATA MODEL OUT "name:--beta 1" "name2:--beta 1 --temperature 0.1" ...
#SBATCH --job-name=porygon-distill
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
DATA=$1 MODEL=$2 OUT=$3; shift 3
mkdir -p "$OUT"
for spec in "$@"; do
  name=${spec%%:*}; flags=${spec#*:}
  echo "=== $name: $flags"
  # shellcheck disable=SC2086  # flags are deliberately split into arguments
  .venv/bin/python -m porygon.expert distill "$MODEL" "$DATA" --out "$OUT/$name.zip" $flags 2>&1 | tee "$OUT/$name.log"
done
