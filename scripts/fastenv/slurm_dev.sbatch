#!/bin/bash
# fastenv 4-CPU dev job on a16 (QOS normal cap): runs the command lines of a file like slurm_bench.sbatch.
#   sbatch scripts/fastenv/slurm_dev.sbatch scripts/fastenv/dev_cmds_update.txt
#SBATCH --job-name=ws-fastenv-dev
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
G=/vol/bitbucket/fr1226/gloria
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PYTHONPATH=$ROOT/src
echo "node $(hostname) cpus $(nproc)"; nvidia-smi -L || true
while read -r line; do
  [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
  echo "=== $line"; eval "$line"
done < "$1"
