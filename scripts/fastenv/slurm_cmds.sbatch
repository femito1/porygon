#!/bin/bash
# fastenv command-list job on the long partition (8 CPUs; submit with --exclude=vm-kn-a16 while netv2 needs it): runs the command lines of a file like slurm_bench.sbatch.
#   sbatch --exclude=vm-kn-a16 scripts/fastenv/slurm_cmds.sbatch FILE
#SBATCH --job-name=ws-fastenv-eval
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
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
