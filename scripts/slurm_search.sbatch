#!/bin/bash
# Search head-to-heads (python -m porygon.search) on a cluster node with its own local Showdown server.
# 16 CPUs so that it runs next to a 30-CPU training job (QOS: 48 CPUs per user); the GPU scores leaves.
#   sbatch scripts/slurm_search.sbatch MODEL [porygon.search args]
#   e.g. ... --opponent search --value-weight 0.5 --n-battles 300
#SBATCH --job-name=porygon-search
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
MODEL=$1; shift
export PATH=$ROOT/.node/bin:$PATH
PORT=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 + 14 ))
scripts/start_servers.sh 1 "$PORT"
trap 'pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $PORT\$" || true' EXIT
echo "search $MODEL $*"
.venv/bin/python -m porygon.search "$MODEL" --port "$PORT" --workers $(( $(nproc) - 2 )) --device cuda "$@"
