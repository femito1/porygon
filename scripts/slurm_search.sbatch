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
OUT=runs/search-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT"
# duplicate battles: pairs of games with the same seeds and the teams swapped, true teams logged
[ -f .showdown-eval/dist/server/index.js ] || scripts/build_eval_server.sh
export PS_DIR=$ROOT/.showdown-eval PORYGON_DUPLICATE=$SLURM_JOB_ID PORYGON_DUPLICATE_LOG=$ROOT/$OUT/battles.jsonl
export PORYGON_DUPLICATE_GROUP='^search'
# Pull the Python environments into the page cache first: on a slow shared disk, modules imported lazily
# during a websocket handshake can take longer than the 10 s handshake timeout.
find .venv/lib/python3.12/site-packages/{websockets,poke_env} -type f \( -name '*.py' -o -name '*.so*' \) -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
scripts/start_servers.sh 1 "$PORT"
trap 'pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $PORT\$" || true' EXIT
echo "search $MODEL $*"
.venv/bin/python -m porygon.search "$MODEL" --port "$PORT" --workers $(( $(nproc) - 2 )) --device cuda \
  --log-dir "$OUT" "$@" 2>&1 | tee "$OUT/search.log" || true
.venv/bin/python scripts/analyze_benchmark.py "$OUT" 2>&1 | tee "$OUT/analysis.txt" || true
