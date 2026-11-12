#!/bin/bash
# Search-distillation data (scripts/gen_search_labels.py): our search vs itself on the duplicate-battle server,
# both sides' decisions + replays + true teams logged. CPU-only job (the data stream's CPU budget); submit from
# the data stream's code dir (scripts/setup_ws_data.sh makes it a working submit dir).
#   sbatch -c 8 scripts/slurm_searchlabels.sbatch GEN CHUNKS N_PER_CHUNK [FAST=0] [--model CKPT] [gen_search_labels.py args]
#   e.g. sbatch -c 8 scripts/slurm_searchlabels.sbatch 9 200 50 --ms 250 --samples 4
#   gen4 full search (fitted eval default, prior 0.25 default with a model):
#   sbatch -p long -c 24 scripts/slurm_searchlabels.sbatch 4 100 50 --model /vol/bitbucket/fr1226/gloria/runs/20261025-195826-301040/final_model.zip --device cuda --ms 500
#SBATCH --job-name=ws-data-searchlabels
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-data/logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
GEN=$1 CHUNKS=$2 N=$3; shift 3
# Throughput options (label jobs only; benchmarks keep the defaults): 1 torch thread for the B=1 prior forward, and
# tcmalloc in the engine workers. FAST=0 as the first argument after GEN CHUNKS N turns both off (A/B measurement).
FAST=1
if [ "${1:-}" = FAST=0 ]; then FAST=0; shift; fi
MODEL=none
if [ "${1:-}" = --model ]; then MODEL=$2; shift 2; fi
if [ "$FAST" = 1 ]; then
  export PORYGON_TORCH_THREADS=1
  for lib in /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 /usr/lib/x86_64-linux-gnu/libjemalloc.so.2; do
    [ -e "$lib" ] && { export PORYGON_ENGINE_MALLOC=$lib; break; }
  done
fi
echo "throughput options: FAST=$FAST torch threads ${PORYGON_TORCH_THREADS:-default}, engine malloc ${PORYGON_ENGINE_MALLOC:-glibc}"
DATA=/vol/bitbucket/fr1226/porygon-ws-data
export PATH=$ROOT/.node/bin:$PATH PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
if [ "$GEN" = 9 ]; then export PORYGON_GEN=9 PYTHONPATH=$ROOT/.engine-gen9:$PYTHONPATH; fi
OUT=$DATA/runs/searchlabels-gen$GEN-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT/teams"
PORT=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 + 13 ))
export PS_DIR=$ROOT/.showdown-eval PORYGON_DUPLICATE=$SLURM_JOB_ID PORYGON_DUPLICATE_LOG=$OUT/battles.jsonl
export PORYGON_DUPLICATE_TEAMS=$OUT/teams PORYGON_DUPLICATE_GROUP='^sla'
find .venv/lib/python3.12/site-packages/{websockets,poke_env} -type f \( -name '*.py' -o -name '*.so*' \) -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
scripts/start_servers.sh 1 "$PORT"
trap 'pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $PORT\$" || true' EXIT
echo "gen$GEN search labels ($MODEL): $CHUNKS x $N battles, $* -> $OUT"
.venv/bin/python scripts/gen_search_labels.py "$MODEL" --out "$OUT" --port "$PORT" --chunks "$CHUNKS" --n-battles "$N" \
  --workers $(( $(nproc) - 1 )) --format "gen${GEN}randombattle" "$@" 2>&1 | tee "$OUT/gen.log" || true
