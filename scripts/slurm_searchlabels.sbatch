#!/bin/bash
# Search-distillation data (scripts/gen_search_labels.py): our search vs itself on the duplicate-battle server,
# both sides' decisions + replays + true teams logged. CPU-only job (the data stream's CPU budget); submit from
# the data stream's code dir (scripts/setup_ws_data.sh makes it a working submit dir).
#   sbatch -c 8 scripts/slurm_searchlabels.sbatch GEN CHUNKS N_PER_CHUNK [gen_search_labels.py args]
#   e.g. sbatch -c 8 scripts/slurm_searchlabels.sbatch 9 200 50 --ms 250 --samples 4
#SBATCH --job-name=ws-data-searchlabels
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-data/logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
GEN=$1 CHUNKS=$2 N=$3; shift 3
DATA=/vol/bitbucket/fr1226/porygon-ws-data
export PATH=$ROOT/.node/bin:$PATH
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
if [ "$GEN" = 9 ]; then export PORYGON_GEN=9 PYTHONPATH=$ROOT/.engine-gen9:$PYTHONPATH; fi
OUT=$DATA/runs/searchlabels-gen$GEN-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT/teams"
PORT=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 + 13 ))
export PS_DIR=$ROOT/.showdown-eval PORYGON_DUPLICATE=$SLURM_JOB_ID PORYGON_DUPLICATE_LOG=$OUT/battles.jsonl
export PORYGON_DUPLICATE_TEAMS=$OUT/teams PORYGON_DUPLICATE_GROUP='^sla'
find .venv/lib/python3.12/site-packages/{websockets,poke_env} -type f \( -name '*.py' -o -name '*.so*' \) -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
scripts/start_servers.sh 1 "$PORT"
trap 'pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $PORT\$" || true' EXIT
echo "gen$GEN search labels: $CHUNKS x $N battles, $* -> $OUT"
.venv/bin/python scripts/gen_search_labels.py none --out "$OUT" --port "$PORT" --chunks "$CHUNKS" --n-battles "$N" \
  --workers $(( $(nproc) - 1 )) --format "gen${GEN}randombattle" "$@" 2>&1 | tee "$OUT/gen.log" || true
