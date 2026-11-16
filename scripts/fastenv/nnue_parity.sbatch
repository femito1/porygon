#!/bin/bash
# NNUE gate 1, step 3 on the cluster: converter parity (scripts/fastenv/nnue_parity.py) on N gen9 fastenv games,
# from a frozen code copy. One short job, no agent; writes $C/nnue/gate1/PARITY.txt (+ PARITY.json) and stage
# lines in $C/nnue/gate1/parity/STATUS (FAILED on a crash). Prepare the copy, then submit after the helper:
#   rsync -a --exclude __pycache__ src scripts fastenv gpucluster2:$C/nnue/gate1/code/    (+ echo COMMIT > code/COMMIT)
#   cd $C && sbatch --dependency=afterany:<helper> scripts/fastenv/nnue_parity.sbatch [GAMES=60]
#SBATCH --job-name=ws-fastenv-nnueparity
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
C=$PWD G=/vol/bitbucket/fr1226/gloria
OUT=$C/nnue/gate1 S=$C/nnue/gate1/parity CODE=$C/nnue/gate1/code
GAMES=${1:-60}
mkdir -p "$S"
st() { echo "$(date -u +%FT%TZ) $*" >> "$S/STATUS"; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%TZ) $*" > "$S/FAILED"; exit 1; }
[ -e "$OUT/PARITY.txt" ] && fail "PARITY.txt exists (move it away to re-run)"
[ -f "$CODE/COMMIT" ] || fail "no frozen code in $CODE"
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src OMP_NUM_THREADS=1
CPUS=${SLURM_CPUS_ON_NODE:-8}
W=$(( CPUS > 3 ? CPUS - 2 : 1 ))
st "start: job ${SLURM_JOB_ID:-?} on $(hostname), CPUs $CPUS, workers $W, games $GAMES, code $(cat "$CODE/COMMIT")"
"$PY" "$CODE/scripts/fastenv/nnue_parity.py" --format gen9randombattle --play "$GAMES" --workers "$W" --seed 916 \
  --dump-dir "$S/dumps" --out "$OUT/PARITY.txt" > "$S/parity.log" 2>&1
rc=$?
[ -s "$OUT/PARITY.txt" ] || fail "nnue_parity.py exit $rc, no PARITY.txt (see $S/parity.log)"
chmod a-w "$OUT/PARITY.txt" "$OUT/PARITY.json"
st "done: $(head -1 "$OUT/PARITY.txt")"
gzip -q "$S"/dumps/dump-*.jsonl || true
exit 0
