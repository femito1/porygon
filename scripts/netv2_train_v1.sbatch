#!/bin/bash
# netv2 stream: a v1 (SB3) PPO experiment arm in the stream's own cluster dir (gloria is read-only: its venv,
# node and models are used by absolute path, everything written goes under $ROOT = the submit dir).
#   cd /vol/bitbucket/fr1226/porygon-ws-netv2
#   sbatch --job-name=ws-netv2-r0 scripts/netv2_train_v1.sbatch NAME START_ZIP TOTAL [porygon.train args]
# runs in runs/arm-NAME-<date>-<job>/, resumes from its own newest checkpoint (else START_ZIP), opponent pool
# runs/arms/NAME/pool seeded with symlinks to gloria's runs/resume/pool snapshots. No results push.
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --requeue
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
GLORIA=/vol/bitbucket/fr1226/gloria
PY=$GLORIA/.venv/bin/python
export PATH=$GLORIA/.node/bin:$PATH
export PYTHONPATH=$ROOT/src
ARM=$1; ARM_START=$2; TOTAL=$3; shift 3
POOL_NFS=runs/arms/$ARM/pool
if ! [ -d "$POOL_NFS" ]; then
  mkdir -p "$POOL_NFS"
  for f in "$GLORIA"/runs/resume/pool/*.pt; do ln -s "$f" "$POOL_NFS/"; done
fi
$PY -c "import torch, porygon; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(), porygon.__file__)"

CORES=$(nproc)
N_SERVERS=$(( CORES / 3 )); SELFPLAY=$(( CORES * 2 )); POOL_BATTLES=$CORES
BASE=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 ))
scripts/start_servers.sh "$N_SERVERS" "$BASE"

LATEST=$(ls -t runs/arm-"$ARM"-*/checkpoints/*.zip runs/arm-"$ARM"-*/final_model.zip 2>/dev/null | head -1 || true)
LATEST=${LATEST:-$ARM_START}
RUN=runs/arm-$ARM-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$RUN"
echo "resuming from $LATEST; git $(cat "$ROOT/GIT_COMMIT" 2>/dev/null || echo unknown); args: $*" | tee "$RUN/manifest.txt"

LOCAL=${TMPDIR:-/tmp}/porygon-$SLURM_JOB_ID; mkdir -p "$LOCAL/run" "$LOCAL/pool"
( ls -t "$POOL_NFS"/*.pt 2>/dev/null | while read -r f; do cp -pL "$f" "$LOCAL/pool/.tmp" && mv "$LOCAL/pool/.tmp" "$LOCAL/pool/${f##*/}"; done ) &
mirror() {  # pool: only new snapshots go back (the seeded ones are symlinks into gloria and must stay untouched)
  rsync -a "$LOCAL/run/" "$RUN/" || return 1
  for f in "$LOCAL"/pool/*.pt; do [ -e "$POOL_NFS/${f##*/}" ] || cp -p "$f" "$POOL_NFS/" || return 1; done
  { ls -t "$LOCAL"/run/checkpoints/*.zip 2>/dev/null || true; } | tail -n +3 | xargs -r rm -f
}
( while sleep 600; do mirror || echo "$(date -u +%T) mirror FAILED" >&2; done ) &
MIRROR=$!
cleanup() {
  kill $MIRROR 2>/dev/null || true
  mirror && rm -rf "$LOCAL"
  pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security" || true
}
trap cleanup EXIT
trap 'exit 143' TERM
( while sleep 300; do
    f=$LOCAL/run/train.log; [ -f "$f" ] || continue
    if [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 2700 ]; then
      echo "$(date -u +%T) train.log silent for 45 min: requeueing" >&2
      scontrol requeue "$SLURM_JOB_ID" || pkill -u "$USER" -f "^$PY -m porygon.train"; break
    fi
  done ) &
$PY -m porygon.train --async-envs --overlap-update --ports "$BASE-$((BASE + N_SERVERS - 1))" --n-battles "$SELFPLAY" \
  --pool-battles "$POOL_BATTLES" --pool-dir "$LOCAL/pool" --run-dir "$LOCAL/run" \
  --resume "$LATEST" --total-steps "$TOTAL" --max-hours 70.5 --checkpoint-every 5000000 \
  "$@" 2>&1 | tee "$LOCAL/run/train.log" || true
kill $MIRROR 2>/dev/null || true; mirror || true
