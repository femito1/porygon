#!/bin/bash
# Model v2 self-play PPO (porygon.v2.train) on a Slurm GPU node of the Imperial DoC cluster.
#
#   cd /vol/bitbucket/fr1226/porygon-ws-netv2 && mkdir -p logs
#   sbatch scripts/slurm_train_v2.sbatch RUN_NAME [extra porygon.v2.train args]
#
# ROOT = the submit dir (this checkout); code from $ROOT/src (first on PYTHONPATH), Showdown from $ROOT/pokemon-showdown
# (already installed; scripts/start_servers.sh builds it), venv and Node from /vol/bitbucket/fr1226/gloria.
# Settings in $ROOT/netv2.env (sourced; never `sbatch --export`, which gets jobs held on this cluster):
#   FORMAT=gen9randombattle  TOTAL=300000000  (total env steps)
#   RESUME=<.pt>     start from this checkpoint when the run has none of its own
#   ANCHOR=<.pt|.zip> evaluated against every --eval-every steps (best_model.pt by that score)
#   POOL_SEED=<dir>  v2 snapshots hard-linked into the run's pool when it is empty
#   CHAIN=<n>        follow-up jobs once this one ends short of TOTAL (counter kept per run in runs/RUN_NAME/chain_left)
#   TRAIN_ARGS="..." extra porygon.v2.train args for every job of every run (positional args after RUN_NAME win)
# netv2.env may branch on $RUN_NAME for per-run settings.
#
# Layout: runs/RUN_NAME/<date>-<job>/ (train.log, stdout.log, tb/, checkpoints/, final_model.pt, best_model.pt) and
# runs/RUN_NAME/pool/ (snapshots). Every job resumes from the newest checkpoint / final model under runs/RUN_NAME, so
# a requeue (node failure, watchdog) or a chained job continues the same run; several RUN_NAMEs never touch each other.
# The run and pool live on the node (tmpfs, counts toward --mem) and are mirrored to runs/RUN_NAME every 10 min and
# at the end (the shared disk can stall for minutes). No results are pushed anywhere.
#SBATCH --job-name=ws-netv2-train
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --requeue
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
RUN_NAME=${1:?usage: sbatch scripts/slurm_train_v2.sbatch RUN_NAME [porygon.v2.train args]}; shift
case $RUN_NAME in */*|.*|"") echo "bad RUN_NAME '$RUN_NAME'" >&2; exit 2;; esac
[ -f netv2.env ] && source netv2.env
PY=${PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}
export PATH=${NODE_BIN:-/vol/bitbucket/fr1226/gloria/.node/bin}:$PATH
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1
FORMAT=${FORMAT:-gen9randombattle}; TOTAL=${TOTAL:-300000000}
if ! "$PY" -c "import torch, porygon.v2.train; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(), porygon.v2.train.__file__)"; then
  # a node with a broken GPU (seen: CUDA unknown error): resubmit the same run elsewhere, excluding this node
  EXCL=$(hostname -s)${SLURM_JOB_EXC_NODES:+,$SLURM_JOB_EXC_NODES}
  echo "no usable CUDA on $(hostname -s): resubmitting $RUN_NAME excluding $EXCL" >&2
  sbatch --job-name="$SLURM_JOB_NAME" --partition="$SLURM_JOB_PARTITION" --exclude="$EXCL" scripts/slurm_train_v2.sbatch "$RUN_NAME" "$@"
  exit 1
fi
[ -f pokemon-showdown/package.json ] || { echo "no Showdown checkout at $ROOT/pokemon-showdown" >&2; exit 1; }

# Take over from a stand-in job of the same run (e.g. the run parked on the long partition while this job waited for
# its GPU): PREEMPT_NAMES (netv2.env) lists job names; any of ours still running gets SIGTERM (the trainer saves
# final_model.pt, the job mirrors and exits) and we wait until it has left the queue, so the two never write the
# same run at once; then the usual newest-checkpoint resume below picks up where it stopped.
for name in ${PREEMPT_NAMES:-}; do
  [ "$name" = "$SLURM_JOB_NAME" ] && continue
  for j in $(squeue -h -u "$USER" -n "$name" -t RUNNING -o %i 2>/dev/null || true); do
    echo "$(date -u +%T) stopping stand-in job $j ($name) before resuming $RUN_NAME"
    scancel --signal=TERM --batch "$j" || true
    for ((t = 0; t < 60; t++)); do
      [ -z "$(squeue -h -j "$j" -o %i 2>/dev/null || true)" ] && break
      sleep 30
    done
    [ -n "$(squeue -h -j "$j" -o %i 2>/dev/null || true)" ] && { echo "stand-in $j still running after 30 min: cancelling" >&2; scancel "$j" || true; sleep 60; }
  done
done

# --- servers: one per ~3 cores, ports from the job id (nodes are shared) ---
# CPUs we really have: the cgroup's (OMP_NUM_THREADS unset — GNU nproc honours it, and we export 1 above, which once
# made every job size itself for 1 core), cross-checked with Slurm's count; the submit plugin may reshape -c.
CORES=$(env -u OMP_NUM_THREADS nproc)
[ -n "${SLURM_CPUS_ON_NODE:-}" ] && [ "$SLURM_CPUS_ON_NODE" -lt "$CORES" ] && CORES=$SLURM_CPUS_ON_NODE
echo "CPUs: nproc $(env -u OMP_NUM_THREADS nproc), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?} -> using $CORES"
MIN_CORES=${MIN_CORES:-24}
if [ "$CORES" -lt "$MIN_CORES" ]; then
  echo "ERROR: only $CORES CPUs on $(hostname -s) ($SLURM_JOB_PARTITION), need >= $MIN_CORES: not starting" >&2; exit 3
fi
N_SERVERS=$(( CORES / 3 )); SELFPLAY=$(( CORES * 2 )); POOL_BATTLES=$CORES
BASE=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 )); LAST=$(( BASE + N_SERVERS - 1 ))
# start_servers.sh writes the no-disk-writes training config only when there is none
CFG=pokemon-showdown/config/config.js
if [ -f "$CFG" ] && ! grep -q nofswriting "$CFG"; then rm -f "$CFG"; fi
stop_servers() { for ((p = BASE; p <= LAST; p++)); do pkill -u "$USER" -f "pokemon-showdown start --skip-build --no-security $p\$" || true; done; }
trap stop_servers EXIT
scripts/start_servers.sh "$N_SERVERS" "$BASE"

# --- run directories ---
RUNS=$ROOT/runs/$RUN_NAME; mkdir -p "$RUNS/pool"
LATEST=$(ls -t "$RUNS"/*/checkpoints/*.pt "$RUNS"/*/final_model.pt 2>/dev/null | head -1 || true)
LATEST=${LATEST:-${RESUME:-}}
JOB_DIR=$RUNS/$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$JOB_DIR"
[ -f "$RUNS/chain_left" ] || echo "${CHAIN:-0}" > "$RUNS/chain_left"
if [ -n "${POOL_SEED:-}" ] && [ -z "$(ls -A "$RUNS/pool" 2>/dev/null || true)" ]; then
  cp -al "$POOL_SEED"/*.pt "$RUNS/pool"/ 2>/dev/null || cp -p "$POOL_SEED"/*.pt "$RUNS/pool"/ || true
fi
echo "run $RUN_NAME job $SLURM_JOB_ID on $(hostname): $FORMAT, resuming from ${LATEST:-scratch}, $SELFPLAY self-play + $POOL_BATTLES pool battles, ports $BASE-$LAST"

LOCAL=${TMPDIR:-/tmp}/ws-netv2-$SLURM_JOB_ID; mkdir -p "$LOCAL/run" "$LOCAL/pool"
export TORCHINDUCTOR_CACHE_DIR=$LOCAL/inductor
# pool copied in the background, newest first (cp -p keeps the mtimes PoolOpponent's "recent" picks use); until
# snapshots arrive, pool battles fall back to the baseline bots (the pool directory is re-listed every battle)
( { ls -t "$RUNS"/pool/*.pt 2>/dev/null || true; } | while read -r f; do
    cp -p "$f" "$LOCAL/pool/.tmp" && mv "$LOCAL/pool/.tmp" "$LOCAL/pool/${f##*/}"; done ) &
mirror() {  # status = the rsyncs'; pruning the local checkpoints (keep the 2 newest) must not fail it
  rsync -a "$LOCAL/run/" "$JOB_DIR/" && rsync -a --exclude .tmp --exclude '*.tmp' "$LOCAL/pool/" "$RUNS/pool/" || return 1
  { ls -t "$LOCAL"/run/checkpoints/*.pt 2>/dev/null || true; } | tail -n +3 | xargs -r rm -f
}
( while sleep 600; do mirror || echo "$(date -u +%T) mirror FAILED" >&2; done ) &
MIRROR=$!
# watchdog: if train.log is silent for 45 min (an eval takes ~10), requeue: the job restarts from its newest checkpoint
( while sleep 300; do
    f=$LOCAL/run/train.log; [ -f "$f" ] || continue
    if [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 2700 ]; then
      echo "$(date -u +%T) train.log silent for 45 min: requeueing" >&2
      mirror || true
      scontrol requeue "$SLURM_JOB_ID" || pkill -u "$USER" -f "porygon.v2.train --format $FORMAT --ports $BASE-"; break
    fi
  done ) &
WATCHDOG=$!
cleanup() {  # also on scancel / time limit (SIGTERM; Slurm allows ~30 s before SIGKILL)
  kill "$MIRROR" "$WATCHDOG" 2>/dev/null || true
  if mirror; then rm -rf "$LOCAL"; else echo "final mirror failed: keeping $LOCAL" >&2; fi
  stop_servers
}
trap cleanup EXIT

# Restart in place (new flags, same allocation): edit netv2.env, `touch runs/RUN_NAME/restart`, then
#   srun --jobid=J --overlap --ntasks=1 pkill -INT -f '^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train'
# The trainer saves final_model.pt and exits; this loop re-reads netv2.env and relaunches it from that checkpoint.
STOPPED=0
trap 'STOPPED=1; kill -TERM "$TRAIN" 2>/dev/null || true' TERM  # the trainer saves final_model.pt and exits
while true; do
  # shellcheck disable=SC2086
  "$PY" -u -m porygon.v2.train --format "$FORMAT" --ports "$BASE-$LAST" --n-battles "$SELFPLAY" \
    --pool-battles "$POOL_BATTLES" --pool-dir "$LOCAL/pool" --run-dir "$LOCAL/run" \
    ${LATEST:+--resume "$LATEST"} ${ANCHOR:+--anchor "$ANCHOR"} --total-steps "$TOTAL" --max-hours 70.5 \
    ${TRAIN_ARGS:-} "$@" >> "$LOCAL/run/stdout.log" 2>&1 &
  TRAIN=$!
  RC=0
  while kill -0 "$TRAIN" 2>/dev/null; do wait "$TRAIN" && RC=0 || RC=$?; done
  if [ "$STOPPED" = 0 ] && [ -f "$RUNS/restart" ] && [ -f "$LOCAL/run/final_model.pt" ]; then
    rm -f "$RUNS/restart"
    [ -f netv2.env ] && source netv2.env
    LATEST=$LOCAL/run/final_model.pt
    echo "$(date -u +%T) restart in place from $LATEST with TRAIN_ARGS='${TRAIN_ARGS:-}'" | tee -a "$LOCAL/run/train.log"
    continue
  fi
  break
done
trap cleanup EXIT
kill "$MIRROR" 2>/dev/null || true; mirror || true
if [ "$RC" -ne 0 ] && [ "$RC" -ne 143 ] && [ "$RC" -ne 130 ]; then  # a crashed trainer must fail the job (not COMPLETED / exit 0)
  echo "ERROR: trainer exited with $RC; tail of stdout.log:" >&2; tail -25 "$LOCAL/run/stdout.log" >&2 || true; exit "$RC"
fi

# --- continue in a new job (72 h walltime) unless the chain is used up or training is done ---
STEPS=$({ grep -oE "steps [0-9,]+ \(" "$LOCAL/run/train.log" 2>/dev/null || true; } | tail -1 | tr -dc 0-9)
LEFT=$(cat "$RUNS/chain_left" 2>/dev/null || echo 0)
if [ "${LEFT:-0}" -gt 0 ] && [ -f "$JOB_DIR/final_model.pt" ] && [ "${STEPS:-0}" -lt "$TOTAL" ]; then
  echo $(( LEFT - 1 )) > "$RUNS/chain_left"
  sbatch scripts/slurm_train_v2.sbatch "$RUN_NAME" "$@"
fi
