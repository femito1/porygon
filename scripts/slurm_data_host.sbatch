#!/bin/bash
# Data-stream host allocation: the successor of 677193, then of itself every 3 days (orchestrator-approved
# 2026-11-09, option A). It keeps the data stream's long-running steps in one a16 allocation:
#   - the inputlog crawler (resumes from its cursor in raw/showdown-replays/state.sqlite; flock: never two at once;
#     a crashed crawler is restarted after 1 h, at most 3 times per job),
#   - the 4-hourly dataset refresh (a private copy of scripts/refresh_datasets.sh; first round after REFRESH_DELAY s,
#     so a round killed at the previous job's end is redone from its last committed state),
#   - optionally one label step: $D/host/labels.args = one line of scripts/slurm_searchlabels.sbatch arguments
#     (tag <job>1: run dir runs/searchlabels-gen<G>-<date>-<job>1, its own port),
#   - netv2's pretraining steps, which netv2 starts itself:
#       srun --jobid=<this job> --overlap --ntasks=1 -c 8 --gres=gpu:1 --mem=14G ...  (limit its threads itself)
# CPU pinning (measured in 677193, 2026-11-09): an overlap step with -c N is bound to the job's first N/2 physical cores
# (8 CPUs = 4 cores + SMT siblings), and --cpu-bind=none does not change that. So every -c 1 / -c 2 step shares the
# first core. Steps that compute use -c 8 and bound their own parallelism, so the kernel spreads them over idle cores.
# The batch step only holds the allocation and NEVER exits on its own: ending it ends every step, netv2's included.
# A TERM (time limit, scancel, or a stray kill) is logged and ignored; at the time limit or on scancel Slurm
# SIGKILLs the job after KillWait anyway. At start, and every CHECK_EVERY x HOLD_S s, it makes sure one successor
# is queued: same partition and QOS, --dependency=afterany:<this job>.
# Stop the chain: touch $D/host/STOP (no further successor is queued), then scancel the pending ws-data-host job.
# Log: $D/logs/ws-data-host.log; every failure is a line containing "HOST ERROR" (the orchestrator check-in greps it).
#   first job: cd $D/code && sbatch --dependency=afterany:677193 scripts/slurm_data_host.sbatch
#SBATCH --job-name=ws-data-host
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-data/logs/slurm-%j.out
# No `set -e` / `set -u` / pipefail on purpose: nothing in here may end the batch step early.
D=${WS_DATA_ROOT:-/vol/bitbucket/fr1226/porygon-ws-data}
SELF=$D/code/scripts/slurm_data_host.sbatch   # the deployed copy (deploy with rsync, a new inode)
HOST=$D/host
LOG=$D/logs/ws-data-host.log
HOLD_S=${HOST_HOLD_S:-300}                    # holder period (s)
CHECK_EVERY=${HOST_CHECK_EVERY:-12}           # successor check every CHECK_EVERY periods (hourly)
REFRESH_DELAY=${HOST_REFRESH_DELAY:-600}      # s before the first refresh round
CRAWL_RETRY_S=${HOST_CRAWL_RETRY_S:-3600}     # s before a crashed crawler is restarted
PY3=${HOST_PY3:-/vol/bitbucket/fr1226/python/cpython-3.12.16/bin/python3}
ME=${USER:-$(id -un)}
J=${SLURM_JOB_ID:?not inside a Slurm job}
RUN=$HOST/run-$J
mkdir -p "$HOST" "$RUN" "$D/logs"

log() { echo "$(date -u +%FT%TZ) [$J] $*" | tee -a "$LOG"; }
err() { log "HOST ERROR: $*"; }

SLEEP_PID=
on_signal() {  # never exit: a stray signal must not end the allocation
  log "batch: $1 received: ignored (at the time limit or on scancel Slurm SIGKILLs the job after KillWait)"
  [ -n "$SLEEP_PID" ] && kill "$SLEEP_PID" 2>/dev/null
}
for s in TERM INT HUP USR1 USR2; do trap "on_signal $s" "$s"; done

STOP_LOGGED=
ensure_successor() {  # exactly one pending ws-data-host job waiting on this one
  if [ -e "$HOST/STOP" ]; then
    [ -n "$STOP_LOGGED" ] || log "chain: $HOST/STOP present: no successor is queued (scancel a pending one by hand)"
    STOP_LOGGED=1; return 0
  fi
  STOP_LOGGED=
  local q rc n out
  q=$(squeue -h -u "$ME" -n ws-data-host -t PD -o "%i %E" 2>&1); rc=$?
  if [ "$rc" -ne 0 ]; then err "chain: squeue failed (rc $rc), retrying at the next check: $q"; return 1; fi
  n=$(grep -c -E "afterany:$J(\(|,| |$)" <<< "$q")
  if [ "$n" -ge 1 ]; then
    [ "$n" -gt 1 ] && err "chain: $n pending successors wait on $J (expected 1): $(tr '\n' ' ' <<< "$q")"
    return 0
  fi
  # stdout only: the submit plugin prints a banner on stderr, which made a successful submission look failed (11-11)
  local errf="$HOST/sbatch-$J.err"
  out=$(cd "$D/code" && sbatch --parsable --job-name=ws-data-host --partition="$SLURM_JOB_PARTITION" \
        ${SLURM_JOB_QOS:+--qos="$SLURM_JOB_QOS"} --dependency="afterany:$J" "$SELF" 2>"$errf"); rc=$?
  out=$(tr -d '[:space:]' <<< "$out")
  if [ "$rc" -eq 0 ] && [[ $out =~ ^[0-9]+(\;.*)?$ ]]; then
    log "chain: queued successor ${out%%;*} (afterany:$J, partition $SLURM_JOB_PARTITION, qos ${SLURM_JOB_QOS:-default})"
  else
    err "chain: self-submission failed (rc $rc), retrying at the next check: stdout '$out' stderr $(tail -c 400 "$errf" | tr '\n' ' ')"
    return 1
  fi
}

declare -A PIDS=() DONE=()
start_crawl() {
  log "crawl: starting (resumes from the cursor in $D/raw/showdown-replays/state.sqlite; log $D/logs/crawl.log)"
  ( cd /tmp && exec srun --overlap --ntasks=1 -c 1 --mem=1G --job-name=crawl \
      flock -n -E 75 "$D/logs/crawl.lock" "$PY3" -I "$D/bin-crawl.py" crawl --root "$D/raw/showdown-replays" \
      --min-interval 0.5 --min-prio 2 --queue "$D/raw/inputlog-queue-hf-v2-ge1500.txt.gz" --queue-min-prio 2 \
      --exit-when-done --min-free-mb 2000 >> "$D/logs/crawl.log" 2>&1 < /dev/null ) &
  PIDS[crawl]=$!; DONE[crawl]=
}
start_refresh() {  # a private copy: bash reads scripts lazily, and an rsync of the original must not break this one
  if ! cp "$D/code/scripts/refresh_datasets.sh" "$RUN/refresh_datasets.sh"; then
    err "refresh: cannot copy $D/code/scripts/refresh_datasets.sh: not started"; return 1
  fi
  log "refresh: starting $RUN/refresh_datasets.sh 14400 $REFRESH_DELAY (log $D/logs/ws-data-refresh.log)"
  ( cd /tmp && exec srun --overlap --ntasks=1 -c 8 --mem=24G --job-name=refresh \
      bash "$RUN/refresh_datasets.sh" 14400 "$REFRESH_DELAY" >> "$D/logs/ws-data-refresh.log" 2>&1 < /dev/null ) &
  PIDS[refresh]=$!; DONE[refresh]=
}
start_labels() {
  local args
  args=$(grep -v -E '^\s*(#|$)' "$HOST/labels.args" 2>/dev/null | head -1)
  if [ -z "$args" ]; then log "labels: no arguments in $HOST/labels.args: no label step"; return 0; fi
  if ! cp "$D/code/scripts/slurm_searchlabels.sbatch" "$RUN/slurm_searchlabels.sbatch"; then
    err "labels: cannot copy slurm_searchlabels.sbatch: not started"; return 1
  fi
  log "labels: starting slurm_searchlabels.sbatch $args (tag ${J}1, -c ${HOST_LABEL_CPUS:-6}, log $RUN/labels.out)"
  # shellcheck disable=SC2086  # $args: one line of arguments, split on purpose
  ( cd /tmp && exec srun --overlap --ntasks=1 -c "${HOST_LABEL_CPUS:-6}" --mem="${HOST_LABEL_MEM:-8G}" \
      --job-name=labels env SLURM_JOB_ID="${J}1" SLURM_SUBMIT_DIR="$D/code" \
      bash "$RUN/slurm_searchlabels.sbatch" $args >> "$RUN/labels.out" 2>&1 < /dev/null ) &
  PIDS[labels]=$!; DONE[labels]=
}

CRAWL_RESTARTS=0 CRAWL_RESTART_AT=0
step_exit() {  # NAME RC: a step's srun returned
  case $1 in
    crawl)
      case $2 in
        0) log "crawl: finished (queue done: --exit-when-done)" ;;
        75) err "crawl: not started: another crawler holds $D/logs/crawl.lock" ;;
        *) if [ "$CRAWL_RESTARTS" -lt 3 ]; then
             CRAWL_RESTART_AT=$(( $(date +%s) + CRAWL_RETRY_S ))
             err "crawl: exited with rc $2 (see $D/logs/crawl.log); restart $((CRAWL_RESTARTS + 1))/3 in $CRAWL_RETRY_S s"
           else
             err "crawl: exited with rc $2 (see $D/logs/crawl.log); 3 restarts used in this job: left stopped"
           fi ;;
      esac ;;
    refresh) err "refresh: exited with rc $2 (it stops itself after two failed rounds; see $D/logs/ws-data-refresh.log)" ;;
    *) log "$1: step exited with rc $2" ;;
  esac
}
check_steps() {
  local name rc
  for name in "${!PIDS[@]}"; do
    [ -n "${DONE[$name]}" ] && continue
    kill -0 "${PIDS[$name]}" 2>/dev/null && continue
    wait "${PIDS[$name]}"; rc=$?
    DONE[$name]=1
    step_exit "$name" "$rc"
  done
  if [ "$CRAWL_RESTART_AT" -gt 0 ] && [ "$(date +%s)" -ge "$CRAWL_RESTART_AT" ]; then
    CRAWL_RESTART_AT=0; CRAWL_RESTARTS=$((CRAWL_RESTARTS + 1)); start_crawl
  fi
}

log "start: $(hostname -s), partition ${SLURM_JOB_PARTITION:-?}, qos ${SLURM_JOB_QOS:-?}, ${SLURM_CPUS_ON_NODE:-?} CPUs, mem ${SLURM_MEM_PER_NODE:-?} MB, gpu $(nvidia-smi -L 2>/dev/null | head -1 | cut -c1-50), ends $(squeue -h -j "$J" -o %e 2>/dev/null)"
ensure_successor
start_crawl
start_refresh
start_labels
i=0
while true; do  # the holder: never leaves this loop
  sleep "$HOLD_S" & SLEEP_PID=$!
  wait "$SLEEP_PID" 2>/dev/null
  SLEEP_PID=
  i=$((i + 1))
  check_steps
  [ $((i % CHECK_EVERY)) -eq 0 ] && ensure_successor
done
