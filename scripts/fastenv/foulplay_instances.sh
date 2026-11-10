#!/bin/bash
# K Foul Play instances for one block of a netsearch slot run (netsearch_slot.sh, OPP=foulplay...): each accepts
# challenges on ws://localhost:PORT as NAME_i and is restarted whenever it exits (Foul Play exits on some search errors,
# e.g. gen9's ZeroDivisionError in its move selection; a process that logged its fatal traceback but is still running
# after 30 s is killed first). Runs until killed (netsearch_slot.sh runs it in its own session and kills that).
# Foul Play is GPL-3.0 and is NOT part of this repository: it runs from its own clone (FP_DIR), read only, with the
# local-login patch already applied (scripts/slurm_foulplay.sbatch applies it).
#   foulplay_instances.sh FP_DIR FP_VENV FP_PYTHONPATH PORT FORMAT MS PARALLELISM LOGDIR NAME...
# Logs: LOGDIR/<NAME>.log ("Waiting for a" = ready; one "restarting (exit N)" line per restart).
FP=$1 VENV=$2 FPPATH=$3 PORT=$4 FORMAT=$5 MS=$6 PAR=$7 LOGDIR=$8; shift 8
FATAL='run.py", line [0-9]*, in <module>'
instance() {
  local name=$1 log=$LOGDIR/$1.log pid start size from g rc
  cd "$FP" || exit 1
  while true; do
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$FPPATH "$VENV/bin/python" run.py --websocket-uri "ws://localhost:$PORT/showdown/websocket" \
      --ps-username "$name" --bot-mode accept_challenge --pokemon-format "$FORMAT" --search-time-ms "$MS" \
      --search-parallelism "$PAR" --run-count 1000000 --log-level INFO < /dev/null >> "$log" 2>&1 &
    pid=$!; start=$(stat -c %s "$log")
    while kill -0 "$pid" 2>/dev/null; do
      sleep 2
      size=$(stat -c %s "$log"); from=$(( size - 65536 > start ? size - 65536 : start ))
      if [ "$(tail -c +$(( from + 1 )) "$log" | grep -c "$FATAL")" != 0 ]; then
        for ((g = 0; g < 30; g++)); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && { echo "watchdog: fatal error logged, pid $pid still running: killed" >> "$log"; kill -9 "$pid"; }
        break
      fi
    done
    wait "$pid"; rc=$?
    echo "restarting (exit $rc)" >> "$log"; sleep 10
  done
}
for name in "$@"; do instance "$name" & done
wait
