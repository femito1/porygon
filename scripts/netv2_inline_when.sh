#!/bin/bash
# Run a command inside a training job at a restart in place, once a file exists (e.g. a pretraining arm's final.pt):
# queues CMD as runs/RUN/INLINE_CMD (played on the job's own CPUs while the trainer is paused), arms the restart
# flag, SIGINTs the lineage's trainer once (resent only if it did not land: a second SIGINT after a landed one
# aborts the save), waits for the relaunch and re-arms the flag. Same mechanics as scripts/netv2_check_at.sh.
#   nohup srun --jobid=718701 --overlap --ntasks=1 -c 1 --mem=1G bash $C/scripts/netv2_inline_when.sh \
#     r1 G9s $C/runs/pretrain/g9-p2mix-s2/final.pt 'bash scripts/netv2_h2h.sbatch "A,temperature=0.5" "B,..." 1000 --gen 9' \
#     > $C/logs/inline-when-p2mix.out 2>&1 &
# STOP: touch runs/RUN/CHECK_STOP.
set -uo pipefail
C=${NV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}
RUN=$1 LIN=$2 WAIT=$3 CMD=$4
LOCAL=${NV2_LOCAL:-/tmp/ws-netv2-$SLURM_JOB_ID}
LOG=$LOCAL/run/$LIN/train.log
PAT=${NV2_TRAINER_PAT:-'^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'}
say() { echo "$(date -u +%T) $*"; }
cnt() { local c; c=$(grep -c "$1" "$LOG" 2>/dev/null); echo "${c:-0}"; }
say "waiting for $WAIT (then: $CMD) in job $SLURM_JOB_ID, lineage $LIN"
until [ -e "$WAIT" ]; do
  if [ -f "$C/runs/$RUN/CHECK_STOP" ]; then say "CHECK_STOP present: exiting"; exit 0; fi
  sleep "${NV2_POLL:-60}"
done
sleep 30  # let the writer finish anything it does after the file appears (final save, logs)
# never interleave with another queued inline command (e.g. a checkpoint check): wait until none is queued
until [ ! -e "$C/runs/$RUN/INLINE_CMD" ] && [ ! -e "$C/runs/$RUN/INLINE_CMD.running" ]; do sleep 30; done
echo "$CMD" > "$C/runs/$RUN/INLINE_CMD"
say "INLINE_CMD: $(cat "$C/runs/$RUN/INLINE_CMD")"
touch "$C/runs/$RUN/restart"
before=$(cnt "signal 2")
pkill -INT -f "$PAT"; say "SIGINT sent ($(pgrep -f "$PAT" | tr '\n' ' '))"
for i in $(seq 1 12); do
  sleep "${NV2_TICK:-10}"
  [ "$(cnt "signal 2")" -gt "$before" ] && break
done
if [ "$(cnt "signal 2")" -le "$before" ]; then
  say "SIGINT did not land within 120 s: resending once"
  pkill -INT -f "$PAT"
  sleep 30
fi
say "landed: $(grep "signal 2" "$LOG" | tail -n 1)"
starts=$(cnt "v2 trainer:")
for i in $(seq 1 540); do
  sleep "${NV2_TICK:-10}"
  [ "$(cnt "v2 trainer:")" -gt "$starts" ] && break
done
touch "$C/runs/$RUN/restart"
say "trainer relaunched: $(grep "v2 trainer:" "$LOG" | tail -n 1 | cut -c1-160); restart flag re-armed"
say "newest h2h result: $(ls -t "$C"/runs/h2h-*-"$SLURM_JOB_ID"/result.txt 2>/dev/null | head -n 1)"
cat "$(ls -t "$C"/runs/h2h-*-"$SLURM_JOB_ID"/result.txt 2>/dev/null | head -n 1)" 2>/dev/null
