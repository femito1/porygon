#!/bin/bash
# A pre-registered checkpoint check, automated inside the training job (laptop-independent). Waits for the lineage's
# first checkpoint at >= STEPS, copies it to runs/RUN/, queues the head-to-head as runs/RUN/INLINE_CMD (played on
# the job's own CPUs while the trainer is paused; scripts/netv2_h2h.sbatch, result runs/h2h-<date>-<job>/result.txt),
# arms the restart flag, SIGINTs the trainer (resent once only if it did not land: a second SIGINT after a landed one
# aborts the save), waits for the relaunch and re-arms the restart flag. Launch from the login node:
#   nohup srun --jobid=718701 --overlap --ntasks=1 -c 1 --mem=1G bash $C/scripts/netv2_check_at.sh \
#     r1 G9 25000000 "$C/runs/pretrain/P1-g9h-dmg-s2.pt,temperature=0.5" 1000 --gen 9 > $C/logs/check-G9-25M.out 2>&1 &
# STOP: touch runs/RUN/CHECK_STOP (the watcher exits at its next poll without doing anything).
# OPP may list several opponents separated by '|' (2026-11-09): one h2h each, in that order, same N and args (e.g. the
# R1b-F 50M check: "v1@46M,temperature=0.5|v1@500M,temperature=0.5" ... --gen 4 --seed 9450).
set -uo pipefail
C=${NV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}
RUN=$1 LIN=$2 STEPS=$3 OPP=$4 N=$5; shift 5
LOCAL=${NV2_LOCAL:-/tmp/ws-netv2-$SLURM_JOB_ID}
LOG=$LOCAL/run/$LIN/train.log
CK=$LOCAL/run/$LIN/checkpoints
PAT=${NV2_TRAINER_PAT:-'^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'}
say() { echo "$(date -u +%T) $*"; }
cnt() { local c; c=$(grep -c "$1" "$LOG" 2>/dev/null); echo "${c:-0}"; }  # matching lines in the lineage log
say "waiting for a $LIN checkpoint >= $STEPS in $CK (job $SLURM_JOB_ID)"
f=""
while [ -z "$f" ]; do
  if [ -f "$C/runs/$RUN/CHECK_STOP" ]; then say "CHECK_STOP present: exiting"; exit 0; fi
  for x in "$CK"/ckpt_*.pt; do
    [ -e "$x" ] || continue
    n=$(basename "$x" .pt); n=${n#ckpt_}
    case $n in *[!0-9]*|"") continue ;; esac
    if [ "$((10#$n))" -ge "$STEPS" ]; then f=$x; break; fi
  done
  [ -n "$f" ] || sleep "${NV2_POLL:-60}"
done
say "checkpoint $f"
dst=$C/runs/$RUN/check-$LIN-$(basename "$f")
cp -p "$f" "$dst.tmp" && mv -f "$dst.tmp" "$dst" || { say "copy failed: using the node-local file"; dst=$f; }
if [ -e "$C/runs/$RUN/INLINE_CMD" ]; then say "an INLINE_CMD is already queued: not overwriting it, exiting"; exit 1; fi
IFS='|' read -r -a OPPS <<< "$OPP"
: > "$C/runs/$RUN/INLINE_CMD.tmp"
for o in "${OPPS[@]}"; do
  printf 'bash scripts/netv2_h2h.sbatch "%s,temperature=0.5" "%s" %s %s\n' "$dst" "$o" "$N" "$*" >> "$C/runs/$RUN/INLINE_CMD.tmp"
done
mv -f "$C/runs/$RUN/INLINE_CMD.tmp" "$C/runs/$RUN/INLINE_CMD"
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
# wait for the inline command to finish and the trainer to come back (<= 90 min), then re-arm the restart flag
starts=$(cnt "v2 trainer:")
for i in $(seq 1 540); do
  sleep "${NV2_TICK:-10}"
  [ "$(cnt "v2 trainer:")" -gt "$starts" ] && break
done
touch "$C/runs/$RUN/restart"
say "trainer relaunched: $(grep "v2 trainer:" "$LOG" | tail -n 1 | cut -c1-160); restart flag re-armed"
for r in $(ls -tr "$C"/runs/h2h-*-"$SLURM_JOB_ID"/result.txt 2>/dev/null | tail -n "${#OPPS[@]}"); do
  say "result: $r"; cat "$r" 2>/dev/null
done
