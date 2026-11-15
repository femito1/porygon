#!/bin/bash
# netv2 2026-11-14, G9t chain stage S2 (orchestrator-approved chain 15:2x, with the persistent-stop fix), a step in G9t's
# job (A16 810367, gpuvm36):
#   setsid nohup srun --jobid=810367 --overlap --ntasks=1 -c 1 --mem=1G -J nv2-g9t-s2 env JOBPP=<job script pid> \
#     END_EPOCH=<job end> bash $C/scripts/netv2_g9t/chain-s2.sh > $C/logs/g9t-chain-s2-<t>.out 2>&1 < /dev/null &
# 1 waits for S1's rule-1 and rule-3 READINGs (runs/g9t/cmp/{25M,drift50}/READING.txt; runs/g9t/chain/FAILED => stop);
# 2 a STOP (rule 1 or 3) => queue the park INLINE (scripts/netv2_g9t/park-g9t.sh: G9t parked whole in runs/g9t-parked,
#   G9T_STARTED run -> parked, runs/g9/PARKED => the job HOLDs, the successor exits at start);
#   both CONTINUE => (if >= 5 h are left before the job's end) queue the branch INLINE (scripts/netv2_g9t/inline-g9t-cd.sh:
#   G9t-cd, rule 2 h2h, rule 2 applied);
#   both after: no async eval running, exactly one trainer whose parent is the job script, no INLINE_CMD / HOLD /
#   WAIT_FOR queued; argv captured; INLINE_CMD via .tmp + mv; runs/g9/restart; ONE SIGINT;
# 3 waits for the INLINE's end; re-arms runs/g9/restart unless G9t was parked; logs G9t's relaunch. STOP before step 2:
#   touch runs/g9t/chain/S2_STOP.
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; G=$C/runs/g9t; K=$G/chain; R=$C/runs/g9
JOBPP=${JOBPP:?set JOBPP = the job script pid}; END=${END_EPOCH:?set END_EPOCH = the job end (epoch s)}
WAIT_UNTIL=${WAIT_UNTIL:-$(date -u -d '2026-11-14 18:40' +%s)}  # the data host 810087 ends ~18:10
J=${SLURM_JOB_ID:?}; TOP=/tmp/ws-netv2-$J/run; L=$TOP/G9t; LOG=$L/train.log
PAT='^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'
st() { echo "$(date -u +%FT%T) S2 $*" | tee -a "$K/STATUS"; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) S2 $*" >> "$K/FAILED"; exit 1; }
cnt() { local c; c=$(grep -c "$1" "$2" 2>/dev/null); echo "${c:-0}"; }
verd() { sed -n 's/^VERDICT //p' "$1/READING.txt" 2>/dev/null | head -n 1; }
mkdir -p "$K"
st "start in job $J (JOBPP $JOBPP, job end $(date -u -d "@$END" +%FT%T)); waiting for rules 1 + 3 (until $(date -u -d "@$WAIT_UNTIL" +%T))"
rule=""
while true; do
  [ -e "$K/FAILED" ] && { st "runs/g9t/chain/FAILED exists: S2 stops (G9t untouched)"; exit 1; }
  [ -e "$K/S2_STOP" ] && { st "S2_STOP: exiting (G9t untouched)"; exit 0; }
  v1=$(verd "$G/cmp/25M"); v3=$(verd "$G/cmp/drift50")
  if [ "$v1" = STOP ]; then rule=1; break; fi
  if [ "$v1" = CONTINUE ] && [ "$v3" = STOP ]; then rule=3; break; fi
  if [ "$v1" = CONTINUE ] && [ "$v3" = CONTINUE ]; then break; fi
  [ "$(date +%s)" -lt "$WAIT_UNTIL" ] || fail "no rule 1 + 3 verdicts by $(date -u -d "@$WAIT_UNTIL" +%T) (rule 1 '${v1:-}', rule 3 '${v3:-}')"
  sleep 60
done
if [ -n "$rule" ]; then st "rule $rule says STOP: parking G9t"
else
  left=$(( END - $(date +%s) ))
  [ "$left" -ge 18000 ] || fail "rules 1 + 3 CONTINUE but only $((left / 60)) min left in job $J (< 5 h): no branch here; main decides (successor 1137377)"
  st "rules 1 + 3 CONTINUE: queuing the G9t-cd branch + rule 2 ($((left / 60)) min left in the job)"
fi
# --- safe point: no async eval running (<= 30 min) ---
for i in $(seq 1 60); do
  ln0=$(grep -n "started (async, pid" "$LOG" | tail -n 1)
  [ -z "$ln0" ] && break
  ln=${ln0%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$ln0"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$ln0")
  if tail -n +"$ln" "$LOG" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null; then break; fi
  sleep 30
done
[ -e "$K/S2_STOP" ] && { st "S2_STOP: exiting (G9t untouched)"; exit 0; }
for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR PARKED; do [ ! -e "$R/$f" ] || fail "runs/g9/$f present: not queuing"; done
[ "$(cat "$R/MODE" 2>/dev/null)" = g9t ] || fail "runs/g9/MODE is not g9t"
ps=$(pgrep -u fr1226 -f "$PAT" || true); set -- $ps
[ $# = 1 ] || fail "expected exactly 1 trainer, got '$ps'"
pid=$1; pp=$(ps -o ppid= -p "$pid" | tr -d ' ')
[ "$pp" = "$JOBPP" ] || fail "trainer $pid has parent $pp, expected $JOBPP"
[ -e "$TOP/final_model.pt" ] || fail "top-level final_model.pt missing (the spooled loop would end the job)"
cp "/proc/$pid/cmdline" "$K/g9t-trainer-argv.bin"; tr '\0' ' ' < "$K/g9t-trainer-argv.bin" > "$K/g9t-trainer-argv.txt"; echo >> "$K/g9t-trainer-argv.txt"
grep -q -- "--run-dir $L " "$K/g9t-trainer-argv.txt" && grep -q -- '--gae-lambda 0.5' "$K/g9t-trainer-argv.txt" || fail "argv is not G9t's (run dir / --gae-lambda 0.5)"
! grep -q -- '--lr-schedule' "$K/g9t-trainer-argv.txt" || fail "argv has an --lr-schedule (G9t = constant lr)"
if [ -n "$rule" ]; then
  printf 'set -u\n# G9t chain: STOP by runs/g9t/PREREG.txt rule %s => park G9t (scripts/netv2_g9t/park-g9t.sh; PARKED)\nC=%s\n' "$rule" "$C" > "$R/INLINE_CMD.tmp"
  cat >> "$R/INLINE_CMD.tmp" <<INL
if PARK_RULE=$rule PARK_QUEUED_FILE="\$C/runs/g9/INLINE_CMD.running" bash "\$C/scripts/netv2_g9t/park-g9t.sh"; then
  echo "\$(date -u +%FT%T) INLINE FINAL: G9t stopped by rule $rule; parked in runs/g9t-parked (PARKED.txt + MD5SUMS); the A16 job HOLDs, the successor exits at start; A16 placement = main" | tee -a "\$C/runs/g9t/chain/STATUS"
else
  echo "\$(date -u +%FT%T) INLINE FAILED: rule $rule STOP but the park failed (runs/g9t/park.log); G9t relaunches" | tee -a "\$C/runs/g9t/chain/STATUS"
  echo "\$(date -u +%FT%T) INLINE park failed (rule $rule)" >> "\$C/runs/g9t/chain/FAILED"
fi
INL
else
  printf 'set -u\n# G9t chain: G9t-cd branch + rule 2 (scripts/netv2_g9t/inline-g9t-cd.sh md5 %s)\n' "$(md5sum < "$C/scripts/netv2_g9t/inline-g9t-cd.sh" | cut -c1-8)" > "$R/INLINE_CMD.tmp"
  echo "bash $C/scripts/netv2_g9t/inline-g9t-cd.sh" >> "$R/INLINE_CMD.tmp"
fi
bash -n "$R/INLINE_CMD.tmp" || fail "INLINE_CMD.tmp syntax error"
cp -p "$R/INLINE_CMD.tmp" "$K/INLINE_CMD.queued"
mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD"
touch "$R/restart"
st "INLINE_CMD queued ($([ -n "$rule" ] && echo "park, rule $rule" || echo "G9t-cd + rule 2")); restart armed; trainer $pid (parent $pp)"
before=$(cnt "signal 2" "$LOG"); starts=$(cnt "v2 trainer:" "$LOG"); idone=$(cnt "inline command done" "$TOP/train.log")
kill -INT "$pid" && st "SIGINT sent to G9t trainer $pid"
for i in $(seq 1 18); do sleep 10; [ "$(cnt "signal 2" "$LOG")" -gt "$before" ] && break; done
st "landed: $(grep "signal 2" "$LOG" | tail -n 1 | cut -c1-160)"
for i in $(seq 1 3000); do sleep 10; [ "$(cnt "inline command done" "$TOP/train.log")" -gt "$idone" ] && break; done  # <= 8.3 h
st "INLINE done: $(grep "inline command" "$TOP/train.log" | tail -n 2 | tr '\n' ' ' | cut -c1-200)"
if [ -f "$C/runs/g9t-parked/PARKED.txt" ]; then
  sleep 120; st "G9t parked; job HOLD: $(grep -E 'HOLD|refused' "$TOP/train.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
  st "S2 done (stop path)"; exit 0
fi
touch "$R/restart"
for i in $(seq 1 180); do sleep 10; [ "$(cnt "v2 trainer:" "$LOG")" -gt "$starts" ] && break; done
st "G9t relaunched: $(grep "v2 trainer:" "$LOG" | tail -n 1 | cut -c1-160); restart re-armed"
st "S2 done"
