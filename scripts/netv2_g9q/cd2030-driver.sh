#!/bin/bash
# netv2 2026-11-23: G9q-cd@2030M (runs/g9q-cd2030/PREREG.txt) = cd1350-driver.sh (0ca711db) with S_MIN 2030M, the dir
# runs/g9q-cd2030, the names, its pinned PREREG md5 and the keep dir: G9q keeps only its latest checkpoint, so the
# first one >= 2030M (ckpt_02030019516.pt, md5 62e8cc66) was copied node-locally to /tmp/ws-netv2-<job>/cd2030-keep/
# before the PREREG; the driver takes the earliest checkpoint >= S_MIN in that dir and the live dir (md5 pinned for
# 2,030,019,516). ONE INLINE_CMD of G9q's job. Run inside the job, from a node-local copy:
#   setsid nohup srun --jobid=1483048 --overlap --ntasks=1 -c 1 --mem=1G -J nv2-cd2030 env JOBPP=<job script pid> \
#     bash -c 'cp $C/scripts/netv2_g9q/cd2030-driver.sh /tmp/nv2-cd2030-$$.sh && exec bash /tmp/nv2-cd2030-$$.sh' \
#     > $C/logs/cd2030-<t>.out 2>&1 < /dev/null &
# STOP before the SIGINT: touch runs/g9q-cd2030/CD_STOP. A refusal or a missing final writes runs/g9q-cd2030/FAILED; a
# failed relaunch check writes runs/g9q-cd2030/relaunch/FAILED (G9q itself is then for the agent to check).
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; S_MIN=2030000000; JOBPP=${JOBPP:?set JOBPP = the job script pid}
PRE_MD5=7a68bf3e; KEEP_S=2030019516; KEEP_MD5=62e8cc66; LT=/tmp/ws-netv2-$SLURM_JOB_ID
PAT='^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'; PU=fr1226
if [ -n "${MOCK_C:-}" ]; then  # laptop mock only
  C=$MOCK_C; LT=$MOCK_LT; PRE_MD5=$MOCK_PRE_MD5; KEEP_MD5=$MOCK_KEEP_MD5; PAT=$MOCK_PAT; PU=$(id -un); sleep() { command sleep 0.05; }
fi
L=$LT/run/G9q; LOG=$L/train.log; D=$C/runs/g9q-cd2030; R=$C/runs/r1; KEEP=$LT/cd2030-keep
mkdir -p "$D"
say() { echo "$(date -u +%T) $*" | tee -a "$D/STATUS"; }
failx() { say "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$D/FAILED"; exit 3; }
relfail() { say "RELAUNCH CHECK FAILED: $*"; mkdir -p "$D/relaunch"; echo "$(date -u +%FT%T) $*" >> "$D/relaunch/FAILED"; }
cnt() { local c; c=$(grep -c "$1" "$LOG" 2>/dev/null); echo "${c:-0}"; }
mask() { tr '\0' '\n' < "$1" | awk 'p == "--resume" {print "<RESUME>"; p = $0; next} {print; p = $0}'; }
if [ ! -s "$D/PREREG.txt" ] || [ -w "$D/PREREG.txt" ] || [ "$(md5sum < "$D/PREREG.txt" | cut -c1-8)" != "$PRE_MD5" ]; then
  failx "no read-only runs/g9q-cd2030/PREREG.txt with md5 $PRE_MD5: not running"
fi
if [ -n "$(ls -A "$D/final" 2>/dev/null)" ]; then say "runs/g9q-cd2030/final is not empty: not running"; exit 2; fi
say "driver start in job $SLURM_JOB_ID (JOBPP $JOBPP, PREREG $PRE_MD5); waiting for a G9q checkpoint >= $S_MIN in $L/checkpoints"
CK=""
while [ -z "$CK" ]; do
  [ -f "$D/CD_STOP" ] && { say "CD_STOP: exiting"; exit 0; }
  for x in "$KEEP"/ckpt_*.pt "$L"/checkpoints/ckpt_*.pt; do  # the earliest >= S_MIN in either dir
    [ -e "$x" ] || continue
    n=$(basename "$x" .pt); n=${n#ckpt_}; case $n in *[!0-9]*|"") continue ;; esac
    if [ "$((10#$n))" -ge "$S_MIN" ] && { [ -z "$CK" ] || [ "$((10#$n))" -lt "$S" ]; }; then CK=$x; S=$((10#$n)); fi
  done
  [ -n "$CK" ] || sleep 60
done
say "checkpoint $CK (S = $S)"
if [ "$S" = "$KEEP_S" ] && [ "$(md5sum < "$CK" | cut -c1-8)" != "$KEEP_MD5" ]; then failx "$CK md5 is not the pinned $KEEP_MD5: not running"; fi
sleep 60  # the eval at this step logs its 'started' line right after the checkpoint
BP=$D/branch-point-$(printf %011d "$S").pt
cp -p "$CK" "$D/branch-point.pt.tmp" && mv -f "$D/branch-point.pt.tmp" "$BP" || failx "could not copy the branch point"
say "branch point copied: $BP (md5 $(md5sum < "$BP" | cut -c1-8))"
for i in $(seq 1 60); do  # no async eval running (<= 30 min)
  st=$(grep -n "started (async, pid" "$LOG" | tail -n 1)
  [ -z "$st" ] && break
  ln=${st%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$st"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$st")
  if tail -n +"$ln" "$LOG" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null; then break; fi
  sleep 30
done
[ -f "$D/CD_STOP" ] && { say "CD_STOP: exiting"; exit 0; }
end=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null | head -n 1); left=$(( $(date -d "$end" +%s 2>/dev/null || echo 0) - $(date +%s) ))
[ "$left" -ge 10800 ] || failx "only ${left}s left in job $SLURM_JOB_ID (end '$end'; need >= 3 h): not queuing"
for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR CHAIN_STOPPED PARKED; do [ ! -e "$R/$f" ] || failx "runs/r1/$f present: not queuing"; done
ps=$(pgrep -u "$PU" -f "$PAT" || true); set -- $ps
[ $# = 1 ] || failx "expected exactly 1 trainer, got '$ps'"
pid=$1; pp=$(ps -o ppid= -p "$pid" | tr -d ' ')
[ "$pp" = "$JOBPP" ] || failx "trainer $pid has parent $pp, expected $JOBPP"
[ -e "$LT/run/final_model.pt" ] || [ -L "$LT/run/final_model.pt" ] || failx "top-level final_model.pt missing (the spooled loop would end the job)"
cp "/proc/$pid/cmdline" "$D/trainer-argv.bin"; chmod a-w "$D/trainer-argv.bin"
tr '\0' ' ' < "$D/trainer-argv.bin" > "$D/trainer-argv.txt"; echo >> "$D/trainer-argv.txt"
grep -q -- '--lr-schedule' "$D/trainer-argv.txt" && failx "argv has an lr schedule flag (G9q is wsd): not queuing"
grep -q -- '--guard-ptop-max 0.6' "$D/trainer-argv.txt" || failx "argv lacks --guard-ptop-max 0.6: not queuing"
cat > "$R/INLINE_CMD.tmp" <<INL
set -u
# G9q-cd@2030 (runs/g9q-cd2030/PREREG.txt $PRE_MD5): G9q from its first checkpoint >= 2030M, 5M cosine to x0.1, own dirs, entropy floor 0.35
C=$C; L=\$LOCAL/run/G9q; S=$S; CK=$BP
export PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
mapfile -d '' -t A < "\$C/runs/g9q-cd2030/trainer-argv.bin"
W=\$C/bin/py-src-24c633f
B=\$LOCAL/run/G9q-cd2030; mkdir -p "\$B"
if [ ! -d "\$B/pool" ]; then cp -rp "\$L/pool" "\$B/pool-copy" && mv "\$B/pool-copy" "\$B/pool"; fi
mkdir -p "\$B/pool-after-S"
for f in "\$B"/pool/*.pt "\$B"/pool/frozen/best-*.pt; do [ -e "\$f" ] || continue; n=\$(basename "\$f" .pt); n=\${n#best-}; case \$n in *[!0-9]*|"") continue ;; esac
  if [ "\$((10#\$n))" -gt "\$S" ]; then mv -f "\$f" "\$B/pool-after-S/"; fi; done
echo "\$(date -u +%T) G9q-cd2030 start: resume \$CK, cooldown \$S -> \$((S + 5000000)) x0.1, pool \$(ls "\$B"/pool/*.pt | wc -l) + frozen \$(ls "\$B"/pool/frozen/*.pt 2>/dev/null | wc -l)" | tee -a "\$C/runs/g9q-cd2030/notes.log"
timeout -s INT -k 300 7200 "\$W" "\${A[@]:1}" --run-dir "\$B" --pool-dir "\$B/pool" --resume "\$CK" --cooldown-start "\$S" --total-steps "\$((S + 5000000))" --final-lr-frac 0.1 --guard-entropy-min 0.35 >> "\$B/stdout.log" 2>&1
rc=\$?
echo "\$(date -u +%T) G9q-cd2030 exit \$rc; \$(grep "saved final_model" "\$B/train.log" | tail -n 1)" | tee -a "\$C/runs/g9q-cd2030/notes.log"
INL
bash -n "$R/INLINE_CMD.tmp" || failx "INLINE_CMD.tmp syntax error: not queuing"
mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD"
touch "$R/restart"
say "INLINE_CMD queued (G9q-cd2030 from $BP); restart flag armed; trainer $pid (parent $pp)"
before=$(cnt "signal 2"); starts=$(cnt "v2 trainer:")
kill -INT "$pid" && say "SIGINT sent to trainer $pid"
for i in $(seq 1 18); do sleep 10; [ "$(cnt "signal 2")" -gt "$before" ] && break; done
say "landed: $(grep "signal 2" "$LOG" | tail -n 1)"
relaunched=""
for i in $(seq 1 1260); do sleep 10; if [ "$(cnt "v2 trainer:")" -gt "$starts" ]; then relaunched=1; break; fi; done  # branch (<= 2 h) + relaunch
touch "$R/restart"
if [ -n "$relaunched" ]; then say "G9q relaunched: $(grep "v2 trainer:" "$LOG" | tail -n 1 | cut -c1-120); restart flag re-armed"
else relfail "no new 'v2 trainer:' line within 3.5 h of the SIGINT (restart flag touched anyway)"; fi
B=$LT/run/G9q-cd2030
if [ -f "$B/final_model.pt" ]; then
  st=$(grep "saved final_model.pt at" "$B/train.log" | tail -n 1 | grep -oE "[0-9,]+ steps" | tr -dc 0-9)
  [ -n "$st" ] && [ "$st" -ge "$((S + 5000000))" ] || failx "the branch ended at '${st:-?}' steps < S + 5M = $((S + 5000000)) (cut short: not the pre-registered net)"
  mkdir -p "$D/final"; f=$(printf %011d "$st").pt
  cp -p "$B/final_model.pt" "$D/final/$f.tmp" && mv -f "$D/final/$f.tmp" "$D/final/$f" && chmod a-w "$D/final/$f" || failx "could not copy the final"
  cp -p "$B/train.log" "$D/branch-train.log"
  say "G9q-cd2030 final at $st steps -> $D/final/$f (md5 $(md5sum < "$D/final/$f" | cut -c1-8)); lr at the end: $(grep -oE "lr [0-9.e-]+" "$B/train.log" | tail -n 1); ent at the end: $(grep " upd " "$B/train.log" | tail -n 1 | grep -oE "ent [0-9.]+")"
  rm -rf "$B/pool" "$B/pool-after-S" && say "branch-local pool copies removed from /tmp (the final, logs and checkpoints stay)"
else
  failx "NO G9q-cd2030 final_model.pt in $B"
fi
# --- relaunch check (PREREG): argv equal up to the two --resume values; the first upd continues the step counter
[ -n "$relaunched" ] || exit 0
saved=$(grep "saved final_model.pt at" "$LOG" | tail -n 1 | grep -oE "at [0-9,]+ steps" | tr -dc 0-9)
npid=""; for i in $(seq 1 30); do ps=$(pgrep -u "$PU" -f "$PAT" || true); set -- $ps; if [ $# = 1 ] && [ "$1" != "$pid" ]; then npid=$1; break; fi; sleep 10; done
if [ -z "$npid" ]; then relfail "no single new trainer process (found '$ps')"; exit 0; fi
npp=$(ps -o ppid= -p "$npid" | tr -d ' '); [ "$npp" = "$JOBPP" ] || relfail "new trainer $npid has parent $npp, expected $JOBPP"
mkdir -p "$D/relaunch"; cp "/proc/$npid/cmdline" "$D/relaunch/trainer-argv.bin"; tr '\0' ' ' < "$D/relaunch/trainer-argv.bin" > "$D/relaunch/trainer-argv.txt"
if diff <(mask "$D/trainer-argv.bin") <(mask "$D/relaunch/trainer-argv.bin") > "$D/relaunch/argv-diff.txt"; then say "relaunch argv (pid $npid) = the recorded argv up to the --resume values"
else relfail "relaunch argv differs beyond --resume (runs/g9q-cd2030/relaunch/argv-diff.txt)"; fi
lr=$(tr '\0' '\n' < "$D/relaunch/trainer-argv.bin" | awk 'p == "--resume" {r = $0} {p = $0} END {print r}')
[ "$lr" = "$LT/run/G9q/final_model.pt" ] || relfail "the last --resume is '$lr', not \$LOCAL/run/G9q/final_model.pt"
ln=$(grep -n "v2 trainer:" "$LOG" | tail -n 1 | cut -d: -f1); u=""
for i in $(seq 1 180); do u=$(tail -n +"$ln" "$LOG" | grep -m1 " upd " || true); [ -n "$u" ] && break; sleep 10; done
us=$(grep -oE "steps [0-9,]+" <<< "$u" | head -n 1 | tr -dc 0-9)
if [ -n "$us" ] && [ -n "$saved" ] && [ "$((us - saved))" -gt 0 ] && [ "$((us - saved))" -lt 200000 ]; then say "first upd after the relaunch: steps $us (saved $saved, +$((us - saved))): counter continues; relaunch check PASS"
else relfail "first upd after the relaunch: '${u:0:80}' (steps '${us:-?}', saved '${saved:-?}')"; fi
say "driver done"
