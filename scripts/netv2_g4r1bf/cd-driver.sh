#!/bin/bash
# netv2 2026-11-24: R1b-F-cd (council test P3, gen4 re-baseline; runs/g4r1bf-cd/PREREG.txt) = scripts/netv2_g9q/cd2030-driver.sh
# (6fbba5a4) for R1b-F's job: the standard cooldown recipe (5M steps cosine lr -> x0.1, entropy floor 0.35, R1b-F's own argv
# otherwise) INLINE in R1b-F's T4 job, from the branch point copied and md5-pinned at launch (runs/g4r1bf-cd/
# branch-point-<11 digits>.pt, r--). Changes vs cd2030: R1b-F's dirs / names; the pinned branch point instead of a wait; the
# branch dir $LOCAL/cd/R1bF-cd is OUTSIDE $LOCAL/run (the job's 10-min mirror would copy its pool to NFS) and its pool is a
# hard-linked copy (no tmpfs cost; the trainer writes snapshots by tmp + rename, so the lineage's files are never touched);
# INLINE limit 8 h (R1b-F ~370 sps: 5M ~ 3.8 h); on success R1BF_CD_READY for wsp's P3 chain; the relaunch check accepts
# exactly one argv change besides the --resume values: --total-steps 1000000000 -> 2000000000 (netv2.env v22's R1b-F cap,
# sourced by this restart in place), nothing else. Run inside the job, from a node-local copy:
#   setsid nohup srun --jobid=1175389 --overlap --ntasks=1 -c 1 --mem=1G -J nv2-g4cd env JOBPP=<job script pid> \
#     bash -c 'cp $C/scripts/netv2_g4r1bf/cd-driver.sh /tmp/nv2-g4cd-$$.sh && exec bash /tmp/nv2-g4cd-$$.sh' \
#     > $C/logs/g4cd-<t>.out 2>&1 < /dev/null &
# STOP before the SIGINT: touch runs/g4r1bf-cd/CD_STOP. A refusal or a missing final writes runs/g4r1bf-cd/FAILED and
# council/p3-gen4/FAILED; a failed relaunch check writes runs/g4r1bf-cd/relaunch/FAILED (R1b-F is then for the agent).
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; JOBPP=${JOBPP:?set JOBPP = the job script pid}
PRE_MD5=4b5ebb2d; S=635038608; BP_MD5=450fd8a6; LT=/tmp/ws-netv2-$SLURM_JOB_ID; INL_LIMIT=28800
TOT_OLD=1000000000; TOT_NEW=2000000000
PAT='^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'; PU=fr1226
if [ -n "${MOCK_C:-}" ]; then  # laptop mock only
  C=$MOCK_C; LT=$MOCK_LT; PRE_MD5=$MOCK_PRE_MD5; S=$MOCK_S; BP_MD5=$MOCK_BP_MD5; PAT=$MOCK_PAT; PU=$(id -un); INL_LIMIT=60
  sleep() { command sleep 0.05; }
fi
L=$LT/run/R1bF; LOG=$L/train.log; D=$C/runs/g4r1bf-cd; R=$C/runs/g4r1bf; P3=$C/council/p3-gen4
BP=$D/branch-point-$(printf %011d "$S").pt
mkdir -p "$D" "$P3"
say() { echo "$(date -u +%T) $*" | tee -a "$D/STATUS"; }
failx() {
  say "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$D/FAILED"
  [ -e "$P3/R1BF_CD_READY" ] || echo "$(date -u +%FT%T) R1b-F cooldown (netv2 runs/g4r1bf-cd): $*" > "$P3/FAILED"
  exit 3
}
relfail() { say "RELAUNCH CHECK FAILED: $*"; mkdir -p "$D/relaunch"; echo "$(date -u +%FT%T) $*" >> "$D/relaunch/FAILED"; }
cnt() { local c; c=$(grep -c "$1" "$LOG" 2>/dev/null); echo "${c:-0}"; }
# argv with the --resume and --total-steps values masked
mask() { tr '\0' '\n' < "$1" | awk 'p == "--resume" {print "<RESUME>"; p = $0; next} p == "--total-steps" {print "<TOTAL>"; p = $0; next} {print; p = $0}'; }
totals() { tr '\0' '\n' < "$1" | awk 'p == "--total-steps" {printf "%s ", $0} {p = $0}' | sed 's/ $//'; }
if [ ! -s "$D/PREREG.txt" ] || [ -w "$D/PREREG.txt" ] || [ "$(md5sum < "$D/PREREG.txt" | cut -c1-8)" != "$PRE_MD5" ]; then
  failx "no read-only runs/g4r1bf-cd/PREREG.txt with md5 $PRE_MD5: not running"
fi
if [ -n "$(ls -A "$D/final" 2>/dev/null)" ]; then say "runs/g4r1bf-cd/final is not empty: not running"; exit 2; fi
[ -f "$D/CD_STOP" ] && { say "CD_STOP: exiting"; exit 0; }
say "driver start in job $SLURM_JOB_ID (JOBPP $JOBPP, PREREG $PRE_MD5); branch point $BP (S = $S, md5 $BP_MD5)"
if [ ! -f "$BP" ] || [ -w "$BP" ] || [ "$(md5sum < "$BP" | cut -c1-8)" != "$BP_MD5" ]; then failx "$BP missing, writable or md5 not the pinned $BP_MD5: not running"; fi
for i in $(seq 1 60); do  # no async eval running (<= 30 min)
  st=$(grep -n "started (async, pid" "$LOG" | tail -n 1)
  [ -z "$st" ] && break
  ln=${st%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$st"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$st")
  if tail -n +"$ln" "$LOG" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null; then break; fi
  sleep 30
done
[ -f "$D/CD_STOP" ] && { say "CD_STOP: exiting"; exit 0; }
end=$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null | head -n 1); left=$(( $(date -d "$end" +%s 2>/dev/null || echo 0) - $(date +%s) ))
[ "$left" -ge 36000 ] || failx "only ${left}s left in job $SLURM_JOB_ID (end '$end'; need >= 10 h): not queuing"
for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR CHAIN_STOPPED PARKED ROLLBACK; do [ ! -e "$R/$f" ] || failx "runs/g4r1bf/$f present: not queuing"; done
ps=$(pgrep -u "$PU" -f "$PAT" || true); set -- $ps
[ $# = 1 ] || failx "expected exactly 1 trainer, got '$ps'"
pid=$1; pp=$(ps -o ppid= -p "$pid" | tr -d ' ')
[ "$pp" = "$JOBPP" ] || failx "trainer $pid has parent $pp, expected $JOBPP"
[ -e "$LT/run/final_model.pt" ] || [ -L "$LT/run/final_model.pt" ] || failx "top-level final_model.pt missing (the spooled loop would end the job)"
cp "/proc/$pid/cmdline" "$D/trainer-argv.bin"; chmod a-w "$D/trainer-argv.bin"
tr '\0' ' ' < "$D/trainer-argv.bin" > "$D/trainer-argv.txt"; echo >> "$D/trainer-argv.txt"
grep -q -- '--lr-schedule\|--cooldown-start' "$D/trainer-argv.txt" && failx "argv has an lr schedule / cooldown flag (R1b-F is constant lr): not queuing"
grep -q -- '--format gen4randombattle' "$D/trainer-argv.txt" || failx "argv is not gen4randombattle: not queuing"
grep -q -- '--guard-anchor-drop 0.0999' "$D/trainer-argv.txt" || failx "argv lacks --guard-anchor-drop 0.0999 (not R1b-F's config): not queuing"
grep -q -- "--run-dir $L " "$D/trainer-argv.txt" || failx "argv lacks --run-dir $L: not queuing"
[ "$(totals "$D/trainer-argv.bin")" = "$TOT_OLD" ] || failx "argv --total-steps is '$(totals "$D/trainer-argv.bin")', not $TOT_OLD: not queuing"
cat > "$R/INLINE_CMD.tmp" <<INL
set -u
# R1b-F-cd (runs/g4r1bf-cd/PREREG.txt $PRE_MD5): R1b-F from its checkpoint at S = $S, 5M cosine to x0.1, entropy floor 0.35,
# own dirs outside the mirrored run dir, pool = hard-linked copy of R1b-F's (snapshots newer than S moved aside)
C=$C; L=\$LOCAL/run/R1bF; S=$S; CK=$BP
export PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
mapfile -d '' -t A < "\$C/runs/g4r1bf-cd/trainer-argv.bin"
W=\$C/bin/py-src-24c633f
B=\$LOCAL/cd/R1bF-cd; mkdir -p "\$B"
if [ ! -d "\$B/pool" ]; then cp -rpl "\$L/pool" "\$B/pool-copy" && mv "\$B/pool-copy" "\$B/pool"; fi
mkdir -p "\$B/pool-after-S"
for f in "\$B"/pool/*.pt "\$B"/pool/frozen/best-*.pt; do [ -e "\$f" ] || continue; n=\$(basename "\$f" .pt); n=\${n#best-}; n=\${n#ppo_}; n=\${n%_steps}; case \$n in *[!0-9]*|"") continue ;; esac
  if [ "\$((10#\$n))" -gt "\$S" ]; then mv -f "\$f" "\$B/pool-after-S/"; fi; done
echo "\$(date -u +%T) R1b-F-cd start: resume \$CK, cooldown \$S -> \$((S + 5000000)) x0.1, pool \$(ls "\$B"/pool/*.pt | wc -l) + frozen \$(ls "\$B"/pool/frozen/*.pt 2>/dev/null | wc -l), moved aside \$(ls "\$B"/pool-after-S | wc -l)" | tee -a "\$C/runs/g4r1bf-cd/notes.log"
timeout -s INT -k 300 $INL_LIMIT "\$W" "\${A[@]:1}" --run-dir "\$B" --pool-dir "\$B/pool" --resume "\$CK" --cooldown-start "\$S" --total-steps "\$((S + 5000000))" --final-lr-frac 0.1 --guard-entropy-min 0.35 >> "\$B/stdout.log" 2>&1
rc=\$?
echo "\$(date -u +%T) R1b-F-cd exit \$rc; \$(grep "saved final_model" "\$B/train.log" | tail -n 1)" | tee -a "\$C/runs/g4r1bf-cd/notes.log"
INL
bash -n "$R/INLINE_CMD.tmp" || failx "INLINE_CMD.tmp syntax error: not queuing"
mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD"
touch "$R/restart"
say "INLINE_CMD queued (R1b-F-cd from $BP); restart flag armed; trainer $pid (parent $pp)"
before=$(cnt "signal 2"); starts=$(cnt "v2 trainer:")
kill -INT "$pid" && say "SIGINT sent to trainer $pid"
for i in $(seq 1 18); do sleep 10; [ "$(cnt "signal 2")" -gt "$before" ] && break; done
say "landed: $(grep "signal 2" "$LOG" | tail -n 1)"
relaunched=""
for i in $(seq 1 3240); do sleep 10; if [ "$(cnt "v2 trainer:")" -gt "$starts" ]; then relaunched=1; break; fi; done  # cooldown (<= 8 h) + relaunch
touch "$R/restart"
if [ -n "$relaunched" ]; then say "R1b-F relaunched: $(grep "v2 trainer:" "$LOG" | tail -n 1 | cut -c1-120); restart flag re-armed"
else relfail "no new 'v2 trainer:' line within 9 h of the SIGINT (restart flag touched anyway)"; fi
B=$LT/cd/R1bF-cd
if [ -f "$B/final_model.pt" ]; then
  st=$(grep "saved final_model.pt at" "$B/train.log" | tail -n 1 | grep -oE "[0-9,]+ steps" | tr -dc 0-9)
  [ -n "$st" ] && [ "$st" -ge "$((S + 5000000))" ] || failx "the branch ended at '${st:-?}' steps < S + 5M = $((S + 5000000)) (cut short: not the pre-registered net)"
  mkdir -p "$D/final"; f=$(printf %011d "$st").pt
  cp -p "$B/final_model.pt" "$D/final/$f.tmp" && mv -f "$D/final/$f.tmp" "$D/final/$f" && chmod a-w "$D/final/$f" || failx "could not copy the final"
  cp -p "$B/train.log" "$D/branch-train.log"; cp -p "$B/stdout.log" "$D/branch-stdout.log" 2>/dev/null || true
  m=$(md5sum < "$D/final/$f" | cut -c1-8)
  say "R1b-F-cd final at $st steps -> $D/final/$f (md5 $m); lr at the end: $(grep -oE "lr [0-9.e-]+" "$B/train.log" | tail -n 1); ent at the end: $(grep " upd " "$B/train.log" | tail -n 1 | grep -oE "ent [0-9.]+")"
  echo "$D/final/$f $m $st" > "$P3/R1BF_CD_READY.tmp" && mv -f "$P3/R1BF_CD_READY.tmp" "$P3/R1BF_CD_READY" && chmod a-w "$P3/R1BF_CD_READY" || failx "could not write council/p3-gen4/R1BF_CD_READY"
  say "council/p3-gen4/R1BF_CD_READY: $(cat "$P3/R1BF_CD_READY")"
  rm -rf "$B/pool" "$B/pool-after-S" "$B/checkpoints" && say "branch-local pool / checkpoints removed from /tmp (final, logs copied)"
else
  failx "NO R1b-F-cd final_model.pt in $B"
fi
# --- relaunch check (PREREG): argv equal up to the --resume values and exactly --total-steps $TOT_OLD -> $TOT_NEW; the
# first upd continues the step counter
[ -n "$relaunched" ] || exit 0
saved=$(grep "saved final_model.pt at" "$LOG" | tail -n 1 | grep -oE "at [0-9,]+ steps" | tr -dc 0-9)
npid=""; for i in $(seq 1 30); do ps=$(pgrep -u "$PU" -f "$PAT" || true); set -- $ps; if [ $# = 1 ] && [ "$1" != "$pid" ]; then npid=$1; break; fi; sleep 10; done
if [ -z "$npid" ]; then relfail "no single new trainer process (found '$ps')"; exit 0; fi
npp=$(ps -o ppid= -p "$npid" | tr -d ' '); [ "$npp" = "$JOBPP" ] || relfail "new trainer $npid has parent $npp, expected $JOBPP"
mkdir -p "$D/relaunch"; cp "/proc/$npid/cmdline" "$D/relaunch/trainer-argv.bin"; tr '\0' ' ' < "$D/relaunch/trainer-argv.bin" > "$D/relaunch/trainer-argv.txt"
if diff <(mask "$D/trainer-argv.bin") <(mask "$D/relaunch/trainer-argv.bin") > "$D/relaunch/argv-diff.txt"; then say "relaunch argv (pid $npid) = the recorded argv up to the --resume / --total-steps values"
else relfail "relaunch argv differs beyond --resume / --total-steps (runs/g4r1bf-cd/relaunch/argv-diff.txt)"; fi
nt=$(totals "$D/relaunch/trainer-argv.bin")
if [ "$nt" = "$TOT_NEW" ]; then say "relaunch --total-steps $nt (was $(totals "$D/trainer-argv.bin")): the v22 cap, the only other change"
else relfail "relaunch --total-steps is '$nt', not $TOT_NEW"; fi
lr=$(tr '\0' '\n' < "$D/relaunch/trainer-argv.bin" | awk 'p == "--resume" {r = $0} {p = $0} END {print r}')
[ "$lr" = "$LT/run/R1bF/final_model.pt" ] || relfail "the last --resume is '$lr', not \$LOCAL/run/R1bF/final_model.pt"
ln=$(grep -n "v2 trainer:" "$LOG" | tail -n 1 | cut -d: -f1); u=""
for i in $(seq 1 180); do u=$(tail -n +"$ln" "$LOG" | grep -m1 " upd " || true); [ -n "$u" ] && break; sleep 10; done
us=$(grep -oE "steps [0-9,]+" <<< "$u" | head -n 1 | tr -dc 0-9)
if [ -n "$us" ] && [ -n "$saved" ] && [ "$((us - saved))" -gt 0 ] && [ "$((us - saved))" -lt 200000 ]; then say "first upd after the relaunch: steps $us (saved $saved, +$((us - saved))): counter continues; relaunch check PASS"
else relfail "first upd after the relaunch: '${u:0:80}' (steps '${us:-?}', saved '${saved:-?}')"; fi
say "driver done"
