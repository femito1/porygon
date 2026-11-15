#!/bin/bash
# netv2 2026-11-14: deployment net for the current main, G9q-cd@600 (runs/rawfp/PREREG.txt (a); user-approved raw-vs-Foul-Play
# chain), ONE INLINE_CMD of job 1135184 (A30, G9q main) = logs/nv2agent-cd300-20261112.sh (2080abe5, the G9q-cd300 branch)
# with S_MIN 600M and the dir runs/g9q-cd600. Run inside the job:
#   setsid nohup srun --jobid=1135184 --overlap --ntasks=1 -c 1 --mem=1G -J nv2-cd600 env JOBPP=<job script pid> \
#     bash $C/scripts/netv2_g9q/cd600-driver.sh > $C/logs/cd600-20261114.out 2>&1 < /dev/null &
# STOP before the SIGINT: touch runs/g9q-cd600/CD_STOP. A refusal or a missing final writes runs/g9q-cd600/FAILED.
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; S_MIN=${S_MIN:-600000000}; JOBPP=${JOBPP:?set JOBPP = the job script pid}
L=/tmp/ws-netv2-$SLURM_JOB_ID/run/G9q; LOG=$L/train.log; D=$C/runs/g9q-cd600; R=$C/runs/r1
PAT='^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'
say() { echo "$(date -u +%T) $*" | tee -a "$C/runs/g9q-cd600/STATUS"; }
mkdir -p "$C/runs/g9q-cd600"
failx() { say "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$C/runs/g9q-cd600/FAILED"; exit 3; }
cnt() { local c; c=$(grep -c "$1" "$LOG" 2>/dev/null); echo "${c:-0}"; }
[ -s "$C/runs/rawfp/PREREG.txt" ] && [ ! -w "$C/runs/rawfp/PREREG.txt" ] || { say "no read-only runs/rawfp/PREREG.txt: not running"; exit 2; }
say "waiting for a G9q checkpoint >= $S_MIN in $L/checkpoints (job $SLURM_JOB_ID)"
CK=""
while [ -z "$CK" ]; do
  [ -f "$D/CD_STOP" ] && { say "CD_STOP: exiting"; exit 0; }
  for x in "$L"/checkpoints/ckpt_*.pt; do
    [ -e "$x" ] || continue
    n=$(basename "$x" .pt); n=${n#ckpt_}; case $n in *[!0-9]*|"") continue ;; esac
    if [ "$((10#$n))" -ge "$S_MIN" ]; then CK=$x; S=$((10#$n)); break; fi
  done
  [ -n "$CK" ] || sleep 60
done
say "checkpoint $CK (S = $S)"
sleep 60  # the eval at this step logs its 'started' line right after the checkpoint
BP=$D/branch-point-$(printf %011d "$S").pt
cp -p "$CK" "$D/branch-point.pt.tmp" && mv -f "$D/branch-point.pt.tmp" "$BP"
say "branch point copied: $BP (md5 $(md5sum < "$BP" | cut -c1-8))"
for i in $(seq 1 60); do  # no async eval running (<= 30 min)
  st=$(grep -n "started (async, pid" "$LOG" | tail -n 1)
  [ -z "$st" ] && break
  ln=${st%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$st"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$st")
  if tail -n +"$ln" "$LOG" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null; then break; fi
  sleep 30
done
[ -f "$D/CD_STOP" ] && { say "CD_STOP: exiting"; exit 0; }
for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR CHAIN_STOPPED PARKED; do [ ! -e "$R/$f" ] || failx "runs/r1/$f present: not queuing"; done
ps=$(pgrep -u fr1226 -f "$PAT" || true); set -- $ps
[ $# = 1 ] || failx "expected exactly 1 trainer, got '$ps'"
pid=$1; pp=$(ps -o ppid= -p "$pid" | tr -d ' ')
[ "$pp" = "$JOBPP" ] || failx "trainer $pid has parent $pp, expected $JOBPP"
[ -e "/tmp/ws-netv2-$SLURM_JOB_ID/run/final_model.pt" ] || [ -L "/tmp/ws-netv2-$SLURM_JOB_ID/run/final_model.pt" ] || failx "top-level final_model.pt missing (the spooled loop would end the job)"
cp "/proc/$pid/cmdline" "$D/trainer-argv.bin"
tr '\0' ' ' < "$D/trainer-argv.bin" > "$D/trainer-argv.txt"; echo >> "$D/trainer-argv.txt"
grep -q -- '--lr-schedule' "$D/trainer-argv.txt" && failx "argv has an lr schedule flag (G9q is wsd): not queuing"
grep -q -- '--guard-ptop-max 0.6' "$D/trainer-argv.txt" || failx "argv lacks --guard-ptop-max 0.6: not queuing"
cat > "$R/INLINE_CMD.tmp" <<INL
set -u
# re-freeze #1 candidate (runs/g9q-cd600/PREREG.txt): G9q-cd@300, 5M cosine to x0.1, own dirs, entropy floor 0.35
C=$C; L=\$LOCAL/run/G9q; S=$S; CK=$BP
export PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
mapfile -d '' -t A < "\$C/runs/g9q-cd600/trainer-argv.bin"
W=\$C/bin/py-src-24c633f
B=\$LOCAL/run/G9q-cd600; mkdir -p "\$B"
if [ ! -d "\$B/pool" ]; then cp -rp "\$L/pool" "\$B/pool-copy" && mv "\$B/pool-copy" "\$B/pool"; fi
mkdir -p "\$B/pool-after-S"
for f in "\$B"/pool/*.pt "\$B"/pool/frozen/best-*.pt; do [ -e "\$f" ] || continue; n=\$(basename "\$f" .pt); n=\${n#best-}; case \$n in *[!0-9]*|"") continue ;; esac
  if [ "\$((10#\$n))" -gt "\$S" ]; then mv -f "\$f" "\$B/pool-after-S/"; fi; done
echo "\$(date -u +%T) G9q-cd600 start: resume \$CK, cooldown \$S -> \$((S + 5000000)) x0.1, pool \$(ls "\$B"/pool/*.pt | wc -l) + frozen \$(ls "\$B"/pool/frozen/*.pt 2>/dev/null | wc -l)" | tee -a "\$C/runs/g9q-cd600/notes.log"
timeout -s INT -k 300 7200 "\$W" "\${A[@]:1}" --run-dir "\$B" --pool-dir "\$B/pool" --resume "\$CK" --cooldown-start "\$S" --total-steps "\$((S + 5000000))" --final-lr-frac 0.1 --guard-entropy-min 0.35 >> "\$B/stdout.log" 2>&1
rc=\$?
echo "\$(date -u +%T) G9q-cd600 exit \$rc; \$(grep "saved final_model" "\$B/train.log" | tail -n 1)" | tee -a "\$C/runs/g9q-cd600/notes.log"
INL
bash -n "$R/INLINE_CMD.tmp" || failx "INLINE_CMD.tmp syntax error: not queuing"
mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD"
touch "$R/restart"
say "INLINE_CMD queued (G9q-cd600 from $BP); restart flag armed; trainer $pid (parent $pp)"
before=$(cnt "signal 2"); starts=$(cnt "v2 trainer:")
kill -INT "$pid" && say "SIGINT sent to trainer $pid"
for i in $(seq 1 18); do sleep 10; [ "$(cnt "signal 2")" -gt "$before" ] && break; done
say "landed: $(grep "signal 2" "$LOG" | tail -n 1)"
for i in $(seq 1 1260); do sleep 10; [ "$(cnt "v2 trainer:")" -gt "$starts" ] && break; done  # branch (<= 2 h) + relaunch
touch "$R/restart"
say "G9q relaunched: $(grep "v2 trainer:" "$LOG" | tail -n 1 | cut -c1-120); restart flag re-armed"
B=/tmp/ws-netv2-$SLURM_JOB_ID/run/G9q-cd600
if [ -f "$B/final_model.pt" ]; then
  st=$(grep "saved final_model.pt at" "$B/train.log" | tail -n 1 | grep -oE "[0-9,]+ steps" | tr -dc 0-9)
  mkdir -p "$D/final"; f=$(printf %011d "$st").pt
  cp -p "$B/final_model.pt" "$D/final/$f.tmp" && mv -f "$D/final/$f.tmp" "$D/final/$f"
  cp -p "$B/train.log" "$D/branch-train.log"
  say "G9q-cd600 final at $st steps -> $D/final/$f (md5 $(md5sum < "$D/final/$f" | cut -c1-8)); lr at the end: $(grep -oE "lr [0-9.e-]+" "$B/train.log" | tail -n 1)"
else
  failx "NO G9q-cd600 final_model.pt in $B"
fi
