#!/bin/bash
# netv2 2026-11-15: the entropy-schedule arm chain (runs/g9qent/PREREG.txt; orchestrator decision ~21:55), a step of the
# A40 job 1173274: trigger (runs/g9qbf/SKIP.txt, or runs/g9qbf/VERDICT.txt + runs/g9qbf-parked/PARKED.txt) -> INLINE #E1
# (fork point, staged pool, MODE g9qent; the chain lifts the park HOLD) -> the arm's first start checked -> its first
# checkpoint >= S0 + 50M -> INLINE #E2 (park the arm, arm-cd, G9q-cd) -> cooled h2h (cmp_step, seq-gain) ->
# runs/g9qent/VERDICT.txt. A guard trip => park, FAIL. Status runs/g9qent-chain/STATUS; FAILED there.
#   setsid nohup srun --jobid=1173274 --overlap --ntasks=1 -c 8 --gres=gpu:1 --mem=14G -J nv2-qent env JOBPP=<job script pid> \
#     bash -c 'cp $C/scripts/netv2_g9qent/chain-e.sh /tmp/nv2-qent-$$.sh && exec bash /tmp/nv2-qent-$$.sh' > $C/logs/qent-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; K=$C/runs/g9qent-chain; E=$C/runs/g9qent; R=$C/runs/g9; H=$C/runs/g9qent-h2h
J=${SLURM_JOB_ID:?}; JOBPP=${JOBPP:?set JOBPP = the job script pid}; LQ=/tmp/ws-netv2-$J/run
PRE_MD5=ecc8fe39; ENV18_MD5=553bfec9; E1_MD5=7fe29335; E2_MD5=c2b080ca; PK_MD5=ed1062b9; CMP_MD5=a80c7da2; SEED=5005001
PAT='^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'
mkdir -p "$K" "$E"
st() { echo "$(date -u +%FT%T) $*" | tee -a "$K/STATUS" >&2; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$K/FAILED"; exit 1; }
md5of() { md5sum < "$1" | cut -c1-8; }
[ ! -e "$K/FAILED" ] || { st "runs/g9qent-chain/FAILED exists: not running"; exit 1; }
if [ ! -s "$E/PREREG.txt" ] || [ -w "$E/PREREG.txt" ] || [ "$(md5of "$E/PREREG.txt")" != "$PRE_MD5" ]; then fail "no read-only runs/g9qent/PREREG.txt with md5 $PRE_MD5"; fi
for x in "$C/scripts/netv2_g9qent/inline-e1.sh $E1_MD5" "$C/scripts/netv2_g9qent/inline-e2.sh $E2_MD5" "$C/scripts/netv2_g9qbf/park.sh $PK_MD5" "$C/scripts/netv2_cmp_step.sh $CMP_MD5"; do
  set -- $x; [ "$(md5of "$1")" = "$2" ] || fail "$1 md5 is not $2"; done
[ -x "$C/bin/py-src-8892762" ] && [ -d "$C/src-8892762" ] && [ ! -w "$C/src-8892762" ] || fail "no read-only src-8892762 / wrapper"
trap 'st "chain step got SIGTERM (job end / cancel): stopping; the A40 state is as the files say"; exit 1' TERM
no_async_eval() {
  local s ln epid estep; s=$(grep -n "started (async, pid" "$1" | tail -n 1); [ -z "$s" ] && return 0
  ln=${s%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$s"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$s")
  tail -n +"$ln" "$1" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null; }
the_trainer() {
  local ps p out=""; ps=$(pgrep -u fr1226 -f "$PAT" || true)
  for p in $ps; do if [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = "$JOBPP" ]; then out="$out $p"; fi; done
  set -- $out; if [ $# = 1 ]; then echo "$1"; fi; }
wait_inline() {  # wait_inline T0 NAME: the INLINE installed at T0 finished (<= 6 h)
  local i; for i in $(seq 1 2160); do
    sleep 10; if [ ! -e "$R/INLINE_CMD" ] && [ ! -e "$R/INLINE_CMD.running" ] && [ -n "$(find "$R" -maxdepth 1 -name 'INLINE_CMD.done-*' -newermt "@$1" 2>/dev/null | head -n 1)" ]; then return 0; fi; done
  fail "$2: INLINE not finished within 6 h"; }
e2_text() {  # the INLINE #E2 text
  cat > "$K/inline-e2.txt" <<INL
set -u
# entropy arm (runs/g9qent/PREREG.txt $PRE_MD5) READING: park the arm -> runs/g9qent-parked, arm-cd, G9q-cd
cp $C/scripts/netv2_g9qent/inline-e2.sh /tmp/nv2-qent-e2-\$\$.sh && cp $C/scripts/netv2_g9qbf/park.sh /tmp/nv2-qent-pk-\$\$.sh || exit 1
[ "\$(md5sum < /tmp/nv2-qent-e2-\$\$.sh | cut -c1-8)" = $E2_MD5 ] && [ "\$(md5sum < /tmp/nv2-qent-pk-\$\$.sh | cut -c1-8)" = $PK_MD5 ] || { echo "\$(date -u +%FT%T) inline-e2 / park md5 mismatch" >> $K/FAILED; exit 1; }
QE_QUEUED=$C/runs/g9/INLINE_CMD.running QE_PARK=/tmp/nv2-qent-pk-\$\$.sh bash /tmp/nv2-qent-e2-\$\$.sh
rm -f /tmp/nv2-qent-e2-\$\$.sh /tmp/nv2-qent-pk-\$\$.sh
INL
  bash -n "$K/inline-e2.txt" || fail "INLINE #E2 text syntax"; }
verdict() { { echo "entropy-schedule arm VERDICT (runs/g9qent/PREREG.txt $PRE_MD5): $1"; echo "$2"
    if [ -f "$H/READING.txt" ]; then sed -n 2,4p "$H/READING.txt"; fi
    if [ -f "$E/TRIPPED" ]; then echo "guard trip: $(head -c 200 "$E/TRIPPED")"; fi
    echo "env md5 at the fork ${ENV_FORK:-?}, at the reading $(md5of "$C/netv2.env")$([ "${ENV_FORK:-}" = "$(md5of "$C/netv2.env")" ] || echo ' (CHANGED: the orchestrator judges)')"
    grep -E "final at" "$K/STATUS" | sed 's/^/  /'; } > "$E/VERDICT.txt.tmp" && mv -f "$E/VERDICT.txt.tmp" "$E/VERDICT.txt" && chmod a-w "$E/VERDICT.txt"
  st "VERDICT $1: $2"; }
st "chain start on $(hostname -s) (job $J step ${SLURM_STEP_ID:-?}, JOBPP $JOBPP, PREREG $PRE_MD5); waiting <= 60 h for runs/g9qbf/SKIP.txt or VERDICT + the fork's park"
# ---- trigger ----
T0=$(date +%s)
until [ -f "$C/runs/g9qbf/SKIP.txt" ] || { [ -f "$C/runs/g9qbf/VERDICT.txt" ] && [ -f "$C/runs/g9qbf-parked/PARKED.txt" ]; }; do
  [ $(( $(date +%s) - T0 )) -lt 216000 ] || fail "no trigger within 60 h"
  sleep 120
done
st "trigger: $([ -f "$C/runs/g9qbf/SKIP.txt" ] && echo "SKIP: $(head -n 1 "$C/runs/g9qbf/SKIP.txt" | cut -c1-80)" || echo "bf16 VERDICT: $(head -n 1 "$C/runs/g9qbf/VERDICT.txt" | cut -c1-100)")"
[ "$(md5of "$C/netv2.env")" = "$ENV18_MD5" ] || fail "live netv2.env md5 $(md5of "$C/netv2.env") is not v18 $ENV18_MD5: the arm's mode is not live"
ENV_FORK=$(md5of "$C/netv2.env")
for i in $(seq 1 60); do [ -f "$R/HOLD" ] && break; sleep 30; done
[ -f "$R/HOLD" ] || fail "the A40 is not in the park HOLD (no runs/g9/HOLD after 30 min)"
for f in INLINE_CMD INLINE_CMD.running; do [ ! -e "$R/$f" ] || fail "runs/g9/$f present: not queuing"; done
[ -z "$(the_trainer)" ] || fail "a trainer is running under the job script (not the park HOLD)"
# ---- INLINE #E1 ----
cat > "$K/inline-e1.txt" <<INL
set -u
# entropy arm (runs/g9qent/PREREG.txt $PRE_MD5) START: fork point, staged pool, MODE g9qent, lift the park HOLD
cp $C/scripts/netv2_g9qent/inline-e1.sh /tmp/nv2-qent-e1-\$\$.sh || exit 1
[ "\$(md5sum < /tmp/nv2-qent-e1-\$\$.sh | cut -c1-8)" = $E1_MD5 ] || { echo "\$(date -u +%FT%T) inline-e1 md5 mismatch" >> $K/FAILED; exit 1; }
bash /tmp/nv2-qent-e1-\$\$.sh; rm -f /tmp/nv2-qent-e1-\$\$.sh
INL
bash -n "$K/inline-e1.txt" || fail "INLINE #E1 text syntax"
cp "$K/inline-e1.txt" "$R/INLINE_CMD.tmp" && mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD" || fail "cannot install INLINE #E1"
t1=$(date +%s); touch "$R/restart"; rm -f "$R/HOLD"; st "INLINE #E1 queued; park HOLD lifted (the next round runs it)"
wait_inline "$t1" E1
[ ! -e "$K/FAILED" ] || fail "INLINE #E1 failed (see above)"
[ "$(cat "$R/MODE")" = g9qent ] || fail "MODE is '$(cat "$R/MODE")' after INLINE #E1"
read -r FP S0 FM < "$E/FORK_POINT"
for i in $(seq 1 180); do if grep -q "v2 trainer:" "$LQ/G9qent/train.log" 2>/dev/null; then break; fi; sleep 10; done
grep -q "v2 trainer:" "$LQ/G9qent/train.log" 2>/dev/null || fail "G9qent did not start within 30 min"
touch "$R/restart"
pid=$(the_trainer); [ -n "$pid" ] || fail "no single G9qent trainer under the job script"
tr '\0' ' ' < "/proc/$pid/cmdline" > "$E/first-argv.txt"; tr '\0' '\n' < "/proc/$pid/environ" | grep '^PYTHONPATH=' > "$E/first-env.txt"
grep -q -- "--ent-schedule linear --ent-final 0.001 --ent-start $S0 --ent-horizon 50000000" "$E/first-argv.txt" || fail "G9qent's argv lacks the schedule with S0 $S0"
grep -q -- "--no-bf16" "$E/first-argv.txt" && grep -q -- "--guard-entropy-min 0.35" "$E/first-argv.txt" && ! grep -q -- "--guard-entropy-min 0.5 " "$E/first-argv.txt" || fail "G9qent's argv: fp32 / entropy floor wrong"
grep -q "src-8892762" "$E/first-env.txt" || fail "G9qent's PYTHONPATH lacks src-8892762"
tr ' ' '\n' < "$E/first-argv.txt" | awk 'p == "--resume" {r = $0} {p = $0} END {print r}' | grep -qx "$FP" || fail "G9qent's last --resume is not the fork point"
# orchestrator conditions 11-15 ~22:20 (runs/g9qent/notes.log): the startup line shows the wsd lr and the ent ramp, fp32; no --actor-stream
sl=$(grep "v2 trainer:" "$LQ/G9qent/train.log" | tail -n 1); echo "$sl" > "$E/first-startup-line.txt"
grep -qF "lr schedule: wsd, lr 7e-05" <<< "$sl" && grep -qF "; ent schedule linear 0.01 -> 0.001 over [$S0, $((S0 + 50000000))]" <<< "$sl" || fail "G9qent's startup line lacks 'lr schedule: wsd, lr 7e-05 ... ent schedule linear 0.01 -> 0.001 over [$S0, $((S0 + 50000000))]': $(cut -c1-60 <<< "$sl")... (runs/g9qent/first-startup-line.txt)"
grep -qF "amp=False" <<< "$sl" || fail "G9qent's startup line does not show amp=False (fp32)"
grep -q -- "--actor-stream" "$E/first-argv.txt" && fail "G9qent's argv has --actor-stream"
st "G9qent first start (trainer $pid): startup line OK (lr schedule wsd, ent ramp 0.01 -> 0.001 over [$S0, $((S0 + 50000000))], amp=False), no --actor-stream; floor 0.35, src-8892762, resume $FP; restart re-armed"
# ---- the arm runs to its first checkpoint >= S0 + 50M (a guard trip ends it) ----
T=$((S0 + 50000000)); tripped=""
while :; do
  if [ -f "$E/TRIPPED" ]; then tripped=1; break; fi
  for x in "$LQ"/G9qent/checkpoints/ckpt_*.pt; do [ -e "$x" ] || continue; n=$(basename "$x" .pt); n=${n#ckpt_}
    case $n in *[!0-9]*|"") continue ;; esac; if [ "$((10#$n))" -ge "$T" ]; then break 2; fi; done
  sleep 120
done
if [ -n "$tripped" ]; then
  st "G9qent guard trip (runs/g9qent/TRIPPED: $(head -c 120 "$E/TRIPPED")) => FAIL; lifting the trip HOLD to park the arm"
  rm -f "$R/HOLD"; for i in $(seq 1 180); do [ -n "$(the_trainer)" ] && break; sleep 10; done
fi
st "the arm reached its reading point ($( [ -n "$tripped" ] && echo "trip" || echo ">= $T")): INLINE #E2"
for i in $(seq 1 120); do no_async_eval "$LQ/G9qent/train.log" && break; sleep 30; done
for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR; do [ ! -e "$R/$f" ] || fail "runs/g9/$f present before INLINE #E2"; done
pid=$(the_trainer); [ -n "$pid" ] || fail "no single G9qent trainer to stop"
cp "/proc/$pid/cmdline" "$K/arm-argv.bin" && chmod a-w "$K/arm-argv.bin" || fail "cannot record the arm's argv"
e2_text
if [ -n "$tripped" ]; then  # a trip: park only (no reading): the branch script parks first; cut it after the park
  sed -i 's#^QE_QUEUED=#QE_PARK_ONLY=1 QE_QUEUED=#' "$K/inline-e2.txt"; fi
cp "$K/inline-e2.txt" "$R/INLINE_CMD.tmp" && mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD" || fail "cannot install INLINE #E2"
t2=$(date +%s); touch "$R/restart"; kill -INT "$pid" && st "INLINE #E2 queued, restart armed, ONE SIGINT to trainer $pid"
wait_inline "$t2" E2
if [ -n "$tripped" ]; then verdict FAIL "a guard trip of the arm (no reading): G9q unchanged"; st "chain done"; exit 0; fi
[ ! -e "$K/FAILED" ] || fail "INLINE #E2 failed (see above)"
A=$(ls "$C"/runs/g9qent-cd/final/*.pt 2>/dev/null); B=$(ls "$C"/runs/g9qent-g9qcd/final/*.pt 2>/dev/null)
[ "$(wc -w <<< "$A")" = 1 ] && [ "$(wc -w <<< "$B")" = 1 ] || fail "want one final each in runs/g9qent-cd/final and runs/g9qent-g9qcd/final"
# ---- cooled h2h ----
mkdir -p "$H"; cp "$C/scripts/netv2_cmp_step.sh" "/tmp/nv2-qent-cmp-$$.sh" && [ "$(md5of "/tmp/nv2-qent-cmp-$$.sh")" = "$CMP_MD5" ] || fail "cmp_step copy / md5"
st "h2h: arm-cd $(basename "$A") vs G9q-cd $(basename "$B"), seed $SEED, 7 workers"
CMP_WORKERS=7 timeout 43200 bash "/tmp/nv2-qent-cmp-$$.sh" 0 "$SEED" "$H" "$C/runs/g9qent-cd/final" "$C/runs/g9qent-g9qcd/final" > "$H/run.log" 2>&1 || fail "h2h exited $? (log $H/run.log)"
rm -f "/tmp/nv2-qent-cmp-$$.sh"
grep -qx "G9s $A" "$H/NETS.txt" && grep -qx "G9p $B" "$H/NETS.txt" && grep -q "^--- all pairs played" "$H/RESULT.txt" || fail "h2h NETS / RESULT check"
read -r x y <<< "$(sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$H/RESULT.txt")"
if [ "${x:-}" = 10 ]; then v=PASS; why="\"arm-cd > G9q-cd\" SUCCESS: the schedule improves the cooled net at +50M => G9q's recipe may adopt it (orchestrator)"
elif [ "${y:-}" = 10 ]; then v=FAIL; why="\"G9q-cd > arm-cd\" SUCCESS: the schedule hurts; G9q unchanged"
elif [ "${x:-}" = 20 ] && [ "${y:-}" = 20 ]; then v="NO GAIN"; why="both FUTILITY: G9q unchanged"
else v=INCONCLUSIVE; why="exits ${x:-?} / ${y:-?}: G9q unchanged"; fi
{ echo "entropy-arm cooled reading (runs/g9qent/PREREG.txt $PRE_MD5), read mechanically by scripts/netv2_g9qent/chain-e.sh"
  echo "$v (\"arm-cd > G9q-cd\" exit ${x:-?}, \"G9q-cd > arm-cd\" exit ${y:-?}; 10 SUCCESS, 20 FUTILITY)"
  echo "A = arm-cd $(basename "$A") (\"G9s\"), B = G9q-cd $(basename "$B") (\"G9p\"); seed $SEED; RESULT.txt md5 $(md5of "$H/RESULT.txt")"
  grep -A1 '^DECISION OF RECORD' "$H/RESULT.txt"
  sed -n '/^--- nested, G9s > G9p:/,/^--- round-robin, G9s > G9p:/p' "$H/RESULT.txt" | grep -E "^--- nested|DECISION|estimate|repeated CI"
  grep -A1 '^--- all pairs played' "$H/RESULT.txt"; } > "$H/READING.txt.tmp" && mv -f "$H/READING.txt.tmp" "$H/READING.txt" && chmod a-w "$H/READING.txt"
verdict "$v" "$why"
st "chain done"
