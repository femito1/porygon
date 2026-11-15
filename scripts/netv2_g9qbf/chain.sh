#!/bin/bash
# netv2 2026-11-15: the bf16 A/B chain (runs/g9qbf/PREREG.txt; orchestrator decision ~20:05, item 3), a step of the A40
# job 1173274: OVERALL.txt -> deploy netv2.env v17 (no job in nv2_pause) -> INLINE #1 (park G9q2, fork point, staged pool,
# microbench, GO => MODE g9qbf) -> verify the fork's first start -> R+25 / R+50 (cmp_step, seq-gain; STOP if "fp32 >
# bf16" succeeds) -> park the fork (INLINE #2) -> runs/g9qbf/VERDICT.txt. Status runs/g9qbf-chain/STATUS; FAILED there.
#   setsid nohup srun --jobid=1173274 --overlap --ntasks=1 -c 4 --gres=gpu:1 --mem=12G -J nv2-qbf env JOBPP=<job script pid> \
#     bash -c 'cp $C/scripts/netv2_g9qbf/chain.sh /tmp/nv2-qbf-$$.sh && exec bash /tmp/nv2-qbf-$$.sh' > $C/logs/qbf-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; K=$C/runs/g9qbf-chain; F=$C/runs/g9qbf; B=$C/runs/g9qbf-bench; R=$C/runs/g9
J=${SLURM_JOB_ID:?}; JOBPP=${JOBPP:?set JOBPP = the job script pid}; LQ=/tmp/ws-netv2-$J/run
PRE_MD5=a77115fe; ENV_PRED=2a7aaea1; ENV17=$C/netv2.env.v17-g9qbf; ENV17_MD5=0f951c39
I1_MD5=f261fc81; PK_MD5=ed1062b9; CMP_MD5=a80c7da2
PAT='^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'
mkdir -p "$K" "$F" "$B"
st() { echo "$(date -u +%FT%T) $*" | tee -a "$K/STATUS" >&2; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$K/FAILED"; exit 1; }
md5of() { md5sum < "$1" | cut -c1-8; }
[ ! -e "$K/FAILED" ] || { st "runs/g9qbf-chain/FAILED exists: not running"; exit 1; }
if [ ! -s "$F/PREREG.txt" ] || [ -w "$F/PREREG.txt" ] || [ "$(md5of "$F/PREREG.txt")" != "$PRE_MD5" ]; then fail "no read-only runs/g9qbf/PREREG.txt with md5 $PRE_MD5"; fi
for x in "$C/scripts/netv2_g9qbf/inline1.sh $I1_MD5" "$C/scripts/netv2_g9qbf/park.sh $PK_MD5" "$C/scripts/netv2_cmp_step.sh $CMP_MD5" "$ENV17 $ENV17_MD5"; do
  set -- $x; [ "$(md5of "$1")" = "$2" ] || fail "$1 md5 is not $2"; done
[ ! -w "$ENV17" ] || fail "$ENV17 is writable (want r--)"
trap 'st "chain step got SIGTERM (job end / cancel): stopping; the A40 state is as the files say"; exit 1' TERM
no_async_eval() {  # no_async_eval TRAINLOG: the last 'started (async' eval has its result line or its pid is gone
  local s ln epid estep; s=$(grep -n "started (async, pid" "$1" | tail -n 1); [ -z "$s" ] && return 0
  ln=${s%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$s"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$s")
  tail -n +"$ln" "$1" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null; }
the_trainer() {  # prints the one trainer pid whose parent is the job script ("" otherwise)
  local ps p out=""; ps=$(pgrep -u fr1226 -f "$PAT" || true)
  for p in $ps; do if [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = "$JOBPP" ]; then out="$out $p"; fi; done
  set -- $out; if [ $# = 1 ]; then echo "$1"; fi; }
inline() {  # inline TEXTFILE TRAINLOG NAME: install runs/g9/INLINE_CMD, arm restart, ONE SIGINT outside an async eval; waits for its .done
  local txt=$1 log=$2 nm=$3 pid t0 i
  for i in $(seq 1 120); do no_async_eval "$log" && break; sleep 30; done
  for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR CHAIN_STOPPED PARKED; do [ ! -e "$R/$f" ] || fail "$nm: runs/g9/$f present: not queuing"; done
  pid=$(the_trainer); [ -n "$pid" ] || fail "$nm: not exactly one trainer under the job script $JOBPP"
  if [ "$nm" = inline1 ]; then cp "/proc/$pid/cmdline" "$B/g9q2-argv.bin" && chmod a-w "$B/g9q2-argv.bin" || fail "cannot record G9q2's argv"; fi
  bash -n "$txt" || fail "$nm: INLINE text syntax"
  cp "$txt" "$R/INLINE_CMD.tmp" && mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD" || fail "$nm: cannot install INLINE_CMD"
  touch "$R/restart"; t0=$(date +%s)
  kill -INT "$pid" && st "$nm: INLINE_CMD queued, restart armed, ONE SIGINT to trainer $pid"
  for i in $(seq 1 1080); do  # <= 3 h
    sleep 10; if [ ! -e "$R/INLINE_CMD" ] && [ ! -e "$R/INLINE_CMD.running" ] && [ -n "$(find "$R" -maxdepth 1 -name 'INLINE_CMD.done-*' -newermt "@$t0" 2>/dev/null | head -n 1)" ]; then break; fi; done
  [ ! -e "$R/INLINE_CMD" ] && [ ! -e "$R/INLINE_CMD.running" ] || fail "$nm: INLINE not finished within 3 h"
  st "$nm: INLINE done ($(ls -t "$R"/INLINE_CMD.done-* | head -n 1))"; }
reading() {  # reading NAME DIR: STOP|NI|UNDECIDED from cmp_step's RESULT (A = G9qbf "G9s", B = G9q "G9p"); writes DIR/READING.txt
  local x y v
  read -r x y <<< "$(sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$2/RESULT.txt")"
  if [ "${y:-}" = 10 ]; then v=STOP; elif [ "${y:-}" = 20 ]; then v=NI; else v=UNDECIDED; fi
  { echo "bf16 A/B reading $1 (runs/g9qbf/PREREG.txt $PRE_MD5), read mechanically by scripts/netv2_g9qbf/chain.sh"
    echo "$1 $v (\"fp32 > bf16\" exit ${y:-?}: 10 SUCCESS = bf16 worse => STOP, 20 FUTILITY = bf16 not >= 2 points worse; \"bf16 > fp32\" exit ${x:-?} (info))"
    echo "A = G9qbf (\"G9s\"), B = G9q (\"G9p\"); RESULT.txt md5 $(md5of "$2/RESULT.txt")"; grep -A1 '^DECISION OF RECORD' "$2/RESULT.txt"
    sed -n '/^--- nested, G9s > G9p:/,/^--- round-robin, G9s > G9p:/p' "$2/RESULT.txt" | grep -E "^--- nested|DECISION|estimate|repeated CI"
    grep -A1 '^--- all pairs played' "$2/RESULT.txt"; } > "$2/READING.txt.tmp" && mv -f "$2/READING.txt.tmp" "$2/READING.txt" && chmod a-w "$2/READING.txt"
  echo "$v"; }
park_fork() {  # INLINE #2: park G9qbf whole (MODE stays g9qbf; v17 then HOLDs)
  local i
  if [ -f "$F/TRIPPED" ] && [ -f "$R/HOLD" ]; then  # a guard trip's HOLD: the pre-registered outcome is FAIL + park, no decision pending
    rm -f "$R/HOLD"; st "removed the G9qbf trip HOLD (PREREG: a trip = FAIL + park); waiting for the relaunch to queue the park"
    for i in $(seq 1 180); do [ -n "$(the_trainer)" ] && break; sleep 10; done
    touch "$R/restart"
  fi
  cat > "$K/inline2.txt" <<INL
set -u
# bf16 A/B (runs/g9qbf/PREREG.txt $PRE_MD5) step 6: park G9qbf -> runs/g9qbf-parked (v17 then HOLDs the A40)
cp $C/scripts/netv2_g9qbf/park.sh /tmp/nv2-qbf-park-\$\$.sh || exit 1
[ "\$(md5sum < /tmp/nv2-qbf-park-\$\$.sh | cut -c1-8)" = $PK_MD5 ] || { echo "\$(date -u +%FT%T) park.sh md5 mismatch" >> $K/FAILED; exit 1; }
PARK_LIN=G9qbf PARK_STATE=runs/g9qbf PARK_DEST=runs/g9qbf-parked PARK_QUEUED=$C/runs/g9/INLINE_CMD.running PARK_WHY="bf16 A/B ended ($1; runs/g9qbf/PREREG.txt step 6); A40 placement = orchestrator / user" bash /tmp/nv2-qbf-park-\$\$.sh
rm -f /tmp/nv2-qbf-park-\$\$.sh
INL
  inline "$K/inline2.txt" "$LQ/G9qbf/train.log" inline2
  [ -f "$C/runs/g9qbf-parked/PARKED.txt" ] || fail "G9qbf park failed (see runs/g9qbf/park.log): the fork trains on"
  st "G9qbf parked: $(head -n 1 "$C/runs/g9qbf-parked/PARKED.txt")"; }
verdict() {  # verdict WORD WHY
  { echo "bf16 A/B VERDICT (runs/g9qbf/PREREG.txt $PRE_MD5): $1"; echo "$2"
    for r in r25 r50; do if [ -f "$C/runs/g9qbf-$r/READING.txt" ]; then echo "--- $r:"; sed -n 2,3p "$C/runs/g9qbf-$r/READING.txt"; fi; done
    if [ -f "$F/TRIPPED" ]; then echo "guard trip: $(head -c 200 "$F/TRIPPED")"; fi
    echo "bench: $(grep -E '^RATIO' "$B/RESULTS.txt" 2>/dev/null)"; } > "$F/VERDICT.txt.tmp" && mv -f "$F/VERDICT.txt.tmp" "$F/VERDICT.txt" && chmod a-w "$F/VERDICT.txt"
  st "VERDICT $1: $2"; }
st "chain start on $(hostname -s) (job $J step ${SLURM_STEP_ID:-?}, JOBPP $JOBPP, PREREG $PRE_MD5); waiting <= 40 h for runs/g9q2/cmp/OVERALL.txt"
# ---- 0. G9q2's OVERALL ----
T0=$(date +%s)
until [ -f "$C/runs/g9q2/cmp/OVERALL.txt" ] && [ ! -w "$C/runs/g9q2/cmp/OVERALL.txt" ]; do
  [ ! -e "$C/runs/g9q2/chain/FAILED" ] || fail "runs/g9q2/chain/FAILED: no OVERALL (no deploy, no park)"
  [ $(( $(date +%s) - T0 )) -lt 144000 ] || fail "no OVERALL.txt within 40 h"
  sleep 120
done
st "OVERALL present: $(sed -n 2p "$C/runs/g9q2/cmp/OVERALL.txt" | cut -c1-150)"
# ---- 1. deploy netv2.env v17 (no job in nv2_pause) ----
if [ "$(md5of "$C/netv2.env")" = "$ENV17_MD5" ]; then st "netv2.env is v17 already ($ENV17_MD5)"
else
  [ "$(md5of "$C/netv2.env")" = "$ENV_PRED" ] || fail "live netv2.env md5 $(md5of "$C/netv2.env") is not the pinned predecessor $ENV_PRED: no deploy"
  t1=$(date +%s)
  while :; do
    busy=""; for r in r1 g9 g4r1bf; do for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR; do if [ -e "$C/runs/$r/$f" ]; then busy="$busy runs/$r/$f"; fi; done; done
    [ -n "$busy" ] || break
    [ $(( $(date +%s) - t1 )) -lt 21600 ] || fail "a job sat in nv2_pause for 6 h ($busy): no deploy"
    if [ ! -e "$K/.busy-noted" ]; then st "deploy waits: $busy"; touch "$K/.busy-noted"; fi; sleep 120
  done
  bk=$C/netv2.env.bak-$(date -u +%Y%m%d-%H%M); cp -p "$C/netv2.env" "$bk" || fail "backup failed"
  cp "$ENV17" "$C/netv2.env.new" && [ "$(md5of "$C/netv2.env.new")" = "$ENV17_MD5" ] && bash -n "$C/netv2.env.new" || fail "staging netv2.env.new failed"
  mv -f "$C/netv2.env.new" "$C/netv2.env" || fail "mv netv2.env.new failed"
  [ "$(md5of "$C/netv2.env")" = "$ENV17_MD5" ] || fail "after the deploy netv2.env md5 is $(md5of "$C/netv2.env")"
  st "DEPLOYED netv2.env v17 ($ENV17_MD5); backup $bk ($(md5of "$bk"))"
fi
# ---- 2-4. INLINE #1: park G9q2 + fork point + staged pool + microbench ----
cat > "$K/inline1.txt" <<INL
set -u
# bf16 A/B (runs/g9qbf/PREREG.txt $PRE_MD5) steps 2-4: park G9q2 -> runs/g9q2-parked, fork point, staged pool, microbench; GO => MODE g9qbf
cp $C/scripts/netv2_g9qbf/inline1.sh /tmp/nv2-qbf-i1-\$\$.sh && cp $C/scripts/netv2_g9qbf/park.sh /tmp/nv2-qbf-pk-\$\$.sh || exit 1
[ "\$(md5sum < /tmp/nv2-qbf-i1-\$\$.sh | cut -c1-8)" = $I1_MD5 ] && [ "\$(md5sum < /tmp/nv2-qbf-pk-\$\$.sh | cut -c1-8)" = $PK_MD5 ] || { echo "\$(date -u +%FT%T) inline1 / park md5 mismatch" >> $B/FAILED; exit 1; }
QBF_QUEUED=$C/runs/g9/INLINE_CMD.running QBF_PARK=/tmp/nv2-qbf-pk-\$\$.sh bash /tmp/nv2-qbf-i1-\$\$.sh
rm -f /tmp/nv2-qbf-i1-\$\$.sh /tmp/nv2-qbf-pk-\$\$.sh
INL
inline "$K/inline1.txt" "$LQ/G9q2/train.log" inline1
if [ -e "$B/FAILED" ]; then fail "INLINE #1 failed: $(tail -n 1 "$B/FAILED") (G9q2 parked: $([ -f "$C/runs/g9q2-parked/PARKED.txt" ] && echo yes || echo no))"; fi
dec=$(sed -n 's/^DECISION //p' "$B/RESULTS.txt" 2>/dev/null | tail -n 1)
st "bench: $(grep -E '^(bf16|fp32|RATIO)' "$B/RESULTS.txt" | tr '\n' ' ' | cut -c1-400)"
if [ "$dec" = SKIP ]; then st "SKIP (runs/g9qbf/SKIP.txt): G9q stays fp32; the A40 HOLDs (G9q2 parked); chain done"; exit 0; fi
[ "$dec" = GO ] && [ "$(cat "$R/MODE")" = g9qbf ] || fail "bench decision '$dec' / MODE '$(cat "$R/MODE")' unexpected"
read -r FP S0 FM < "$F/FORK_POINT"
# ---- 4b. the fork's first start ----
for i in $(seq 1 180); do if grep -q "v2 trainer:" "$LQ/G9qbf/train.log" 2>/dev/null; then break; fi; sleep 10; done
grep -q "v2 trainer:" "$LQ/G9qbf/train.log" 2>/dev/null || fail "G9qbf did not start within 30 min (MODE g9qbf; see $LQ/train.log)"
touch "$R/restart"
pid=$(the_trainer); [ -n "$pid" ] || fail "no single G9qbf trainer under the job script"
tr '\0' ' ' < "/proc/$pid/cmdline" > "$F/first-argv.txt"
grep -q -- "--no-bf16" "$F/first-argv.txt" && fail "G9qbf's first argv still has --no-bf16"
grep -q -- "--seed" "$F/first-argv.txt" && fail "G9qbf's first argv has --seed"
tr ' ' '\n' < "$F/first-argv.txt" | awk 'p == "--resume" {r = $0} {p = $0} END {print r}' | grep -qx "$FP" || fail "G9qbf's last --resume is not the fork point $FP"
grep -q -- "--run-dir $LQ/G9qbf --pool-dir $LQ/G9qbf/pool" "$F/first-argv.txt" || fail "G9qbf's argv lacks its lineage dirs"
# the recipe check: the fork's argv = the bench's bf16 arm argv up to the bench limits and the run / pool / resume values
norm() { tr ' ' '\n' | sed '/^$/d' | awk 'p == "--run-dir" || p == "--pool-dir" || p == "--resume" || p == "--total-steps" || p == "--max-hours" {p = $0; next} $0 == "--run-dir" || $0 == "--pool-dir" || $0 == "--resume" || $0 == "--total-steps" || $0 == "--max-hours" || $0 == "--eval-every" || $0 == "--snapshot-every" || $0 == "--checkpoint-every" {p = $0; next} p == "--eval-every" || p == "--snapshot-every" || p == "--checkpoint-every" {p = $0; next} {print; p = $0}' | sort; }
if diff <(sed 's#^[^ ]* ##' "$B/bf16-argv.txt" | norm) <(sed 's#^[^ ]* ##' "$F/first-argv.txt" | norm) > "$F/first-argv-vs-bench.diff"; then st "G9qbf first start (trainer $pid): argv = the bf16 bench arm's (bench limits / dirs / resume aside); resume $FP; restart flag re-armed"
else st "ALERT: G9qbf's argv differs from the bf16 arm's beyond limits / dirs (runs/g9qbf/first-argv-vs-bench.diff): $(head -c 300 "$F/first-argv-vs-bench.diff" | tr '\n' ' ')"
  mkdir -p "$K/argv"; echo "$(date -u +%FT%T) G9qbf first argv != the bf16 bench arm's (runs/g9qbf/first-argv-vs-bench.diff); the chain goes on" >> "$K/argv/FAILED"; fi
# ---- 5. readings ----
run_cmp() {  # run_cmp NAME STEP SEED WORKERS
  local D=$C/runs/g9qbf-$1; mkdir -p "$D"
  cp "$C/scripts/netv2_cmp_step.sh" "/tmp/nv2-qbf-cmp-$$.sh" && [ "$(md5of "/tmp/nv2-qbf-cmp-$$.sh")" = "$CMP_MD5" ] || fail "$1: cmp_step copy / md5"
  st "$1: waiting for G9qbf's and G9q's first pool snapshots >= $2, then h2h seed $3 ($4 workers)"
  CMP_WORKERS=$4 timeout 172800 bash "/tmp/nv2-qbf-cmp-$$.sh" "$2" "$3" "$D" "$C/runs/g9/*/G9qbf/pool" "$C/runs/r1/*/G9q/pool" > "$D/run.log" 2>&1 &
  local cp=$! rc
  while kill -0 "$cp" 2>/dev/null; do  # a guard trip of the fork ends the A/B (FAIL): stop waiting / playing
    if [ -f "$F/TRIPPED" ]; then st "$1: G9qbf guard trip (runs/g9qbf/TRIPPED) => stop the reading"; pkill -TERM -P "$cp" 2>/dev/null; kill -TERM "$cp" 2>/dev/null; sleep 5; rm -f "/tmp/nv2-qbf-cmp-$$.sh"; return 1; fi
    sleep 60; done
  wait "$cp"; rc=$?; rm -f "/tmp/nv2-qbf-cmp-$$.sh"
  [ "$rc" = 0 ] || fail "$1: h2h exited $rc (log $D/run.log)"
  grep -q "^G9s $C/runs/g9/[^ ]*/G9qbf/pool/" "$D/NETS.txt" && grep -q "^G9p $C/runs/r1/[^ ]*/G9q/pool/" "$D/NETS.txt" || fail "$1: NETS.txt sides wrong"
  grep -q "^--- all pairs played" "$D/RESULT.txt" || fail "$1: RESULT.txt incomplete"; }
if ! run_cmp r25 $((S0 + 25000000)) 1625251 4; then park_fork "FAIL: guard trip before R+25"; verdict FAIL "guard trip before R+25 (runs/g9qbf/TRIPPED): G9q stays fp32"; st "chain done"; exit 0; fi
v25=$(reading R+25 "$C/runs/g9qbf-r25"); st "R+25 READING $v25 ($(sed -n 2p "$C/runs/g9qbf-r25/READING.txt" | cut -d'(' -f2 | cut -c1-120))"
if [ "$v25" = STOP ] || [ -f "$F/TRIPPED" ]; then park_fork "FAIL at R+25"; verdict FAIL "R+25 $v25$([ -f "$F/TRIPPED" ] && echo ' + guard trip'): bf16 worse => G9q stays fp32"; st "chain done"; exit 0; fi
if ! run_cmp r50 $((S0 + 50000000)) 1650501 4; then park_fork "FAIL: guard trip before R+50"; verdict FAIL "R+25 $v25; guard trip before R+50 (runs/g9qbf/TRIPPED): G9q stays fp32"; st "chain done"; exit 0; fi
v50=$(reading R+50 "$C/runs/g9qbf-r50"); st "R+50 READING $v50 ($(sed -n 2p "$C/runs/g9qbf-r50/READING.txt" | cut -d'(' -f2 | cut -c1-120))"
park_fork "R+50 $v50"
if [ -f "$F/TRIPPED" ] || [ "$v50" = STOP ]; then verdict FAIL "R+25 $v25, R+50 $v50$([ -f "$F/TRIPPED" ] && echo ' + guard trip'): G9q stays fp32"
elif [ "$v50" = NI ]; then verdict PASS "R+25 $v25, R+50 $v50 (bf16 not >= 2 points worse at +50M) => the orchestrator switches G9q to bf16 at its next restart in place"
else verdict INCONCLUSIVE "R+25 $v25, R+50 $v50: G9q stays fp32"; fi
st "chain done"
