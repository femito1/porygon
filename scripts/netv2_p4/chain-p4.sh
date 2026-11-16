#!/bin/bash
# netv2 2026-11-16: the P4 data-lever chain (runs/p4/PREREG.txt; orchestrator decision 14:5x), a step of the A40 job
# 1173274 (in v18's park HOLD on G9qent): P4 READY -> BC-P4 (P3's wrapper) -> B1 BC-P4 vs BC-P3 (runs/p4/bc/READING.txt)
# -> bridge + netv2.env v19 deploy + MODE g9p4 -> G9-P4's first start checked -> R25 / R50 vs G9q (runs/p4/cmp/) ->
# OVERALL (CANDIDATE, or park G9-P4 and HOLD). Status runs/p4/STATUS; any failure appends runs/p4/FAILED and ends it.
#   setsid nohup srun --jobid=1173274 --overlap --ntasks=1 -c 8 --gres=gpu:1 --mem=36G -J nv2-p4 env JOBPP=<job script pid> \
#     bash -c 'cp $C/scripts/netv2_p4/chain-p4.sh /tmp/nv2-p4-$$.sh && exec bash /tmp/nv2-p4-$$.sh' > $C/logs/p4-<t>.out 2>&1 < /dev/null &
# Laptop mock of its INLINE and argv check: scripts/netv2_p4/mock-p4.sh (P4_C / P4_D / P4_LQ / P4_SLEEP / P4_GPY: dry runs only).
set -uo pipefail
C=${P4_C:-/vol/bitbucket/fr1226/porygon-ws-netv2}; P=$C/runs/p4; R=$C/runs/g9; S9=$C/runs/g9p4
D=${P4_D:-/vol/bitbucket/fr1226/porygon-ws-data/datasets/frozen}
J=${SLURM_JOB_ID:?}; JOBPP=${JOBPP:?set JOBPP = the job script pid}; LQ=${P4_LQ:-/tmp/ws-netv2-$J/run}; Z=${P4_SLEEP:-1}
GPY=${P4_GPY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}
PRE_MD5=2479eb4f; ENV18=553bfec9; ENV19=b705bb93; BR_MD5=e6c894ae; AC_MD5=a7d5205d; PK_MD5=ed1062b9; CMP_MD5=a80c7da2; BCW_MD5=1886db62
P3=$C/runs/pretrain/P3-g9-s2.pt; P3_MD5=3e600174; P4=$C/runs/pretrain/P4-g9-s2.pt; BCOUT=$C/runs/pretrain/g9-p4-s2
QP=$C/runs/g9/20261109-021316-741849/G9q/pool; Q25=$QP/00025038951.pt; Q25_MD5=b1cb200f; Q50=$QP/00050038810.pt; Q50_MD5=29b0d2af
SEED_B1=4410041; SEED_R25=4425251; SEED_R50=4450501
PAT='^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'
mkdir -p "$P" "$P/bc" "$P/cmp" "$S9"
st() { echo "$(date -u +%FT%T) $*" | tee -a "$P/STATUS" >&2; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$P/FAILED"; exit 1; }
md5of() { md5sum < "$1" | cut -c1-8; }
zz() { sleep $(( $1 * Z )); }
pub() {  # pub TMP FINAL: publish read-only
  mv -f "$1" "$2" && chmod a-w "$2"; }
[ ! -e "$P/FAILED" ] || { st "runs/p4/FAILED exists: not running"; exit 1; }
if [ ! -s "$P/PREREG.txt" ] || [ -w "$P/PREREG.txt" ] || [ "$(md5of "$P/PREREG.txt")" != "$PRE_MD5" ]; then fail "no read-only runs/p4/PREREG.txt with md5 $PRE_MD5"; fi
for x in "$C/scripts/netv2_p4/inline-bridge.sh $BR_MD5" "$C/scripts/netv2_p4/argvcheck.py $AC_MD5" "$C/scripts/netv2_g9qbf/park.sh $PK_MD5" \
         "$C/scripts/netv2_cmp_step.sh $CMP_MD5" "$C/logs/nv2agent-p2-20261109.sh $BCW_MD5" "$C/netv2.env.v19-g9p4 $ENV19" "$P3 $P3_MD5" "$Q25 $Q25_MD5" "$Q50 $Q50_MD5"; do
  set -- $x; [ -f "$1" ] && [ "$(md5of "$1")" = "$2" ] || fail "$1 missing or md5 not $2"; done
[ ! -w "$C/netv2.env.v19-g9p4" ] && [ -s "$P/ref-g9q2-first-argv.txt" ] && [ ! -w "$P/ref-g9q2-first-argv.txt" ] && [ -s "$P/ref-g9q-live-argv.txt" ] && [ ! -w "$P/ref-g9q-live-argv.txt" ] || fail "staged v19 / argv references not read-only"
trap 'st "chain step got SIGTERM (job end / cancel)"; echo "$(date -u +%FT%T) SIGTERM (job end / cancel): the A40 state is as the files say" >> "$P/FAILED"; exit 1' TERM
no_async_eval() {  # no_async_eval TRAIN_LOG: the last async eval (if any) has finished
  local s ln epid estep; s=$(grep -n "started (async, pid" "$1" 2>/dev/null | tail -n 1); [ -z "$s" ] && return 0
  ln=${s%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$s"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$s")
  tail -n +"$ln" "$1" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null; }
the_trainer() {  # the single trainer under the job script (empty if none or several)
  local ps p out=""; ps=$(pgrep -u "$(id -u)" -f "$PAT" || true)
  for p in $ps; do if [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = "$JOBPP" ]; then out="$out $p"; fi; done
  set -- $out; if [ $# = 1 ]; then echo "$1"; fi; }
rundir_of() { tr '\0' '\n' < "/proc/$1/cmdline" 2>/dev/null | awk 'p == "--run-dir" {r = $0} {p = $0} END {print r}'; }
in_pause() {  # in_pause RUN: a job of runs/RUN sits (or will sit) in nv2_pause
  local f; for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR; do [ -e "$C/runs/$1/$f" ] && return 0; done; return 1; }
wait_inline() {  # wait_inline T0 NAME: the INLINE installed at T0 finished (<= 6 h)
  local i; for i in $(seq 1 2160); do
    zz 10; if [ ! -e "$R/INLINE_CMD" ] && [ ! -e "$R/INLINE_CMD.running" ] && [ -n "$(find "$R" -maxdepth 1 -name 'INLINE_CMD.done-*' -newermt "@$1" 2>/dev/null | head -n 1)" ]; then return 0; fi; done
  fail "$2: INLINE not finished within 6 h"; }
dor() { sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$1/RESULT.txt"; }
cmp_run() {  # cmp_run STEP SEED OUT A_DIRS B_DIRS LIMIT_S
  local rc; cp "$C/scripts/netv2_cmp_step.sh" "/tmp/nv2-p4-cmp-$$.sh" && [ "$(md5of "/tmp/nv2-p4-cmp-$$.sh")" = "$CMP_MD5" ] || return 1
  mkdir -p "$3"; CMP_WORKERS=7 timeout "$6" bash "/tmp/nv2-p4-cmp-$$.sh" "$1" "$2" "$3" "$4" "$5" > "$3/run.log" 2>&1; rc=$?; rm -f "/tmp/nv2-p4-cmp-$$.sh"; return $rc; }
park_p4() {  # park_p4 WHY: ONE INLINE (park.sh) + ONE SIGINT; v19 then HOLDs the A40 on the parked G9-P4
  local why=$1 i pid t
  if [ -f "$C/runs/g9p4-parked/PARKED.txt" ]; then st "G9-P4 already parked"; return 0; fi
  for i in $(seq 1 180); do grep -q " upd [0-9]* steps " "$LQ/G9P4/train.log" 2>/dev/null && break; zz 10; done  # park.sh needs steps > 0
  for i in $(seq 1 120); do no_async_eval "$LQ/G9P4/train.log" && break; zz 30; done
  for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR; do [ ! -e "$R/$f" ] || fail "park: runs/g9/$f present (G9-P4 not training here?): park it by hand ($why)"; done
  pid=$(the_trainer); [ -n "$pid" ] && [ "$(rundir_of "$pid")" = "$LQ/G9P4" ] || fail "park: no single G9-P4 trainer to stop ($why)"
  cat > "$P/inline-park.txt" <<INL
set -u
# G9-P4 PARK (runs/p4/PREREG.txt $PRE_MD5): $why
cp $C/scripts/netv2_g9qbf/park.sh /tmp/nv2-p4-pk-\$\$.sh || exit 1
[ "\$(md5sum < /tmp/nv2-p4-pk-\$\$.sh | cut -c1-8)" = $PK_MD5 ] || { echo "\$(date -u +%FT%T) park.sh md5 mismatch" >> $P/FAILED; exit 1; }
PARK_LIN=G9P4 PARK_STATE=runs/g9p4 PARK_DEST=runs/g9p4-parked PARK_QUEUED=$R/INLINE_CMD.running PARK_WHY="$why" bash /tmp/nv2-p4-pk-\$\$.sh 2>&1 | tee -a $P/STATUS | tail -n 1
rm -f /tmp/nv2-p4-pk-\$\$.sh
INL
  bash -n "$P/inline-park.txt" || fail "park INLINE text syntax"
  cp "$P/inline-park.txt" "$R/INLINE_CMD.tmp" && mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD" || fail "cannot install the park INLINE"
  t=$(date +%s); touch "$R/restart"; kill -INT "$pid" && st "park INLINE queued, restart armed, ONE SIGINT to G9-P4 trainer $pid ($why)"
  wait_inline "$t" park
  [ -f "$C/runs/g9p4-parked/PARKED.txt" ] || fail "park: no runs/g9p4-parked/PARKED.txt after the INLINE (runs/g9p4/park.log)"
  st "G9-P4 PARKED: $(head -n 1 "$C/runs/g9p4-parked/PARKED.txt" | cut -c1-200)"; }
st "chain start on $(hostname -s) (job $J step ${SLURM_STEP_ID:-?}, JOBPP $JOBPP, PREREG $PRE_MD5)"
# ---- 1. the P4 freeze ----
T0=$(date +%s)
while :; do
  n=0; RD=""; for f in "$D"/p4-gen9-*/READY; do [ -e "$f" ] || continue; n=$((n + 1)); RD=$f; done
  [ "$n" -le 1 ] || fail "$n P4 freezes with READY in $D: want exactly one"
  [ "$n" = 1 ] && break
  [ $(( $(date +%s) - T0 )) -lt 86400 ] || fail "no P4 READY within 24 h"
  zz 300
done
PD=$(dirname "$RD"); SH=$PD/v2pre-human-shuf
ntr=$(ls "$SH" 2>/dev/null | grep -cE '^shard-[0-9]+\.npz$'); nva=$(ls "$SH" 2>/dev/null | grep -cE '^shard-val-[0-9]+\.npz$')
[ "$ntr" -gt 0 ] && [ "$nva" -gt 0 ] || fail "P4 layout: $SH has $ntr training / $nva val shards"
VJ=""; for v in "$SH/verify.json" "$PD/verify.json"; do if [ -f "$v" ]; then VJ=$v; break; fi; done
[ -n "$VJ" ] && "$GPY" -I -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('ok') is True else 1)" "$VJ" || fail "P4: no verify.json with top-level ok true ($SH, $PD)"
{ echo "P4 freeze (runs/p4/PREREG.txt $PRE_MD5): $PD (READY $(stat -c %y "$RD" | cut -c1-19))"
  echo "shards: $ntr training + $nva val in $SH; verify $VJ ok; summary.json md5 $( [ -f "$PD/summary.json" ] && md5of "$PD/summary.json" || echo none)"
  [ -f "$PD/summary.json" ] && head -c 1500 "$PD/summary.json"; echo; } > "$P/bc/DATA.txt.tmp" && pub "$P/bc/DATA.txt.tmp" "$P/bc/DATA.txt" 2>/dev/null || true
st "P4 READY: $PD ($ntr training + $nva val shards, $VJ ok)"
# ---- 2. BC-P4 (P3's exact wrapper and args; only the dataset changes) ----
if [ -f "$P4" ] && [ -s "$S9/P4.md5" ] && [ "$(md5of "$P4")" = "$(cat "$S9/P4.md5")" ]; then st "BC-P4 exists ($P4, md5 $(cat "$S9/P4.md5")): skip training"
else
  [ ! -e "$P4" ] || fail "$P4 exists without a matching runs/g9p4/P4.md5"
  cp "$C/logs/nv2agent-p2-20261109.sh" "/tmp/nv2-p4-bc-$$.sh" && [ "$(md5of "/tmp/nv2-p4-bc-$$.sh")" = "$BCW_MD5" ] || fail "BC wrapper copy / md5"
  if [ -f "$BCOUT/final.pt" ] && [ ! -e "$BCOUT.lock" ] && grep -q "g9-p4-s2 exit 0" "$P/bc/pretrain-run.log" 2>/dev/null; then st "BC-P4 trained before ($BCOUT/final.pt): publishing it"
  else
    st "BC-P4: bash nv2agent-p2-20261109.sh g9-p4-s2 $SH --stream-gb 1.0 --eval-every 5000 (limit 20 h)"
    timeout -k 300 72000 bash "/tmp/nv2-p4-bc-$$.sh" g9-p4-s2 "$SH" --stream-gb 1.0 --eval-every 5000 > "$P/bc/pretrain-run.log" 2>&1; rc=$?
    rm -f "/tmp/nv2-p4-bc-$$.sh"
    [ "$rc" = 0 ] && [ -f "$BCOUT/final.pt" ] && grep -q "g9-p4-s2 exit 0" "$P/bc/pretrain-run.log" || fail "BC-P4 exited $rc (log runs/p4/bc/pretrain-run.log, $BCOUT/stdout.log)"
  fi
  cp -p "$BCOUT/final.pt" "$P4.tmp" && pub "$P4.tmp" "$P4" || fail "cannot publish $P4"
  [ "$(md5of "$P4")" = "$(md5of "$BCOUT/final.pt")" ] || fail "$P4 copy md5 mismatch"
  md5of "$P4" > "$S9/P4.md5.tmp" && pub "$S9/P4.md5.tmp" "$S9/P4.md5" || fail "cannot write runs/g9p4/P4.md5"
  st "BC-P4 done: $P4 (md5 $(cat "$S9/P4.md5")); $(grep '"event": "val"' "$BCOUT/stdout.log" | tail -n 1 | grep -oE '"step": [0-9]+|"policy_ce": [0-9.]+|"value_bce": [0-9.]+' | tr '\n' ' ')"
fi
# ---- 3. B1: BC-P4 vs BC-P3 (informational, no stop rule) ----
B1=$P/bc/h2h
if [ -f "$P/bc/READING.txt" ]; then st "B1 READING exists: $(sed -n 2p "$P/bc/READING.txt")"
else
  mkdir -p "$P/bc/nets/A" "$P/bc/nets/B"
  for x in "A $P4" "B $P3"; do set -- $x; f=$P/bc/nets/$1/00000000000.pt
    if [ ! -f "$f" ]; then cp -p "$2" "$f.tmp" && pub "$f.tmp" "$f" || fail "B1 net copy $2"; fi
    [ "$(md5of "$f")" = "$(md5of "$2")" ] || fail "B1 net $f md5"; done
  st "B1: BC-P4 vs BC-P3, seed $SEED_B1, 7 workers"
  cmp_run 0 "$SEED_B1" "$B1" "$P/bc/nets/A" "$P/bc/nets/B" 43200 || fail "B1 h2h exited (log $B1/run.log)"
  grep -qx "G9s $P/bc/nets/A/00000000000.pt" "$B1/NETS.txt" && grep -qx "G9p $P/bc/nets/B/00000000000.pt" "$B1/NETS.txt" && grep -q "^--- all pairs played" "$B1/RESULT.txt" || fail "B1 NETS / RESULT check"
  read -r x y <<< "$(dor "$B1")"
  if [ "${x:-}" = 10 ]; then v='"BC-P4 > BC-P3" SUCCESS'; elif [ "${y:-}" = 10 ]; then v='"BC-P3 > BC-P4" SUCCESS'
  elif [ "${x:-}" = 20 ] && [ "${y:-}" = 20 ]; then v="BOTH FUTILITY (gap below ~2 points)"; else v="UNDECIDED (exits ${x:-?} / ${y:-?})"; fi
  { echo "B1 READING (runs/p4/PREREG.txt $PRE_MD5; informational, no stop rule), read mechanically by scripts/netv2_p4/chain-p4.sh"
    echo "B1 $v (\"BC-P4 > BC-P3\" exit ${x:-?}, \"BC-P3 > BC-P4\" exit ${y:-?}; 10 SUCCESS, 20 FUTILITY, 0 undecided at n_max)"
    echo "A = BC-P4 $P4 (md5 $(cat "$S9/P4.md5"), \"G9s\"), B = BC-P3 $P3 (md5 $P3_MD5, \"G9p\"); seed $SEED_B1; RESULT.txt md5 $(md5of "$B1/RESULT.txt")"
    grep -A1 '^DECISION OF RECORD' "$B1/RESULT.txt"
    sed -n '/^--- nested, G9s > G9p:/,/^--- round-robin, G9s > G9p:/p' "$B1/RESULT.txt" | grep -E "^--- nested|DECISION|estimate|repeated CI"
    grep -A3 '^--- all pairs played' "$B1/RESULT.txt"
    echo "BC-P4's last held-out line (its own held-out set; P3's is runs/pretrain/g9-p3-s2/stdout.log, other rows): $(grep '"event": "val"' "$BCOUT/stdout.log" 2>/dev/null | tail -n 1 | cut -c1-400)"
  } > "$P/bc/READING.txt.tmp" && pub "$P/bc/READING.txt.tmp" "$P/bc/READING.txt"
  st "B1 READING: $v"
fi
# ---- 4. G9-P4's start: bridge -> v19 deploy -> MODE g9p4 -> ONE SIGINT ----
if [ -f "$S9/G9P4_STARTED" ]; then st "G9-P4 started before ($(cat "$S9/G9P4_STARTED")): readings only"
else
  [ ! -e "$P/NO_BRIDGE" ] || fail "runs/p4/NO_BRIDGE exists (orchestrator veto of the bridge / v19 deploy): G9-P4 not started; the A40 HOLDs"
  [ "$(md5of "$C/netv2.env")" = "$ENV18" ] || fail "live netv2.env md5 $(md5of "$C/netv2.env") is not v18 $ENV18"
  [ "$(cat "$R/MODE")" = g9qent ] && [ -f "$C/runs/g9qent-parked/PARKED.txt" ] || fail "runs/g9/MODE '$(cat "$R/MODE")' is not the parked g9qent"
  for i in $(seq 1 60); do [ -f "$R/HOLD" ] && break; zz 30; done
  [ -f "$R/HOLD" ] || fail "the A40 is not in the park HOLD (no runs/g9/HOLD after 30 min)"
  for f in INLINE_CMD INLINE_CMD.running WAIT_FOR; do [ ! -e "$R/$f" ] || fail "runs/g9/$f present: not queuing"; done
  [ -z "$(the_trainer)" ] || fail "a trainer is running under the job script (not the park HOLD)"
  [ ! -e "$C/runs/g9qbf/G9QBF_STARTED" ] || fail "runs/g9qbf/G9QBF_STARTED exists: no bridge lineage"
  T1=$(date +%s); until ! in_pause r1 && ! in_pause g4r1bf; do
    [ $(( $(date +%s) - T1 )) -lt 21600 ] || fail "runs/r1 or runs/g4r1bf in nv2_pause for 6 h: no deploy window"; zz 60; done
  cat > "$P/inline-bridge.txt" <<INL
set -u
# G9-P4 BRIDGE (runs/p4/PREREG.txt $PRE_MD5): stage the bridge lineage G9qbf, MODE g9qbf, lift the park HOLD
cp $C/scripts/netv2_p4/inline-bridge.sh /tmp/nv2-p4-br-\$\$.sh || exit 1
[ "\$(md5sum < /tmp/nv2-p4-br-\$\$.sh | cut -c1-8)" = $BR_MD5 ] || { echo "\$(date -u +%FT%T) inline-bridge md5 mismatch" >> $P/FAILED; exit 1; }
bash /tmp/nv2-p4-br-\$\$.sh; rm -f /tmp/nv2-p4-br-\$\$.sh
INL
  bash -n "$P/inline-bridge.txt" || fail "bridge INLINE text syntax"
  cp "$P/inline-bridge.txt" "$R/INLINE_CMD.tmp" && mv -f "$R/INLINE_CMD.tmp" "$R/INLINE_CMD" || fail "cannot install the bridge INLINE"
  t1=$(date +%s); touch "$R/restart"; rm -f "$R/HOLD"; st "bridge INLINE queued, restart armed, park HOLD lifted"
  pid=""; for i in $(seq 1 270); do  # <= 45 min for the bridge trainer (restart kept armed: any exit re-reads netv2.env)
    touch "$R/restart"; [ ! -e "$P/FAILED" ] || fail "the bridge INLINE refused (see STATUS)"
    pid=$(the_trainer); if [ -n "$pid" ] && [ "$(rundir_of "$pid")" = "$LQ/G9qbf" ]; then break; fi; pid=""; zz 10; done
  [ -n "$pid" ] || fail "no bridge trainer (--run-dir $LQ/G9qbf) within 45 min of the bridge INLINE"
  st "bridge trainer $pid up (MODE $(cat "$R/MODE"))"
  dep=""; for i in $(seq 1 240); do  # <= 2 h for a deploy window (no job of r1 / g9 / g4r1bf in nv2_pause)
    touch "$R/restart"
    if ! in_pause r1 && ! in_pause g4r1bf && ! in_pause g9 && [ -n "$(the_trainer)" ]; then dep=1; break; fi; zz 30; done
  if [ -z "$dep" ]; then
    echo g9qent > "$R/MODE.tmp" && mv -f "$R/MODE.tmp" "$R/MODE"; touch "$R/restart"
    for i in $(seq 1 120); do no_async_eval "$LQ/G9qbf/train.log" && break; zz 30; done
    pid=$(the_trainer); [ -n "$pid" ] && kill -INT "$pid"
    fail "no deploy window within 2 h of the bridge: MODE g9qent restored, ONE SIGINT to ${pid:-no trainer} (v18 HOLDs on the parked arm)"; fi
  bak=$C/netv2.env.bak-$(date -u +%Y%m%d-%H%M)
  [ "$(md5of "$C/netv2.env")" = "$ENV18" ] || fail "live netv2.env changed to $(md5of "$C/netv2.env") during the bridge"
  cp -p "$C/netv2.env" "$bak" && cp "$C/netv2.env.v19-g9p4" "$C/netv2.env.new" || fail "deploy: backup / .new copy"
  [ "$(md5of "$C/netv2.env.new")" = "$ENV19" ] && bash -n "$C/netv2.env.new" || fail "deploy: netv2.env.new md5 / syntax"
  mv -f "$C/netv2.env.new" "$C/netv2.env" && [ "$(md5of "$C/netv2.env")" = "$ENV19" ] || fail "deploy: mv / live md5"
  st "DEPLOYED netv2.env v19 ($ENV19 = v18 + MODE g9p4); backup $bak ($ENV18)"
  cp -p "$R/MODE" "$R/MODE.g9qbf-before-g9p4"; echo g9p4 > "$R/MODE.tmp" && mv -f "$R/MODE.tmp" "$R/MODE" || fail "cannot write MODE g9p4"
  touch "$R/restart"; st "MODE g9p4 written, restart armed"
  for i in $(seq 1 180); do  # the bridge's startup line (<= 30 min), unless it already exited into G9-P4
    pid=$(the_trainer); if [ -n "$pid" ] && [ "$(rundir_of "$pid")" = "$LQ/G9P4" ]; then break; fi
    grep -q "v2 trainer:" "$LQ/G9qbf/train.log" 2>/dev/null && break; touch "$R/restart"; zz 10; done
  for i in $(seq 1 120); do no_async_eval "$LQ/G9qbf/train.log" && break; touch "$R/restart"; zz 30; done
  pid=$(the_trainer)
  if [ -n "$pid" ] && [ "$(rundir_of "$pid")" = "$LQ/G9qbf" ]; then touch "$R/restart"; kill -INT "$pid" && st "ONE SIGINT to the bridge trainer $pid"
  else st "the bridge trainer is gone (now: ${pid:-none} $( [ -n "$pid" ] && rundir_of "$pid")): no SIGINT"; fi
  pid=""; for i in $(seq 1 270); do  # <= 45 min for G9-P4's trainer and its startup line
    p=$(the_trainer); if [ -n "$p" ] && [ "$(rundir_of "$p")" = "$LQ/G9P4" ] && grep -q "v2 trainer:" "$LQ/G9P4/train.log" 2>/dev/null; then pid=$p; break; fi
    if [ -n "$p" ] && [ "$(rundir_of "$p")" = "$LQ/G9qbf" ] && [ "$i" -gt 60 ]; then fail "the bridge trainer $p still runs 10 min after the SIGINT"; fi
    touch "$R/restart"; zz 10; done
  [ -n "$pid" ] || fail "G9-P4 did not start within 45 min (MODE $(cat "$R/MODE"); $LQ/train.log)"
  touch "$R/restart"
  { grep -E "saved final_model" "$LQ/G9qbf/train.log" 2>/dev/null | tail -n 1; } | sed 's/^/bridge: /' >> "$C/runs/g9qbf/BRIDGE.txt" || true
  cp "/proc/$pid/cmdline" "$S9/first-argv.bin" && chmod a-w "$S9/first-argv.bin" || fail "cannot record G9-P4's argv"
  tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(PYTHONPATH|PORYGON_PS)=' > "$S9/first-env.txt"
  sl=$(grep "v2 trainer:" "$LQ/G9P4/train.log" | tail -n 1); echo "$sl" > "$S9/first-startup-line.txt"
  "$GPY" -I "$C/scripts/netv2_p4/argvcheck.py" "$S9/first-argv.bin" "$P/ref-g9q2-first-argv.txt" "$P/ref-g9q-live-argv.txt" "$P4" "$P3" > "$S9/first-argv-check.txt" 2>&1; ac=$?
  why=""
  [ "$ac" = 0 ] || why="argv check FAIL (runs/g9p4/first-argv-check.txt)"
  grep -q "^PYTHONPATH=$C/src-24c633f" "$S9/first-env.txt" || why="$why; PYTHONPATH lacks src-24c633f"
  grep -qx "PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11" "$S9/first-env.txt" || why="$why; PORYGON_PS not ps-2026-11"
  grep -qF "amp=False" <<< "$sl" && grep -qF "resume=$P4, steps=0" <<< "$sl" || why="$why; startup line lacks amp=False / resume=$P4, steps=0"
  grep -q " $(cat "$S9/P4.md5") g9 $J$" "$S9/G9P4_STARTED" 2>/dev/null || why="$why; runs/g9p4/G9P4_STARTED '$(cat "$S9/G9P4_STARTED" 2>/dev/null)'"
  if [ -n "$why" ]; then park_p4 "G9-P4 first-start checks failed: ${why#; }"; fail "G9-P4 first-start checks: ${why#; } (G9-P4 parked)"; fi
  st "G9-P4 first start (trainer $pid): argv = G9q2's first argv up to P4 / lineage / seed and = G9q's live recipe up to P4; src-24c633f; ps-2026-11; amp=False; resume=BC-P4 steps 0; restart re-armed"
fi
# ---- 5. R25 / R50 vs G9q ----
TS=$(date -d "$(cut -d' ' -f1 "$S9/G9P4_STARTED")" +%s 2>/dev/null || date +%s)
one() {  # one NAME STEP SEED WANT WMD5 DEADLINE_H -> echoes SUCCESS-P4 | SUCCESS-G9q | FUTILITY | UNDECIDED
  local name=$1 step=$2 seed=$3 want=$4 wmd5=$5 dl=$6 O=$P/cmp/$1 left x y v fu
  if [ -f "$O/READING.txt" ]; then sed -n 2p "$O/READING.txt" | cut -d' ' -f2; return 0; fi
  left=$(( TS + dl * 3600 - $(date +%s) )); [ "$left" -gt 600 ] || fail "$name: deadline passed before the h2h"
  st "$name: waiting for G9-P4's first pool snapshot >= $step, then h2h vs G9q $(basename "$want") seed $seed (limit $((left / 60)) min)"
  cmp_run "$step" "$seed" "$O" "$C/runs/g9/*/G9P4/pool $LQ/G9P4/pool" "$QP" "$left" || fail "$name: h2h exited (log $O/run.log)"
  grep -q "^G9p $want$" "$O/NETS.txt" || fail "$name: NETS.txt names $(sed -n 's/^G9p //p' "$O/NETS.txt") not $want"
  [ "$(sed -n 's/^md5 g9s [0-9a-f]* g9p \([0-9a-f]*\)$/\1/p' "$O/NETS.txt" | cut -c1-8)" = "$wmd5" ] || fail "$name: G9q snapshot md5 is not $wmd5"
  grep -qE "^G9s ($C/runs/g9/[^ ]*/G9P4/pool/|$LQ/G9P4/pool/)[0-9]+\.pt$" "$O/NETS.txt" || fail "$name: the A side is not a G9-P4 pool snapshot"
  grep -q "^--- all pairs played" "$O/RESULT.txt" || fail "$name: RESULT.txt incomplete"
  read -r x y <<< "$(dor "$O")"
  if [ "${x:-}" = 10 ]; then v=SUCCESS-P4; elif [ "${y:-}" = 10 ]; then v=SUCCESS-G9q; elif [ "${x:-}" = 20 ] && [ "${y:-}" = 20 ]; then v=FUTILITY; else v=UNDECIDED; fi
  fu=$( [ -f "$S9/FU.started" ] && echo "G9-P4 guard FALLBACK ran ($(head -c 120 "$S9/FU.started"); reason $(head -c 120 "$S9/FU.reason" 2>/dev/null)): confounded, the orchestrator judges" || echo "no guard fallback of G9-P4 (runs/g9p4/FU.started absent)")
  { echo "G9-P4 PREREG reading $name (runs/p4/PREREG.txt $PRE_MD5), read mechanically by scripts/netv2_p4/chain-p4.sh"
    echo "$name $v (\"G9-P4 > G9q\" exit ${x:-?}, \"G9q > G9-P4\" exit ${y:-?}; 10 SUCCESS, 20 FUTILITY, 0 undecided at n_max)"
    echo "A = G9-P4 $(sed -n 's/^G9s //p' "$O/NETS.txt") (\"G9s\"), B = G9q $want (\"G9p\"); seed $seed; RESULT.txt md5 $(md5of "$O/RESULT.txt")"
    grep -A1 '^DECISION OF RECORD' "$O/RESULT.txt"
    sed -n '/^--- nested, G9s > G9p:/,/^--- round-robin, G9s > G9p:/p' "$O/RESULT.txt" | grep -E "^--- nested|DECISION|estimate|repeated CI"
    grep -A3 '^--- all pairs played' "$O/RESULT.txt"; echo "$fu; env md5 $(md5of "$C/netv2.env")"
  } > "$O/READING.txt.tmp" && pub "$O/READING.txt.tmp" "$O/READING.txt"
  st "$name READING $v"; echo "$v"; }
v25=$(one 25M 25000000 "$SEED_R25" "$Q25" "$Q25_MD5" 24) || exit 1
v50=$(one 50M 50000000 "$SEED_R50" "$Q50" "$Q50_MD5" 36) || exit 1
if [ "$v50" = SUCCESS-P4 ]; then o="CANDIDATE: R50 \"G9-P4 > G9q\" SUCCESS => G9-P4 is the candidate for the next main run; main decides its placement; G9-P4 keeps training on the A40 meanwhile"
else o="NOT A CANDIDATE: R50 $v50 (not \"G9-P4 > G9q\" SUCCESS) => G9-P4 parked at its state at the reading (runs/g9p4-parked/); the A40 HOLDs for main / the user"; fi
{ echo "G9-P4 PREREG overall reading (runs/p4/PREREG.txt $PRE_MD5): R25 $v25 (informational), R50 $v50 (decides)"; echo "$o"
  echo "--- B1:"; cat "$P/bc/READING.txt"; for s in 25M 50M; do echo "--- $s:"; cat "$P/cmp/$s/READING.txt"; done
} > "$P/cmp/OVERALL.txt.tmp" && pub "$P/cmp/OVERALL.txt.tmp" "$P/cmp/OVERALL.txt"
st "OVERALL: R25 $v25, R50 $v50 => ${o%%:*}"
if [ "$v50" != SUCCESS-P4 ]; then park_p4 "R50 $v50: not a candidate (runs/p4/PREREG.txt); A40 placement = orchestrator / user"; fi
st "chain done"
