#!/bin/bash
# netv2 2026-11-15, G9q2 readings chain (runs/g9q2/PREREG.txt 5d218a36; USER decision 11-15 ~14:05), in the data host
# (h2h-only step): R25 (seed 9925): G9q2's first pool snapshot >= 25M vs G9q pool/00025038951.pt; R50 (seed 9926): G9q2's
# first pool snapshot >= 50M vs G9q pool/00050038810.pt; each via scripts/netv2_cmp_step.sh (a80c7da2; A "G9s" = G9q2,
# B "G9p" = G9q; seq-gain design, both directions, nested order = record) -> runs/g9q2/cmp/{25M,50M}/RESULT.txt +
# READING.txt (r--: "STEP SUCCESS|FUTILITY|UNDECIDED"); then runs/g9q2/cmp/OVERALL.txt (r--): EITHER SUCCESS / BOTH
# FUTILITY / INCONCLUSIVE. Measurements only: nothing here touches G9q2. Status runs/g9q2/chain/STATUS; a failure
# appends runs/g9q2/chain/FAILED and ends the chain. Each step waits for its snapshots (cmp_step's own wait loop),
# R25 <= 36 h, R50 <= 60 h after this chain starts.
#   setsid nohup srun --jobid=<data host> --overlap --ntasks=1 -c 8 --gres=gpu:1 --mem=8G -J nv2-g9q2-cmp \
#     bash $C/scripts/netv2_g9q2/cmp-chain.sh > $C/logs/g9q2-cmp-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; G=$C/runs/g9q2; K=$G/chain; mkdir -p "$K" "$G/cmp"
QP=$C/runs/g9/20261109-021316-741849/G9q/pool
st() { echo "$(date -u +%FT%T) $*" | tee -a "$K/STATUS" >&2; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$K/FAILED"; exit 1; }
[ ! -e "$K/FAILED" ] || { st "runs/g9q2/chain/FAILED exists: not running"; exit 1; }
[ "$(md5sum < "$G/PREREG.txt" | cut -c1-8)" = 5d218a36 ] && [ ! -w "$G/PREREG.txt" ] || fail "PREREG md5 is not 5d218a36 (or writable)"
T0=$(date +%s)
trap 'fail "killed by SIGTERM (the host job ended or the step was cancelled): relaunch in the next host"' TERM
dor() { sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$1/RESULT.txt"; }
cmp_run() {  # cmp_run STEP SEED OUT S_DIRS P_DIRS LIMIT_S
  local lim=$6
  cp "$C/scripts/netv2_cmp_step.sh" "/tmp/nv2-g9q2-cmp-$$.sh" || return 1
  [ "$(md5sum < "/tmp/nv2-g9q2-cmp-$$.sh" | cut -c1-8)" = a80c7da2 ] || { echo "netv2_cmp_step.sh md5 is not a80c7da2"; return 1; }
  CMP_WORKERS=7 timeout "$lim" bash "/tmp/nv2-g9q2-cmp-$$.sh" "$1" "$2" "$3" "$4" "$5"; local rc=$?; rm -f "/tmp/nv2-g9q2-cmp-$$.sh"; return $rc; }
reading() {  # reading STEPNAME DIR -> echoes SUCCESS|FUTILITY|UNDECIDED and writes DIR/READING.txt
  local x y v
  read -r x y <<< "$(dor "$2")"
  if [ "$x" = 10 ] || [ "$y" = 10 ]; then v=SUCCESS; elif [ "$x" = 20 ] && [ "$y" = 20 ]; then v=FUTILITY; else v=UNDECIDED; fi
  { echo "G9q2 PREREG reading $1 (runs/g9q2/PREREG.txt 5d218a36), read mechanically by scripts/netv2_g9q2/cmp-chain.sh"
    echo "$1 $v (\"G9q2 > G9q\" exit $x, \"G9q > G9q2\" exit $y; 10 SUCCESS, 20 FUTILITY, 0 undecided at n_max)"
    echo "RESULT.txt md5 $(md5sum < "$2/RESULT.txt" | cut -c1-8)"; grep -A1 '^DECISION OF RECORD' "$2/RESULT.txt"
    grep -A3 '^--- all pairs played' "$2/RESULT.txt"; } > "$2/READING.txt.tmp" && mv -f "$2/READING.txt.tmp" "$2/READING.txt" && chmod a-w "$2/READING.txt"
  echo "$v"; }
one() {  # one NAME STEP SEED SNAP_EXPECTED SNAP_MD5 DEADLINE_H
  local name=$1 step=$2 seed=$3 want=$4 wmd5=$5 dl=$6 D=$G/cmp/$1 left v
  mkdir -p "$D"
  if [ -f "$D/READING.txt" ]; then st "$name: READING.txt exists: $(sed -n 2p "$D/READING.txt")"; sed -n 2p "$D/READING.txt" | cut -d' ' -f2; return 0; fi
  left=$(( T0 + dl * 3600 - $(date +%s) )); [ "$left" -gt 600 ] || fail "$name: deadline passed before the h2h"
  st "$name: waiting for G9q2's first pool snapshot >= $step, then h2h vs G9q $(basename "$want") seed $seed (limit $((left / 60)) min)"
  cmp_run "$step" "$seed" "$D" "$C/runs/g9/*/G9q2/pool" "$QP" "$left" > "$D/run.log" 2>&1 || fail "$name: h2h exited $? (log $D/run.log)"
  grep -q "^G9p $QP/$(basename "$want")" "$D/NETS.txt" || fail "$name: NETS.txt names $(sed -n 's/^G9p //p' "$D/NETS.txt") not $(basename "$want")"
  [ "$(sed -n 's/^md5 g9s [0-9a-f]* g9p \([0-9a-f]*\)$/\1/p' "$D/NETS.txt" | cut -c1-8)" = "$wmd5" ] || fail "$name: G9q snapshot md5 is not $wmd5"
  grep -q "^G9s $C/runs/g9/[^ ]*/G9q2/pool/" "$D/NETS.txt" || fail "$name: the A side is not a G9q2 pool snapshot"
  grep -q "^--- all pairs played" "$D/RESULT.txt" || fail "$name: RESULT.txt incomplete"
  v=$(reading "$name" "$D"); st "$name READING $v ($(sed -n 2p "$D/READING.txt" | cut -d'(' -f2))"; echo "$v"; }
st "chain start on $(hostname -s) (job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?})"
v25=$(one 25M 25000000 9925 "$QP/00025038951.pt" b1cb200f 36) || exit 1
v50=$(one 50M 50000000 9926 "$QP/00050038810.pt" 29b0d2af 60) || exit 1
if [ "$v25" = SUCCESS ] || [ "$v50" = SUCCESS ]; then o="EITHER SUCCESS: the two seeds differ by more than noise at a matched step => single-seed recipe comparisons are bounded by seed variance (gaps and intervals in cmp/{25M,50M}/RESULT.txt)"
elif [ "$v25" = FUTILITY ] && [ "$v50" = FUTILITY ]; then o="BOTH FUTILITY: seed variance at matched steps below ~2 points (theta1 52%)"
else o="INCONCLUSIVE (neither rule's condition met)"; fi
{ echo "G9q2 PREREG overall reading (runs/g9q2/PREREG.txt 5d218a36): R25 $v25, R50 $v50"; echo "$o"
  for s in 25M 50M; do echo "--- $s:"; cat "$G/cmp/$s/READING.txt"; done; } > "$G/cmp/OVERALL.txt.tmp" && mv -f "$G/cmp/OVERALL.txt.tmp" "$G/cmp/OVERALL.txt" && chmod a-w "$G/cmp/OVERALL.txt"
st "OVERALL: R25 $v25, R50 $v50 => $o"
st "chain done"
