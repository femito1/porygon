#!/bin/bash
# netv2 2026-11-15: READING R800 (runs/g9q-cd800/PREREG.txt): G9q-cd@800 (A, cmp_step's "G9s" label) vs G9q-cd@605M (B,
# "G9p") in-process via scripts/netv2_cmp_step.sh (a80c7da2, unmodified; STEP 0 = no step constraint; seq-gain design
# f12fec7c; seed 8006051). A driver step in the data host: waits <= 10 h for the one file in runs/g9q-cd800/final/ (md5 =
# the branch driver's STATUS line; stops on runs/g9q-cd800/FAILED), plays (limit 6 h), reads mechanically: IMPROVING iff
# "cd800 > cd605" SUCCESS (nested order of record) -> runs/g9q-cd800/READING.txt (r--). Failure: runs/g9q-cd800/r800/FAILED.
#   setsid nohup srun --jobid=1105603 --overlap --ntasks=1 -c 6 --gres=gpu:1 --mem=14G -J nv2-r800 \
#     bash -c 'cp $C/scripts/netv2_g9q/r800-chain.sh /tmp/nv2-r800-$$.sh && exec bash /tmp/nv2-r800-$$.sh' \
#     > $C/logs/r800-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; D=$C/runs/g9q-cd800; K=$D/r800; mkdir -p "$K"
PRE_MD5=1a1b9fa4; SEED=8006051; BD=$C/runs/g9q-cd600/final; B605=$BD/00605037107.pt; B605_MD5=7c28fac8
st() { echo "$(date -u +%FT%T) $*" | tee -a "$K/STATUS" >&2; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$K/FAILED"; exit 1; }
[ ! -e "$K/FAILED" ] || { st "runs/g9q-cd800/r800/FAILED exists: not running"; exit 1; }
if [ -e "$D/READING.txt" ]; then st "runs/g9q-cd800/READING.txt exists: nothing to do"; exit 0; fi
if [ ! -s "$D/PREREG.txt" ] || [ -w "$D/PREREG.txt" ] || [ "$(md5sum < "$D/PREREG.txt" | cut -c1-8)" != "$PRE_MD5" ]; then fail "no read-only runs/g9q-cd800/PREREG.txt with md5 $PRE_MD5"; fi
if [ "$(ls "$BD"/*.pt 2>/dev/null | wc -l)" != 1 ] || [ "$(md5sum < "$B605" | cut -c1-8)" != "$B605_MD5" ]; then fail "runs/g9q-cd600/final must hold only 00605037107.pt (md5 $B605_MD5)"; fi
trap 'fail "killed by SIGTERM (the host job ended or the step was cancelled): rerun the same pre-registration in the next host"' TERM
T0=$(date +%s); st "chain start on $(hostname -s) (job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}); waiting <= 10 h for runs/g9q-cd800/final/*.pt"
A=""
while [ -z "$A" ]; do
  if [ -e "$D/FAILED" ]; then fail "runs/g9q-cd800/FAILED: no cd800 net ($(tail -n 1 "$D/FAILED"))"; fi
  n=$(ls "$D"/final/*.pt 2>/dev/null | wc -l)
  if [ "$n" -gt 1 ]; then fail "$n files in runs/g9q-cd800/final (want 1)"; fi
  if [ "$n" = 1 ]; then A=$(ls "$D"/final/*.pt); break; fi
  if [ $(( $(date +%s) - T0 )) -ge 36000 ]; then fail "no cd800 final within 10 h"; fi
  sleep 60
done
want=""; for i in $(seq 1 20); do want=$(grep "G9q-cd800 final at" "$D/STATUS" 2>/dev/null | tail -n 1 | sed -n 's/.*(md5 \([0-9a-f]*\)).*/\1/p'); [ -n "$want" ] && break; sleep 15; done
got=$(md5sum < "$A" | cut -c1-8)
if [ -z "$want" ] || [ "$got" != "$want" ]; then fail "cd800 final $A md5 $got != the driver's STATUS md5 '${want:-none}'"; fi
st "A = G9q-cd@800 $A (md5 $got); B = G9q-cd@605M $B605 ($B605_MD5); seed $SEED; 5 workers"
cp "$C/scripts/netv2_cmp_step.sh" "/tmp/nv2-r800-cmp-$$.sh" || fail "could not copy netv2_cmp_step.sh"
[ "$(md5sum < "/tmp/nv2-r800-cmp-$$.sh" | cut -c1-8)" = a80c7da2 ] || fail "netv2_cmp_step.sh md5 is not a80c7da2"
CMP_WORKERS=5 timeout 21600 bash "/tmp/nv2-r800-cmp-$$.sh" 0 "$SEED" "$K" "$D/final" "$BD" > "$K/run.log" 2>&1; rc=$?
rm -f "/tmp/nv2-r800-cmp-$$.sh"
[ "$rc" = 0 ] || fail "h2h exited $rc (log $K/run.log)"
grep -qx "G9s $A" "$K/NETS.txt" || fail "NETS.txt: A is not $A"
grep -qx "G9p $B605" "$K/NETS.txt" || fail "NETS.txt: B is not $B605"
m=$(sed -n 's/^md5 g9s \([0-9a-f]*\) g9p \([0-9a-f]*\)$/\1 \2/p' "$K/NETS.txt")
[ "${m:0:8} ${m:33:8}" = "$got $B605_MD5" ] || fail "NETS.txt md5s '$m' are not $got / $B605_MD5"
grep -q "^--- all pairs played" "$K/RESULT.txt" || fail "RESULT.txt incomplete"
read -r x y <<< "$(sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$K/RESULT.txt")"
case "${x:-}" in
  10) v="IMPROVING: \"G9q still improving\" (\"cd800 > cd605\" SUCCESS: the 95% repeated CI's lower bound > 50%)" ;;
  20) v="NOT SHOWN: \"cd800 > cd605\" FUTILITY (a gain of >= 2 points is unlikely)" ;;
  *) v="NOT SHOWN: \"cd800 > cd605\" undecided (exit '${x:-?}')" ;;
esac
case "${y:-}" in 10) w="; \"cd605 > cd800\" SUCCESS (cd800 is weaker)" ;; 20) w="; \"cd605 > cd800\" FUTILITY" ;; *) w="; \"cd605 > cd800\" exit '${y:-?}'" ;; esac
{ echo "R800 PREREG reading (runs/g9q-cd800/PREREG.txt $PRE_MD5), read mechanically by scripts/netv2_g9q/r800-chain.sh"
  echo "R800 $v$w"
  echo "A = G9q-cd@800 $(basename "$A") (md5 $got) vs B = G9q-cd@605M 00605037107.pt ($B605_MD5); raw T 0.5; seed $SEED; RESULT.txt md5 $(md5sum < "$K/RESULT.txt" | cut -c1-8) (labels there: G9s = A, G9p = B)"
  grep -A1 '^DECISION OF RECORD' "$K/RESULT.txt"
  sed -n '/^--- nested, G9s > G9p:/,/^--- round-robin, G9s > G9p:/p' "$K/RESULT.txt" | grep -E "^--- nested|DECISION|estimate|repeated CI|overrun"
  grep -A1 '^--- all pairs played' "$K/RESULT.txt"; } > "$D/READING.txt.tmp" && mv -f "$D/READING.txt.tmp" "$D/READING.txt" && chmod a-w "$D/READING.txt"
st "READING: $(sed -n 2p "$D/READING.txt")"
st "chain done"
