#!/bin/bash
# netv2 2026-11-19: READING R1350 (runs/g9q-cd1350/PREREG.txt): G9q-cd@1350 (A, cmp_step's "G9s" label) vs G9q-cd@805
# (B, "G9p") in-process via scripts/netv2_cmp_step.sh (a80c7da2, unmodified; STEP 0 = no step constraint; seq-gain design
# f12fec7c; seed 8135061) = r800-chain.sh (7a98d34d) with A / B / seed / dirs changed, plus the F1350 gate: on SUCCESS it
# releases the HELD slot job runs/conf1350/JOBID (scontrol release); on NOT SHOWN it cancels it (scancel) and appends a
# WITHDRAWN line to $C/LEDGER-OUTBOX.md; on FAILED the job stays held for main. A driver step in the data host: waits
# <= 10 h for the one file in runs/g9q-cd1350/final/ (md5 = the branch driver's STATUS line; stops on
# runs/g9q-cd1350/FAILED), plays (limit 6 h), reads mechanically -> runs/g9q-cd1350/READING.txt (r--).
# Failure: runs/g9q-cd1350/r1350/FAILED.
#   setsid nohup srun --jobid=1172609 --overlap --ntasks=1 -c 6 --gres=gpu:1 --mem=14G -J nv2-r1350 \
#     bash -c 'cp $C/scripts/netv2_g9q/r1350-chain.sh /tmp/nv2-r1350-$$.sh && exec bash /tmp/nv2-r1350-$$.sh' \
#     > $C/logs/r1350-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=${MOCK_C:-/vol/bitbucket/fr1226/porygon-ws-netv2}; D=$C/runs/g9q-cd1350; K=$D/r1350; mkdir -p "$K"
PRE_MD5=342e63f8; SEED=8135061; BD=$C/runs/g9q-cd800/final; B805=$BD/00805029873.pt; B805_MD5=d0c5b6cf
CMP_MD5=a80c7da2; WAIT_S=36000; POLL_S=60
if [ -n "${MOCK_C:-}" ]; then PRE_MD5=$MOCK_PRE_MD5; B805_MD5=$MOCK_B805_MD5; CMP_MD5=$MOCK_CMP_MD5; WAIT_S=$MOCK_WAIT_S; POLL_S=1; fi  # laptop mock only
FPJ_FILE=$C/runs/conf1350/JOBID
st() { echo "$(date -u +%FT%T) $*" | tee -a "$K/STATUS" >&2; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
fpj() { local j; j=$(cat "$FPJ_FILE" 2>/dev/null | tr -dc 0-9); echo "${j:-}"; }
fail() { st "FAILED: $* (F1350 job '$(fpj)' left as it is: HELD for main)"; echo "$(date -u +%FT%T) $*" >> "$K/FAILED"; exit 1; }
[ ! -e "$K/FAILED" ] || { st "runs/g9q-cd1350/r1350/FAILED exists: not running"; exit 1; }
if [ -e "$D/READING.txt" ]; then st "runs/g9q-cd1350/READING.txt exists: nothing to do"; exit 0; fi
if [ ! -s "$D/PREREG.txt" ] || [ -w "$D/PREREG.txt" ] || [ "$(md5sum < "$D/PREREG.txt" | cut -c1-8)" != "$PRE_MD5" ]; then fail "no read-only runs/g9q-cd1350/PREREG.txt with md5 $PRE_MD5"; fi
if [ "$(ls "$BD"/*.pt 2>/dev/null | wc -l)" != 1 ] || [ "$(md5sum < "$B805" | cut -c1-8)" != "$B805_MD5" ]; then fail "runs/g9q-cd800/final must hold only 00805029873.pt (md5 $B805_MD5)"; fi
trap 'fail "killed by SIGTERM (the host job ended or the step was cancelled): rerun the same pre-registration in the next host"' TERM
T0=$(date +%s); st "chain start on $(hostname -s) (job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}); F1350 job '$(fpj)'; waiting <= $((WAIT_S / 3600)) h for runs/g9q-cd1350/final/*.pt"
A=""
while [ -z "$A" ]; do
  if [ -e "$D/FAILED" ]; then fail "runs/g9q-cd1350/FAILED: no cd1350 net ($(tail -n 1 "$D/FAILED"))"; fi
  n=$(ls "$D"/final/*.pt 2>/dev/null | wc -l)
  if [ "$n" -gt 1 ]; then fail "$n files in runs/g9q-cd1350/final (want 1)"; fi
  if [ "$n" = 1 ]; then A=$(ls "$D"/final/*.pt); break; fi
  if [ $(( $(date +%s) - T0 )) -ge "$WAIT_S" ]; then fail "no cd1350 final within $((WAIT_S / 3600)) h"; fi
  sleep "$POLL_S"
done
want=""; for i in $(seq 1 20); do want=$(grep "G9q-cd1350 final at" "$D/STATUS" 2>/dev/null | tail -n 1 | sed -n 's/.*(md5 \([0-9a-f]*\)).*/\1/p'); [ -n "$want" ] && break; sleep 15; done
got=$(md5sum < "$A" | cut -c1-8)
if [ -z "$want" ] || [ "$got" != "$want" ]; then fail "cd1350 final $A md5 $got != the driver's STATUS md5 '${want:-none}'"; fi
st "A = G9q-cd@1350 $A (md5 $got); B = G9q-cd@805 $B805 ($B805_MD5); seed $SEED; 5 workers"
cp "$C/scripts/netv2_cmp_step.sh" "/tmp/nv2-r1350-cmp-$$.sh" || fail "could not copy netv2_cmp_step.sh"
[ "$(md5sum < "/tmp/nv2-r1350-cmp-$$.sh" | cut -c1-8)" = "$CMP_MD5" ] || fail "netv2_cmp_step.sh md5 is not $CMP_MD5"
CMP_WORKERS=5 timeout 21600 bash "/tmp/nv2-r1350-cmp-$$.sh" 0 "$SEED" "$K" "$D/final" "$BD" > "$K/run.log" 2>&1; rc=$?
rm -f "/tmp/nv2-r1350-cmp-$$.sh"
[ "$rc" = 0 ] || fail "h2h exited $rc (log $K/run.log)"
grep -qx "G9s $A" "$K/NETS.txt" || fail "NETS.txt: A is not $A"
grep -qx "G9p $B805" "$K/NETS.txt" || fail "NETS.txt: B is not $B805"
m=$(sed -n 's/^md5 g9s \([0-9a-f]*\) g9p \([0-9a-f]*\)$/\1 \2/p' "$K/NETS.txt")
[ "${m:0:8} ${m:33:8}" = "$got $B805_MD5" ] || fail "NETS.txt md5s '$m' are not $got / $B805_MD5"
grep -q "^--- all pairs played" "$K/RESULT.txt" || fail "RESULT.txt incomplete"
read -r x y <<< "$(sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$K/RESULT.txt")"
case "${x:-}" in
  10) v="IMPROVING: SUCCESS (\"cd1350 > cd805\" SUCCESS: the 95% repeated CI's lower bound > 50%) => G9q-cd@1350 is the new deployment-net candidate; F1350 released" ;;
  20) v="NOT SHOWN: \"cd1350 > cd805\" FUTILITY (a gain of >= 2 points is unlikely) => cd@805 stays; F1350 cancelled" ;;
  *) v="NOT SHOWN: \"cd1350 > cd805\" undecided (exit '${x:-?}') => cd@805 stays; F1350 cancelled" ;;
esac
case "${y:-}" in 10) w="; \"cd805 > cd1350\" SUCCESS (cd1350 is weaker: a warning for main)" ;; 20) w="; \"cd805 > cd1350\" FUTILITY" ;; *) w="; \"cd805 > cd1350\" exit '${y:-?}'" ;; esac
{ echo "R1350 PREREG reading (runs/g9q-cd1350/PREREG.txt $PRE_MD5), read mechanically by scripts/netv2_g9q/r1350-chain.sh"
  echo "R1350 $v$w"
  echo "A = G9q-cd@1350 $(basename "$A") (md5 $got) vs B = G9q-cd@805 00805029873.pt ($B805_MD5); raw T 0.5; seed $SEED; RESULT.txt md5 $(md5sum < "$K/RESULT.txt" | cut -c1-8) (labels there: G9s = A, G9p = B)"
  grep -A1 '^DECISION OF RECORD' "$K/RESULT.txt"
  sed -n '/^--- nested, G9s > G9p:/,/^--- round-robin, G9s > G9p:/p' "$K/RESULT.txt" | grep -E "^--- nested|DECISION|estimate|repeated CI|overrun"
  grep -A1 '^--- all pairs played' "$K/RESULT.txt"; } > "$D/READING.txt.tmp" && mv -f "$D/READING.txt.tmp" "$D/READING.txt" && chmod a-w "$D/READING.txt"
st "READING: $(sed -n 2p "$D/READING.txt")"
# --- F1350 gate (PREREG): release on SUCCESS, cancel + WITHDRAWN on NOT SHOWN
J=$(fpj)
if [ -z "$J" ]; then st "F1350: no job id in runs/conf1350/JOBID: nothing released or cancelled (main to act)"; st "chain done"; exit 0; fi
q=$(squeue -h -j "$J" -o "%T %r" 2>/dev/null | head -n 1)
if [ "${x:-}" = 10 ]; then
  if scontrol release "$J" 2>>"$K/STATUS"; then st "F1350: released slot job $J (was '$q'; dependency afterany:1228316 still applies)"
  else st "F1350: scontrol release $J FAILED (state '$q'): main to release by hand"; fi
else
  if scancel "$J" 2>>"$K/STATUS"; then st "F1350: cancelled slot job $J (was '$q')"
  else st "F1350: scancel $J FAILED (state '$q'): main to cancel by hand"; fi
  outbox "$(date -u +%FT%H:%MZ) | netv2 | WITHDRAWN | $J | F1350 (G9q-cd@1350 vs FP gen9) not played: R1350 $(sed -n 2p "$D/READING.txt" | cut -c7- | cut -d'=' -f1 | sed 's/ *$//'); runs/g9q-cd1350/READING.txt; slot claim withdrawn"
fi
st "chain done"
