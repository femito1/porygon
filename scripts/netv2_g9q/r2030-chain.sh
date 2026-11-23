#!/bin/bash
# netv2 2026-11-23: READING R2030 (runs/g9q-cd2030/PREREG.txt): G9q-cd@2030 (A, cmp_step's "G9s" label) vs G9q-cd@1350
# (B, "G9p") in-process via scripts/netv2_cmp_step.sh (a80c7da2, unmodified; STEP 0 = no step constraint; seq-gain design
# f12fec7c; seed 8203061) = r1350-chain.sh (33c06131) with A / B / seed / dirs / names changed, plus the placement change:
# this chain is a 1-CPU step in G9q's job 1483048 (ends 11-26), and it runs the harness as an srun --overlap step in the
# RUNNING ws-data-host job with the most time left (>= 1.5 h), so it survives data's host hand-over (1174320 ends 11-23
# 18:13Z; successor 1545813). If that host ends during the h2h, the harness runs once more in the next host (it skips the
# shards already complete: atomic files, fixed per-shard seeds => the same pairs and looks). F2030 gate: on SUCCESS it
# releases the HELD slot job runs/conf2030/JOBID (scontrol release); on NOT SHOWN it cancels it (scancel) and appends a
# WITHDRAWN line to $C/LEDGER-OUTBOX.md; on FAILED the job stays held for main. Waits <= 10 h for the one file in
# runs/g9q-cd2030/final/ (md5 = the branch driver's STATUS line; stops on runs/g9q-cd2030/FAILED), plays (limit 6 h),
# reads mechanically -> runs/g9q-cd2030/READING.txt (r--). Failure: runs/g9q-cd2030/r2030/FAILED.
#   setsid nohup srun --jobid=1483048 --overlap --ntasks=1 -c 1 --mem=1G -J nv2-r2030 \
#     bash -c 'cp $C/scripts/netv2_g9q/r2030-chain.sh /tmp/nv2-r2030-$$.sh && exec bash /tmp/nv2-r2030-$$.sh' \
#     > $C/logs/r2030-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=${MOCK_C:-/vol/bitbucket/fr1226/porygon-ws-netv2}; D=$C/runs/g9q-cd2030; K=$D/r2030; mkdir -p "$K"
PRE_MD5=7a68bf3e; SEED=8203061; BD=$C/runs/g9q-cd1350/final; B1350=$BD/01355036714.pt; B1350_MD5=6cf40c0c
CMP_MD5=a80c7da2; WAIT_S=36000; POLL_S=60; HOST_WAIT_S=86400; MIN_LEFT=5400; U=fr1226
if [ -n "${MOCK_C:-}" ]; then PRE_MD5=$MOCK_PRE_MD5; B1350_MD5=$MOCK_B1350_MD5; CMP_MD5=$MOCK_CMP_MD5; WAIT_S=$MOCK_WAIT_S; POLL_S=1; HOST_WAIT_S=$MOCK_WAIT_S; U=$(id -un); fi  # laptop mock only
FPJ_FILE=$C/runs/conf2030/JOBID
st() { echo "$(date -u +%FT%T) $*" | tee -a "$K/STATUS" >&2; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
fpj() { local j; j=$(cat "$FPJ_FILE" 2>/dev/null | tr -dc 0-9); echo "${j:-}"; }
fail() { st "FAILED: $* (F2030 job '$(fpj)' left as it is: HELD for main)"; echo "$(date -u +%FT%T) $*" >> "$K/FAILED"; exit 1; }
# the RUNNING ws-data-host job with the most time left, if >= MIN_LEFT s
pick_host() {
  local now best="" bl=0 j e l
  now=$(date +%s)
  while read -r j e; do
    [ -n "$j" ] || continue
    l=$(( $(date -d "$e" +%s 2>/dev/null || echo 0) - now ))
    if [ "$l" -ge "$MIN_LEFT" ] && [ "$l" -gt "$bl" ]; then best=$j; bl=$l; fi
  done < <(squeue -h -u "$U" -n ws-data-host -t RUNNING -o "%i %e" 2>/dev/null)
  echo "$best"
}
running() { [ "$(squeue -h -j "$1" -o %T 2>/dev/null | head -n 1)" = RUNNING ]; }
[ ! -e "$K/FAILED" ] || { st "runs/g9q-cd2030/r2030/FAILED exists: not running"; exit 1; }
if [ -e "$D/READING.txt" ]; then st "runs/g9q-cd2030/READING.txt exists: nothing to do"; exit 0; fi
if [ ! -s "$D/PREREG.txt" ] || [ -w "$D/PREREG.txt" ] || [ "$(md5sum < "$D/PREREG.txt" | cut -c1-8)" != "$PRE_MD5" ]; then fail "no read-only runs/g9q-cd2030/PREREG.txt with md5 $PRE_MD5"; fi
if [ "$(ls "$BD"/*.pt 2>/dev/null | wc -l)" != 1 ] || [ "$(md5sum < "$B1350" | cut -c1-8)" != "$B1350_MD5" ]; then fail "runs/g9q-cd1350/final must hold only 01355036714.pt (md5 $B1350_MD5)"; fi
trap 'fail "killed by SIGTERM (G9q job ended or the chain step was cancelled): rerun the same pre-registration"' TERM
T0=$(date +%s); st "chain start on $(hostname -s) (job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}); F2030 job '$(fpj)'; waiting <= $((WAIT_S / 3600)) h for runs/g9q-cd2030/final/*.pt"
A=""
while [ -z "$A" ]; do
  if [ -e "$D/FAILED" ]; then fail "runs/g9q-cd2030/FAILED: no cd2030 net ($(tail -n 1 "$D/FAILED"))"; fi
  n=$(ls "$D"/final/*.pt 2>/dev/null | wc -l)
  if [ "$n" -gt 1 ]; then fail "$n files in runs/g9q-cd2030/final (want 1)"; fi
  if [ "$n" = 1 ]; then A=$(ls "$D"/final/*.pt); break; fi
  if [ $(( $(date +%s) - T0 )) -ge "$WAIT_S" ]; then fail "no cd2030 final within $((WAIT_S / 3600)) h"; fi
  sleep "$POLL_S"
done
want=""; for i in $(seq 1 20); do want=$(grep "G9q-cd2030 final at" "$D/STATUS" 2>/dev/null | tail -n 1 | sed -n 's/.*(md5 \([0-9a-f]*\)).*/\1/p'); [ -n "$want" ] && break; sleep 15; done
got=$(md5sum < "$A" | cut -c1-8)
if [ -z "$want" ] || [ "$got" != "$want" ]; then fail "cd2030 final $A md5 $got != the driver's STATUS md5 '${want:-none}'"; fi
st "A = G9q-cd@2030 $A (md5 $got); B = G9q-cd@1350 $B1350 ($B1350_MD5); seed $SEED; 5 workers"
CMP=$K/cmp-step-copy.sh
cp "$C/scripts/netv2_cmp_step.sh" "$CMP.tmp" && mv -f "$CMP.tmp" "$CMP" || fail "could not copy netv2_cmp_step.sh"
[ "$(md5sum < "$CMP" | cut -c1-8)" = "$CMP_MD5" ] || fail "netv2_cmp_step.sh md5 is not $CMP_MD5"
T1=$(date +%s); rc=1
for att in 1 2; do
  H=""; T2=$(date +%s)
  while [ -z "$H" ]; do
    H=$(pick_host); [ -n "$H" ] && break
    if [ $(( $(date +%s) - T2 )) -ge "$HOST_WAIT_S" ]; then fail "no RUNNING ws-data-host job with >= $((MIN_LEFT / 60)) min left within $((HOST_WAIT_S / 3600)) h"; fi
    sleep "$POLL_S"
  done
  left=$(( 21600 - ($(date +%s) - T1) )); [ "$left" -gt 60 ] || fail "the 6 h h2h limit is used up"
  st "h2h attempt $att in data host job $H (ends $(squeue -h -j "$H" -o %e 2>/dev/null | head -n 1)); limit ${left}s; $(ls "$K"/h2h/shard-*.jsonl 2>/dev/null | wc -l) shards already complete"
  # a clean Slurm environment: srun from inside this job's step would inherit its CPU binding
  ( for v in $(compgen -e | grep -E '^SLURM'); do unset "$v"; done
    exec srun --jobid="$H" --overlap --ntasks=1 -c 6 --gres=gpu:1 --mem=14G -J nv2-r2030-h2h \
      env CMP_WORKERS=5 timeout "$left" bash "$CMP" 0 "$SEED" "$K" "$D/final" "$BD" ) >> "$K/run.log" 2>&1 < /dev/null; rc=$?
  [ "$rc" = 0 ] && break
  if [ "$att" = 1 ] && ! running "$H"; then st "h2h attempt 1: rc $rc and host job $H is no longer RUNNING: once more in the next host"; continue; fi
  fail "h2h exited $rc in host $H (attempt $att; log $K/run.log)"
done
[ "$rc" = 0 ] || fail "h2h exited $rc"
grep -qx "G9s $A" "$K/NETS.txt" || fail "NETS.txt: A is not $A"
grep -qx "G9p $B1350" "$K/NETS.txt" || fail "NETS.txt: B is not $B1350"
m=$(sed -n 's/^md5 g9s \([0-9a-f]*\) g9p \([0-9a-f]*\)$/\1 \2/p' "$K/NETS.txt")
[ "${m:0:8} ${m:33:8}" = "$got $B1350_MD5" ] || fail "NETS.txt md5s '$m' are not $got / $B1350_MD5"
grep -q "^--- all pairs played" "$K/RESULT.txt" || fail "RESULT.txt incomplete"
read -r x y <<< "$(sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$K/RESULT.txt")"
case "${x:-}" in
  10) v="IMPROVING: SUCCESS (\"cd2030 > cd1350\" SUCCESS: the 95% repeated CI's lower bound > 50%) => G9q-cd@2030 is the new deployment-net candidate; F2030 released" ;;
  20) v="NOT SHOWN: \"cd2030 > cd1350\" FUTILITY (a gain of >= 2 points is unlikely) => cd@1350 stays; F2030 cancelled" ;;
  *) v="NOT SHOWN: \"cd2030 > cd1350\" undecided (exit '${x:-?}') => cd@1350 stays; F2030 cancelled" ;;
esac
case "${y:-}" in 10) w="; \"cd1350 > cd2030\" SUCCESS (cd2030 is weaker: a warning for main)" ;; 20) w="; \"cd1350 > cd2030\" FUTILITY" ;; *) w="; \"cd1350 > cd2030\" exit '${y:-?}'" ;; esac
{ echo "R2030 PREREG reading (runs/g9q-cd2030/PREREG.txt $PRE_MD5), read mechanically by scripts/netv2_g9q/r2030-chain.sh"
  echo "R2030 $v$w"
  echo "A = G9q-cd@2030 $(basename "$A") (md5 $got) vs B = G9q-cd@1350 01355036714.pt ($B1350_MD5); raw T 0.5; seed $SEED; RESULT.txt md5 $(md5sum < "$K/RESULT.txt" | cut -c1-8) (labels there: G9s = A, G9p = B)"
  grep -A1 '^DECISION OF RECORD' "$K/RESULT.txt"
  sed -n '/^--- nested, G9s > G9p:/,/^--- round-robin, G9s > G9p:/p' "$K/RESULT.txt" | grep -E "^--- nested|DECISION|estimate|repeated CI|overrun"
  grep -A1 '^--- all pairs played' "$K/RESULT.txt"; } > "$D/READING.txt.tmp" && mv -f "$D/READING.txt.tmp" "$D/READING.txt" && chmod a-w "$D/READING.txt"
st "READING: $(sed -n 2p "$D/READING.txt")"
# --- F2030 gate (PREREG): release on SUCCESS, cancel + WITHDRAWN on NOT SHOWN
J=$(fpj)
if [ -z "$J" ]; then st "F2030: no job id in runs/conf2030/JOBID: nothing released or cancelled (main to act)"; st "chain done"; exit 0; fi
q=$(squeue -h -j "$J" -o "%T %r" 2>/dev/null | head -n 1)
if [ "${x:-}" = 10 ]; then
  if scontrol release "$J" 2>>"$K/STATUS"; then st "F2030: released slot job $J (was '$q')"
  else st "F2030: scontrol release $J FAILED (state '$q'): main to release by hand"; fi
else
  if scancel "$J" 2>>"$K/STATUS"; then st "F2030: cancelled slot job $J (was '$q')"
  else st "F2030: scancel $J FAILED (state '$q'): main to cancel by hand"; fi
  outbox "$(date -u +%FT%H:%MZ) | netv2 | WITHDRAWN | $J | F2030 (G9q-cd@2030 vs FP gen9 200k) not played: R2030 $(sed -n 2p "$D/READING.txt" | cut -c7- | cut -d'=' -f1 | sed 's/ *$//'); runs/g9q-cd2030/READING.txt; slot claim withdrawn"
fi
st "chain done"
