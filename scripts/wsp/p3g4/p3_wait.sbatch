#!/bin/bash
# wsp P3 READY poller (PREREG $W/council/p3-gen4/PREREG.txt). Submitted by p3_chain.sbatch stage i when arm (i) is
# done and netv2's R1BF_CD_READY is not there yet (the slot is then released). One short pass, then:
#   - $D/VERDICT.txt, $D/FAILED or $D/STOP exists          => stop (nothing to do / stopped by hand)
#   - netv2's $C/council/p3-gen4/FAILED exists               => $D/FAILED (no arm (ii)), stop
#   - netv2's R1BF_CD_READY exists                           => CLAIM line in $W/LEDGER-OUTBOX.md and submit stage ii
#     (p3_chain.sbatch ii) after every slot-sized job of ours already queued or running (>= 20 CPUs on training /
#     long: the shared benchmark slot), stop
#   - more than 48 h since arm (i) ended ($D/WAIT_START)     => $D/FAILED, stop
#   - else the next pass in 30 min.
#SBATCH --job-name=ws-wsp-p3wait
#SBATCH --partition=training
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/p3wait.out
#SBATCH --open-mode=append
set -uo pipefail
W=/vol/bitbucket/fr1226/porygon-ws-wsp; C=/vol/bitbucket/fr1226/porygon-ws-netv2
D=$W/council/p3-gen4; K=$D/code; NV=$C/council/p3-gen4; J=${SLURM_JOB_ID:?}; WAIT_H=48
st() { echo "$(date -u +%FT%TZ) [$J wait] $*" | tee -a "$D/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$W/LEDGER-OUTBOX.md" ) 9> "$W/LEDGER-OUTBOX.lock"; }
fail() { echo "$(date -u +%FT%TZ) job $J (READY poller): $1" >> "$D/FAILED"; st "FAILED: $1"; }
for f in VERDICT.txt FAILED STOP; do [ -e "$D/$f" ] && { echo "$D/$f exists: poller stops"; exit 0; }; done
if [ -f "$NV/FAILED" ]; then fail "netv2's R1b-F-cd cooldown FAILED ($(head -c 200 "$NV/FAILED")): no arm (ii)"; exit 0; fi
if [ -f "$NV/R1BF_CD_READY" ]; then
  deps=$(squeue -u "$USER" -h -o '%i|%P|%C|%c|%j' 2>/dev/null |
         awk -F'|' -v me="$J" '$1 != me && $2 ~ /training|long/ && ($3 >= 20 || $4 >= 20) {print $1}' | paste -sd: -)
  jid=$(cd "$W" && sbatch --parsable --partition=training ${deps:+--dependency=afterany:$deps} "$K/p3_chain.sbatch" ii) ||
    { fail "READY seen but stage ii could not be submitted"; exit 0; }
  outbox "$(date -u +%FT%H:%MZ) | wsp | CLAIM | $jid | P3 arm (ii) (council P3, main-signed 11-24; PREREG $D/PREREG.txt): raw R1b-F-cd T 0.5 vs Foul Play gen4 at a FIXED 200,000 iterations per state, 1000 games on arm (i)'s team pairs (seed 4031124), then VERDICT.txt; 24 CPUs training, ~2 h${deps:+; afterany:$deps}; RELEASE via \$W/LEDGER-OUTBOX.md"
  st "READY seen ($(head -c 200 "$NV/R1BF_CD_READY")): stage ii submitted as job $jid${deps:+ (afterany:$deps)}; CLAIM line written"
  exit 0
fi
start=$(cat "$D/WAIT_START" 2>/dev/null) || { fail "no WAIT_START"; exit 0; }
if [ $(( $(date -u +%s) - start )) -gt $(( WAIT_H * 3600 )) ]; then
  fail "no R1BF_CD_READY within $WAIT_H h of arm (i)'s end: no arm (ii)"; exit 0
fi
jid=$(cd "$W" && sbatch --parsable --partition=training --begin=now+30minutes "$K/p3_wait.sbatch") ||
  { fail "could not resubmit the READY poller"; exit 0; }
echo "$(date -u +%FT%TZ) READY absent; next pass $jid"
