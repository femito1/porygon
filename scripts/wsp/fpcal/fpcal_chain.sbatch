#!/bin/bash
# wsp fpcal (PREREG $WSP/fpcal/PREREG.txt): raw G9q-cd@805M (T 0.5) vs Foul Play gen9 at two FIXED Foul Play budgets,
# ONE benchmark-slot job (ledger-claimed): arm L (50,000 iterations per 1000 ms state) -> arm H (800,000) -> READING.txt
# (fpcal_read.py: each arm, paired differences vs the time-based conf800 run 1173380 on the same team pairs, slope per
# 4x budget) -> one ledger RELEASE line in $WSP/LEDGER-OUTBOX.md. A failed arm writes FAILED-<arm>; the job goes on.
#   cd $WSP && sbatch --dependency=afterany:1173672 fpcal/code/fpcal_chain.sbatch
#SBATCH --job-name=ws-wsp-fpcal
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
WSP=/vol/bitbucket/fr1226/porygon-ws-wsp; C=/vol/bitbucket/fr1226/porygon-ws-netv2
D=$WSP/fpcal; K=$D/code; J=${SLURM_JOB_ID:?}
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
NET=$C/runs/g9q-cd800/final/00805029873.pt; NET_SHA=e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f
REF=$C/runs/foulplay-net-20261116-173816-1173380   # time-based 1000 ms, same seed: the T point (median 210k/state)
SEED=8800222; ARMS="L:50000 H:800000"; PRE_MD5=5a68f926
st() { echo "$(date -u +%FT%T) $*" | tee -a "$D/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$WSP/LEDGER-OUTBOX.md" ) 9> "$WSP/LEDGER-OUTBOX.lock"; }
fail() { st "FAILED-$1: $2"; echo "$(date -u +%FT%T) $2" >> "$D/FAILED-$1"; }
st "job $J on $(hostname -s) (${SLURM_CPUS_ON_NODE:-?} CPUs) start"
if [ "$(md5sum < "$D/PREREG.txt" | cut -c1-8)" != "$PRE_MD5" ] || [ -w "$D/PREREG.txt" ]; then
  fail prereg "PREREG md5 / mode wrong"; outbox "$(date -u +%FT%H:%MZ) | wsp | RELEASE | $J | fpcal did not run (PREREG check); slot FREE"; exit 1
fi
if [ "$(sha256sum < "$NET" | cut -c1-64)" != "$NET_SHA" ]; then
  fail net "net sha256 mismatch"; outbox "$(date -u +%FT%H:%MZ) | wsp | RELEASE | $J | fpcal did not run (net sha); slot FREE"; exit 1
fi
READ_ARGS=("T=210000=$REF")
for arm in $ARMS; do
  name=${arm%%:*}; it=${arm#*:}
  st "arm $name: Foul Play fixed $it iterations per 1000 ms, 1000 games, seed $SEED"
  bash -c "cd $C && SLURM_SUBMIT_DIR=$C bash $K/fp9_iters.sbatch $NET 1000 --gen 9 --seed $SEED --fp-iters $it --temperature 0.5" > "$D/arm-$name.log" 2>&1
  R=$(ls -td "$D"/runs/foulplay-net-*-"$J"-"$it" 2>/dev/null | head -n 1)
  if [ -z "$R" ] || [ ! -f "$R/result.txt" ]; then fail "$name" "no result (see arm-$name.log)"; continue; fi
  if [ -f "$R/ITERS_MISMATCH" ]; then fail "$name" "iteration check: $(cat "$R/ITERS_MISMATCH")"; continue; fi
  st "arm $name: $(head -n 1 "$R/result.txt" | sed 's/^.*: //')"
  READ_ARGS+=("$name=$it=$R")
done
{ echo "fpcal reading (PREREG md5 $PRE_MD5), job $J, $(date -u +%FT%TZ)"; "$PY" "$K/fpcal_read.py" "${READ_ARGS[@]}"; } > "$D/READING.txt.tmp" 2>&1 &&
  mv "$D/READING.txt.tmp" "$D/READING.txt" && chmod a-w "$D/READING.txt" || fail read "fpcal_read.py failed"
st "READING: $(tr '\n' ' ' < "$D/READING.txt" 2>/dev/null | cut -c1-400)"
outbox "$(date -u +%FT%H:%MZ) | wsp | RELEASE | $J | fpcal done ($(ls "$D"/FAILED-* 2>/dev/null | wc -l) failed stages); slot FREE"
st "done"
