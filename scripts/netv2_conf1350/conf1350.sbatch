#!/bin/bash
# netv2 2026-11-19: F1350 = 1000-game confirmation of raw G9q-cd@1350 vs Foul Play gen9, ONE slot job
# (runs/g9q-cd1350/PREREG.txt) = conf800.sbatch (9fb4e117) stage (ii) only, with the net lookup (the one file in
# runs/g9q-cd1350/final/, md5 = the branch driver's STATUS line), an R1350 gate (runs/g9q-cd1350/READING.txt line 2 must
# start "R1350 IMPROVING: SUCCESS"), the seed 8135222 and paths changed -> RESULTS.txt -> one ledger RELEASE line.
# Submitted HELD; r1350-chain.sh releases it on R1350 SUCCESS (or cancels it):
#   cd $C && sbatch --hold --dependency=afterany:1228316 scripts/netv2_conf1350/conf1350.sbatch
#SBATCH --job-name=ws-netv2-conf1350
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; cd "$C" || exit 1
D=$C/runs/conf1350; mkdir -p "$D"; J=${SLURM_JOB_ID:?}
PRE_MD5=342e63f8
ND=$C/runs/g9q-cd1350; PRE=$ND/PREREG.txt
FP9=/vol/bitbucket/fr1226/porygon-ws-data/code/scripts/netv2_foulplay_fp9.sbatch; FP9_MD5=7476be8b
SEED_II=8135222
st() { echo "$(date -u +%FT%T) $*" | tee -a "$D/STATUS"; }
md5of() { md5sum < "$1" | cut -c1-8; }
failstage() { st "FAILED-$1: $2"; echo "$(date -u +%FT%T) $2" >> "$D/FAILED-$1"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
newest() { ls -td "$C"/runs/"$1"-*-"$J" 2>/dev/null | head -n 1; }
# reading: CONFIRMED iff the lower 95% bound (point - half-width of the pair-mean interval) > 50
reading() {
  local p h
  p=$(grep -oE ': [0-9.]+% ± [0-9.]+%' <<< "$1" | head -n 1 | sed -E 's/^: ([0-9.]+)% ± ([0-9.]+)%/\1/')
  h=$(grep -oE ': [0-9.]+% ± [0-9.]+%' <<< "$1" | head -n 1 | sed -E 's/^: ([0-9.]+)% ± ([0-9.]+)%/\2/')
  if [ -z "$p" ] || [ -z "$h" ]; then echo "UNREADABLE result line"; return 0; fi
  awk -v p="$p" -v h="$h" 'BEGIN { lo = p - h; hi = p + h; printf "[%.1f, %.1f] => %s", lo, hi, (lo > 50 ? "CONFIRMED (lower 95% bound > 50%)" : "NOT CONFIRMED (lower 95% bound <= 50%)") }'
}
short() { grep -oE ': [0-9.]+% ± [0-9.]+%' <<< "$1" | head -n 1 | sed 's/^: //' || true; }
norun() { st "$1: nothing run"; echo "$(date -u +%FT%T) $1" > "$D/FAILED-$2"; outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | F1350 (G9q-cd@1350 vs FP gen9) did not run ($1); slot FREE"; exit 1; }
st "job $J on $(hostname -s) (${SLURM_CPUS_ON_NODE:-?} CPUs) start"
if [ "$(md5of "$PRE")" != "$PRE_MD5" ] || [ -w "$PRE" ]; then norun "PREREG md5 / mode wrong" prereg; fi
# the R1350 gate (PREREG): play only after a SUCCESS reading
if ! sed -n 2p "$ND/READING.txt" 2>/dev/null | grep -q '^R1350 IMPROVING: SUCCESS'; then norun "runs/g9q-cd1350/READING.txt is missing or not SUCCESS" gate; fi
# the net: the one file in runs/g9q-cd1350/final/, md5 = the driver's STATUS line
n=$(ls "$ND"/final/*.pt 2>/dev/null | wc -l)
if [ "$n" != 1 ]; then norun "$n files in runs/g9q-cd1350/final (want 1)" net; fi
NET=$(ls "$ND"/final/*.pt)
want=$(grep "G9q-cd1350 final at" "$ND/STATUS" 2>/dev/null | tail -n 1 | sed -n 's/.*(md5 \([0-9a-f]*\)).*/\1/p')
NET_MD5=$(md5of "$NET")
if [ -z "$want" ] || [ "$NET_MD5" != "$want" ]; then norun "net md5 $NET_MD5 != the driver's STATUS md5 '${want:-none}'" net; fi
grep -q "(md5 $NET_MD5)" "$ND/READING.txt" || norun "net md5 $NET_MD5 is not the one in READING.txt" net
st "net G9q-cd@1350 = $NET (md5 $NET_MD5); R1350 SUCCESS"
res_ii="not run"; rd_ii=""
if [ "$(md5of "$FP9")" != "$FP9_MD5" ]; then failstage ii "fp9 harness md5"; res_ii="FAILED (harness md5)"
else
  st "F1350 G9q-cd@1350 vs Foul Play gen9, 1000 games, seed $SEED_II"
  if bash "$FP9" "$NET" 1000 --gen 9 --seed "$SEED_II" --temperature 0.5 > "$D/ii.log" 2>&1 && R=$(newest foulplay-net) && [ -n "$R" ] && [ -f "$R/result.txt" ]; then
    res_ii="$(head -n 1 "$R/result.txt")"; rd_ii="$(reading "$res_ii")"; st "F1350 $res_ii => $rd_ii"
  else failstage ii "harness exited / no result.txt (log $D/ii.log)"; res_ii="FAILED"; fi
fi
{ echo "F1350: G9q-cd@1350 1000-game confirmation vs Foul Play gen9 (runs/g9q-cd1350/PREREG.txt $PRE_MD5), job $J; net $(basename "$NET") md5 $NET_MD5; measurement, reading per the PREREG"
  echo "vs Foul Play gen9 [fp9 harness, seed $SEED_II (new)]: $res_ii${rd_ii:+ => $rd_ii}"
  echo "context only (not a paired test): G9q-cd@805 vs Foul Play gen9 57.8% ± 2.8% [55.0, 60.6] (runs/conf800/RESULTS.txt)"
  for f in "$D"/FAILED-*; do if [ -e "$f" ]; then echo "failed: $f"; fi; done
  true; } > "$D/RESULTS.txt.tmp"
mv -f "$D/RESULTS.txt.tmp" "$D/RESULTS.txt" && chmod a-w "$D/RESULTS.txt"
st "RESULTS written"
fl=""; for f in "$D"/FAILED-*; do if [ -e "$f" ]; then fl="$fl $(basename "$f")"; fi; done
outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | F1350 G9q-cd@1350 1000-g confirmation done: vs FP gen9 $(short "$res_ii") ${rd_ii:-$res_ii}; failed stages:${fl:- none}; runs/conf1350/RESULTS.txt; slot FREE"
st "chain done (RELEASE line in $C/LEDGER-OUTBOX.md)"
