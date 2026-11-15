#!/bin/bash
# netv2 2026-11-15: 1000-game confirmations of raw G9q-cd@800, ONE slot job (runs/conf800/PREREG.txt) = conf605.sbatch
# (562a8e68) with the net lookup (the one file in runs/g9q-cd800/final/, md5 = the branch driver's STATUS line; waits
# <= 3 h), new seeds and paths: (i) vs our pure gen9 search (seed 8800111) -> (ii) vs Foul Play gen9 (fp9 harness, seed
# 8800222) -> RESULTS.txt -> one ledger RELEASE line. A failed stage writes runs/conf800/FAILED-<stage>; the job goes on.
#   cd $C && sbatch --dependency=afterany:1172540 scripts/netv2_conf800/conf800.sbatch
#SBATCH --job-name=ws-netv2-conf800
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; cd "$C" || exit 1
D=$C/runs/conf800; mkdir -p "$D"; J=${SLURM_JOB_ID:?}
PRE_MD5=c6a38724
ND=$C/runs/g9q-cd800
FP9=/vol/bitbucket/fr1226/porygon-ws-data/code/scripts/netv2_foulplay_fp9.sbatch; FP9_MD5=7476be8b
VS=$C/scripts/netv2_vs_search.sbatch; VS_MD5=7d74e0ad
SEED_I=8800111; SEED_II=8800222
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
norun() { st "$1: nothing run"; echo "$(date -u +%FT%T) $1" > "$D/FAILED-$2"; outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | G9q-cd@800 confirmation chain did not run ($1); slot FREE"; exit 1; }
st "job $J on $(hostname -s) (${SLURM_CPUS_ON_NODE:-?} CPUs) start"
if [ "$(md5of "$D/PREREG.txt")" != "$PRE_MD5" ] || [ -w "$D/PREREG.txt" ]; then norun "PREREG md5 / mode wrong" prereg; fi
# the net: the one file in runs/g9q-cd800/final/ (wait <= 3 h while the branch may still run), md5 = the driver's STATUS line
NET=""; t0=$(date +%s)
while [ -z "$NET" ]; do
  if [ -e "$ND/FAILED" ]; then norun "runs/g9q-cd800/FAILED: no G9q-cd@800 net" net; fi
  n=$(ls "$ND"/final/*.pt 2>/dev/null | wc -l)
  if [ "$n" -gt 1 ]; then norun "$n files in runs/g9q-cd800/final (want 1)" net; fi
  if [ "$n" = 1 ]; then NET=$(ls "$ND"/final/*.pt); break; fi
  if [ $(( $(date +%s) - t0 )) -ge 10800 ]; then norun "no G9q-cd@800 net after 3 h" net; fi
  if [ $(( ($(date +%s) - t0) % 1800 )) -lt 60 ]; then st "waiting for runs/g9q-cd800/final/*.pt"; fi
  sleep 60
done
want=""; for i in $(seq 1 20); do want=$(grep "G9q-cd800 final at" "$ND/STATUS" 2>/dev/null | tail -n 1 | sed -n 's/.*(md5 \([0-9a-f]*\)).*/\1/p'); [ -n "$want" ] && break; sleep 15; done
NET_MD5=$(md5of "$NET")
if [ -z "$want" ] || [ "$NET_MD5" != "$want" ]; then norun "net md5 $NET_MD5 != the driver's STATUS md5 '${want:-none}'" net; fi
st "net G9q-cd@800 = $NET (md5 $NET_MD5)"
res_i="not run"; res_ii="not run"; rd_i=""; rd_ii=""
# --- (i) vs our pure gen9 search ---
if [ "$(md5of "$VS")" != "$VS_MD5" ]; then failstage i "vs-search harness md5"; res_i="FAILED (harness md5)"
else
  st "(i) G9q-cd@800 vs pure gen9 search, 1000 games, seed $SEED_I"
  if bash "$VS" "$NET,temperature=0.5" 1000 --gen 9 --seed "$SEED_I" > "$D/i.log" 2>&1 && R=$(newest vs-search) && [ -n "$R" ] && [ -f "$R/result.txt" ]; then
    res_i="$(head -n 1 "$R/result.txt")"; rd_i="$(reading "$res_i")"; st "(i) $res_i => $rd_i"
  else failstage i "harness exited / no result.txt (log $D/i.log)"; res_i="FAILED"; fi
fi
# --- (ii) vs Foul Play gen9 ---
if [ "$(md5of "$FP9")" != "$FP9_MD5" ]; then failstage ii "fp9 harness md5"; res_ii="FAILED (harness md5)"
else
  st "(ii) G9q-cd@800 vs Foul Play gen9, 1000 games, seed $SEED_II"
  if bash "$FP9" "$NET" 1000 --gen 9 --seed "$SEED_II" --temperature 0.5 > "$D/ii.log" 2>&1 && R=$(newest foulplay-net) && [ -n "$R" ] && [ -f "$R/result.txt" ]; then
    res_ii="$(head -n 1 "$R/result.txt")"; rd_ii="$(reading "$res_ii")"; st "(ii) $res_ii => $rd_ii"
  else failstage ii "harness exited / no result.txt (log $D/ii.log)"; res_ii="FAILED"; fi
fi
{ echo "G9q-cd@800 1000-game confirmations (runs/conf800/PREREG.txt $PRE_MD5), job $J; net $(basename "$NET") md5 $NET_MD5; measurements, readings per the PREREG"
  echo "(i) vs pure gen9 search [vs_search, seed $SEED_I (new)]: $res_i${rd_i:+ => $rd_i}"
  echo "(ii) vs Foul Play gen9 [fp9 harness, seed $SEED_II (new)]: $res_ii${rd_ii:+ => $rd_ii}"
  for f in "$D"/FAILED-*; do if [ -e "$f" ]; then echo "failed: $f"; fi; done
  true; } > "$D/RESULTS.txt.tmp"
mv -f "$D/RESULTS.txt.tmp" "$D/RESULTS.txt" && chmod a-w "$D/RESULTS.txt"
st "RESULTS written"
fl=""; for f in "$D"/FAILED-*; do if [ -e "$f" ]; then fl="$fl $(basename "$f")"; fi; done
outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | G9q-cd@800 1000-g confirmations done: (i) vs pure gen9 search $(short "$res_i") ${rd_i:-$res_i}; (ii) vs FP gen9 $(short "$res_ii") ${rd_ii:-$res_ii}; failed stages:${fl:- none}; runs/conf800/RESULTS.txt; slot FREE"
st "chain done (RELEASE line in $C/LEDGER-OUTBOX.md)"
