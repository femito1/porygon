#!/bin/bash
# netv2 2026-11-15: 1000-game confirmations of raw G9q-cd@605M, ONE slot job (runs/conf605/PREREG.txt): (i) vs our pure
# gen9 search (seed 738201 = G9p-cd's 864111) -> (ii) vs Foul Play gen9 (fp9 harness, seed 5150417) -> RESULTS.txt ->
# one ledger RELEASE line. Stages are independent: a failure writes runs/conf605/FAILED-<stage> and the job goes on.
#   cd $C && sbatch scripts/netv2_conf605/conf605.sbatch
#SBATCH --job-name=ws-netv2-conf605
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; cd "$C" || exit 1
D=$C/runs/conf605; mkdir -p "$D"; J=${SLURM_JOB_ID:?}
PRE_MD5=517b679c
NET=$C/runs/g9q-cd600/final/00605037107.pt; NET_MD5=7c28fac8
FP9=/vol/bitbucket/fr1226/porygon-ws-data/code/scripts/netv2_foulplay_fp9.sbatch; FP9_MD5=7476be8b
VS=$C/scripts/netv2_vs_search.sbatch; VS_MD5=7d74e0ad
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
st "job $J on $(hostname -s) (${SLURM_CPUS_ON_NODE:-?} CPUs) start"
[ "$(md5of "$D/PREREG.txt")" = "$PRE_MD5" ] && [ ! -w "$D/PREREG.txt" ] || { st "PREREG md5 / mode wrong: nothing run"; echo "PREREG check" > "$D/FAILED-prereg"; outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | G9q-cd@605M confirmation chain did not run (PREREG check failed); slot FREE"; exit 1; }
[ "$(md5of "$NET")" = "$NET_MD5" ] || { st "net md5 wrong: nothing run"; echo "net md5" > "$D/FAILED-net"; outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | G9q-cd@605M confirmation chain did not run (net md5); slot FREE"; exit 1; }
st "net G9q-cd@605M = $NET (md5 $NET_MD5)"
res_i="not run"; res_ii="not run"; rd_i=""; rd_ii=""
# --- (i) vs our pure gen9 search ---
if [ "$(md5of "$VS")" != "$VS_MD5" ]; then failstage i "vs-search harness md5"; res_i="FAILED (harness md5)"
else
  st "(i) G9q-cd@605M vs pure gen9 search, 1000 games, seed 738201"
  if bash "$VS" "$NET,temperature=0.5" 1000 --gen 9 --seed 738201 > "$D/i.log" 2>&1 && R=$(newest vs-search) && [ -n "$R" ] && [ -f "$R/result.txt" ]; then
    res_i="$(head -n 1 "$R/result.txt")"; rd_i="$(reading "$res_i")"; st "(i) $res_i => $rd_i"
  else failstage i "harness exited / no result.txt (log $D/i.log)"; res_i="FAILED"; fi
fi
# --- (ii) vs Foul Play gen9 ---
if [ "$(md5of "$FP9")" != "$FP9_MD5" ]; then failstage ii "fp9 harness md5"; res_ii="FAILED (harness md5)"
else
  st "(ii) G9q-cd@605M vs Foul Play gen9, 1000 games, seed 5150417"
  if bash "$FP9" "$NET" 1000 --gen 9 --seed 5150417 --temperature 0.5 > "$D/ii.log" 2>&1 && R=$(newest foulplay-net) && [ -n "$R" ] && [ -f "$R/result.txt" ]; then
    res_ii="$(head -n 1 "$R/result.txt")"; rd_ii="$(reading "$res_ii")"; st "(ii) $res_ii => $rd_ii"
  else failstage ii "harness exited / no result.txt (log $D/ii.log)"; res_ii="FAILED"; fi
fi
{ echo "G9q-cd@605M 1000-game confirmations (runs/conf605/PREREG.txt $PRE_MD5), job $J; measurements, readings per the PREREG"
  echo "(i) vs pure gen9 search [vs_search, seed 738201; pairs with G9p-cd 864111 43.4 +- 2.8]: $res_i${rd_i:+ => $rd_i}"
  echo "(ii) vs Foul Play gen9 [fp9 harness, seed 5150417 (new)]: $res_ii${rd_ii:+ => $rd_ii}"
  for f in "$D"/FAILED-*; do if [ -e "$f" ]; then echo "failed: $f"; fi; done
  true; } > "$D/RESULTS.txt.tmp"
mv -f "$D/RESULTS.txt.tmp" "$D/RESULTS.txt" && chmod a-w "$D/RESULTS.txt"
st "RESULTS written"
fl=""; for f in "$D"/FAILED-*; do if [ -e "$f" ]; then fl="$fl $(basename "$f")"; fi; done
outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | G9q-cd@605M 1000-g confirmations done: (i) vs pure gen9 search $(short "$res_i") ${rd_i:-$res_i}; (ii) vs FP gen9 $(short "$res_ii") ${rd_ii:-$res_ii}; failed stages:${fl:- none}; runs/conf605/RESULTS.txt; slot FREE"
st "chain done (RELEASE line in $C/LEDGER-OUTBOX.md)"
