#!/bin/bash
# netv2 2026-11-14: raw nets vs Foul Play (+ G9q-cd vs our search), ONE slot job (runs/rawfp/PREREG.txt): (b) G9q-cd@600 vs
# Foul Play gen9 -> (d) G9q-cd@600 vs pure gen9 search -> (c) R1b-F@300M vs Foul Play gen4 -> RESULTS.txt -> one ledger
# RELEASE line. Stages are independent: a failure writes runs/rawfp/FAILED-<stage> and the job goes on.
#   cd $C && sbatch --dependency=afterany:1172518 scripts/netv2_rawfp/rawfp.sbatch
#SBATCH --job-name=ws-netv2-rawfp
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; cd "$C" || exit 1
D=$C/runs/rawfp; mkdir -p "$D"; J=${SLURM_JOB_ID:?}
PRE_MD5=f0a508e3
FP9=/vol/bitbucket/fr1226/porygon-ws-data/code/scripts/netv2_foulplay_fp9.sbatch; FP9_MD5=7476be8b
FP4=$C/scripts/netv2_foulplay.sbatch; FP4_MD5=35e7e30b
VS=$C/scripts/netv2_vs_search.sbatch; VS_MD5=7d74e0ad
R4=$C/runs/g4r1bf/prior-R1bF-300M.pt; R4_MD5=c3ec9433
st() { echo "$(date -u +%FT%T) $*" | tee -a "$D/STATUS"; }
md5of() { md5sum < "$1" | cut -c1-8; }
failstage() { st "FAILED-$1: $2"; echo "$(date -u +%FT%T) $2" >> "$D/FAILED-$1"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
newest() { ls -td "$C"/runs/"$1"-*-"$J" 2>/dev/null | head -n 1; }
st "job $J on $(hostname -s) (${SLURM_CPUS_ON_NODE:-?} CPUs) start"
[ "$(md5of "$D/PREREG.txt")" = "$PRE_MD5" ] && [ ! -w "$D/PREREG.txt" ] || { st "PREREG md5 / mode wrong: nothing run"; outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | raw-vs-FP chain did not run (PREREG check failed); slot FREE"; exit 1; }
res_b="not run"; res_d="not run"; res_c="not run"
# --- the cd600 net (b, d) ---
NET=""
for i in $(seq 1 180); do
  NET=$(ls "$C"/runs/g9q-cd600/final/*.pt 2>/dev/null | head -n 1)
  [ -n "$NET" ] && break
  [ -e "$C/runs/g9q-cd600/FAILED" ] && break
  sleep 60
done
if [ -z "$NET" ]; then
  failstage b "no runs/g9q-cd600/final/*.pt (cd600 FAILED: $(tail -n 1 "$C/runs/g9q-cd600/FAILED" 2>/dev/null))"; failstage d "skipped with (b): no cd600 final"
  res_b="SKIPPED (no cd600 final)"; res_d="SKIPPED (no cd600 final)"
else
  st "net G9q-cd600 = $NET (md5 $(md5of "$NET"))"
  # --- (b) ---
  if [ "$(md5of "$FP9")" != "$FP9_MD5" ]; then failstage b "fp9 harness md5"; res_b="FAILED (harness md5)"
  else
    st "(b) G9q-cd600 vs Foul Play gen9, 400 games, seed 843443"
    if bash "$FP9" "$NET" 400 --gen 9 --seed 843443 --temperature 0.5 > "$D/b.log" 2>&1 && R=$(newest foulplay-net) && [ -f "$R/result.txt" ]; then
      res_b="$(head -n 1 "$R/result.txt")"; st "(b) $res_b"
    else failstage b "harness exited / no result.txt (log $D/b.log)"; res_b="FAILED"; fi
  fi
  # --- (d) ---
  if [ "$(md5of "$VS")" != "$VS_MD5" ]; then failstage d "vs-search harness md5"; res_d="FAILED (harness md5)"
  else
    st "(d) G9q-cd600 vs pure gen9 search, 400 games, seed 738164"
    if bash "$VS" "$NET,temperature=0.5" 400 --gen 9 --seed 738164 > "$D/d.log" 2>&1 && R=$(newest vs-search) && [ -f "$R/result.txt" ]; then
      res_d="$(head -n 1 "$R/result.txt")"
      p=$(grep -oE ': [0-9.]+%' <<< "$res_d" | head -n 1 | tr -dc '0-9.')
      if [ -n "$p" ] && awk -v x="$p" 'BEGIN { exit !(x >= 50) }'; then res_d="$res_d => >= 50%: PROPOSE the 1000-game confirmation (standing rule)"; else res_d="$res_d => < 50%: no confirmation"; fi
      st "(d) $res_d"
    else failstage d "harness exited / no result.txt (log $D/d.log)"; res_d="FAILED"; fi
  fi
fi
# --- (c) ---
if [ "$(md5of "$FP4")" != "$FP4_MD5" ] || [ "$(md5of "$R4")" != "$R4_MD5" ]; then failstage c "harness / net md5"; res_c="FAILED (md5)"
else
  st "(c) R1b-F@300M vs Foul Play gen4, 400 games, seed 701384"
  if bash "$FP4" "$R4" 400 --gen 4 --seed 701384 --temperature 0.5 > "$D/c.log" 2>&1 && R=$(newest foulplay-net) && [ -f "$R/result.txt" ] && grep -q "prior-R1bF-300M" "$R/result.txt"; then
    res_c="$(head -n 1 "$R/result.txt")"; st "(c) $res_c"
  else failstage c "harness exited / no result.txt (log $D/c.log)"; res_c="FAILED"; fi
fi
{ echo "raw nets vs Foul Play + G9q-cd vs search (runs/rawfp/PREREG.txt $PRE_MD5), job $J; measurements only"
  echo "(b) G9q-cd@600 ($NET) vs Foul Play gen9 [fp9 harness, seed 843443; G9p-cd 943190 46.2 +- 4.6]: $res_b"
  echo "(d) G9q-cd@600 vs pure gen9 search [seed 738164; G9p-cd 845624 50.5 +- 4.5]: $res_d"
  echo "(c) R1b-F@300M vs Foul Play gen4 [netv2_foulplay, seed 701384; v1@600M 701384 46.2 +- 3.8 (600 g)]: $res_c"
  ls "$D"/FAILED-* 2>/dev/null | sed 's/^/failed: /'; } > "$D/RESULTS.txt.tmp" && mv -f "$D/RESULTS.txt.tmp" "$D/RESULTS.txt" && chmod a-w "$D/RESULTS.txt"
st "RESULTS written"
short() { grep -oE ': [0-9.]+% ± [0-9.]+%' <<< "$1" | head -n 1 | sed 's/^: //' || true; }
outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | raw-vs-FP chain done: (b) G9q-cd@600 vs FP gen9 $(short "$res_b" || echo "$res_b"); (d) vs pure gen9 search $(short "$res_d"); (c) R1b-F@300M vs FP gen4 $(short "$res_c"); failed stages: $(ls "$D"/FAILED-* 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')none-else; runs/rawfp/RESULTS.txt; slot FREE"
st "chain done (RELEASE line in $C/LEDGER-OUTBOX.md)"
