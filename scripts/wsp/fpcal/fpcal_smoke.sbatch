#!/bin/bash
# wsp fpcal smoke (11-17, outside the slot): the frozen $WSP/fpcal/code harness (fixed iterations + fp_iters_check.py)
# on 4 games at N 50,000 and 4 games at N 800,000 (K 2 Foul Play instances, seed 8800223, not a slot run). PASS iff
# each arm wrote ITERS_OK, no ITERS_MISMATCH, and its check saw N-count states. Writes $WSP/fpcal/smoke/RESULT and
# moves the run dirs to $WSP/fpcal/smoke/.
#   cd $WSP && sbatch fpcal/code/fpcal_smoke.sbatch
#SBATCH --job-name=ws-wsp-fpsmoke
#SBATCH --partition=training
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
WSP=/vol/bitbucket/fr1226/porygon-ws-wsp; C=/vol/bitbucket/fr1226/porygon-ws-netv2
D=$WSP/fpcal; K=$D/code; S=$D/smoke; J=${SLURM_JOB_ID:?}
NET=$C/runs/g9q-cd800/final/00805029873.pt
mkdir -p "$S"; rm -f "$S/RESULT"
ok=1; lines=()
for it in 50000 800000; do
  bash -c "cd $C && SLURM_SUBMIT_DIR=$C bash $K/fp9_iters.sbatch $NET 4 --gen 9 --k 2 --seed 8800223 --fp-iters $it --temperature 0.5" > "$S/arm-$it-$J.log" 2>&1
  R=$(ls -td "$D"/runs/foulplay-net-*-"$J"-"$it" 2>/dev/null | head -n 1)
  if [ -z "$R" ]; then ok=0; lines+=("N $it: no run dir"); continue; fi
  mv "$R" "$S/"; R=$S/$(basename "$R")
  games=$(wc -l < "$R/results.jsonl" 2>/dev/null || echo 0)
  if [ -f "$R/ITERS_MISMATCH" ]; then ok=0; lines+=("N $it: MISMATCH $(cat "$R/ITERS_MISMATCH")"); continue; fi
  if ! grep -q "OK (.*N [1-9]" "$R/ITERS_OK" 2>/dev/null; then ok=0; lines+=("N $it: no ITERS_OK with N states"); continue; fi
  lines+=("N $it: $games games; $(cat "$R/ITERS_OK"); fp0.log counts: $(grep -oE 'Iterations [0-9]+: [0-9]+' "$R/fp0.log" | awk '{print $3}' | sort -n | uniq -c | awk '{printf "%sx%s ", $1, $2}')")
done
{ [ "$ok" = 1 ] && echo "PASS job $J $(date -u +%FT%TZ)" || echo "FAIL job $J $(date -u +%FT%TZ)"; printf '%s\n' "${lines[@]}"; } > "$S/RESULT"
cat "$S/RESULT"
[ "$ok" = 1 ]
