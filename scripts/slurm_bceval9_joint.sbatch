#!/bin/bash
# Joint pre-registered gen9 BC evaluation (data + netv2; plan in .orchestra/data/STATUS.md "Joint pre-registered BC
# evaluation"). One job on the shared long slot; every stage runs netv2's harness (porygon.v2.evaluate.load_player)
# from netv2's cluster dir, so all settings match. Duplicate pairs, both nets raw at T 0.5, 95% CI from pair means.
#   C. BC (v1) vs P1 (v2) head to head, 1000 games; the pair-mean winner is "the better net" for D
#   A. BC vs our standard gen9 search (MODEL none, stock eval, 4 x 1000 ms), 400 games
#   B. P1 vs the same search, 400 games
#   D. the better net vs Foul Play --gen 9, 400 games
#   sbatch [--dependency=afterany:J] scripts/slurm_bceval9_joint.sbatch BC_ZIP P1_PT
# BC_ZIP may not exist yet at submit time (training inside another job): the job waits up to 1 h for it.
#SBATCH --job-name=ws-data-bceval9
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-data/logs/slurm-%j.out
set -uo pipefail
BC=$1 P1=$2
NV=/vol/bitbucket/fr1226/porygon-ws-netv2
LOG=/vol/bitbucket/fr1226/porygon-ws-data/logs/bceval9-$SLURM_JOB_ID.txt
for _ in $(seq 60); do [ -f "$BC" ] && break; echo "waiting for $BC"; sleep 60; done
[ -f "$BC" ] || { echo "no BC net at $BC" | tee -a "$LOG"; exit 2; }
[ -f "$P1" ] || { echo "no P1 net at $P1" | tee -a "$LOG"; exit 2; }
export SLURM_SUBMIT_DIR=$NV
cd "$NV"
T=",temperature=0.5"
stage() {  # NAME SCRIPT ARGS...: runs one netv2 stage, appends its result line to LOG
  local name=$1; shift
  echo "=== $(date -Is) stage $name: $*" | tee -a "$LOG"
  local before; before=$(ls -d runs/*/ 2>/dev/null | sort)
  bash "$@"
  local out; out=$(comm -13 <(echo "$before") <(ls -d runs/*/ 2>/dev/null | sort) | grep -- "-$SLURM_JOB_ID/$" | tail -1)
  echo "$name ${out:-?}: $(head -1 "${out:-/dev/null}result.txt" 2>/dev/null)" | tee -a "$LOG"
}
echo "BC=$BC P1=$P1 job $SLURM_JOB_ID $(date -Is)" | tee -a "$LOG"
stage C scripts/netv2_h2h.sbatch "$BC$T" "$P1$T" 1000 --gen 9
RATE=$(grep "^C " "$LOG" | grep -o ': [0-9.]*%' | head -1 | tr -d ': %')
if [ -n "$RATE" ] && awk "BEGIN{exit !($RATE >= 50)}"; then BETTER=$BC; else BETTER=$P1; fi
echo "better net (stage C pair mean ${RATE:-?}% for BC): $BETTER" | tee -a "$LOG"
SEARCH=(ms=1000 samples=4 eval_config=stock prior_weight=0)
stage A scripts/netv2_vs_search.sbatch "$BC$T" 400 --gen 9 "${SEARCH[@]}"
stage B scripts/netv2_vs_search.sbatch "$P1$T" 400 --gen 9 "${SEARCH[@]}"
stage D scripts/netv2_foulplay.sbatch "$BETTER" 400 --gen 9 --temperature 0.5
echo "=== $(date -Is) done" | tee -a "$LOG"
