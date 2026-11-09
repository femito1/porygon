#!/bin/bash
# Pre-registered gen9 label-teacher test (data stream, orchestrator-approved 2026-11-09; STATUS.md "Search-label
# targets and teacher strength"). Search vs search on the duplicate server through netv2's harness
# (porygon.v2.evaluate specs), 400 games = 200 duplicate pairs per arm, 95% CI from pair means:
#   a. G9p-prior search (v2 net as the prior, prior_weight 0.25, 4 x 500 ms) vs pure search (4 x 500 ms)
#   b. pure search 4 x 1000 ms vs pure search 4 x 500 ms
# Both sides: stock eval, 4 samples. Decision: the label generator moves to the config with the best win-rate gain
# per CPU-second. If (a) is a clear gain, the next slot use is G9p-prior search vs Foul Play gen9.
#   sbatch [--dependency=afterany:J] scripts/slurm_teacher_test9.sbatch PRIOR_PT|POINTER.txt
#SBATCH --job-name=ws-data-teacher9
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-data/logs/slurm-%j.out
set -uo pipefail
PRIOR=$1
# PRIOR may be a .txt pointer file holding the checkpoint path, read at job start (so the "latest" net is used).
case "$PRIOR" in *.txt) PRIOR=$(grep -v "^#" "$PRIOR" | head -1 | tr -d "[:space:]") ;; esac
NV=/vol/bitbucket/fr1226/porygon-ws-netv2
LOG=/vol/bitbucket/fr1226/porygon-ws-data/logs/teacher9-$SLURM_JOB_ID.txt
[ -f "$PRIOR" ] || { echo "no prior net at $PRIOR" | tee -a "$LOG"; exit 2; }
echo "prior $PRIOR ($(stat -c '%y' "$PRIOR")) sha1 $(sha1sum "$PRIOR" | cut -c1-12), job $SLURM_JOB_ID $(date -Is)" | tee -a "$LOG"
export SLURM_SUBMIT_DIR=$NV
cd "$NV"
COMMON=samples=4,eval_config=stock,workers=11  # 2 sides x 11 engine workers on 24 CPUs (one shard, 3 battles at a
H2H=(--gen 9 --shards 1 --concurrency 3)       # time: 3 x 2 x 4 sample searches ~ 22 workers, as the search benchmarks)
stage() {  # NAME SCRIPT ARGS...: one netv2 stage; appends its result line to LOG
  local name=$1; shift
  echo "=== $(date -Is) arm $name: $*" | tee -a "$LOG"
  local before; before=$(ls -d runs/*/ 2>/dev/null | sort)
  bash "$@"
  local out; out=$(comm -13 <(echo "$before") <(ls -d runs/*/ 2>/dev/null | sort) | grep -- "-$SLURM_JOB_ID/$" | tail -1)
  echo "$name ${out:-?}: $(head -1 "${out:-/dev/null}result.txt" 2>/dev/null)" | tee -a "$LOG"
}
stage a scripts/netv2_h2h.sbatch "search:$PRIOR,ms=500,$COMMON,prior_weight=0.25" \
  "search:none,ms=500,$COMMON,prior_weight=0" 400 "${H2H[@]}"
stage b scripts/netv2_h2h.sbatch "search:none,ms=1000,$COMMON,prior_weight=0" \
  "search:none,ms=500,$COMMON,prior_weight=0" 400 "${H2H[@]}"
echo "=== $(date -Is) done" | tee -a "$LOG"
