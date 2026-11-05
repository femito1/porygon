#!/bin/bash
# Raw-network head-to-heads (no search): every MODEL against REF, all pairings in parallel, one Showdown
# server each (porygon.tournament, sampled actions as in training). Random teams per battle.
#   sbatch scripts/slurm_h2h_nets.sbatch N_BATTLES REF MODEL [MODEL ...]
#SBATCH --job-name=porygon-h2h-nets
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
N=$1; REF=$2; shift 2
export PATH=$ROOT/.node/bin:$PATH
OUT=runs/h2h-nets-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT"
BASE=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 ))
scripts/start_servers.sh "$#" "$BASE"
trap 'pkill -u "$USER" -f "pokemon-showdown start --skip-build --no-security" || true' EXIT
i=0
for M in "$@"; do
  name=$(echo "$M" | sed 's#^runs/##; s#/checkpoints##; s#/#_#g; s#\.zip$##')
  .venv/bin/python -m porygon.tournament "$M" "$REF" --baselines --port $(( BASE + i )) --n-battles "$N" \
    --concurrency 24 > "$OUT/$name.txt" 2>&1 &
  i=$(( i + 1 ))
done
wait
for f in "$OUT"/*.txt; do echo "== $f"; grep -E " vs " "$f" | head -1; done | tee "$OUT/summary.txt"
