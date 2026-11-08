#!/bin/bash
# Records the fixed v2 diagnostic position set (v2diag_record.py): v1 500M vs itself, K servers x N games.
#   sbatch --partition=long -c 6 scripts/wsp/v2diag_record.sbatch N K
#SBATCH --job-name=ws-wsp-v2pos
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export PYTHONPATH=$WSP/v2diag/src:$WSP/scripts/wsp
export OMP_NUM_THREADS=1
N=$1; K=$2
OUT=$WSP/v2diag/pos; mkdir -p "$OUT"
BASE=$(( 33000 + (SLURM_JOB_ID % 1000) * 8 ))
PS=$GLORIA/.showdown-eval
for ((i = 0; i < K; i++)); do
  (cd "$PS" && setsid nohup node pokemon-showdown start --skip-build --no-security $((BASE + i)) < /dev/null > "$OUT/showdown-$i.log" 2>&1 &)
done
for ((i = 0; i < K; i++)); do
  for ((t = 0; t < 2400; t++)); do grep -q listening "$OUT/showdown-$i.log" 2>/dev/null && break; sleep 0.5; done
done
trap 'for ((i=0;i<K;i++)); do pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $((BASE+i))\$" || true; done' EXIT
sleep 10
for ((i = 0; i < K; i++)); do
  python scripts/wsp/v2diag_record.py ckpt/ref/m500.zip --port $((BASE + i)) --n-battles "$N" --out "$OUT/p$i" > "$OUT/log-$i.txt" 2>&1 &
done
wait
cat "$OUT"/log-*.txt | tail -n 4
