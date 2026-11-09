#!/bin/bash
# Supervised v2 pretraining arms (porygon.v2.pretrain) in one small GPU job, run one after another.
#   sbatch --job-name=ws-netv2-pre scripts/netv2_pretrain.sbatch OUT_PREFIX SHARD_GLOB GEN ARM...
# ARM = name:seed:model_json[|extra pretrain args], e.g. 'dmg:0:{"n_layers":4,"ffn_mult":1}' or
#   'c005:2:{"n_layers":4,"ffn_mult":1}|--value-coef 0.05 --no-bf16'. Output: runs/pretrain/OUT_PREFIX-<name>-s<seed>/.
# PRETRAIN_ARGS (env, default "--epochs 6 --val-frac 0.06") go before each arm's extra args (argparse: last wins).
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
export PYTHONPATH=$ROOT/src OMP_NUM_THREADS=4
PREFIX=$1; SHARDS=$2; GEN=$3; shift 3
echo "pretrain $PREFIX on $SHARDS (gen $GEN), git $(cat "$ROOT/GIT_COMMIT" 2>/dev/null), $(nvidia-smi --query-gpu=name --format=csv,noheader)"
for arm in "$@"; do
  extra=""
  case $arm in *"|"*) extra=${arm#*|}; arm=${arm%%|*} ;; esac
  name=${arm%%:*}; rest=${arm#*:}; seed=${rest%%:*}; cfg=${rest#*:}
  OUT=runs/pretrain/$PREFIX-$name-s$seed
  [ -f "$OUT/final.pt" ] && { echo "skip $OUT (done)"; continue; }
  echo "== $OUT  cfg=$cfg  extra=$extra"
  # shellcheck disable=SC2086
  $PY -u -m porygon.v2.pretrain --shards "$SHARDS" --gen "$GEN" --out "$OUT" --model-json "$cfg" --seed "$seed" \
    ${PRETRAIN_ARGS:---epochs 6 --val-frac 0.06} $extra 2>&1 | grep -vE '"event": "train"' | cut -c1-800 | tail -n 40
done
