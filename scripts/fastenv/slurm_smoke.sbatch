#!/bin/bash
# fastenv smoke test on one 24-CPU node: (1) same-node env-only throughput of the old pipeline (local Showdown
# servers + poke-env workers, random actions) and of fastenv; (2) gen4 PPO from scratch on fastenv (v1 encoder,
# MaskedPolicy, AsyncPPO with overlapped update). Checkpoints go to runs/smoke-<job>/ for porygon.fastenv_eval.
#   sbatch --nodelist=vm-kn-a16 scripts/fastenv/slurm_smoke.sbatch [HOURS=10] [train args]
#SBATCH --job-name=ws-fastenv-smoke
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
HOURS=${1:-10}; shift || true
G=/vol/bitbucket/fr1226/gloria
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PYTHONPATH=$ROOT/src
echo "node $(hostname) cpus $(nproc) $(lscpu | grep 'Model name' | sed 's/ \+/ /g')"; nvidia-smi -L || true
cat $G/pokemon-showdown/dist/data/*.js $G/pokemon-showdown/dist/data/random-battles/*/*.js* > /dev/null 2>&1 || true
node scripts/fastenv/simspeed.js gen4randombattle 200

# (1a) old pipeline, env only: servers from a node-local copy of Showdown (gloria stays read-only)
PS=${TMPDIR:-/tmp}/ps-$SLURM_JOB_ID
rsync -a --exclude logs --exclude .git $G/pokemon-showdown/ "$PS/" && mkdir -p "$PS/logs/repl"
BASE=$((20000 + SLURM_JOB_ID % 20000)); NS=6
for ((i = 0; i < NS; i++)); do
  (cd "$PS" && setsid nohup node pokemon-showdown start --skip-build --no-security $((BASE + i)) < /dev/null > "$ROOT/logs/showdown-$SLURM_JOB_ID-$i.log" 2>&1 &)
done
for ((i = 0; i < NS; i++)); do
  for ((t = 0; t < 600; t++)); do grep -q listening "$ROOT/logs/showdown-$SLURM_JOB_ID-$i.log" 2>/dev/null && break; sleep 1; done
done
sleep 20
for fmt in gen4randombattle gen9randombattle; do
  echo "=== old pipeline env-only $fmt, 46 poke-env workers, $NS servers"
  timeout 600 $PY scripts/fastenv/profile_showdown_env.py --port $BASE --servers $NS --workers 46 --seconds 90 --format $fmt 2>&1 | tail -3
done
pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security" || true
# (1b) fastenv env only, same node
for fmt in gen4randombattle gen9randombattle; do
  echo "=== fastenv env-only $fmt"
  $PY scripts/fastenv/bench_fastenv.py --format $fmt --workers 23 --battles 48 --seconds 60 --async
done

# (2) smoke training run
RUN=runs/smoke-$SLURM_JOB_ID; mkdir -p "$RUN"
echo "=== training: $RUN, $HOURS h"
$PY -m porygon.train --fastenv 22x48 --format gen4randombattle --total-steps 150000000 --max-hours "$HOURS" \
  --rollout 40960 --batch-size 4096 --n-epochs 2 --autocast bf16 --overlap-update --eval-every 0 \
  --checkpoint-every 10000000 --run-dir "$RUN" "$@" 2>&1 | grep -v "^|  \|^---" | tee "$RUN/train.log" | tail -n 200
echo "done"
