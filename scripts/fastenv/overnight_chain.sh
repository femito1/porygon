#!/bin/bash
# Overnight chain inside fastenv's job (2026-11-09): wait for the S arm's trainer to exit, then
#   1. per-sample equality evaluation (F vs S at 1M / 2M, T 0.5, + bots) and the collector-level comparison;
#   2. Oct-19 gap Test 1: arm L-W = diag2's L command with poke-env-exact wait rows, 1 h (2.5M checkpoint);
#   3. L-W vs Oct-19@2.5M (2000 games) and L-W vs L@2.5M (1000 games).
#   in_job.sh JOB 'scripts/fastenv/overnight_chain.sh'
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
G=/vol/bitbucket/fr1226/gloria
PY=$G/.venv/bin/python
while pgrep -u "$USER" -f "^$PY -u -m porygon.v2.train --collector showdown" > /dev/null; do sleep 30; done
echo "$(date -u +%T) S done"; tail -2 runs/g9smoke-sd-onpol/train.log | cut -c1-120
scripts/fastenv/eval_g9_persample.sh runs/g9smoke-fe-onpol runs/g9smoke-sd-onpol 16
PYTHONPATH=$ROOT/combo/src $PY scripts/fastenv/compare_runs.py runs/g9smoke-fe-onpol runs/g9smoke-sd-onpol --at 0.5e6 1e6 2e6
echo "$(date -u +%T) persample done"
export PYTHONPATH=$ROOT/src OMP_NUM_THREADS=1 PORYGON_PS=$G/pokemon-showdown PORYGON_NODE=$G/.node/bin/node PATH=$G/.node/bin:$PATH
W=runs/gapW-${SLURM_JOB_ID:-x}; mkdir -p $W
$PY -m porygon.train --fastenv 5x48 --fastenv-lockstep --fastenv-wait-rows --format gen4randombattle --total-steps 52000000 \
  --max-hours 1.0 --eval-every 0 --checkpoint-every 2500000 --run-dir $W > $W/train.log 2>&1
echo "$(date -u +%T) L-W exit $?"
$PY scripts/fastenv/tb_scalars.py $W time/fps train/explained_variance train/entropy_loss train/clip_fraction train/approx_kl
c=$(ls $W/checkpoints/ppo_*_steps.zip 2>/dev/null | awk -F'ppo_|_steps' '{d=$2-2500000; if (d<0) d=-d; if (d<5000) print}' | head -1)
EV="$PY -m porygon.fastenv_eval --format gen4randombattle --workers 15 --battles 32"
if [ -n "$c" ]; then
  echo "=== $c vs old-ckpts/ppo_2499840_steps.pt"; $EV --games 2000 --a "$c" --b old-ckpts/ppo_2499840_steps.pt 2>&1 | tail -1
  echo "=== $c vs runs/diag2-lock-710562/checkpoints/ppo_2499840_steps.zip"
  $EV --games 1000 --a "$c" --b runs/diag2-lock-710562/checkpoints/ppo_2499840_steps.zip 2>&1 | tail -1
  $EV --games 600 --a "$c" --b heuristic 2>&1 | tail -1
fi
echo "$(date -u +%T) chain done"
