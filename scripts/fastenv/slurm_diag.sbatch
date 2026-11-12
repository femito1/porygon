#!/bin/bash
# Diagnostic A/B on one 24-CPU long node, both from scratch, gen4, the exact thesis recipe in fp32/TF32 (no bf16),
# self-play only, checkpoints every 2.5M steps, sharing the GPU:
#   F: fastenv (6 workers x 48 battles) + AsyncPPO          O: old pipeline (10 local servers, 60 poke-env battles,
#      --async-envs --overlap-update) + AsyncPPO
# Question: does the old run's per-sample learning (old@2.5M) reproduce on fastenv (F) and on today's async code (O)?
#   sbatch --nodelist=vm-kn-a16 scripts/fastenv/slurm_diag.sbatch [HOURS=1.7]
#SBATCH --job-name=ws-fastenv-diag
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
HOURS=${1:-1.7}
G=/vol/bitbucket/fr1226/gloria
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PYTHONPATH=$ROOT/src
echo "node $(hostname) cpus $(nproc)"; nvidia-smi -L || true
find $G/.venv/lib/python3.12/site-packages/{torch,poke_env,stable_baselines3,numpy} -type f -size +100k -exec cat {} + > /dev/null 2>&1
PS=${TMPDIR:-/tmp}/ps-$SLURM_JOB_ID
rsync -a --exclude logs --exclude .git $G/pokemon-showdown/ "$PS/" && mkdir -p "$PS/logs/repl"
BASE=$((20000 + SLURM_JOB_ID % 20000)); NS=10
for ((i = 0; i < NS; i++)); do
  (cd "$PS" && setsid nohup node pokemon-showdown start --skip-build --no-security $((BASE + i)) < /dev/null > "$ROOT/logs/showdown-$SLURM_JOB_ID-$i.log" 2>&1 &)
done
for ((i = 0; i < NS; i++)); do
  for ((t = 0; t < 900; t++)); do grep -q listening "$ROOT/logs/showdown-$SLURM_JOB_ID-$i.log" 2>/dev/null && break; sleep 1; done
done
sleep 20
COMMON="--format gen4randombattle --total-steps 150000000 --max-hours $HOURS --overlap-update --eval-every 0 --checkpoint-every 2500000"
F=runs/diag-fast-$SLURM_JOB_ID; O=runs/diag-old-$SLURM_JOB_ID; mkdir -p $F $O
$PY -m porygon.train --fastenv 6x48 $COMMON --run-dir $F > $F/train.log 2>&1 &
PF=$!
$PY -m porygon.train --async-envs --n-battles 60 --ports $BASE-$((BASE + NS - 1)) $COMMON --run-dir $O > $O/train.log 2>&1 &
PO=$!
wait $PF; echo "fast exit $?"; wait $PO; echo "old exit $?"
pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security" || true
