#!/bin/bash
# porygon.train on fastenv in a 4-CPU a16 job (QOS normal cap): 3 Node workers + the trainer, GPU learner.
#   sbatch scripts/fastenv/slurm_train_dev.sbatch NAME HOURS [porygon.train args]
#SBATCH --job-name=ws-fastenv-train4
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
NAME=$1; HOURS=$2; shift 2
G=/vol/bitbucket/fr1226/gloria
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PYTHONPATH=$ROOT/src
echo "node $(hostname) cpus $(nproc) $(lscpu | grep 'Model name' | sed 's/ \+/ /g')"; nvidia-smi -L || true
RUN=runs/$NAME-$SLURM_JOB_ID; mkdir -p "$RUN"
$PY -m porygon.train --fastenv 3x48 --format gen4randombattle --total-steps 150000000 --max-hours "$HOURS" \
  --overlap-update --eval-every 0 --run-dir "$RUN" "$@" > "$RUN/train.log" 2>&1
echo "exit $?"
