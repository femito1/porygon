#!/bin/bash
# fastenv dev allocation on the long partition: holds the stream's slot; work runs inside it with
#   scripts/fastenv/in_job.sh JOBID 'CMD'      (srun --overlap; PY / PORYGON_NODE / PORYGON_PS preset)
#   sbatch [--nodelist=vm-kn-a16] scripts/fastenv/slurm_hold.sbatch
# 20 CPUs (not 24) leaves the long QOS room (48/user) for the benchmark slot + wsp's 4-CPU watcher.
#SBATCH --job-name=ws-fastenv-dev
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=44G
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
G=/vol/bitbucket/fr1226/gloria
echo "node $(hostname) SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?} nproc $(env -u OMP_NUM_THREADS nproc)"; nvidia-smi -L || true
# pre-read the venv's big libraries once (slow shared disk)
SITE=$($G/.venv/bin/python -c 'import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))')
( cd "$SITE" && timeout 1200 find torch triton nvidia -name '*.so*' -size +1M -exec cat {} + > /dev/null ) || true
echo "$(date -u +%T) pre-read done"
sleep infinity
