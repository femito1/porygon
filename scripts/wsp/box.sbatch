#!/bin/bash
# wsp analysis box: holds 1 GPU + 8 CPUs; analysis commands run inside it with
#   srun --jobid=$JOB --overlap --ntasks=1 bash -lc 'source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh; python ...'
#SBATCH --job-name=ws-wsp-box
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=3-00:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
echo "box on $(hostname) $(date)"; nvidia-smi -L
# local scratch copy of the decoded data for fast reads
mkdir -p /tmp/wsp-$SLURM_JOB_ID
while true; do sleep 600; done
