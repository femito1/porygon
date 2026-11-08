#!/bin/bash
# Profile the v2 PPO update and actor inference on one GPU (no battles): scripts/profile_v2_update.py.
#   sbatch --job-name=ws-netv2-prof scripts/netv2_profile.sbatch
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
export PYTHONPATH=$ROOT/src
FX=$ROOT/fixtures/gen4
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== baseline shape: 7 epochs x 1024, no move-slot trim"
$PY -u scripts/profile_v2_update.py "$FX" --epochs 7 --batch 1024 --updates 2 --model-json '{"trim_moves": false}'
echo "== 3 epochs x 4096, no trim"
$PY -u scripts/profile_v2_update.py "$FX" --epochs 3 --batch 4096 --updates 2 --model-json '{"trim_moves": false}'
echo "== 3 epochs x 4096, trim (default), profiled"
$PY -u scripts/profile_v2_update.py "$FX" --epochs 3 --batch 4096 --updates 2 --profile
echo "== 3 epochs x 4096, trim, compiled damage"
$PY -u scripts/profile_v2_update.py "$FX" --epochs 3 --batch 4096 --updates 2 --compile
