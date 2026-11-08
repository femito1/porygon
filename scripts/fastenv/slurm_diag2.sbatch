#!/bin/bash
# Diagnostic 2 (one 24-CPU long node, fastenv only, gen4 from scratch, thesis recipe in fp32, lr schedule of the
# Oct-19 reference run: --total-steps 52M), both arms sharing the GPU:
#   L: SB3's own lockstep PPO over fastenv_vec.FastVecEnv (poke-env wait-step semantics) - the reference run's setup
#   A: AsyncPPO + overlapped update over fastenv_sb3.FastBattles - the async collection design
# Then the evaluation list (cmds file, 2nd arg). Question: at 2.5M, does L match old@2.5M (~50%)? Is A below L?
#   sbatch --exclude=vm-kn-a16 [--dependency=afterany:JOB] scripts/fastenv/slurm_diag2.sbatch HOURS CMDS
#SBATCH --job-name=ws-fastenv-diag2
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
HOURS=${1:-3}; CMDS=${2:-}
G=/vol/bitbucket/fr1226/gloria
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PYTHONPATH=$ROOT/src
echo "node $(hostname) cpus $(nproc) $(lscpu | grep 'Model name' | sed 's/ \+/ /g')"; nvidia-smi -L || true
find $G/.venv/lib/python3.12/site-packages/{torch,poke_env,stable_baselines3,numpy} -type f -size +100k -exec cat {} + > /dev/null 2>&1
if [ -n "$CMDS" ]; then  # evaluations first (minutes)
  while read -r line; do
    [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
    echo "=== $line"; eval "$line"
  done < "$CMDS"
fi
COMMON="--format gen4randombattle --total-steps 52000000 --max-hours $HOURS --eval-every 0 --checkpoint-every 2500000"
L=runs/diag2-lock-$SLURM_JOB_ID; A=runs/diag2-async-$SLURM_JOB_ID; mkdir -p $L $A
$PY -m porygon.train --fastenv 5x48 --fastenv-lockstep $COMMON --run-dir $L > $L/train.log 2>&1 &
PL=$!
$PY -m porygon.train --fastenv 5x48 --overlap-update $COMMON --run-dir $A > $A/train.log 2>&1 &
PA=$!
wait $PL; echo "lockstep exit $?"; wait $PA; echo "async exit $?"
