#!/bin/bash
# fastenv helper job (the stream's one helper slot; dove via --partition=training): holds the allocation and runs
# queued step files, so long steps need no srun client on the login node.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch scripts/fastenv/slurm_helper.sbatch
# Queue a step: write jobq/NAME.sh.tmp, then mv it to jobq/NAME.sh (atomic). Each step runs once, in its own session
# (setsid), cwd = this dir, env as in in_job.sh (PY, PORYGON_NODE, PORYGON_PS = gloria's pinned build, PATH, PYTHONPATH):
#   log jobq/NAME.log, exit code jobq/NAME.rc (written at the end), session id jobq/NAME.pgid
#   (stop one step: in_job.sh JOB 'kill -- -$(cat jobq/NAME.pgid)'). The step file moves to jobq/done/.
# Quick commands still go through in_job.sh (srun --overlap).
#SBATCH --job-name=ws-fastenv-helper
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
G=/vol/bitbucket/fr1226/gloria
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PYTHONPATH=$PWD/src
echo "$(date -u +%FT%T) node $(hostname) SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?} cpuset $(taskset -pc $$ | sed 's/.*: //')"
nvidia-smi -L || true
# pre-read the venv's big libraries once, in the background (slow shared disk)
SITE=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
( cd "$SITE" && timeout 1200 find torch triton nvidia -name '*.so*' -size +1M -exec cat {} + > /dev/null; echo "$(date -u +%T) pre-read done" ) &
Q=$PWD/jobq; mkdir -p "$Q/done"
while true; do
  for f in "$Q"/*.sh; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .sh); mv "$f" "$Q/done/$n.sh"
    setsid bash -c 'echo $$ > "$1.pgid"; bash "$2" > "$1.log" 2>&1; echo $? > "$1.rc"' _ "$Q/$n" "$Q/done/$n.sh" &
    echo "$(date -u +%T) started $n"
  done
  sleep 15
done
