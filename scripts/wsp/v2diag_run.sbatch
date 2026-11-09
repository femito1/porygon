#!/bin/bash
# Small CPU job for v2 diagnosis: runs the given shell script (sourced env: v2diag/src snapshot first).
#   sbatch --partition=long -c 4 scripts/wsp/v2diag_run.sbatch SCRIPT.sh
#SBATCH --job-name=ws-wsp-v2diag
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export PYTHONPATH=$WSP/v2diag/src:$WSP/scripts/wsp
export OMP_NUM_THREADS=4
S=$(mktemp /tmp/v2diag-XXXX.sh); cp "$1" $S; bash $S; rm -f $S
