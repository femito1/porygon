#!/bin/bash
#SBATCH --job-name=ws-data-convert
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time=2-00:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-data/logs/slurm-%j.out
# Convert battles to first-person training data (scripts/convert_replays.py) on the cluster.
#
#   sbatch scripts/slurm_convert.sbatch MODE FORMAT OUT INPUT... [-- convert_replays args]
#     MODE: corpus (human replays with input logs: re-simulated), runs (search-label runs with streams),
#           logged-obs (old --log-oracle benchmark runs, v1 only)
#   e.g. sbatch scripts/slurm_convert.sbatch corpus gen9randombattle $D/datasets/human \
#            $D/raw/corpus/gen9randombattle/part-*.jsonl.gz -- --min-rating 1500
#        sbatch scripts/slurm_convert.sbatch runs gen9randombattle $D/datasets/searchlabels $D/runs/searchlabels-gen9-*
# Inside an already running job of the stream instead (shares its CPUs):
#   srun --jobid=JOB --overlap --ntasks=1 bash scripts/slurm_convert.sbatch MODE FORMAT OUT INPUT... -- --workers 3
#
# Uses gloria's venv and node read-only, our code first on PYTHONPATH, gen9 engine build first for gen9.
# Showdown builds per commit (corpus mode): scripts/build_showdown_versions.sh -> $D/showdown/builds.
set -euo pipefail
D=/vol/bitbucket/fr1226/porygon-ws-data
G=/vol/bitbucket/fr1226/gloria
CODE=${CODE:-$D/code}
MODE=$1 FORMAT=$2 OUT=$3
shift 3
INPUTS=()
while (($#)) && [[ $1 != -- ]]; do INPUTS+=("$1"); shift; done
[[ ${1:-} == -- ]] && shift
GEN=${FORMAT:3:1}
export PORYGON_GEN=$GEN PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=$CODE/src
[[ $GEN == 9 ]] && export PYTHONPATH=$G/.engine-gen9:$PYTHONPATH
export PATH=$G/.node/bin:$PATH
export PORYGON_COMMIT=$(cat "$CODE/COMMIT" 2>/dev/null || echo unknown)
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-4} > 1 ? ${SLURM_CPUS_PER_TASK:-4} - 1 : 1 ))
mkdir -p "$OUT" "$D/logs"
cd "$CODE"
echo "convert $MODE $FORMAT -> $OUT (${#INPUTS[@]} inputs, $WORKERS workers, code $PORYGON_COMMIT) on $(hostname)"
exec "$G/.venv/bin/python" scripts/convert_replays.py "$MODE" "${INPUTS[@]}" --format "$FORMAT" --out "$OUT" \
  --builds "$D/showdown/builds" --workers "$WORKERS" "$@"
