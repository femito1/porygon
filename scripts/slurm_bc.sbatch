#!/bin/bash
# Behaviour-cloning training (python -m porygon.bc train) on one GPU. Submit from the data-stream code dir
# (scripts/setup_ws_data.sh), never with --export:
#   cd /vol/bitbucket/fr1226/porygon-ws-data/code
#   sbatch scripts/slurm_bc.sbatch DATA_DIR [DATA_DIR ...] [-- porygon.bc train args]
#   e.g. sbatch scripts/slurm_bc.sbatch ../datasets/encoded/gen9randombattle/v1 -- --epochs 4 --min-rating 1500
#        sbatch --partition=a40 scripts/slurm_bc.sbatch DATA -- --max-hours 20     (then: -- --resume RUN_DIR)
# Output: runs/bc-<date>-<job>/ (= porygon-ws-data/runs/) unless --out is given. Shards are first copied to
# node-local disk when they fit (the shared disk is slow for many random reads; one sequential copy is fast).
# --mem: the stream loader holds --buffer-rows decisions (default 1M ~ 8 GB obs) twice plus prefetch; --loader
# ram needs ~8.2 KB x decisions (gen4/gen9 encoder, D = 4079) on top: raise --mem for it.
#SBATCH --job-name=ws-data-bc
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
DATA=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do DATA+=("$(realpath "$1")"); shift; done
[ "${1:-}" = -- ] && shift
[ ${#DATA[@]} -gt 0 ] || { echo "usage: sbatch scripts/slurm_bc.sbatch DATA_DIR [...] [-- train args]" >&2; exit 1; }
export PYTHONPATH=$ROOT/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PORYGON_COMMIT="$(cat "$ROOT/COMMIT" 2>/dev/null || echo unknown)"
OUT=runs/bc-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID
case " $* " in *" --out "*|*" --resume "*) OUT="" ;; esac
echo "host $(hostname), $(nproc) CPUs, $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "commit $PORYGON_COMMIT; data ${DATA[*]}; args $*"

# Copy shards to node-local disk when they fit in half of its free space.
LOCAL=${TMPDIR:-/tmp}/ws-data-bc-$SLURM_JOB_ID
trap 'rm -rf "$LOCAL"' EXIT
USE=()
need=$(du -sk "${DATA[@]}" | awk '{s += $1} END {print s}')
mkdir -p "$LOCAL"
free=$(df -k --output=avail "$LOCAL" | tail -1)
if [ "$need" -lt $(( free / 2 )) ]; then
  t=$(date +%s); i=0
  for d in "${DATA[@]}"; do mkdir -p "$LOCAL/$i"; cp -r "$d"/. "$LOCAL/$i/"; USE+=("$LOCAL/$i"); i=$((i + 1)); done
  echo "copied $(( need / 1024 )) MB of shards to $LOCAL in $(( $(date +%s) - t )) s"
else
  USE=("${DATA[@]}"); echo "shards ($(( need / 1024 )) MB) do not fit $LOCAL ($(( free / 1024 )) MB free): reading in place"
fi
# The torch/CUDA libraries page-fault very slowly off the shared disk: pull them in with sequential reads.
find .venv/lib/python3.12/site-packages/{torch,nvidia,triton} -name '*.so*' -print0 2>/dev/null \
  | xargs -0 -P4 -n16 cat > /dev/null || true

.venv/bin/python -m porygon.bc train "${USE[@]}" ${OUT:+--out "$OUT"} --device cuda \
  --load-workers $(( $(nproc) > 3 ? $(nproc) - 2 : 1 )) "$@"
