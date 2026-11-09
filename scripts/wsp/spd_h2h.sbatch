#!/bin/bash
# In-process raw-network head-to-head, one shard per job or array task (scripts/wsp/spd_h2h.py on fastenv's
# simulator: no Showdown server, no poke-env, nets on the GPU). Shard K plays duplicate pairs [K*N, (K+1)*N).
#   sbatch [--array=0-3%1] scripts/wsp/spd_h2h.sbatch OUT A B N [spd_h2h.py run options: --seed S --format F ...]
#   sbatch [--array=...] scripts/wsp/spd_h2h.sbatch --list FILE   # one "OUT A B N [options]" per line, run in turn
#                                                                    # (one job start for several pairings)
#   python scripts/wsp/spd_h2h.py merge OUT          # pooled win rate, pair-mean 95% interval (light: login node ok)
# A / B: PATH[,temperature=T] (.zip / pool .pt; 500M aliases), v2:PATH[,temperature=T] (add --v2-src
# $WSP/spd/netv2-<commit>/src), random; B also heuristic | max_power. Relative paths are relative to $WSP.
# Throughput (8 CPUs = 7 workers x 48 battles + this process, one T4/A16): see .orchestra/wsp/STATUS.md "spd".
#SBATCH --job-name=ws-wsp-spd
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=1
[ "${1:-}" = "--list" ] || [ $# -ge 4 ] || { echo "usage: sbatch spd_h2h.sbatch OUT A B N_PAIRS_PER_SHARD [run options] | --list FILE" >&2; exit 2; }
FE=${SPD_FASTENV:-$WSP/spd/fastenv-b9b8a85}
export SPD_FASTENV_JS=$FE/fastenv/js PORYGON_PS=${PORYGON_PS:-$GLORIA/pokemon-showdown} PORYGON_NODE=$GLORIA/.node/bin/node
[ -f "$SPD_FASTENV_JS/host.js" ] || { echo "no vendored fastenv in $FE (run scripts/wsp/spd_vendor.sh)" >&2; exit 1; }
NCPU=$(python -c 'import os; print(len(os.sched_getaffinity(0)))')
echo "$(date -u +%FT%TZ) node $(hostname), $NCPU CPUs (SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-?}), $(nvidia-smi -L 2>/dev/null | head -1)"
[ "$NCPU" -ge 2 ] || { echo "only $NCPU CPU(s) allocated" >&2; exit 1; }
# the shared disk stalls page faults on big files: pull the Python env into the page cache with sequential reads
find $GLORIA/.venv/lib/python3.12/site-packages/{torch,stable_baselines3,gymnasium} -type f \( -name '*.py' -o -name '*.so*' \) \
  -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
if [ "$1" = "--list" ]; then
  rc=0
  while read -r line; do
    [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
    eval "set -- $line"
    echo "=== $(date -u +%T) $line"
    python scripts/wsp/spd_h2h.py run --a "$2" --b "$3" --pairs "$4" --out "$1" "${@:5}" < /dev/null || rc=1
  done < "$2"
  exit $rc
fi
OUT=$1; A=$2; B=$3; N=$4; shift 4
python scripts/wsp/spd_h2h.py run --a "$A" --b "$B" --pairs "$N" --out "$OUT" "$@"
