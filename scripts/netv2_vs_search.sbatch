#!/bin/bash
# Raw network vs our search (porygon.search.SearchPlayer), duplicate pairs; we report the network's win rate.
#   sbatch scripts/netv2_vs_search.sbatch NET N [--gen 4|9] [--search SPEC] [--seed TAG] [--shards S]
#                                         [--concurrency C] [key=value ...]
# Default search: gen4 = the 500M v1 net as the prior with the default (fitted) eval, 4 x 1 s, prior 0.25;
# gen9 = search:none (pure engine search, stock eval). key=value options are appended to the search spec, e.g.
# ms=500 samples=8 prior_weight=0 eval_config=stock (porygon.v2.evaluate docstring). --search replaces the spec
# (e.g. --search search:runs/v2/net.pt for a v2 prior). Engine workers: nproc - 3, split over the shards
# (default 1); concurrent battles per shard default workers / samples.
# Output: runs/vs-search-<date>-<job>/ (result.txt, manifest.txt, shard-*/).
#SBATCH --job-name=ws-netv2-vs-search
#SBATCH --partition=long
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
source scripts/netv2_common.sh
ALL="$*"
NET=$1; N=$2; shift 2
GEN=4; SEARCH=; SHARDS=1; CONC=; SEED=$SLURM_JOB_ID; OPTS=
while [ $# -gt 0 ]; do
  case "$1" in
    --gen) GEN=$2; shift 2 ;; --search) SEARCH=$2; shift 2 ;; --shards) SHARDS=$2; shift 2 ;;
    --concurrency) CONC=$2; shift 2 ;; --seed) SEED=$2; shift 2 ;;
    *=*) OPTS+=",$1"; shift ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
done
if [ -z "$SEARCH" ]; then
  if [ "$GEN" = 9 ]; then SEARCH=search:none; else SEARCH=search:$GLORIA/runs/20261025-195826-301040/final_model.zip; fi
fi
SEARCH=$SEARCH$OPTS
netv2_gen "$GEN"
WORKERS=$(( ($NPROC - 3) / SHARDS )); [ "$WORKERS" -ge 1 ] || WORKERS=1
SAMPLES=$(echo "$SEARCH" | grep -o 'samples=[0-9]*' | cut -d= -f2 || true)
[ -n "$CONC" ] || CONC=$(( WORKERS / ${SAMPLES:-4} > 0 ? WORKERS / ${SAMPLES:-4} : 1 ))
netv2_out vs-search
netv2_manifest vs-search "$ALL (search spec: $SEARCH; $SHARDS shards x $WORKERS workers, $CONC concurrent)" "$NET" "$SEARCH"
trap netv2_stop_servers EXIT
netv2_preread "$GLORIA"/.venv/lib/python3.12/site-packages/{torch,poke_engine}
netv2_shards "$NET" "$SEARCH" "$N" "$SHARDS" "$CONC" --workers "$WORKERS"
