#!/bin/bash
# Refresh the search-label datasets every INTERVAL seconds (incremental: only games not yet converted), inside a
# running ws-data job:
#   srun --jobid=J --overlap --ntasks=1 -c 2 bash scripts/refresh_searchlabels.sh [GEN=9] [INTERVAL=14400]
# Converts every $D/runs/searchlabels-gen<GEN>-* run with the v1, v2 and v2pre (netv2 pretraining) encoders in one
# pass into $D/datasets/searchlabels and $D/datasets/v2pre/gen<GEN>-searchlabels (atomic version swaps). Stops
# after two failed refreshes in a row. Log: $D/logs/ws-data-refresh.log (when started as documented in STATUS.md).
set -uo pipefail
GEN=${1:-9}
INTERVAL=${2:-14400}
D=/vol/bitbucket/fr1226/porygon-ws-data
NS=${NETV2_SRC:-$D/netv2-src/cc6d8dd/src}
cd "$D/code"
fails=0
while true; do
  runs=$(ls -d "$D"/runs/searchlabels-gen$GEN-* 2>/dev/null)
  echo "=== $(date -Is) refresh gen$GEN: $(echo $runs | wc -w) runs"
  out=$(nice -n 10 bash scripts/slurm_convert.sbatch runs gen${GEN}randombattle "$D/datasets/searchlabels" $runs -- \
        --workers 2 --incremental --encoder porygon.encoder:embed_battle --encoder porygon.v2.encoder:encode \
        --encoder v2pre --extra-src "$NS" --v2pre-out "$D/datasets/v2pre/gen$GEN-searchlabels" 2>&1)
  rc=$?
  echo "$out" | grep -E "^incremental|^unit|\"games\"|\"decisions\"|mask equal|illegal|Error|Traceback|unit error"
  if ((rc != 0)) || grep -q -E "Traceback|unit error" <<< "$out"; then
    fails=$((fails + 1))
    echo "refresh failed (rc=$rc, $fails in a row)"
    echo "$out" | tail -20
    ((fails >= 2)) && { echo "two failures in a row: stopping"; exit 1; }
  else
    fails=0
  fi
  sleep "$INTERVAL"
done
