#!/bin/bash
# Refresh the data stream's datasets every INTERVAL seconds, incrementally (only games not yet converted), inside a
# running ws-data job:
#   srun --jobid=J --overlap --ntasks=1 -c 2 bash scripts/refresh_datasets.sh [INTERVAL=14400]
# 1. gen9 search labels: every $D/runs/searchlabels-gen9-* run -> $D/datasets/searchlabels (v1, v2) and
#    $D/datasets/v2pre/gen9-searchlabels;
# 2. gen9 human replays: corpus records with an input log (`prepare`, the crawl keeps adding input logs) ->
#    $D/datasets/human (v1, v2) and $D/datasets/v2pre/gen9-human; bc-exclude applied.
# v1, v2 and v2pre are encoded in one replay pass (identical decisions); outputs swap atomically. Stops after two
# failed rounds in a row; a round is skipped while another conversion holds $D/datasets/.convert.lock
# (`flock $D/datasets/.convert.lock <cmd>` for one-off conversions). Log: $D/logs/ws-data-refresh.log (when started as documented in data/STATUS.md).
set -uo pipefail
INTERVAL=${1:-14400}
FIRST_DELAY=${2:-0}  # seconds before the first round (restarts keep the schedule)
D=/vol/bitbucket/fr1226/porygon-ws-data
NS=${NETV2_SRC:-$D/netv2-src/cc6d8dd/src}
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
CORPUS=$D/raw/corpus/gen9randombattle
ENC=(--encoder porygon.encoder:embed_battle --encoder porygon.v2.encoder:encode --encoder v2pre --extra-src "$NS")
export PATH=/vol/bitbucket/fr1226/gloria/.node/bin:$PATH PYTHONPATH=$D/code/src PYTHONDONTWRITEBYTECODE=1
cd "$D/code"
fails=0
check() {  # $1 = output of one conversion; returns 1 on failure
  grep -E "^incremental|^unit|\"games\"|\"decisions\"|mask equal|illegal|not exact|Error|Traceback|unit error" <<< "$1"
  ! grep -q -E "Traceback|unit error" <<< "$1"
}
LOCK=$D/datasets/.convert.lock  # any conversion that holds it makes this round skip (no CPU overlap)
sleep "$FIRST_DELAY"
while true; do
  exec 9> "$LOCK"
  if ! flock -n 9; then
    echo "=== $(date -Is) another conversion holds $LOCK: skipping this round"; exec 9>&-; sleep "$INTERVAL"; continue
  fi
  ok=1
  echo "=== $(date -Is) search labels: $(ls -d "$D"/runs/searchlabels-gen9-* 2>/dev/null | wc -l) runs"
  out=$(nice -n 10 bash scripts/slurm_convert.sbatch runs gen9randombattle "$D/datasets/searchlabels" \
        $(ls -d "$D"/runs/searchlabels-gen9-*) -- --workers 2 --incremental "${ENC[@]}" \
        --v2pre-out "$D/datasets/v2pre/gen9-searchlabels" 2>&1) || ok=0
  check "$out" || ok=0
  if ls -d "$D"/runs/searchlabels-gen4-* > /dev/null 2>&1; then  # gen4 full-search labels (long-slot jobs)
    echo "=== $(date -Is) gen4 search labels: $(ls -d "$D"/runs/searchlabels-gen4-* | wc -l) runs"
    out=$(nice -n 10 bash scripts/slurm_convert.sbatch runs gen4randombattle "$D/datasets/searchlabels" \
          $(ls -d "$D"/runs/searchlabels-gen4-*) -- --workers 2 --incremental "${ENC[@]}" \
          --v2pre-out "$D/datasets/v2pre/gen4-searchlabels" 2>&1) || ok=0
    check "$out" || ok=0
  fi
  echo "=== $(date -Is) human replays"
  out=$(nice -n 10 "$PY" scripts/convert_replays.py prepare "$CORPUS"/part-*.jsonl.gz --format gen9randombattle \
        --out "$D/datasets/human-prep" --inputlogs "$D"/raw/inputlogs/gen9randombattle/part-*.jsonl.gz \
        "$D"/raw/showdown-replays/inputlogs/gen9randombattle/*.jsonl.gz 2>&1) || ok=0
  echo "$out" | tail -1
  for f in hi rest; do
    out=$(nice -n 10 bash scripts/slurm_convert.sbatch corpus gen9randombattle "$D/datasets/human" \
          "$D/datasets/human-prep/$f.jsonl.gz" -- --workers 2 --incremental --bc-exclude "$CORPUS/bc-exclude.json" \
          "${ENC[@]}" --v2pre-out "$D/datasets/v2pre/gen9-human" 2>&1) || ok=0
    check "$out" || ok=0
  done
  if ((ok)); then fails=0; else
    fails=$((fails + 1)); echo "round failed ($fails in a row)"; echo "$out" | tail -20
    ((fails >= 2)) && { echo "two failed rounds in a row: stopping"; exit 1; }
  fi
  exec 9>&-  # release the lock while sleeping
  sleep "$INTERVAL"
done
