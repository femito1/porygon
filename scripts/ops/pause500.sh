#!/bin/bash
# Pause (SIGSTOP, not kill) the 4 x 500 ms gen9 label search that runs in 677193's batch step, at a chunk boundary.
# Killing it would end the batch script and with it the whole allocation (crawler, refresh, netv2's P2 steps).
set -u
D=/vol/bitbucket/fr1226/porygon-ws-data; R=$D/runs/searchlabels-gen9-20261108-181027-677193
PY=$(pgrep -u fr1226 -f "scripts/gen_search_labels.py none --out $R" | head -1)
NODE=$(pgrep -u fr1226 -f "pokemon-showdown start --skip-build --no-security 39101$" | head -1)
[ -n "$PY" ] && [ -n "$NODE" ] || { echo "label processes not found: py=$PY node=$NODE"; exit 1; }
n0=$(grep -c "chunk" "$R/gen.log")
for _ in $(seq 240); do [ "$(grep -c chunk "$R/gen.log")" -gt "$n0" ] && break; sleep 15; done
KIDS=$(pgrep -P "$PY" | tr '\n' ' ')
kill -STOP "$PY" $KIDS "$NODE"
echo "$(date -Is) paused at: $(tail -1 "$R/gen.log" | cut -c1-120)"
echo "stopped pids: python $PY, workers $KIDS, node $NODE (resume: kill -CONT them)"
ps -o pid,stat,args -p "$PY" $KIDS "$NODE" | cut -c1-100
