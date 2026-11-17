#!/bin/bash
# wsp loss analysis, step 1 (11-16): logged in-process gen9 games of raw G9q-cd@805M (T 0.5), run as a v2diag
# oneshot inside the tracker pass (4 CPUs + T4; the one-wsp-job rule). Code: $WSP/losses/code = ws/wsp spd_h2h.py +
# spd_host.js with --game-log. M1 = cd@805 vs itself (loser vs winner contrast); M2 = cd@805 vs BC-P3 (the human-BC
# init: a human-imitation comparison on the same positions). Output $WSP/losses/{m1,m2}/ (h2h records + games/).
set -uo pipefail
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
C=/vol/bitbucket/fr1226/porygon-ws-netv2
L=$WSP/losses; K=$L/code
export SPD_FASTENV_JS=$WSP/spd/fastenv-b9b8a85/fastenv/js PORYGON_PS=$GLORIA/pokemon-showdown PORYGON_NODE=$GLORIA/.node/bin/node
export OMP_NUM_THREADS=1
NET=$C/runs/g9q-cd800/final/00805029873.pt; P3=$C/runs/pretrain/P3-g9-s2.pt
NCPU=$(python -c 'import os; print(len(os.sched_getaffinity(0)))'); W=$(( NCPU > 2 ? NCPU - 1 : 1 ))
echo "$(date -u +%FT%TZ) losses: $(hostname) $NCPU CPUs, $W workers" | tee -a $L/STATUS
run() {  # OUT B PAIRS SEED
  python $K/spd_h2h.py run --a "v2:$NET,temperature=0.5" --b "$2" --pairs "$3" --shard 0 --seed "$4" \
    --out "$L/$1" --format gen9randombattle --workers "$W" --battles 32 --v2-src "$C/src-24c633f" --device cuda \
    --progress 120 --game-log "$L/$1/games" > "$L/$1.log" 2>&1
  echo "$(date -u +%FT%TZ) $1 rc=$? $(python $K/spd_h2h.py merge $L/$1 2>&1 | tail -n 1)" | tee -a $L/STATUS
}
run m1 "v2:$NET,temperature=0.5" 1500 9161101
run m2 "v2:$P3,temperature=0.5" 1000 9161102
for d in m1 m2; do gzip -f $L/$d/games/*.jsonl 2>/dev/null; done
echo "$(date -u +%FT%TZ) losses: DONE" | tee -a $L/STATUS
