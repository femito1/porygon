#!/bin/bash
# wsp loss analysis, step 2: loss_patterns.py over the logged games (M1, M2) and fp_games.py over the real Foul Play
# runs; run inside a wsp job (srun --overlap into the tracker pass). Output $WSP/losses/{patterns.json,patterns.txt,fp_games.txt}
set -uo pipefail
WSP=/vol/bitbucket/fr1226/porygon-ws-wsp; L=$WSP/losses; K=$L/code; P=/vol/bitbucket/fr1226/gloria/.venv/bin/python
R=/vol/bitbucket/fr1226/porygon-ws-netv2/runs
$P -I $K/loss_patterns.py --moves $L/moves9.json --out $L/patterns.json --examples 3 M1=$L/m1/games M2=$L/m2/games > $L/patterns.txt 2>&1
echo "patterns rc=$?" >> $L/STATUS
$P -I $K/fp_games.py $R/foulplay-net-20261116-173816-1173380 > $L/fp_games_805.txt 2>&1
$P -I $K/fp_games.py $R/foulplay-net-20261116-173816-1173380 $R/foulplay-net-20261115-163340-1173215 $R/foulplay-net-20261114-202039-1172567 > $L/fp_games_pooled.txt 2>&1
echo "$(date -u +%FT%TZ) analysis done" >> $L/STATUS
