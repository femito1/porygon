#!/bin/bash
# eds: corrected oracle relabels, 2 workers (1 thread each) inside the box job
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=1
python scripts/wsp/eds_relabel.py 0 2 > out/eds/relabel-0.log 2>&1 &
python scripts/wsp/eds_relabel.py 1 2 > out/eds/relabel-1.log 2>&1 &
wait
echo ALLDONE
