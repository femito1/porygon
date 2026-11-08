#!/bin/bash
# usage (login node): nohup srun --jobid=674320 --overlap --ntasks=1 bash scripts/wsp/beh_run.sh SCRIPT ARGS... > out/beh/srun_SCRIPT.log 2>&1 < /dev/null &
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=2
python scripts/wsp/"$@"
