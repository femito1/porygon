#!/bin/bash
# usage (inside srun): bash scripts/wsp/att_bg.sh LOGNAME cmd args...   -> runs detached, logs to $WSP/logs/LOGNAME.log
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=2
mkdir -p $WSP/logs
log=$WSP/logs/$1.log; shift
setsid nohup bash -c "$*" > $log 2>&1 < /dev/null &
echo "started pid $! -> $log"
