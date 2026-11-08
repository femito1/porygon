#!/bin/bash
# usage (login node): nohup srun --jobid=674320 --overlap --ntasks=1 bash scripts/wsp/edw_run.sh SCRIPT ARGS... > out/edw/LOG 2>&1 < /dev/null &
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=2
mkdir -p out/edw
for s in "$@"; do
  echo "=== $s $(date)"
  python scripts/wsp/$s || echo "FAILED: $s"
done
echo "=== all done $(date)"
