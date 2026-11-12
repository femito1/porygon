#!/bin/bash
# dyn: detached runner for dyn_func.py + dyn_func_an.py (launch with nohup srun ... from the login node)
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=2
python scripts/wsp/dyn_func.py out/dyn /tmp/wsp-674320 >> out/dyn/func.log 2>&1
python scripts/wsp/dyn_func_an.py out/dyn data > out/dyn/func_an.txt 2>&1
echo DONE >> out/dyn/func.log
