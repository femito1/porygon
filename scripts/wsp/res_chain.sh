#!/bin/bash
# sequential res runs inside the analysis box: res_chain.sh A|B
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=2
if [ "$1" = A ]; then
  until [ -f out/res/probes_500M.json ]; do sleep 30; done
  python scripts/wsp/res_use.py 500M > out/res/use_500M.log 2>&1
else
  python scripts/wsp/res_probe.py 600M --no-raw > out/res/probe_600M.log 2>&1
  python scripts/wsp/res_use.py 600M > out/res/use_600M.log 2>&1
  python scripts/wsp/res_mlp.py 600M > out/res/mlp_600M.log 2>&1
fi
