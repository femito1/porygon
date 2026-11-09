#!/bin/bash
# netv2 wrapper for wsp's in-process head-to-head (vendored copy, ws/wsp ee76994): spd_h2h.py run ARGS... (v2 nets)
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; V=$C/vendor/wsp-ee76994; G=/vol/bitbucket/fr1226/gloria
export SPD_FASTENV_JS=$V/fastenv-b9b8a85/fastenv/js PORYGON_PS=$G/pokemon-showdown PORYGON_NODE=$G/.node/bin/node OMP_NUM_THREADS=1
exec $G/.venv/bin/python $V/spd_h2h.py "$@"
