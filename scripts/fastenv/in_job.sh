#!/bin/bash
# Run a command inside fastenv's running long job (the stream's only cluster slot), with the env it needs:
#   ssh gpucluster2 'cd /vol/bitbucket/fr1226/porygon-ws-fastenv && scripts/fastenv/in_job.sh JOBID CMD...'
# e.g. scripts/fastenv/in_job.sh 687853 \$PY -m porygon.fastenv_eval --a A.zip --b B.pt --games 1000 --workers 3
set -uo pipefail
JOB=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
G=/vol/bitbucket/fr1226/gloria
exec srun --jobid="$JOB" --overlap --ntasks=1 bash -c "cd '$ROOT' && export PY=$G/.venv/bin/python \
  PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown PATH=$G/.node/bin:\$PATH PYTHONPATH='$ROOT/src' \
  && eval \"\$0\" \"\$@\"" "$@"
