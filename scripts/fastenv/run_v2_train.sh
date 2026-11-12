#!/bin/bash
# netv2's v2 trainer (patched: patches/netv2-fastenv-collector.patch) from the combo tree (fastenv src + netv2's
# porygon.v2 + the patched train.py; COMBO=<dir> picks another tree), inside fastenv's job, pinned to NCPU of the
# job's CPUs:
#   scripts/fastenv/in_job.sh JOB 'scripts/fastenv/run_v2_train.sh NCPU RUN_DIR [porygon.v2.train args]'
set -uo pipefail
NCPU=$1; RUN=$2; shift 2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
G=/vol/bitbucket/fr1226/gloria
export PYTHONPATH=${COMBO:-$ROOT/combo}/src OMP_NUM_THREADS=1 PATH=$G/.node/bin:$PATH
export PORYGON_PS=${PORYGON_PS:-$G/pokemon-showdown} PORYGON_NODE=$G/.node/bin/node
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/ws-fastenv-inductor-$$
# CPU_SKIP=k: leave the last k of the job's CPUs to another run (default: take the last NCPU)
CPUS=$("$G/.venv/bin/python" -c "import os; c = sorted(os.sched_getaffinity(0)); k = ${CPU_SKIP:-0}; print(','.join(map(str, c[len(c) - k - $NCPU:len(c) - k])))")
mkdir -p "$RUN"
echo "$(date -u +%T) $(hostname) cpus $CPUS: porygon.v2.train $*" | tee -a "$RUN/launch.log"
exec taskset -c "$CPUS" "$G/.venv/bin/python" -u -m porygon.v2.train --run-dir "$RUN" "$@" >> "$RUN/stdout.log" 2>&1
