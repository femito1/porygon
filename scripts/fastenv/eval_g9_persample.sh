#!/bin/bash
# Per-sample equality evaluation (pre-registered in .orchestra/fastenv/STATUS.md): fastenv-collector arm F vs
# Showdown-collector arm S at their first checkpoints >= 1M and >= 2M, raw nets at T 0.5, in-process duplicate pairs;
# each also vs the heuristic / max-power bots.   in_job.sh JOB 'scripts/fastenv/eval_g9_persample.sh F_RUN S_RUN [NCPU]'
set -uo pipefail
F=$1; S=$2; NCPU=${3:-8}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
G=/vol/bitbucket/fr1226/gloria
export PYTHONPATH=$ROOT/combo/src OMP_NUM_THREADS=1 PORYGON_PS=$G/pokemon-showdown PORYGON_NODE=$G/.node/bin/node
CPUS=$("$G/.venv/bin/python" -c "import os; c = sorted(os.sched_getaffinity(0)); print(','.join(map(str, c[:$NCPU])))")
EV="taskset -c $CPUS $G/.venv/bin/python -m porygon.fastenv_eval --format gen9randombattle --workers $(( NCPU - 1 )) --battles 32"
# bot games: few slots so that most slots finish whole duplicate pairs (games >> 2 x slots), or the CI uses few pairs
EVB="taskset -c $CPUS $G/.venv/bin/python -m porygon.fastenv_eval --format gen9randombattle --workers 4 --battles 24"
ck() { ls "$1"/checkpoints/ckpt_*.pt | awk -F'ckpt_|[.]pt' -v n="$2" '$2 + 0 >= n' | sort | head -1; }
for n in 1000000 2000000; do
  f=$(ck "$F" $n); s=$(ck "$S" $n)
  echo "=== @${n}: F=$f S=$s"
  [ -n "$f" ] && [ -n "$s" ] && $EV --games 1000 --a "v2:$f,temperature=0.5" --b "v2:$s,temperature=0.5" 2>&1 | tail -1
  for x in "$f" "$s"; do
    [ -n "$x" ] || continue
    for bot in heuristic max_power; do $EVB --games 800 --a "v2:$x,temperature=0.5" --b $bot 2>&1 | tail -1; done
  done
done
