#!/bin/bash
# End-to-end gen9 v2 check of the fastenv pipeline: the same net vs poke-env's baselines on a real Showdown server
# (porygon.v2.evaluate, poke-env players) — compare with the in-process numbers (porygon.fastenv_eval --b heuristic /
# max_power: JS bot ports, JS v2 encoder). Own server from PS_DIR (default ps-sets-pinned = 7b02eb3), pinned CPUs.
#   in_job.sh JOB 'CPU_SKIP=k scripts/fastenv/net_vs_bots_showdown.sh NCPU NET_SPEC GAMES OUT [FORMAT]'
set -uo pipefail
NCPU=$1; NET=$2; GAMES=$3; OUT=$4; FMT=${5:-gen9randombattle}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
G=/vol/bitbucket/fr1226/gloria
PS=${PS_DIR:-$ROOT/ps-sets-pinned}
export PYTHONPATH=${COMBO:-$ROOT/combo}/src OMP_NUM_THREADS=1 PATH=$G/.node/bin:$PATH
CPUS=$("$G/.venv/bin/python" -c "import os; c = sorted(os.sched_getaffinity(0)); k = ${CPU_SKIP:-0}; print(','.join(map(str, c[len(c) - k - $NCPU:len(c) - k])))")
PORT=$(( 31000 + (${SLURM_JOB_ID:-0} % 500) * 4 + ${PORT_OFF:-0} ))
[ -f "$PS/config/config.js" ] || { cp "$PS/config/config-example.js" "$PS/config/config.js"; printf '\nexports.nofswriting = true;\nexports.repl = false;\n' >> "$PS/config/config.js"; }
mkdir -p "$OUT" "$PS/logs/repl"
(cd "$PS" && setsid nohup taskset -c "$CPUS" node pokemon-showdown start --skip-build --no-security "$PORT" < /dev/null > "$OUT/showdown.log" 2>&1 &)
trap 'pkill -u "$USER" -f "pokemon-showdown start --skip-build --no-security $PORT\$" || true' EXIT
for ((t = 0; t < 1200; t++)); do grep -q listening "$OUT/showdown.log" 2>/dev/null && break; sleep 0.5; done
for bot in heuristic max_power; do
  taskset -c "$CPUS" "$G/.venv/bin/python" -m porygon.v2.evaluate "$NET" "$bot" --n-battles "$GAMES" --port "$PORT" \
    --format "$FMT" --out "$OUT/$bot" --concurrency 16 --device cpu 2>&1 | tail -2
done
