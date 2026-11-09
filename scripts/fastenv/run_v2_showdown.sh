#!/bin/bash
# netv2's v2 trainer with its own Showdown collector (local servers + poke-env battle processes), inside fastenv's
# job: the per-sample reference arm for --collector fastenv. Servers from our own build (PS_DIR, default
# ps-sets-pinned = 7b02eb3, the same commit as gloria), pinned with the trainer to NCPU of the job's CPUs.
#   scripts/fastenv/in_job.sh JOB 'scripts/fastenv/run_v2_showdown.sh NCPU NSERVERS RUN_DIR [porygon.v2.train args]'
set -uo pipefail
NCPU=$1; NSRV=$2; RUN=$3; shift 3
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
G=/vol/bitbucket/fr1226/gloria
PS=${PS_DIR:-$ROOT/ps-sets-pinned}
export PYTHONPATH=$ROOT/combo/src OMP_NUM_THREADS=1 PATH=$G/.node/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/ws-fastenv-inductor-$$
CPUS=$("$G/.venv/bin/python" -c "import os; c = sorted(os.sched_getaffinity(0)); print(','.join(map(str, c[-$NCPU:])))")
BASE=$(( 30000 + (${SLURM_JOB_ID:-0} % 1000) * 16 )); LAST=$(( BASE + NSRV - 1 ))
if [ ! -f "$PS/config/config.js" ]; then  # training config: no disk state, no REPL sockets
  cp "$PS/config/config-example.js" "$PS/config/config.js"
  printf '\nexports.nofswriting = true;\nexports.repl = false;\n' >> "$PS/config/config.js"
fi
mkdir -p "$ROOT/logs" "$PS/logs/repl" "$RUN"
stop() { for ((p = BASE; p <= LAST; p++)); do pkill -u "$USER" -f "pokemon-showdown start --skip-build --no-security $p\$" || true; done; }
trap stop EXIT
for ((p = BASE; p <= LAST; p++)); do
  (cd "$PS" && setsid nohup taskset -c "$CPUS" node pokemon-showdown start --skip-build --no-security "$p" < /dev/null \
    > "$ROOT/logs/showdown-$p.log" 2>&1 &)
done
for ((p = BASE; p <= LAST; p++)); do
  for ((t = 0; t < 1200; t++)); do grep -q listening "$ROOT/logs/showdown-$p.log" 2>/dev/null && continue 2; sleep 0.5; done
  echo "Showdown on port $p failed to start" >&2; tail -5 "$ROOT/logs/showdown-$p.log" >&2; exit 1
done
"$G/.venv/bin/python" - "$BASE" "$NSRV" <<'PYEOF' || echo "warning: websocket warm-up failed" >&2
import asyncio, sys
import websockets
async def warm(port):
    for _ in range(60):
        try:
            async with websockets.connect(f"ws://localhost:{port}/showdown/websocket", open_timeout=60):
                return
        except Exception:
            await asyncio.sleep(2)
base, n = int(sys.argv[1]), int(sys.argv[2])
asyncio.run(asyncio.wait_for(asyncio.gather(*(warm(base + i) for i in range(n))), 900))
PYEOF
echo "$(date -u +%T) $(hostname) cpus $CPUS servers $BASE-$LAST: porygon.v2.train $*" | tee -a "$RUN/launch.log"
taskset -c "$CPUS" "$G/.venv/bin/python" -u -m porygon.v2.train --collector showdown --ports "$BASE-$LAST" \
  --run-dir "$RUN" "$@" >> "$RUN/stdout.log" 2>&1
