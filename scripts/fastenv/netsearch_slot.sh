#!/bin/bash
# Network + search slot runner (test plan: .orchestra/fastenv/NETSEARCH-DESIGN.md §6). Every arm of a plan plays one
# opponent on fastenv's own copy of the duplicate-battle benchmark server ($C/.showdown-eval: smogon 7b02eb3 + the
# duplicate patch, code-identical to netv2's), through netv2's harness (porygon.v2.evaluate via netsearch_h2h.py, so
# `netsearch:` specs work), in blocks of BLOCK_PAIRS duplicate pairs with the arms rotated every round. Block j of
# every arm plays on a fresh server seeded "${TAG}b${j}": all arms see the same team pairs. After each block a look
# (netsearch_slot.py look): with FUTILITY set, an arm whose upper 95% bound falls below it stops; empty = fixed n.
#   bash scripts/fastenv/netsearch_slot.sh PLAN.env [OUT]     (inside a job: netsearch_slot.sbatch, or a helper step)
# The first call copies this tree's src + scripts and netv2's src/porygon into OUT/code and re-runs itself from that
# copy, so syncs during the run cannot change it. OUT default: $C/runs/netsearch-NAME-<date>-<job>.
# PLAN.env (bash, sourced): NAME GEN TAG OPP ARMS=(spec ...) BLOCK_PAIRS BLOCKS CONC FUTILITY MIN_CPUS
#   [BLOCK_TIMEOUT_MIN (default 240)] [DRY_P=(p ...): NS_DRY=1 writes synthetic games with these win rates instead
#   of playing (laptop dry runs of the control flow)].
# Control: touch OUT/STOP = stop after the current block; OUT/STOP_NOW = kill the current block and stop.
# Output: OUT/manifest.txt, OUT/looks.jsonl, OUT/result.txt + result.json (netsearch_slot.py report), arm-K/block-J/.
set -uo pipefail
PLAN=$1
C=${NS_ROOT:-/vol/bitbucket/fr1226/porygon-ws-fastenv}
G=${NS_GLORIA:-/vol/bitbucket/fr1226/gloria}
NV_LIVE=${NS_NETV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}
JOB=${SLURM_JOB_ID:-local$$}
DRY=${NS_DRY:-0}

if [ -z "${NS_SNAPSHOT:-}" ]; then
  source "$PLAN"
  OUT=${2:-$C/runs/netsearch-$NAME-$(date -u +%Y%m%d-%H%M%S)-$JOB}
  [ ! -e "$OUT/manifest.txt" ] || { echo "$OUT already holds a run" >&2; exit 1; }
  mkdir -p "$OUT/code/netv2src" || exit 1
  SRC=$(cd "$(dirname "$0")/../.." && pwd)
  rsync -a --exclude __pycache__ "$SRC/src" "$SRC/scripts" "$OUT/code/" || exit 1
  [ ! -f "$SRC/COMMIT" ] || cp "$SRC/COMMIT" "$OUT/code/"
  if [ "$DRY" != 1 ]; then
    rsync -a --exclude __pycache__ "$NV_LIVE/src/porygon" "$OUT/code/netv2src/" || exit 1
    cp "$NV_LIVE/COMMIT" "$OUT/code/netv2src/COMMIT" 2>/dev/null || true
  fi
  cp "$PLAN" "$OUT/plan.env"
  printf '%s\n' "${ARMS[@]}" > "$OUT/arms.txt"
  NS_SNAPSHOT=1 exec bash "$OUT/code/scripts/fastenv/netsearch_slot.sh" "$OUT/plan.env" "$OUT"
fi

OUT=$2
CODE=$OUT/code
source "$OUT/plan.env"
FORMAT=gen${GEN}randombattle
BLOCK_TIMEOUT_MIN=${BLOCK_TIMEOUT_MIN:-240}
PS=$C/.showdown-eval
PY=${NS_PY:-$G/.venv/bin/python}
export PATH=$G/.node/bin:$PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PORYGON_GEN=$GEN NETV2_SRC=$CODE/netv2src
export PYTHONPATH=$CODE/src
[ "$GEN" != 9 ] || export PYTHONPATH=$G/.engine-gen9:$CODE/src
SLOT="$PY $CODE/scripts/fastenv/netsearch_slot.py"
NARMS=${#ARMS[@]}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/run.log"; }

SERVER_W= SERVER_G= EVAL_W= EVAL_G=  # waiter pids (exit code = the command's) and session ids (to kill)
cleanup() {
  [ -z "$EVAL_G" ] || kill -- -"$EVAL_G" 2>/dev/null
  [ -z "$SERVER_G" ] || kill -- -"$SERVER_G" 2>/dev/null
}
# spawn PGIDFILE CMD...: CMD in its own session, in the background; $! = a waiter whose exit code is CMD's, and
# PGIDFILE gets the session id (kill -- -ID stops CMD and its children). setsid --wait forks only when it must.
spawn() {
  local f=$1; shift
  rm -f "$f"
  setsid --wait bash -c 'echo $$ > "$0"; exec "$@"' "$f" "$@" &
}
pgid_of() {  # PGIDFILE WAITER: the session id once written (empty if the waiter exits first)
  local t
  for ((t = 0; t < 100; t++)); do [ -s "$1" ] && { cat "$1"; return; }; kill -0 "$2" 2>/dev/null || return; sleep 0.1; done
}
trap cleanup EXIT
trap 'log "signal: stopping"; exit 143' TERM INT

# ---------------------------------------------------------------- checks and manifest
NCPU=${SLURM_CPUS_ON_NODE:-$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo 0)}
if [ "$DRY" != 1 ] && [ "$NCPU" -lt "${MIN_CPUS:-1}" ]; then
  log "only $NCPU CPUs, the plan needs ${MIN_CPUS}: not running"; exit 2
fi
{
  echo "kind: netsearch-slot $NAME"
  echo "job: $JOB on $(hostname) ($NCPU cpus; cpuset $(taskset -pc $$ 2>/dev/null | sed 's/.*: //')), started $(date -u +%FT%TZ)"
  echo "plan: $OUT/plan.env sha256 $(sha256sum "$OUT/plan.env" | cut -c1-16)"
  sed 's/^/  | /' "$OUT/plan.env"
  echo "fastenv commit: $(cat "$CODE/COMMIT" 2>/dev/null || echo unknown); code sha256 (src *.py, scripts/fastenv *.py *.sh): $(cd "$CODE" && find src scripts/fastenv -maxdepth 3 \( -name '*.py' -o -name '*.sh' \) | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16)"
  if [ "$DRY" != 1 ]; then
    echo "netv2 src: $NV_LIVE/src/porygon (commit $(cat "$CODE/netv2src/COMMIT" 2>/dev/null || echo unknown)), copy sha256 $(cd "$CODE/netv2src" && find porygon -name '*.py' | LC_ALL=C sort | xargs cat | sha256sum | cut -c1-16)"
    echo "showdown: $PS"; sed 's/^/  | /' "$PS/SOURCE.txt"
    echo "  code sha256 now: $(cd "$PS" && find dist config data sim lib server -type f ! -path config/chat-plugins/seasons.json | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)"
    echo "engine: $(ls "$G"/.engine-gen9/poke_engine/*.so 2>/dev/null | head -1) sha256 $(sha256sum "$G"/.engine-gen9/poke_engine/*.so 2>/dev/null | head -1 | cut -c1-16)"
    echo "python: $PY ($("$PY" -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__)' 2>&1))"
  fi
  echo "format: $FORMAT  seed tag: $TAG (block j: server seed ${TAG}b<j>)  block: $BLOCK_PAIRS pairs x $BLOCKS blocks, $CONC concurrent  futility: ${FUTILITY:-none (fixed n)}"
  for spec in "$OPP" "${ARMS[@]}"; do
    echo "spec: $spec"
    base=${spec%%,*}; base=${base#netsearch:}; base=${base#search:}
    [ ! -f "$base" ] || echo "  file: $(readlink -f "$base") $(stat -L -c '%s bytes, mtime %y' "$base") sha256 $(sha256sum "$base" | cut -c1-64)"
    [ "$DRY" = 1 ] || echo "  check: $("$PY" "$CODE/scripts/fastenv/netsearch_h2h.py" --check "$spec" 2>&1 | tail -2 | tr '\n' ' ')"
  done
} > "$OUT/manifest.txt" 2>&1
cat "$OUT/manifest.txt"

# ---------------------------------------------------------------- servers and blocks
if [ "$DRY" != 1 ]; then
  log "pre-reading the Python environment"
  SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  timeout 1800 find "$SITE"/{torch,poke_env,websockets,numpy} "$G/.engine-gen9" -type f \( -name '*.py' -o -name '*.so*' \) \
    -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null 2>&1 || true
  PORT=
  for ((p = 52008 + (${SLURM_JOB_ID:-$$} % 800) * 16, i = 0; i < 8; i++, p++)); do
    if python3 -c "import socket, sys; s = socket.socket(); s.bind(('', $p)); s.close()" 2>/dev/null; then PORT=$p; break; fi
  done
  [ -n "$PORT" ] || { log "no free port"; exit 1; }
  log "port $PORT"
fi

start_server() {  # DIR SEED: a duplicate-battle server on $PORT logging to DIR/battles.jsonl
  local dir=$1 seed=$2 t
  (cd "$PS" && spawn "$dir/server.pgid" env PORYGON_DUPLICATE="$seed" PORYGON_DUPLICATE_LOG="$dir/battles.jsonl" \
     PORYGON_DUPLICATE_GROUP='^pa' node pokemon-showdown start --skip-build --no-security "$PORT" \
     < /dev/null > "$dir/showdown.log" 2>&1; wait $!) &
  SERVER_W=$!
  SERVER_G=$(pgid_of "$dir/server.pgid" "$SERVER_W")
  for ((t = 0; t < 2400; t++)); do
    grep -q listening "$dir/showdown.log" 2>/dev/null && break
    kill -0 "$SERVER_W" 2>/dev/null || { log "server died: $(tail -3 "$dir/showdown.log")"; return 1; }
    sleep 0.5
  done
  grep -q listening "$dir/showdown.log" || { log "server on $PORT not listening"; return 1; }
  # the first connection loads the server's connection code (slow disk): open one before the players do
  "$PY" - "$PORT" <<'PYEOF' || log "warning: no websocket handshake on $PORT"
import asyncio, sys
import websockets

async def warm(port: int) -> None:
    for _ in range(60):
        try:
            async with websockets.connect(f"ws://localhost:{port}/showdown/websocket", open_timeout=60):
                return
        except Exception:
            await asyncio.sleep(5)
    raise SystemExit(f"port {port}: no handshake")

asyncio.run(asyncio.wait_for(warm(int(sys.argv[1])), 1800))
PYEOF
}

stop_server() {
  [ -n "$SERVER_W" ] || return 0
  [ -z "$SERVER_G" ] || kill -- -"$SERVER_G" 2>/dev/null
  for ((t = 0; t < 60; t++)); do kill -0 "$SERVER_W" 2>/dev/null || break; sleep 0.5; done
  [ -z "$SERVER_G" ] || kill -9 -- -"$SERVER_G" 2>/dev/null
  wait "$SERVER_W" 2>/dev/null
  SERVER_W= SERVER_G=
}

play_block() {  # K J: arm K's block J; returns the client's exit code
  local k=$1 j=$2 spec=${ARMS[$(($1 - 1))]} dir=$OUT/arm-$1/block-$2 seed=${TAG}b$2 rc waited=0
  local n=$((2 * BLOCK_PAIRS)) tag=${JOB}a$1b$2
  mkdir -p "$dir"
  case "$spec" in netsearch:*) spec="$spec,log=$dir/gate.jsonl" ;; esac
  if [ "$DRY" = 1 ]; then
    $SLOT fake "$dir" --games "$n" --p "${DRY_P[$(($1 - 1))]}" --a "pa$tag" --seed "$seed" > "$dir/evaluate.log" 2>&1
    return
  fi
  start_server "$dir" "$seed" || return 1
  spawn "$dir/client.pgid" env PORYGON_DUPLICATE="$seed" PORYGON_DUPLICATE_LOG="$dir/battles.jsonl" \
    PORYGON_DUPLICATE_GROUP='^pa' "$PY" "$CODE/scripts/fastenv/netsearch_h2h.py" "$spec" "$OPP" --n-battles "$n" \
    --port "$PORT" --format "$FORMAT" --out "$dir" --concurrency "$CONC" --name-tag "$tag" --print-every 10 \
    > "$dir/evaluate.log" 2>&1
  EVAL_W=$!
  EVAL_G=$(pgid_of "$dir/client.pgid" "$EVAL_W")
  while kill -0 "$EVAL_W" 2>/dev/null; do
    sleep 10; waited=$((waited + 10))
    if [ -e "$OUT/STOP_NOW" ] || [ "$waited" -ge $((BLOCK_TIMEOUT_MIN * 60)) ]; then
      log "arm $k block $j: killing the client after ${waited}s ($( [ -e "$OUT/STOP_NOW" ] && echo STOP_NOW || echo timeout))"
      [ -z "$EVAL_G" ] || kill -- -"$EVAL_G" 2>/dev/null; sleep 20; [ -z "$EVAL_G" ] || kill -9 -- -"$EVAL_G" 2>/dev/null
      break
    fi
  done
  wait "$EVAL_W"; rc=$?
  EVAL_W= EVAL_G=
  stop_server
  return $rc
}

log "start: $NARMS arms x $BLOCKS blocks of $BLOCK_PAIRS pairs vs $OPP (tag $TAG)"
for ((j = 1; j <= BLOCKS; j++)); do
  for ((r = 0; r < NARMS; r++)); do
    k=$(( (r + j - 1) % NARMS + 1 ))  # round j starts with arm j (mod the number of arms)
    [ ! -e "$OUT/arm-$k/STOPPED" ] || continue
    [ ! -e "$OUT/STOP" ] && [ ! -e "$OUT/STOP_NOW" ] || { log "stop file: ending the run"; break 2; }
    log "arm $k block $j: start (seed ${TAG}b$j)"
    play_block "$k" "$j"; rc=$?
    games=$(cat "$OUT/arm-$k/block-$j/results.jsonl" 2>/dev/null | wc -l)
    log "arm $k block $j: client exit $rc, $games games: $(tail -1 "$OUT/arm-$k/block-$j/evaluate.log" 2>/dev/null)"
    if [ "$rc" != 0 ] && [ "$games" -lt $((2 * BLOCK_PAIRS * 9 / 10)) ]; then
      [ -e "$OUT/STOP_NOW" ] || { touch "$OUT/HARNESS_FAIL"; log "HARNESS FAILURE in arm $k block $j: the run stops (void; replay after the fix)"; }
      break 2
    fi
    $SLOT look "$OUT" "$k" --blocks-max "$BLOCKS" ${FUTILITY:+--futility "$FUTILITY"} | tee -a "$OUT/run.log"
    [ "${PIPESTATUS[0]}" != 20 ] || { touch "$OUT/arm-$k/STOPPED"; log "arm $k: FUTILITY stop after block $j"; }
  done
done
$SLOT report "$OUT" --json "$OUT/result.json" | tee "$OUT/result.txt"
log "done"
