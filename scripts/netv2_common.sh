# Shared by scripts/netv2_*.sbatch (sourced, not run): environment, run directory + manifest, duplicate-battle
# Showdown servers. ROOT = the submit dir (a netv2 cluster copy with src/, scripts/, .showdown-eval/ and
# pokemon-showdown/); venvs, node, the gen9 engine and models come from the shared main copy GLORIA, read only.
# The git commit of the synced code: env NETV2_COMMIT at submission (sbatch passes the environment), else the
# file $ROOT/COMMIT.

GLORIA=${NETV2_GLORIA:-/vol/bitbucket/fr1226/gloria}
PY=$GLORIA/.venv/bin/python
PS=$ROOT/.showdown-eval
export PATH=$GLORIA/.node/bin:$PATH
NPROC=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)  # GNU nproc honours OMP_NUM_THREADS
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
NETV2_PORTS=() NETV2_LOGS=()

# netv2_gen GEN: PYTHONPATH, PORYGON_GEN and FORMAT for gen 4 or 9
netv2_gen() {
  GEN=$1
  FORMAT=gen${GEN}randombattle
  export PYTHONPATH=$ROOT/src
  case "$GEN" in
    4) export PORYGON_GEN=4 ;;
    9) [ -f "$GLORIA/.engine-gen9/poke_engine/__init__.py" ] || { echo "no gen9 engine in $GLORIA/.engine-gen9" >&2; exit 1; }
       export PORYGON_GEN=9 PYTHONPATH=$GLORIA/.engine-gen9:$ROOT/src ;;
    *) echo "--gen must be 4 or 9 (got $GEN)" >&2; exit 1 ;;
  esac
}

# netv2_out KIND: OUT = $ROOT/runs/KIND-<date>-<job>
netv2_out() {
  OUT=$ROOT/runs/$1-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID
  mkdir -p "$OUT"
}

# netv2_manifest KIND "ORIGINAL ARGS" SPEC...: OUT/manifest.txt with what this run played and the code it ran
netv2_manifest() {
  local kind=$1 args=$2; shift 2
  local commit=${NETV2_COMMIT:-$(cat "$ROOT/COMMIT" 2>/dev/null || true)}
  {
    echo "kind: $kind"
    echo "job: $SLURM_JOB_ID on $(hostname) ($NPROC cpus), started $(date -Is)"
    echo "args: $args"
    echo "gen: $GEN  format: $FORMAT  seed: $SEED"
    echo "commit: ${commit:-unknown}"
    echo "src sha256: $(cd "$ROOT" && find src/ -name '*.py' | sort | xargs cat | sha256sum | cut -c1-16 || true)"
    echo "python: $PY  PYTHONPATH: $PYTHONPATH"
    local s base
    for s in "$@"; do
      echo "spec: $s"
      base=${s%%,*}; base=${base#search:}
      if [ -f "$base" ]; then
        echo "  file: $(readlink -f "$base") $(stat -L -c '%s bytes, mtime %y' "$base" || true)"
        echo "  sha256: $(sha256sum "$base" | cut -c1-64 || true)"
      fi
    done
  } > "$OUT/manifest.txt"
  cat "$OUT/manifest.txt"
}

# Pull the Python environment into the page cache: on the slow shared disk, modules imported lazily during a
# websocket handshake can take longer than its 10 s timeout.
netv2_preread() {
  find "$GLORIA"/.venv/lib/python3.12/site-packages/{websockets,poke_env} "$@" -type f \( -name '*.py' -o -name '*.so*' \) \
    -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
}

# netv2_server PORT DIR SEED GROUP: a duplicate-battle server (patches/showdown-duplicate-battles.patch) that
# logs every battle to DIR/battles.jsonl; its pairs are seeded from SEED; the player matching GROUP gets team A
# in the first game of each pair. Servers in one job need distinct SEEDs, or they replay the same games.
netv2_server() {
  local port=$1 dir=$2 seed=$3 group=$4
  mkdir -p "$dir"
  if [ ${#NETV2_PORTS[@]} -eq 0 ]; then
    if [ ! -f "$PS/config/config.js" ]; then
      cp "$PS/config/config-example.js" "$PS/config/config.js"
      printf '\nexports.nofswriting = true;\nexports.repl = false;\n' >> "$PS/config/config.js"
    fi
    [ -f "$PS/dist/server/index.js" ] || (cd "$PS" && node build)
    grep -q porygonDuplicate "$PS/dist/server/room-battle.js" ||
      { echo "$PS is not the duplicate-battle server (scripts/build_eval_server.sh)" >&2; exit 1; }
    mkdir -p "$PS/logs/repl"
  fi
  (cd "$PS" && PORYGON_DUPLICATE=$seed PORYGON_DUPLICATE_LOG=$dir/battles.jsonl PORYGON_DUPLICATE_GROUP=$group \
     PORYGON_DUPLICATE_TEAMS=${NETV2_TEAMS:+$dir/teams} \
     setsid nohup node pokemon-showdown start --skip-build --no-security "$port" < /dev/null > "$dir/showdown-$port.log" 2>&1 &)
  [ -z "${NETV2_TEAMS:-}" ] || mkdir -p "$dir/teams"
  NETV2_PORTS+=("$port")
  NETV2_LOGS+=("$dir/showdown-$port.log")
}

netv2_stop_servers() {
  local port
  for port in "${NETV2_PORTS[@]}"; do
    pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $port\$" || true
  done
}

# wait until every server listens, then open one websocket each (the first connection loads a server's
# connection code, which on a slow disk outlasts the clients' handshake timeout)
netv2_wait_servers() {
  local i t
  for i in "${!NETV2_PORTS[@]}"; do
    for ((t = 0; t < 2400; t++)); do
      grep -q "listening" "${NETV2_LOGS[$i]}" 2>/dev/null && continue 2
      sleep 0.5
    done
    echo "Showdown on port ${NETV2_PORTS[$i]} failed to start:" >&2; tail -20 "${NETV2_LOGS[$i]}" >&2; exit 1
  done
  "$PY" - "${NETV2_PORTS[@]}" <<'PYEOF' || echo "warning: no websocket handshake on some port" >&2
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

async def main(ports: list[int]) -> None:
    await asyncio.wait_for(asyncio.gather(*(warm(p) for p in ports)), 3600)

asyncio.run(main([int(p) for p in sys.argv[1:]]))
PYEOF
  echo "Showdown servers up on ports ${NETV2_PORTS[*]}"
}

# netv2_base_port: 16 ports per job, apart from the porygon-* scripts' 20000-51999 range
netv2_base_port() { echo $(( 52000 + (SLURM_JOB_ID % 800) * 16 )); }

# netv2_shards A B N SHARDS CONCURRENCY [EVALUATE ARGS...]: SHARDS (server, porygon.v2.evaluate) pairs in
# parallel, about N/SHARDS games each (whole pairs), into OUT/shard-<i>/; then the pooled result.
netv2_shards() {
  local a=$1 b=$2 n=$3 shards=$4 conc=$5; shift 5
  local per=$(( (n + shards - 1) / shards )); per=$(( per + per % 2 ))
  local base i pids=()
  base=$(netv2_base_port)
  for ((i = 0; i < shards; i++)); do netv2_server $(( base + i )) "$OUT/shard-$i" "${SEED}s$i" '^pa'; done
  netv2_wait_servers
  for ((i = 0; i < shards; i++)); do
    PORYGON_DUPLICATE=${SEED}s$i PORYGON_DUPLICATE_LOG=$OUT/shard-$i/battles.jsonl PORYGON_DUPLICATE_GROUP='^pa' \
      "$PY" -m porygon.v2.evaluate "$a" "$b" --n-battles "$per" --port $(( base + i )) --format "$FORMAT" \
      --out "$OUT/shard-$i" --concurrency "$conc" --name-tag "${SLURM_JOB_ID}x$i" "$@" > "$OUT/shard-$i/evaluate.log" 2>&1 &
    pids+=($!)
  done
  echo "$shards shards x $per games, $conc concurrent battles each"
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || echo "shard $i exited with $?: tail $OUT/shard-$i/evaluate.log" >&2; done
  tail -n 2 "$OUT"/shard-*/evaluate.log || true
  "$PY" "$ROOT/scripts/netv2_pool_ci.py" "$OUT" 2>&1 | tee "$OUT/result.txt" || true
}
