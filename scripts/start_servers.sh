#!/usr/bin/env bash
# Start N local Showdown servers on ports BASE..BASE+N-1 (no auth, no throttling).
# Usage: scripts/start_servers.sh [N=1] [BASE=8000]    Stop: scripts/stop_servers.sh
# PS_DIR: another Showdown checkout (default pokemon-showdown/), e.g. the duplicate-battle one (build_eval_server.sh)
set -euo pipefail
N=${1:-1}; BASE=${2:-8000}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PS=${PS_DIR:-$ROOT/pokemon-showdown}
if [ ! -f "$PS/config/config.js" ]; then
  # Training servers share one directory: no disk state (avoids write races), no REPL sockets.
  cp "$PS/config/config-example.js" "$PS/config/config.js"
  printf '\nexports.nofswriting = true;\nexports.repl = false;\n' >> "$PS/config/config.js"
fi
mkdir -p "$ROOT/logs" "$PS/logs/repl"
(cd "$PS" && node build)  # once; concurrent per-server builds clobber each other
for ((i = 0; i < N; i++)); do
  PORT=$((BASE + i))
  (cd "$PS" && setsid nohup node pokemon-showdown start --skip-build --no-security "$PORT" < /dev/null > "$ROOT/logs/showdown-$PORT.log" 2>&1 &)
done
for ((i = 0; i < N; i++)); do
  PORT=$((BASE + i))
  for ((t = 0; t < 1200; t++)); do  # up to 10 min: startup reads ~1k files, slow on a busy network disk
    grep -q "listening" "$ROOT/logs/showdown-$PORT.log" 2>/dev/null && continue 2
    sleep 0.5
  done
  echo "Showdown on port $PORT failed to start:" >&2; tail -20 "$ROOT/logs/showdown-$PORT.log" >&2; exit 1
done
# The first websocket connection makes a server load its connection code, which on a slow disk can take longer
# than the clients' handshake timeout (poke-env, Foul Play): open one per server here and wait for it.
if [ -x "$ROOT/.venv/bin/python" ]; then
  "$ROOT/.venv/bin/python" - "$BASE" "$N" <<'PYEOF' || echo "warning: no websocket handshake on some port" >&2
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

async def main(base: int, n: int) -> None:
    await asyncio.wait_for(asyncio.gather(*(warm(base + i) for i in range(n))), 3600)

asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
PYEOF
fi
echo "Showdown servers up on ports $BASE-$((BASE + N - 1))"
