#!/bin/bash
# Search player (or raw network) vs Foul Play on one node: a local Showdown server, K Foul Play
# instances (1 core each, 1 s per move) and scripts/vs_foulplay.py. Results: runs/foulplay-<job>/.
# Foul Play is GPL-3.0 and NOT vendored: the first run clones it into FP_DIR (default: next to the
# repo) with its own venv, and patches its login for the local --no-security server.
#   sbatch scripts/slurm_foulplay.sbatch MODEL [N_PER_OPPONENT=50] [K_OPPONENTS=4] [vs_foulplay args]
#SBATCH --job-name=porygon-foulplay
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
MODEL=$1; N=${2:-50}; K=${3:-4}; shift $(( $# < 3 ? $# : 3 ))
export PATH=$ROOT/.node/bin:$ROOT/.cargo/bin:$PATH CARGO_HOME=$ROOT/.cargo RUSTUP_HOME=$ROOT/.rustup
FP=${FP_DIR:-$(dirname "$ROOT")/foul-play}

if ! [ -x "$FP/.venv/bin/python" ]; then
  [ -d "$FP" ] || git clone -q https://github.com/pmariglia/foul-play.git "$FP"
  python3 -m venv "$FP/.venv"
  "$FP/.venv/bin/pip" install -q requests websockets==14.1 python-dateutil==2.8.0
  "$FP/.venv/bin/pip" install -q --no-cache-dir poke-engine==0.0.48 \
    --config-settings="build-args=--features poke-engine/gen4 --no-default-features"
fi
python3 - "$FP/fp/websocket_client.py" <<'EOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]); s = p.read_text()
marker = "patched for the Porygon benchmark"
if marker not in s:
    a = "        guest_login = self.password is None\n\n        if guest_login:"
    b = ("        guest_login = self.password is None\n\n"
         "        if guest_login and (\"localhost\" in self.address or \"127.0.0.1\" in self.address):\n"
         f"            # local --no-security server: no assertion needed ({marker})\n"
         "            await self.send_message(\"\", [\"/trn \" + self.username + \",0,\"])\n"
         "            await asyncio.sleep(3)\n"
         "            return self.username\n\n"
         "        if guest_login:")
    assert s.count(a) == 1, "Foul Play's login changed; update the patch"
    p.write_text(s.replace(a, b))
EOF

OUT=runs/foulplay-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT"
LOGS=${TMPDIR:-/tmp}/porygon-fp-$SLURM_JOB_ID; mkdir -p "$LOGS"  # Foul Play logs are chatty: keep them off the shared disk
PORT=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 + 15 ))
scripts/start_servers.sh 1 "$PORT"
LOOPS=()
trap 'kill "${LOOPS[@]}" 2>/dev/null || true
      pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $PORT\$" || true
      pkill -u "$USER" -f "run.py --websocket-uri ws://localhost:$PORT/" || true
      cp "$LOGS"/fp0.log "$OUT"/ 2>/dev/null || true; rm -rf "$LOGS"' EXIT

NAMES=()
for ((i = 0; i < K; i++)); do
  NAMES+=("fp${SLURM_JOB_ID}x$i")
  # restarted if it exits (e.g. "timed out during handshake" while the server is still slow to start)
  (cd "$FP" && while true; do .venv/bin/python run.py --websocket-uri "ws://localhost:$PORT/showdown/websocket" \
     --ps-username "fp${SLURM_JOB_ID}x$i" --bot-mode accept_challenge --pokemon-format gen4randombattle \
     --search-time-ms 1000 --run-count 1000000 --log-level INFO; echo "restarting"; sleep 10; done) \
     < /dev/null >> "$LOGS/fp$i.log" 2>&1 &
  LOOPS+=($!)
done
for ((i = 0; i < K; i++)); do  # importing on a slow shared disk can take minutes
  for ((t = 0; t < 1800; t++)); do grep -qs "Waiting for a" "$LOGS/fp$i.log" && continue 2; sleep 1; done
  echo "Foul Play instance $i did not start:" >&2; tail -20 "$LOGS/fp$i.log" >&2; exit 1
done
echo "$K Foul Play instances up on port $PORT"

# our search: ~4 engine workers per simultaneous battle; leave a core per Foul Play instance + overhead
WORKERS=$(( $(nproc) - K - 3 ))
.venv/bin/python scripts/vs_foulplay.py "$MODEL" --player search --port "$PORT" --opponents "${NAMES[@]}" \
  --n "$N" --workers "$WORKERS" --results "$OUT/results.jsonl" "$@" 2>&1 | tee "$OUT/vs.log" || true
python3 - "$OUT/results.jsonl" <<'EOF'
import json, math, sys
r = [json.loads(line) for line in open(sys.argv[1])]
w, l = sum(x["status"] == "win" for x in r), sum(x["status"] == "loss" for x in r)
n = w + l
print(f"RESULT: {w}-{l} of {len(r)} battles = {w / max(n, 1):.1%} (±{1.96 * math.sqrt(max(w, 1) * max(l, 1) / max(n, 1) ** 3):.1%})")
EOF
