#!/bin/bash
# Search player (or raw network) vs Foul Play on one node: a local Showdown server, K Foul Play
# instances (1 core each, 1 s per move) and scripts/vs_foulplay.py. Results: runs/foulplay-<job>/.
# Foul Play is GPL-3.0 and NOT vendored: the first run clones it into FP_DIR (default: next to the
# repo) with its own venv, and patches its login for the local --no-security server.
#   sbatch scripts/slurm_foulplay.sbatch MODEL [N_PER_OPPONENT=50] [K_OPPONENTS=4] [--gen 9] [--fp-parallelism P] [vs_foulplay args]
# --gen 9: gen9randombattle. Foul Play uses its default (gen9, terastallization) engine in FP_DIR/.venv-gen9; we use
# the gen9 engine in .engine-gen9 (pip install --target of a GEN=9 scripts/build_engine.sh wheel) ahead of .venv's.
# MODEL must be "none" (pure engine search: the network is gen4-only), with --prior-weight 0 and an --eval-config
# (default for gen9: stock; the fitted one is src/porygon/data/eval_gen9_r1.json).
# --fp-parallelism P: Foul Play searches 2P states per move, P at a time (default 1: 2 states x 1 s on one core,
# 2 core-seconds per move; our default search is 4 x 1 s = 4 core-seconds). Each instance gets P cores.
#SBATCH --job-name=porygon-foulplay
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
MODEL=$1; N=${2:-50}; K=${3:-4}
case "$N$K" in *-*) echo "N_PER_OPPONENT and K_OPPONENTS come before any option (got N=$N K=$K)" >&2; exit 1 ;; esac
shift $(( $# < 3 ? $# : 3 ))
FP_PAR=1; GEN=4
while [ "${1:-}" = "--fp-parallelism" ] || [ "${1:-}" = "--gen" ]; do
  case "$1" in --fp-parallelism) FP_PAR=$2 ;; --gen) GEN=$2 ;; esac; shift 2
done
FORMAT=gen${GEN}randombattle
FP_VENV=.venv; FP_FEATURES=poke-engine/gen4
if [ "$GEN" = 9 ]; then
  FP_VENV=.venv-gen9; FP_FEATURES=poke-engine/terastallization
  [ "$MODEL" = none ] || { echo "--gen 9 needs MODEL none (the network is gen4-only)" >&2; exit 1; }
  [ -f "$ROOT/.engine-gen9/poke_engine/__init__.py" ] || { echo "no gen9 engine in .engine-gen9" >&2; exit 1; }
  export PORYGON_GEN=9 PYTHONPATH=$ROOT/.engine-gen9:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
fi
export PATH=$ROOT/.node/bin:$ROOT/.cargo/bin:$PATH CARGO_HOME=$ROOT/.cargo RUSTUP_HOME=$ROOT/.rustup
FP=${FP_DIR:-$(dirname "$ROOT")/foul-play}

if ! [ -x "$FP/$FP_VENV/bin/python" ]; then
  [ -d "$FP" ] || git clone -q https://github.com/pmariglia/foul-play.git "$FP"
  # the standalone CPython 3.12 the other venvs use (node images moved /usr/bin/python3 to 3.14)
  PY312=${PY312:-/vol/bitbucket/fr1226/python/cpython-3.12.16/bin/python3}; [ -x "$PY312" ] || PY312=python3
  "$PY312" -m venv "$FP/$FP_VENV"
  "$FP/$FP_VENV/bin/pip" install -q requests websockets==14.1 python-dateutil==2.8.0
  "$FP/$FP_VENV/bin/pip" install -q --no-cache-dir poke-engine==0.0.48 \
    --config-settings="build-args=--features $FP_FEATURES --no-default-features"
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
# duplicate battles: pairs of games with the same seeds and the teams swapped, true teams logged
[ -f .showdown-eval/dist/server/index.js ] || scripts/build_eval_server.sh
export PS_DIR=$ROOT/.showdown-eval PORYGON_DUPLICATE=$SLURM_JOB_ID PORYGON_DUPLICATE_LOG=$ROOT/$OUT/battles.jsonl
export PORYGON_DUPLICATE_TEAMS=$ROOT/$OUT/teams; mkdir -p "$PORYGON_DUPLICATE_TEAMS"  # true teams (--oracle)
export PORYGON_DUPLICATE_GROUP='^fp'
scripts/start_servers.sh 1 "$PORT"
LOOPS=()
trap 'kill "${LOOPS[@]}" 2>/dev/null || true
      pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $PORT\$" || true
      pkill -u "$USER" -f "run.py --websocket-uri ws://localhost:$PORT/" || true
      cp "$LOGS"/fp0.log "$OUT"/ 2>/dev/null || true; rm -rf "$LOGS"' EXIT

# Pull the Python environments into the page cache first: on a slow shared disk, modules imported lazily
# during a websocket handshake can take longer than the 10 s handshake timeout.
find "$FP/$FP_VENV" "$FP/fp" .venv/lib/python3.12/site-packages/{websockets,poke_env} -type f \( -name '*.py' -o -name '*.so*' \) -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
NAMES=()
for ((i = 0; i < K; i++)); do
  NAMES+=("fp${SLURM_JOB_ID}x$i")
  # restarted if it exits (e.g. "timed out during handshake" while the server is still slow to start)
  (cd "$FP" && while true; do "$FP_VENV"/bin/python run.py --websocket-uri "ws://localhost:$PORT/showdown/websocket" \
     --ps-username "fp${SLURM_JOB_ID}x$i" --bot-mode accept_challenge --pokemon-format "$FORMAT" \
     --search-time-ms 1000 --search-parallelism "$FP_PAR" --run-count 1000000 --log-level INFO; echo "restarting"; sleep 10; done) \
     < /dev/null >> "$LOGS/fp$i.log" 2>&1 &
  LOOPS+=($!)
done
for ((i = 0; i < K; i++)); do  # importing on a slow shared disk can take minutes
  for ((t = 0; t < 1800; t++)); do grep -qs "Waiting for a" "$LOGS/fp$i.log" && continue 2; sleep 1; done
  echo "Foul Play instance $i did not start:" >&2; tail -20 "$LOGS/fp$i.log" >&2; exit 1
done
echo "$K Foul Play instances up on port $PORT"

# our search: ~4 engine workers per simultaneous battle; leave FP_PAR cores per Foul Play instance + overhead
WORKERS=$(( $(nproc) - K * FP_PAR - 3 ))
echo "nproc $(nproc), $K Foul Play instances x $FP_PAR cores, $WORKERS engine workers for our search"
.venv/bin/python scripts/vs_foulplay.py "$MODEL" --player search --port "$PORT" --opponents "${NAMES[@]}" \
  --n "$N" --workers "$WORKERS" --results "$OUT/results.jsonl" --log-dir "$OUT" --format "$FORMAT" "$@" 2>&1 | tee "$OUT/vs.log" || true
.venv/bin/python scripts/analyze_benchmark.py "$OUT" 2>&1 | tee "$OUT/analysis.txt" || true
python3 - "$OUT/results.jsonl" <<'EOF'
import json, math, sys
r = [json.loads(line) for line in open(sys.argv[1])]
w, l = sum(x["status"] == "win" for x in r), sum(x["status"] == "loss" for x in r)
n = w + l
print(f"RESULT: {w}-{l} of {len(r)} battles = {w / max(n, 1):.1%} (±{1.96 * math.sqrt(max(w, 1) * max(l, 1) / max(n, 1) ** 3):.1%})")
EOF
