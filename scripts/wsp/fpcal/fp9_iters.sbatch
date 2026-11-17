#!/bin/bash
# wsp fpcal (11-17): COPY of data's scripts/netv2_foulplay_fp9.sbatch (md5 7476be8b, the conf800 harness) with ONE
# change: Foul Play's MCTS runs a FIXED iteration count, --fp-iters N per 1000 ms of nominal budget (2 states x N,
# 4 x N/2 early), through scripts/wsp/fpcal/fp_fixed_iters.py, instead of --search-time-ms 1000 wall clock.
# Bookkeeping only besides that: OUT = $WSP/fpcal/runs/foulplay-net-<date>-<job>-<N> (not netv2's runs/), cpu.txt
# (lscpu), and a background check (fp_iters_check.py) that every "Iterations i: V" line of Foul Play's logs is N or
# N/2 (rounded up to 1000), or 1000 on a single-option state (11-17 fix, see fp_iters_check.py): a mismatch writes
# OUT/ITERS_MISMATCH and stops the games (the arm is then void). Run with cwd = the netv2
# cluster dir (src/, scripts/, .showdown-eval/ read only), like conf800 runs the fp9 harness:
#   cd $C && bash $WSP/fpcal/code/fp9_iters.sbatch NET N --gen 9 --seed S --fp-iters ITERS [vs_foulplay args]
#SBATCH --job-name=ws-netv2-foulplay
#SBATCH --partition=long
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
source scripts/netv2_common.sh
ALL="$*"
NET=$1; N=$2; shift 2
GEN=4; K=$(( $NPROC - 4 < 20 ? $NPROC - 4 : 20 )); FP_PAR=1; SEED=$SLURM_JOB_ID; FP_ITERS=
while [ $# -gt 0 ]; do
  case "$1" in
    --gen) GEN=$2; shift 2 ;; --k) K=$2; shift 2 ;; --fp-parallelism) FP_PAR=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;; --fp-iters) FP_ITERS=$2; shift 2 ;; *) break ;;
  esac
done
netv2_gen "$GEN"
[[ "$FP_ITERS" =~ ^[0-9]+000$ ]] || { echo "--fp-iters N required (a multiple of 1000, got '$FP_ITERS')" >&2; exit 2; }
WRAP=/vol/bitbucket/fr1226/porygon-ws-wsp/fpcal/code/fp_fixed_iters.py
[ -f "$WRAP" ] || { echo "no $WRAP" >&2; exit 1; }
# wsp: results under the wsp cluster dir, one dir per arm
netv2_out() { OUT=/vol/bitbucket/fr1226/porygon-ws-wsp/fpcal/runs/$1-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID-$FP_ITERS; mkdir -p "$OUT"; }
FP=${FP_DIR:-/vol/bitbucket/fr1226/foul-play}
FP_VENV=.venv; if [ "$GEN" = 9 ]; then FP_VENV=.venv-gen9; fi
# data-stream fix: gen9 Foul Play imports poke_engine from gloria's .engine-gen9 (its .venv-gen9 has none), exactly
# as gloria's scripts/slurm_foulplay.sbatch launches it (the 2026-11-07 gen9 baselines: PYTHONPATH=.engine-gen9:src).
FP_PYTHONPATH=; if [ "$GEN" = 9 ]; then FP_PYTHONPATH=$GLORIA/.engine-gen9:$GLORIA/src; fi
[ -x "$FP/$FP_VENV/bin/python" ] || { echo "no Foul Play venv $FP/$FP_VENV (run slurm_foulplay.sbatch once)" >&2; exit 1; }
grep -q "patched for the Porygon benchmark" "$FP/fp/websocket_client.py" ||
  { echo "Foul Play's local login is not patched (slurm_foulplay.sbatch does it)" >&2; exit 1; }
PER=$(( (N + K - 1) / K ))
netv2_out foulplay-net
netv2_manifest foulplay-net "$ALL (K=$K x $PER games, fp-parallelism $FP_PAR, Foul Play $(git -C "$FP" rev-parse --short HEAD 2>/dev/null || true), FIXED $FP_ITERS iterations per 1000 ms via $(md5sum < "$WRAP" | cut -c1-8))" "$NET"
lscpu > "$OUT/cpu.txt" 2>&1 || true
LOGS=${TMPDIR:-/tmp}/ws-netv2-fp-$SLURM_JOB_ID; mkdir -p "$LOGS"  # Foul Play logs are chatty: off the shared disk
PORT=$(netv2_base_port)
LOOPS=()
trap 'kill "${LOOPS[@]}" 2>/dev/null || true; netv2_stop_servers
      pkill -u "$USER" -f "run.py --websocket-uri ws://localhost:$PORT/" || true
      cp "$LOGS"/fp0.log "$OUT"/ 2>/dev/null || true; rm -rf "$LOGS"' EXIT
netv2_server "$PORT" "$OUT" "$SEED" '^fp'
netv2_preread "$FP/$FP_VENV" "$FP/fp" "$GLORIA"/.venv/lib/python3.12/site-packages/torch
netv2_wait_servers
NAMES=()
for ((i = 0; i < K; i++)); do
  NAMES+=("fp${SLURM_JOB_ID}x$i")
  # restarted if it exits (e.g. "timed out during handshake" while the server is still slow to start)
  (cd "$FP" && while true; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FP_PYTHONPATH" "$FP_VENV"/bin/python "$WRAP" --iters-per-s "$FP_ITERS" run.py \
     --websocket-uri "ws://localhost:$PORT/showdown/websocket" --ps-username "fp${SLURM_JOB_ID}x$i" \
     --bot-mode accept_challenge --pokemon-format "$FORMAT" --search-time-ms 1000 --search-parallelism "$FP_PAR" \
     --run-count 1000000 --log-level INFO; echo "restarting"; sleep 10; done) < /dev/null >> "$LOGS/fp$i.log" 2>&1 &
  LOOPS+=($!)
done
for ((i = 0; i < K; i++)); do  # importing on a slow shared disk can take minutes
  for ((t = 0; t < 1800; t++)); do grep -qs "Waiting for a" "$LOGS/fp$i.log" && continue 2; sleep 1; done
  echo "Foul Play instance $i did not start:" >&2; tail -20 "$LOGS/fp$i.log" >&2; exit 1
done
echo "$K Foul Play instances up on port $PORT"
# wsp: every logged per-state iteration count must be FP_ITERS (1000 ms states) or FP_ITERS/2 rounded up to 1000
# (500 ms states), or 1000 on a single-option state (poke-engine's own quick exit; fp_iters_check.py); checked every
# 60 s and once more after the games; a mismatch voids the arm and stops the games
CHECK=$(dirname "$WRAP")/fp_iters_check.py
[ -f "$CHECK" ] || { echo "no $CHECK" >&2; exit 1; }
iters_check() {
  local msg
  if msg=$("$PY" -I "$CHECK" "$FP_ITERS" "$LOGS"/fp*.log 2>&1); then echo "$(date -u +%FT%T) $msg" > "$OUT/ITERS_OK"; return 0; fi
  echo "$(date -u +%FT%T) $msg" > "$OUT/ITERS_MISMATCH"; return 1
}
( while sleep 60; do
    iters_check || { pkill -u "$USER" -f "^$PY scripts/vs_foulplay.py .*--port $PORT " || true; exit 0; }
  done ) &
LOOPS+=($!)
"$PY" scripts/vs_foulplay.py "$NET" --player policy --port "$PORT" --opponents "${NAMES[@]}" --n "$PER" \
  --results "$OUT/results.jsonl" --format "$FORMAT" "$@" 2>&1 | tee "$OUT/vs.log" || true
iters_check || true  # the games of the last minute
"$PY" scripts/netv2_pool_ci.py "$OUT" 2>&1 | tee "$OUT/result.txt" || true
