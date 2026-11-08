#!/bin/bash
# Raw network (v1 or v2) vs Foul Play, duplicate pairs: one duplicate-battle server, K Foul Play instances
# (1 s per state, --fp-parallelism P cores each, its default compute at P = 1) and scripts/vs_foulplay.py
# --player policy (sampling; --deterministic for argmax). N = total games (N/K per Foul Play instance).
#   sbatch scripts/netv2_foulplay.sbatch NET N [--gen 4|9] [--k K] [--fp-parallelism P] [--seed TAG] [vs_foulplay args]
# Foul Play (GPL-3.0, not vendored) runs from FP_DIR (default /vol/bitbucket/fr1226/foul-play), read only: its
# venvs (.venv gen4, .venv-gen9) and the local-login patch must already be there (slurm_foulplay.sbatch sets them up).
# Default K = min(20, nproc - 4). Output: runs/foulplay-net-<date>-<job>/ (result.txt, results.jsonl, battles.jsonl).
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
GEN=4; K=$(( $NPROC - 4 < 20 ? $NPROC - 4 : 20 )); FP_PAR=1; SEED=$SLURM_JOB_ID
while [ $# -gt 0 ]; do
  case "$1" in
    --gen) GEN=$2; shift 2 ;; --k) K=$2; shift 2 ;; --fp-parallelism) FP_PAR=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;; *) break ;;
  esac
done
netv2_gen "$GEN"
FP=${FP_DIR:-/vol/bitbucket/fr1226/foul-play}
FP_VENV=.venv; if [ "$GEN" = 9 ]; then FP_VENV=.venv-gen9; fi
[ -x "$FP/$FP_VENV/bin/python" ] || { echo "no Foul Play venv $FP/$FP_VENV (run slurm_foulplay.sbatch once)" >&2; exit 1; }
grep -q "patched for the Porygon benchmark" "$FP/fp/websocket_client.py" ||
  { echo "Foul Play's local login is not patched (slurm_foulplay.sbatch does it)" >&2; exit 1; }
PER=$(( (N + K - 1) / K ))
netv2_out foulplay-net
netv2_manifest foulplay-net "$ALL (K=$K x $PER games, fp-parallelism $FP_PAR, Foul Play $(git -C "$FP" rev-parse --short HEAD 2>/dev/null || true))" "$NET"
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
  (cd "$FP" && while true; do PYTHONDONTWRITEBYTECODE=1 PYTHONPATH= "$FP_VENV"/bin/python run.py \
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
"$PY" scripts/vs_foulplay.py "$NET" --player policy --port "$PORT" --opponents "${NAMES[@]}" --n "$PER" \
  --results "$OUT/results.jsonl" --format "$FORMAT" "$@" 2>&1 | tee "$OUT/vs.log" || true
"$PY" scripts/netv2_pool_ci.py "$OUT" 2>&1 | tee "$OUT/result.txt" || true
