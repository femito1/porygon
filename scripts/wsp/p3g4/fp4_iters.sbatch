#!/bin/bash
# wsp P3 (2026-11-24): gen4 Foul Play at a FIXED iteration count, for both P3 arms. COPY of the fpcal harness
# scripts/wsp/fpcal/fp9_iters.sbatch (md5 82a207a6 = data's netv2_foulplay_fp9.sbatch 7476be8b, the conf800 / F2030
# harness, + Foul Play under fp_fixed_iters.py + the iteration check; for --gen 4 it is netv2's netv2_foulplay.sbatch
# 35e7e30b, the v2g4-10 harness, + the same two changes). Changes, all bookkeeping except (1):
#  (1) --player search: OUR SEARCH PLAYER, run exactly as gloria's scripts/slurm_foulplay.sbatch runs it (the fp4-04
#      harness): gloria's code (cwd gloria, PYTHONPATH gloria/src, gloria/scripts/vs_foulplay.py, gloria/.venv and
#      its gen4 engine), --workers = nproc - K * fp-parallelism - 3 (fp4-04: 24 - 4 - 3 = 17), and the job's own
#      OMP_NUM_THREADS / MKL_NUM_THREADS, as gloria's harness left them (netv2_common.sh sets both to 1). No --log-dir
#      (decision logs: disk). --player policy (default) = fp9_iters.sbatch's raw-network player, unchanged.
#  (2) WRAP / CHECK = the P3 copies in $W/council/p3-gen4/code/ (fp_fixed_iters.py with the gen4 guards).
#  (3) OUT = $W/council/p3-gen4/runs/foulplay-<--tag>-<date>-<job>; fp0.log kept gzipped; PYTHONDONTWRITEBYTECODE=1
#      for our side as well (gloria and netv2 are read only).
# Foul Play: $FP/.venv (gen4, stock poke-engine 0.0.48 gen4 build), K instances x --fp-parallelism 1, 1000 ms nominal
# per state => FIXED --fp-iters N iterations per 1000 ms (2 states x N; early 4 x N/2). Duplicate-battle server =
# netv2's .showdown-eval (netv2_common.sh netv2_server: pairs seeded from --seed, Foul Play = team A in game 0).
# Run with cwd = the netv2 cluster dir (its src/, scripts/, .showdown-eval/ read only):
#   cd $C && SLURM_SUBMIT_DIR=$C bash $W/council/p3-gen4/code/fp4_iters.sbatch NET N --gen 4 --player policy|search \
#     --k K --seed S --fp-iters ITERS --tag NAME [--port-offset 0-15] [vs_foulplay args]
#SBATCH --job-name=ws-wsp-fp4iters
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
# the job's own OpenMP / MKL thread settings, before netv2_common.sh sets both to 1: the search player gets them back,
# as gloria's slurm_foulplay.sbatch (the fp4-04 harness) left them alone
OMP0=${OMP_NUM_THREADS-@unset@}; MKL0=${MKL_NUM_THREADS-@unset@}
source scripts/netv2_common.sh
ALL="$*"
NET=$1; N=$2; shift 2
GEN=4; K=; FP_PAR=1; SEED=$SLURM_JOB_ID; FP_ITERS=; PLAYER=policy; TAG=net; PORT_OFF=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gen) GEN=$2; shift 2 ;; --k) K=$2; shift 2 ;; --fp-parallelism) FP_PAR=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;; --fp-iters) FP_ITERS=$2; shift 2 ;; --player) PLAYER=$2; shift 2 ;;
    --tag) TAG=$2; shift 2 ;; --port-offset) PORT_OFF=$2; shift 2 ;; *) break ;;
  esac
done
netv2_gen "$GEN"
[[ "$FP_ITERS" =~ ^[0-9]+000$ ]] || { echo "--fp-iters N required (a multiple of 1000, got '$FP_ITERS')" >&2; exit 2; }
case "$PLAYER" in policy|search) ;; *) echo "--player policy|search (got '$PLAYER')" >&2; exit 2 ;; esac
[[ "$TAG" =~ ^[A-Za-z0-9_.]+$ ]] || { echo "--tag: letters, digits, _ . only (got '$TAG')" >&2; exit 2; }
[[ "$PORT_OFF" =~ ^([0-9]|1[0-5])$ ]] || { echo "--port-offset 0-15 (got '$PORT_OFF')" >&2; exit 2; }
[ -n "$K" ] || K=$(( $NPROC - 4 < 20 ? $NPROC - 4 : 20 ))
P3=/vol/bitbucket/fr1226/porygon-ws-wsp/council/p3-gen4
WRAP=$P3/code/fp_fixed_iters.py
[ -f "$WRAP" ] || { echo "no $WRAP" >&2; exit 1; }
netv2_out() { OUT=$P3/runs/$1-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT"; }
FP=${FP_DIR:-/vol/bitbucket/fr1226/foul-play}
FP_VENV=.venv; if [ "$GEN" = 9 ]; then FP_VENV=.venv-gen9; fi
# gen9 Foul Play imports poke_engine from gloria's .engine-gen9 (its .venv-gen9 has none); gen4 Foul Play uses its
# own venv's engine (as in netv2_foulplay.sbatch)
FP_PYTHONPATH=; if [ "$GEN" = 9 ]; then FP_PYTHONPATH=$GLORIA/.engine-gen9:$GLORIA/src; fi
[ -x "$FP/$FP_VENV/bin/python" ] || { echo "no Foul Play venv $FP/$FP_VENV (run slurm_foulplay.sbatch once)" >&2; exit 1; }
grep -q "patched for the Porygon benchmark" "$FP/fp/websocket_client.py" ||
  { echo "Foul Play's local login is not patched (slurm_foulplay.sbatch does it)" >&2; exit 1; }
WORKERS=
if [ "$PLAYER" = search ]; then
  WORKERS=$(( NPROC - K * FP_PAR - 3 ))  # gloria's slurm_foulplay.sbatch: leave FP_PAR cores per Foul Play + overhead
  [ "$WORKERS" -ge 1 ] || { echo "no cores left for our search ($NPROC cpus, K $K)" >&2; exit 2; }
fi
PER=$(( (N + K - 1) / K ))
export PYTHONDONTWRITEBYTECODE=1
netv2_out "foulplay-$TAG"
netv2_manifest "foulplay-$TAG" "$ALL (player $PLAYER, K=$K x $PER games, fp-parallelism $FP_PAR${WORKERS:+, $WORKERS engine workers}, Foul Play $(git -C "$FP" rev-parse --short HEAD 2>/dev/null || true), FIXED $FP_ITERS iterations per 1000 ms via $(md5sum < "$WRAP" | cut -c1-8))" "$NET"
if [ "$PLAYER" = search ]; then
  { echo "search player code: $GLORIA (scripts/vs_foulplay.py $(md5sum < "$GLORIA/scripts/vs_foulplay.py" | cut -c1-8)," \
         "src sha256 $(cd "$GLORIA" && find src/ -name '*.py' | sort | xargs cat | sha256sum | cut -c1-16)," \
         "engine $(md5sum < "$GLORIA"/.venv/lib/python3.12/site-packages/poke_engine/poke_engine.cpython-312-x86_64-linux-gnu.so | cut -c1-8));" \
         "its OMP_NUM_THREADS $OMP0, MKL_NUM_THREADS $MKL0 (the job's own)"
  } | tee -a "$OUT/manifest.txt"
fi
lscpu > "$OUT/cpu.txt" 2>&1 || true
LOGS=${TMPDIR:-/tmp}/ws-wsp-fp4-$SLURM_JOB_ID-$TAG; mkdir -p "$LOGS"  # Foul Play logs are chatty: off the shared disk
PORT=$(( $(netv2_base_port) + PORT_OFF ))  # a job's 16-port block: one port per arm run, so runs never share one
LOOPS=()
trap 'kill "${LOOPS[@]}" 2>/dev/null || true; netv2_stop_servers
      pkill -u "$USER" -f "run.py --websocket-uri ws://localhost:$PORT/" || true
      gzip -c "$LOGS"/fp0.log > "$OUT"/fp0.log.gz 2>/dev/null || true; rm -rf "$LOGS"' EXIT
netv2_server "$PORT" "$OUT" "$SEED" '^fp'
netv2_preread "$FP/$FP_VENV" "$FP/fp" "$GLORIA"/.venv/lib/python3.12/site-packages/{torch,poke_engine} "$GLORIA/src"
netv2_wait_servers
NAMES=()
for ((i = 0; i < K; i++)); do
  NAMES+=("fp${SLURM_JOB_ID}x$i")
  # restarted if it exits (e.g. "timed out during handshake" while the server is still slow to start)
  (cd "$FP" && while true; do PYTHONPATH="$FP_PYTHONPATH" "$FP_VENV"/bin/python "$WRAP" --iters-per-s "$FP_ITERS" run.py \
     --websocket-uri "ws://localhost:$PORT/showdown/websocket" --ps-username "fp${SLURM_JOB_ID}x$i" \
     --bot-mode accept_challenge --pokemon-format "$FORMAT" --search-time-ms 1000 --search-parallelism "$FP_PAR" \
     --run-count 1000000 --log-level INFO; echo "restarting"; sleep 10; done) < /dev/null >> "$LOGS/fp$i.log" 2>&1 &
  LOOPS+=($!)
done
for ((i = 0; i < K; i++)); do  # importing on a slow shared disk can take minutes
  for ((t = 0; t < 1800; t++)); do grep -qs "Waiting for a" "$LOGS/fp$i.log" && continue 2; sleep 1; done
  echo "Foul Play instance $i did not start:" >&2; tail -20 "$LOGS/fp$i.log" >&2; exit 1
done
echo "$K Foul Play instances up on port $PORT ($(grep -h -m1 '^fp_fixed_iters:' "$LOGS/fp0.log" || echo 'no launcher line'))"
# every logged per-state iteration count must be FP_ITERS (1000 ms states) or FP_ITERS/2 rounded up to 1000
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
if [ "$PLAYER" = search ]; then
  ENVU=(); ENVS=()
  if [ "$OMP0" = @unset@ ]; then ENVU+=(-u OMP_NUM_THREADS); else ENVS+=("OMP_NUM_THREADS=$OMP0"); fi
  if [ "$MKL0" = @unset@ ]; then ENVU+=(-u MKL_NUM_THREADS); else ENVS+=("MKL_NUM_THREADS=$MKL0"); fi
  (cd "$GLORIA" && env "${ENVU[@]}" "${ENVS[@]}" PYTHONPATH="$GLORIA/src" "$PY" scripts/vs_foulplay.py "$NET" \
     --player search --port "$PORT" --opponents "${NAMES[@]}" --n "$PER" --workers "$WORKERS" \
     --results "$OUT/results.jsonl" --format "$FORMAT" "$@") 2>&1 | tee "$OUT/vs.log" || true
else
  "$PY" scripts/vs_foulplay.py "$NET" --player policy --port "$PORT" --opponents "${NAMES[@]}" --n "$PER" \
    --results "$OUT/results.jsonl" --format "$FORMAT" "$@" 2>&1 | tee "$OUT/vs.log" || true
fi
iters_check || true  # the games of the last minute
"$PY" scripts/netv2_pool_ci.py "$OUT" 2>&1 | tee "$OUT/result.txt" || true
