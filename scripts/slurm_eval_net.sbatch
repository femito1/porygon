#!/bin/bash
# Raw-network (no search) benchmarks on the duplicate-battle Showdown: pairs of games with the same seeds and the
# teams swapped (PORYGON_DUPLICATE*), so results are scored by pair means. Submit from the data-stream code dir
# (scripts/setup_ws_data.sh); full runs need the shared 24-CPU --partition=long slot (claim it in
# .orchestra/ledger.md first); the defaults below are that slot. Never --export.
#   sbatch scripts/slurm_eval_net.sbatch MODE NET N [--ref REF] [args for the underlying program]
# MODE
#   search9  NET (gen9) vs our gen9 search: porygon.search none --opponent policy --opp-model NET (pure engine
#            search, 4 x 1 s, stock eval unless --eval-config src/porygon/data/eval_gen9_r1.json); N battles.
#   fp9      NET (gen9) vs gen9 Foul Play (1 core, 1 s per move each): scripts/vs_foulplay.py NET --player policy,
#            K = nproc - 3 Foul Play instances, N battles per instance.
#   search4  NET (gen4) vs our gen4 search (REF prior 0.25, fitted eval, 4 x 1 s); N battles.
#   ppo4     NET vs REF, both raw networks (porygon.tournament, sampled actions), gen4; N battles.
#   h2h9     the same in gen9 (--ref required).
# Env overrides for runs inside another job (srun --overlap): EVAL_PORT, EVAL_K (Foul Play instances),
# PS_SKIP_BUILD=1 (do not rebuild the Showdown dist another job is serving from).
# REF (search4 / ppo4) defaults to gloria's 500M PPO network. Output: runs/evalnet-MODE-<date>-<job>/ with
# battles.jsonl (duplicate log), the program's log, summary.txt ("NET win rate: x% ± per game | pairs: y% ± z%").
# Pool several runs: .venv/bin/python -m porygon.bc summarize DIR [DIR ...] --ours REGEX (regex in summary.txt).
# Dry run (a few games) inside the data stream's running job, sharing its CPUs (short searches, few workers):
#   srun --jobid=JOB --overlap --ntasks=1 bash -c 'export SLURM_SUBMIT_DIR=$PWD PS_SKIP_BUILD=1; \
#     bash scripts/slurm_eval_net.sbatch search9 NET 4 --workers 2 --concurrency 1 --ms 300 --samples 2'
#   (fp9: EVAL_K=2 ... fp9 NET 2). Each run's servers live on their own port (+11), so other jobs' are untouched.
#SBATCH --job-name=ws-data-evalnet
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
MODE=$1; NET=$(realpath "$2"); N=$3; shift 3
REF=/vol/bitbucket/fr1226/gloria/runs/20261025-195826-301040/final_model.zip
if [ "${1:-}" = --ref ]; then REF=$(realpath "$2"); shift 2; fi
G=/vol/bitbucket/fr1226/gloria
[ -f .showdown-eval/dist/server/index.js ] && [ -L .venv ] || { echo "run scripts/setup_ws_data.sh first" >&2; exit 1; }
case "$MODE" in search9|fp9|h2h9) GEN=9 ;; search4|ppo4) GEN=4 ;; *) echo "unknown MODE $MODE" >&2; exit 1 ;; esac
FORMAT=gen${GEN}randombattle
export PATH=$ROOT/.node/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$ROOT/src
[ "$GEN" = 9 ] && export PORYGON_GEN=9 PYTHONPATH=$ROOT/.engine-gen9:$ROOT/src
DEV=cpu; nvidia-smi -L > /dev/null 2>&1 && DEV=cuda
OUT=runs/evalnet-$MODE-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT"
PORT=${EVAL_PORT:-$(( 20000 + (SLURM_JOB_ID % 2000) * 16 + 11 ))}  # +13 is slurm_searchlabels, +14/15 search/foulplay
echo "$MODE: NET $NET, REF $REF, N $N, $FORMAT, $(nproc) CPUs on $(hostname), args $*; commit $(cat COMMIT 2>/dev/null)" \
  | tee "$OUT/manifest.txt"
# Our private copy of the duplicate-battle server (setup_ws_data.sh): start_servers.sh writes into PS_DIR.
export PS_DIR=$ROOT/.showdown-eval PORYGON_DUPLICATE=$SLURM_JOB_ID PORYGON_DUPLICATE_LOG=$ROOT/$OUT/battles.jsonl
export PORYGON_DUPLICATE_TEAMS=$ROOT/$OUT/teams; mkdir -p "$PORYGON_DUPLICATE_TEAMS"
# Player names: porygon.search: "search<tag>" vs "opppolicy<tag>"; vs_foulplay: "pol<tag>x<i>" vs "fp<job>x<i>";
# porygon.tournament: "porygon0x<r>" (NET, first model) vs "porygon1x<r>". GROUP fixes who gets team A in
# game 1 of a pair (it must match exactly one of the two names); OURS picks NET's games.
case "$MODE" in
  search*) GROUP='^search'; OURS='^opppolicy' ;;
  fp9) GROUP='^fp'; OURS='^pol' ;;
  *) GROUP='^porygon0x'; OURS='^porygon0x' ;;
esac
export PORYGON_DUPLICATE_GROUP=$GROUP
find .venv/lib/python3.12/site-packages/{websockets,poke_env,torch} -type f \( -name '*.py' -o -name '*.so*' \) \
  -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
scripts/start_servers.sh 1 "$PORT"
LOOPS=()
FP=/vol/bitbucket/fr1226/foul-play
LOGS=${TMPDIR:-/tmp}/ws-data-fp-$SLURM_JOB_ID
trap 'kill "${LOOPS[@]}" 2>/dev/null || true
      pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $PORT\$" || true
      pkill -u "$USER" -f "run.py --websocket-uri ws://localhost:$PORT/" || true
      [ -d "$LOGS" ] && { cp "$LOGS"/fp0.log "$OUT"/ 2>/dev/null || true; rm -rf "$LOGS"; }' EXIT
WORKERS=$(( $(nproc) > 3 ? $(nproc) - 2 : 1 ))
CONC=$(( WORKERS / 5 > 1 ? WORKERS / 5 : 1 ))  # ~4-5 engine workers per simultaneous searched battle
case "$MODE" in
  search9)
    .venv/bin/python -m porygon.search none --opponent policy --opp-model "$NET" --n-battles "$N" --format "$FORMAT" \
      --port "$PORT" --workers "$WORKERS" --concurrency "$CONC" --device "$DEV" --log-dir "$OUT" "$@" 2>&1 \
      | tee "$OUT/search.log" || true ;;
  search4)
    .venv/bin/python -m porygon.search "$REF" --opponent policy --opp-model "$NET" --n-battles "$N" --format "$FORMAT" \
      --port "$PORT" --workers "$WORKERS" --concurrency "$CONC" --device "$DEV" --log-dir "$OUT" "$@" 2>&1 \
      | tee "$OUT/search.log" || true ;;
  ppo4|h2h9)
    [ "$MODE" = h2h9 ] && [ "$REF" = "$G/runs/20261025-195826-301040/final_model.zip" ] && \
      { echo "h2h9 needs --ref (a gen9 network)" >&2; exit 1; }
    .venv/bin/python -m porygon.tournament "$NET" "$REF" --baselines --format "$FORMAT" --port "$PORT" \
      --n-battles "$N" --concurrency "$(( $(nproc) * 2 ))" "$@" 2>&1 | tee "$OUT/tournament.log" || true ;;
  fp9)
    grep -q "patched for the Porygon benchmark" "$FP/fp/websocket_client.py" || \
      { echo "Foul Play's local-login patch is missing: not patching gloria's shared copy; ask the orchestrator" >&2; exit 1; }
    K=${EVAL_K:-$(( $(nproc) > 4 ? $(nproc) - 3 : 2 ))}
    mkdir -p "$LOGS"
    find "$FP/.venv-gen9" "$FP/fp" -type f \( -name '*.py' -o -name '*.so*' \) -print0 2>/dev/null \
      | xargs -0 -P4 -n64 cat > /dev/null || true
    NAMES=()
    for ((i = 0; i < K; i++)); do
      NAMES+=("fp${SLURM_JOB_ID}x$i")
      (cd "$FP" && while true; do .venv-gen9/bin/python run.py --websocket-uri "ws://localhost:$PORT/showdown/websocket" \
         --ps-username "fp${SLURM_JOB_ID}x$i" --bot-mode accept_challenge --pokemon-format "$FORMAT" \
         --search-time-ms 1000 --search-parallelism 1 --run-count 1000000 --log-level INFO; echo restarting; sleep 10; done) \
         < /dev/null >> "$LOGS/fp$i.log" 2>&1 &
      LOOPS+=($!)
    done
    for ((i = 0; i < K; i++)); do
      for ((t = 0; t < 1800; t++)); do grep -qs "Waiting for a" "$LOGS/fp$i.log" && continue 2; sleep 1; done
      echo "Foul Play instance $i did not start:" >&2; tail -20 "$LOGS/fp$i.log" >&2; exit 1
    done
    echo "$K Foul Play instances up on port $PORT"
    .venv/bin/python scripts/vs_foulplay.py "$NET" --player policy --port "$PORT" --opponents "${NAMES[@]}" \
      --n "$N" --results "$OUT/results.jsonl" --format "$FORMAT" --device "$DEV" "$@" 2>&1 | tee "$OUT/vs.log" || true ;;
esac
case "$MODE" in search*) .venv/bin/python scripts/analyze_benchmark.py "$OUT" > "$OUT/analysis.txt" 2>&1 || true ;; esac
{ echo "$MODE NET=$NET REF=$REF"; echo -n "NET win rate, "
  .venv/bin/python -m porygon.bc summarize "$OUT" --ours "$OURS"; } | tee "$OUT/summary.txt"
