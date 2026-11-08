#!/usr/bin/env bash
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --job-name=ws-fastenv-eng-spike
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-fastenv/engine_spike/logs/slurm-%j.out
# The engine-fidelity spike on one 4-CPU node: build the spike engines (unless present), generate team pairs,
# play every (gen, sim, policy) cell with 4 parallel workers, then time each simulator alone on one core.
#   sbatch run_spike.sbatch [PAIRS=2000] [REPS=4]
# Layout under $W: repo/{src,fastenv/js,scripts/fastenv/engine_spike} (rsync'd), pe-src/ (patched .poke-engine copy).
set -euo pipefail
W=/vol/bitbucket/fr1226/porygon-ws-fastenv/engine_spike
G=/vol/bitbucket/fr1226/gloria
PAIRS=${1:-2000}
REPS=${2:-4}
S=$W/repo/scripts/fastenv/engine_spike
OUT=$W/out
mkdir -p "$OUT" "$W/logs"
export PORYGON_PS=$G/pokemon-showdown NODE=$G/.node/bin/node PY=$G/.venv/bin/python
cat "$PY" "$NODE" > /dev/null; cat "$G"/.node/bin/node > /dev/null   # pre-read off the slow disk
hostname; nproc; date
[ -f "$W/eng9/poke_engine/__init__.py" ] && [ -f "$W/eng4/poke_engine/__init__.py" ] || W=$W bash "$S/build_spike_engine.sh"
cd "$S"
for g in 9 4; do
  [ -s "$OUT/pairs$g.jsonl" ] || "$NODE" teams.js "$PAIRS" gen${g}randombattle 1 > "$OUT/pairs$g.jsonl"
  "$NODE" dex_tables.js $g > "$OUT/tables$g.json"
done
date
# task lines: name|command
tasks=$OUT/tasks.txt
: > "$tasks"
for g in 9 4; do
  for pol in random greedy; do
    echo "ps$g-$pol|$NODE ps_games.js --teams $OUT/pairs$g.jsonl --format gen${g}randombattle --policy $pol --reps $REPS" >> "$tasks"
    echo "eng$g-$pol|env PORYGON_GEN=$g PYTHONPATH=$W/eng$g:$W/repo/src $PY engine_games.py --teams $OUT/pairs$g.jsonl --tables $OUT/tables$g.json --policy $pol --reps $REPS" >> "$tasks"
  done
  echo "native$g-random|env PORYGON_GEN=$g PYTHONPATH=$W/eng$g:$W/repo/src $PY engine_games.py --native --teams $OUT/pairs$g.jsonl --reps $REPS" >> "$tasks"
done
run_task() { name=${1%%|*}; cmd=${1#*|}; $cmd > "$OUT/$name.jsonl" 2> "$OUT/$name.err" || echo "FAILED $name"; echo "done $name $(date +%T)"; }
export -f run_task; export OUT
tr '\n' '\0' < "$tasks" | xargs -0 -P 4 -I{} bash -c 'run_task "$@"' _ {}
date
# timing: each simulator alone on one core (the others idle), same pairs and policy (random), after a warm-up
for g in 9 4; do
  echo "gen$g" >> "$OUT/timing.txt"
  "$NODE" ps_games.js --teams "$OUT/pairs$g.jsonl" --format gen${g}randombattle --policy random --reps 1 --count 50 > /dev/null 2>&1
  "$NODE" ps_games.js --teams "$OUT/pairs$g.jsonl" --format gen${g}randombattle --policy random --reps 2 --count 300 2>> "$OUT/timing.txt" > /dev/null
  env PORYGON_GEN=$g PYTHONPATH=$W/eng$g:$W/repo/src $PY engine_games.py --teams "$OUT/pairs$g.jsonl" --policy random --reps 2 --count 300 2>> "$OUT/timing.txt" > /dev/null
  env PORYGON_GEN=$g PYTHONPATH=$W/eng$g:$W/repo/src $PY engine_games.py --native --teams "$OUT/pairs$g.jsonl" --reps 20 --count 300 2>> "$OUT/timing.txt" > /dev/null
done
cat "$OUT/timing.txt"
date
