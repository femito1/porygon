#!/bin/bash
# Engine-only self-play (python -m porygon.selfplay) on one node: training data for the leaf evaluation, or
# evaluation head-to-heads. No Showdown server, no GPU work (the partitions require one).
#   sbatch scripts/slurm_selfplay.sbatch [--gen 9] MODE PAIRS [porygon.selfplay args]
#   --gen 9: gen9 random battles with the gen9 engine in .engine-gen9 ahead of .venv's (as slurm_foulplay.sbatch):
#            .venv/bin/python -m pip install --no-deps --target .engine-gen9 <wheel from WHEEL_DIR=... GEN=9 build_engine.sh>
#   e.g. ... data 20000 --iterations 20000 --explore 0.1
#        ... h2h 2000 --a runs/evalfit/r1/full.json --b stock
#SBATCH --job-name=porygon-selfplay
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
GEN=4; PY=.venv/bin/python
if [ "${1:-}" = "--gen" ]; then GEN=$2; shift 2; fi
export PORYGON_GEN=$GEN PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
if [ "$GEN" != 4 ]; then
  [ -f "$ROOT/.engine-gen$GEN/poke_engine/__init__.py" ] || { echo "no gen$GEN engine in .engine-gen$GEN" >&2; exit 1; }
  export PYTHONPATH=$ROOT/.engine-gen$GEN:$PYTHONPATH
fi
MODE=$1; PAIRS=$2; shift 2
export PATH=$ROOT/.node/bin:$PATH
OUT=runs/selfplay-$MODE-$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$OUT"
node scripts/gen_teams.js "$PAIRS" "gen${GEN}randombattle" > "$OUT/teams.jsonl"
echo "selfplay gen$GEN $MODE $PAIRS $*"
"$PY" -m porygon.selfplay "$MODE" --teams "$OUT/teams.jsonl" --out "$OUT" --workers "$(nproc)" "$@" \
  2>&1 | tee "$OUT/selfplay.log"
