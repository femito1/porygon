#!/bin/bash
# Same-node, same-CPU-count env throughput: the old pipeline (local Showdown server + poke-env workers + Python
# encoder) vs fastenv, random legal actions, no learner. 4-CPU a16 dev job.
#   sbatch scripts/fastenv/slurm_compare.sbatch
#SBATCH --job-name=ws-fastenv-cmp
#SBATCH --partition=a16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
G=/vol/bitbucket/fr1226/gloria
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PYTHONPATH=$ROOT/src
echo "node $(hostname) cpus $(nproc) $(lscpu | grep 'Model name' | sed 's/ \+/ /g')"
# warm the slow shared disk: Python env (torch, poke-env) and Showdown data
find $G/.venv/lib/python3.12/site-packages/{torch,poke_env,stable_baselines3,numpy} -type f -size +100k -exec cat {} + > /dev/null 2>&1
cat $G/pokemon-showdown/dist/data/*.js $G/pokemon-showdown/dist/data/random-battles/*/*.js* > /dev/null 2>&1
$PY -c "import porygon.env" && echo "imports warm"
node scripts/fastenv/simspeed.js gen4randombattle 100
PS=${TMPDIR:-/tmp}/ps-$SLURM_JOB_ID
rsync -a --exclude logs --exclude .git $G/pokemon-showdown/ "$PS/" && mkdir -p "$PS/logs/repl"
PORT=$((20000 + SLURM_JOB_ID % 20000))
(cd "$PS" && setsid nohup node pokemon-showdown start --skip-build --no-security $PORT < /dev/null > "$ROOT/logs/showdown-$SLURM_JOB_ID.log" 2>&1 &)
for ((t = 0; t < 600; t++)); do grep -q listening "$ROOT/logs/showdown-$SLURM_JOB_ID.log" 2>/dev/null && break; sleep 1; done
sleep 10
for fmt in gen4randombattle gen9randombattle; do
  for w in 4 8; do
    echo "=== old pipeline $fmt: $w poke-env workers, 1 server"
    timeout 900 $PY scripts/fastenv/profile_showdown_env.py --port $PORT --workers $w --seconds 90 --format $fmt 2>&1 | tail -2
  done
done
pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $PORT" || true
for fmt in gen4randombattle gen9randombattle; do
  for w in 3 4; do
    echo "=== fastenv $fmt: $w workers x 48"
    $PY scripts/fastenv/bench_fastenv.py --format $fmt --workers $w --battles 48 --seconds 60 --async
  done
done
