#!/bin/bash
# fastenv throughput benchmarks on one node. Runs each argument line of BENCH (a file of commands, one per line,
# executed from the stream dir with PY / NODE / PORYGON_PS set) and logs to logs/slurm-<id>.out.
#   sbatch scripts/fastenv/slurm_bench.sbatch scripts/fastenv/bench_cmds.txt
#SBATCH --job-name=ws-fastenv-bench
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
G=/vol/bitbucket/fr1226/gloria
export PY=$G/.venv/bin/python PORYGON_NODE=$G/.node/bin/node PORYGON_PS=$G/pokemon-showdown
export PATH=$G/.node/bin:$PATH PYTHONPATH=$ROOT/src
echo "node $(hostname) cpus $(nproc) $(lscpu | grep 'Model name' | sed 's/ \+/ /g')"; nvidia-smi -L || true
# pre-read the big files the first imports touch (slow shared disk)
cat $G/pokemon-showdown/dist/data/*.js $G/pokemon-showdown/dist/data/random-battles/*/*.js* > /dev/null 2>&1 || true
while read -r line; do
  [ -z "$line" ] || [ "${line:0:1}" = "#" ] && continue
  echo "=== $line"; eval "$line"
done < "$1"
