#!/bin/bash
# On-policy self-play rollouts (MODEL vs itself) logged per decision for gradient studies (selfplay_log.py).
#   sbatch --partition=long -c 24 scripts/wsp/selfplay_log.sbatch MODEL N_PER_PROC N_PROCS
#SBATCH --job-name=ws-wsp-selfplay
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=1
MODEL=$1; N=$2; K=$3
OUT=$WSP/out/selfplay/$SLURM_JOB_ID; mkdir -p "$OUT"
BASE=$(( 31000 + (SLURM_JOB_ID % 1000) * 16 ))
PS=$GLORIA/.showdown-eval
find $GLORIA/.venv/lib/python3.12/site-packages/{websockets,poke_env,torch,stable_baselines3} -type f \( -name '*.py' -o -name '*.so*' \) -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
for ((i = 0; i < K; i++)); do
  (cd "$PS" && setsid nohup node pokemon-showdown start --skip-build --no-security $((BASE + i)) < /dev/null > "$OUT/showdown-$i.log" 2>&1 &)
done
for ((i = 0; i < K; i++)); do
  for ((t = 0; t < 2400; t++)); do grep -q listening "$OUT/showdown-$i.log" 2>/dev/null && break; sleep 0.5; done
done
trap 'for ((i=0;i<K;i++)); do pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $((BASE+i))\$" || true; done' EXIT
sleep 20
for ((i = 0; i < K; i++)); do
  python scripts/wsp/selfplay_log.py "$MODEL" --port $((BASE + i)) --n-battles "$N" --out "$OUT/p$i" > "$OUT/log-$i.txt" 2>&1 &
done
wait
echo done; ls -la "$OUT"/p*
