#!/bin/bash
# Duplicate-pair raw-network head-to-heads: every MODEL A_i vs the reference B, one duplicate Showdown server
# (gloria's .showdown-eval, run read-only, never rebuilt) and one pairing process per A_i, all in parallel.
#   sbatch --partition=long -c 24 scripts/wsp/h2h_dup.sbatch N_BATTLES B A1 [A2 ...]
# Output: $WSP/out/h2h/<job>/<name>/{battles.jsonl,log.txt}; summary.txt (duplicate-pair 95% CIs).
#SBATCH --job-name=ws-wsp-h2h
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=1
N=$1; B=$2; shift 2
OUT=$WSP/out/h2h/$SLURM_JOB_ID; mkdir -p "$OUT"
BASE=$(( 30000 + (SLURM_JOB_ID % 1000) * 16 ))
PS=$GLORIA/.showdown-eval
find $GLORIA/.venv/lib/python3.12/site-packages/{websockets,poke_env,torch,stable_baselines3} -type f \( -name '*.py' -o -name '*.so*' \) -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true
i=0; names=()
for A in "$@"; do
  name=$(basename "$(dirname "$A")")_$(basename "$A" | sed 's/\.\(zip\|pt\)$//'); name=${name#.}
  mkdir -p "$OUT/$name"; names+=("$name")
  PORT=$(( BASE + i ))
  (cd "$PS" && PORYGON_DUPLICATE="wsp-$SLURM_JOB_ID-$i" PORYGON_DUPLICATE_LOG="$OUT/$name/battles.jsonl" \
     PORYGON_DUPLICATE_GROUP='^wspa' setsid nohup node pokemon-showdown start --skip-build --no-security "$PORT" \
     < /dev/null > "$OUT/$name/showdown.log" 2>&1 &)
  i=$(( i + 1 ))
done
for ((j = 0; j < i; j++)); do
  for ((t = 0; t < 2400; t++)); do grep -q listening "$OUT/${names[$j]}/showdown.log" 2>/dev/null && break; sleep 0.5; done
done
python - "$BASE" "$i" <<'PYEOF' || echo "warning: websocket warm-up failed" >&2
import asyncio, sys, websockets
async def warm(port):
    for _ in range(120):
        try:
            async with websockets.connect(f"ws://localhost:{port}/showdown/websocket", open_timeout=60): return
        except Exception: await asyncio.sleep(5)
async def main(b, n): await asyncio.gather(*(warm(b + k) for k in range(n)))
asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
PYEOF
trap 'pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $BASE" || true; for ((j=0;j<i;j++)); do pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $((BASE+j))\$" || true; done' EXIT
echo "h2h $N battles: B=$B vs $* (servers $BASE..$((BASE+i-1)))"
j=0
for A in "$@"; do
  python scripts/wsp/h2h_dup.py "$A" "$B" --port $(( BASE + j )) --n-battles "$N" --out "$OUT/${names[$j]}" \
    > "$OUT/${names[$j]}/log.txt" 2>&1 &
  j=$(( j + 1 ))
done
wait
for n in "${names[@]}"; do echo "== $n vs $B: $(python scripts/wsp/h2h_dup.py --score "$OUT/$n")"; done | tee "$OUT/summary.txt"
