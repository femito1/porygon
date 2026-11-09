#!/bin/bash
# Same node, same CPUs: in-process head-to-heads (spd_h2h.py) vs the Showdown harness (h2h_dup.py + one duplicate
# server per pairing, as h2h_dup.sbatch), plus the temperature-option check. A test job (8 CPUs, < 1 h), not a benchmark.
#   sbatch scripts/wsp/spd_bench.sbatch [OUT=$WSP/out/spd/bench-<job>] [SD_PAIRINGS=6] [SD_GAMES=400] [IP_PAIRS=1000]
# In-process: 500M_temp05 vs 500M (pairs [4000, 5000) of the seed-1 validation experiment, mergeable with
# spd_h2h.sbatch shards 0-3), A/A 500M vs 500M, 500M at temperature 0.5 vs 500M (should equal 500M_temp05).
# Showdown: SD_PAIRINGS x (500M_temp05 vs 500M), SD_GAMES games each, all in parallel. CPU = the job cgroup's usage.
#SBATCH --job-name=ws-wsp-spdbench
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=1
OUT=${1:-$WSP/out/spd/bench-$SLURM_JOB_ID}; P=${2:-6}; G=${3:-400}; NP=${4:-1000}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
FE=${SPD_FASTENV:-$WSP/spd/fastenv-b9b8a85}
export SPD_FASTENV_JS=$FE/fastenv/js PORYGON_PS=${PORYGON_PS:-$GLORIA/pokemon-showdown} PORYGON_NODE=$GLORIA/.node/bin/node
A=ckpt/edits/500M_temp05.pt; B=ckpt/ref/m500.zip
NCPU=$(python -c 'import os; print(len(os.sched_getaffinity(0)))')
echo "$(date -u +%FT%TZ) node $(hostname), $NCPU CPUs (SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE:-?}), $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2), $(nvidia-smi -L 2>/dev/null | head -1)" | tee "$OUT/node.txt"
[ "$NCPU" -ge 4 ] || { echo "only $NCPU CPUs" >&2; exit 1; }
find $GLORIA/.venv/lib/python3.12/site-packages/{torch,stable_baselines3,gymnasium,websockets,poke_env} -type f \
  \( -name '*.py' -o -name '*.so*' \) -print0 2>/dev/null | xargs -0 -P4 -n64 cat > /dev/null || true

# CPU-seconds used by everything in this job so far (cgroup v2 job_<id> cpu.stat, else v1 cpuacct), -1 if unknown
cgcpu() { python - <<'PYEOF'
import os
rel = open("/proc/self/cgroup").read().split("\n")
v2 = [l.split("::", 1)[1] for l in rel if l.startswith("0::")]
out = -1
if v2:
    parts = v2[0].strip("/").split("/")
    for i in range(len(parts), 0, -1):
        d = "/sys/fs/cgroup/" + "/".join(parts[:i])
        if parts[i - 1].startswith("job_") and os.path.exists(d + "/cpu.stat"):
            out = next(int(l.split()[1]) for l in open(d + "/cpu.stat") if l.startswith("usage_usec")) / 1e6
            break
else:
    for l in rel:
        if "cpuacct" in l:
            p = l.split(":", 2)[2]
            while p and "job_" in p and not p.rstrip("/").rsplit("/", 1)[-1].startswith("job_"):
                p = p.rstrip("/").rsplit("/", 1)[0]
            f = f"/sys/fs/cgroup/cpu,cpuacct{p}/cpuacct.usage"
            if os.path.exists(f):
                out = int(open(f).read()) / 1e9
print(out)
PYEOF
}
now() { date +%s.%N; }
phase() {  # phase NAME CMD...: wall seconds and job CPU-seconds of one command
  local name=$1; shift
  local c0 t0 c1 t1; c0=$(cgcpu); t0=$(now)
  "$@"; local rc=$?
  c1=$(cgcpu); t1=$(now)
  python -c "import sys; n, t0, t1, c0, c1, rc = sys.argv[1:]; w = float(t1) - float(t0); c = float(c1) - float(c0) if float(c0) >= 0 else float('nan'); print(f'PHASE {n}: rc {rc}, wall {w:.1f} s, job CPU {c:.0f} s = {c / w:.2f} cores busy')" \
    "$name" "$t0" "$t1" "$c0" "$c1" "$rc" | tee -a "$OUT/phases.txt"
}

# --- 1. in-process (7 workers x 48 battles + this process) ---------------------------------------------------
IP=(python scripts/wsp/spd_h2h.py run --pairs "$NP" --progress 0)
phase ip_temp05 "${IP[@]}" --a "$A" --b "$B" --shard 4 --seed 1 --out "$WSP/out/spd/val-temp05" 2>&1 | tee "$OUT/ip_temp05.log"
phase ip_aa "${IP[@]}" --a "$B" --b "$B" --seed 1 --out "$WSP/out/spd/aa-500M" 2>&1 | tee "$OUT/ip_aa.log"
phase ip_T05 "${IP[@]}" --a "$B,temperature=0.5" --b "$B" --seed 1 --out "$WSP/out/spd/T05-500M" 2>&1 | tee "$OUT/ip_T05.log"

# --- 2. offline: 500M at temperature 0.5 has the same action distribution as the 500M_temp05 edit ---------------
python - "$A" "$B" <<'PYEOF' 2>&1 | tee "$OUT/temp_check.txt"
import sys, zipfile, numpy as np, torch
from porygon.model import load_policy
zf = zipfile.ZipFile("data/decisions.npz")
def head(name, n):  # first n rows of an uncompressed .npz member, without reading the rest
    with zf.open(name) as f:
        major, _ = np.lib.format.read_magic(f)
        shape, _, dt = (np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0)(f)
        k = n * int(np.prod(shape[1:]))
        return np.frombuffer(f.read(k * dt.itemsize), dtype=dt).reshape(n, *shape[1:])
obs, mask = head("obs.npy", 5000).astype(np.float32), head("mask.npy", 5000).astype(np.int8)
pe, pb = load_policy(sys.argv[1], "cuda"), load_policy(sys.argv[2], "cuda")
x = {"observation": torch.as_tensor(obs, device="cuda"), "action_mask": torch.as_tensor(mask, device="cuda")}
m = x["action_mask"].bool()
with torch.no_grad():
    le = pe.get_distribution(x).distribution.logits
    lb = pb.get_distribution(x).distribution.logits
    q = torch.softmax(torch.where(m, lb / 0.5, torch.tensor(float("-inf"), device="cuda")), -1)
    d = float((q - torch.softmax(le, -1)).abs().max())
print(f"temperature check on {len(obs)} logged positions: max |P(500M, T=0.5) - P(500M_temp05)| = {d:.2e}")
PYEOF

# --- 3. Showdown harness: P pairings in parallel, one duplicate server + one h2h_dup.py process each ----------
BASE=$(( 40000 + (SLURM_JOB_ID % 1000) * 16 )); PS=$GLORIA/.showdown-eval
trap 'for ((j=0;j<P;j++)); do pkill -u "$USER" -f "^node pokemon-showdown start --skip-build --no-security $((BASE+j))\$" || true; done' EXIT
for ((i = 0; i < P; i++)); do
  mkdir -p "$OUT/sd$i"
  (cd "$PS" && PORYGON_DUPLICATE="spd-$SLURM_JOB_ID-$i" PORYGON_DUPLICATE_LOG="$OUT/sd$i/battles.jsonl" \
     PORYGON_DUPLICATE_GROUP='^wspa' setsid nohup node pokemon-showdown start --skip-build --no-security $((BASE + i)) \
     < /dev/null > "$OUT/sd$i/showdown.log" 2>&1 &)
done
for ((i = 0; i < P; i++)); do
  for ((t = 0; t < 1200; t++)); do grep -q listening "$OUT/sd$i/showdown.log" 2>/dev/null && break; sleep 0.5; done
done
python - "$BASE" "$P" <<'PYEOF' || echo "warning: websocket warm-up failed" >&2
import asyncio, sys, websockets
async def warm(port):
    for _ in range(120):
        try:
            async with websockets.connect(f"ws://localhost:{port}/showdown/websocket", open_timeout=60): return
        except Exception: await asyncio.sleep(5)
async def main(b, n): await asyncio.gather(*(warm(b + k) for k in range(n)))
asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
PYEOF
sd_all() {
  for ((i = 0; i < P; i++)); do
    timeout 1800 python scripts/wsp/h2h_dup.py "$A" "$B" --port $((BASE + i)) --n-battles "$G" --out "$OUT/sd$i" \
      > "$OUT/sd$i/log.txt" 2>&1 &
  done
  wait
}
phase sd_x$P sd_all
python scripts/wsp/h2h_dup.py --score "$OUT"/sd* | tee "$OUT/sd_score.txt"
python - "$OUT" "$P" <<'PYEOF' | tee "$OUT/sd_turns.txt"
import json, sys, glob, numpy as np
t = [json.loads(l)["turns"] for f in glob.glob(f"{sys.argv[1]}/sd*/battles.jsonl") for l in open(f)]
print(f"Showdown games {len(t)}, mean turns {np.mean(t):.2f} ± {1.96 * np.std(t) / np.sqrt(len(t)):.2f}")
PYEOF
python scripts/wsp/spd_h2h.py merge "$WSP/out/spd/val-temp05" | tee "$OUT/merge_temp05.txt"
cat "$OUT/phases.txt"
