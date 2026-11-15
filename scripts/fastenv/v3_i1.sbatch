#!/bin/bash
# V3 I1 inference bench (.orchestra/fastenv/V3-I1-PREREG.md): one helper job, no games, no Rust. Leaf observations
# from random walks below P2's 30 roots; parity of E16 / G32 / G16 vs eager fp32; server-loop throughput per config
# and B, uncontended then with 11 engine-search + 3 spin burners (killed by PID). The code is copied into the run dir.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch scripts/fastenv/v3_i1.sbatch
# Output: runs/v3i1-<date>-<job>/ (manifest.txt, i1/i1.json, i1/i1-rows.jsonl, i1.log).
#SBATCH --job-name=ws-fastenv-v3i1
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
C=$PWD
G=/vol/bitbucket/fr1226/gloria
NV=$C/frozen/netv2-fba02fb-20261110
NET=$C/nets/g9p-cd-104.71M.pt
ROOTS=$C/runs/v3gates-20261112-014733-1127567/p2/p2-roots.pkl
PY=$G/.venv/bin/python
RUN=$C/runs/v3i1-$(date -u +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-0}
mkdir -p "$RUN/code"
cp -r src scripts "$RUN/code/"
cp COMMIT "$RUN/code/" 2>/dev/null || true
CODE=$RUN/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PORYGON_GEN=9 NETV2_SRC=$NV/src
export PYTHONPATH=$G/.engine-gen9:$CODE/src
{
  echo "kind: v3 I1 inference bench (prereg .orchestra/fastenv/V3-I1-PREREG.md, md5 in HANDOFF)"
  echo "time: $(date -u +%FT%TZ) node $(hostname) SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?} cpuset $(taskset -pc $$ | sed 's/.*: //')"
  nvidia-smi -L 2>&1 | head -2
  nvidia-smi -q -d CLOCK 2>&1 | grep -A4 -E "^ +(Clocks|Max Clocks)$" | head -12
  echo "fastenv commit: $(cat "$CODE/COMMIT" 2>/dev/null || echo unknown); v3_i1 sha256 $(sha256sum "$CODE/scripts/fastenv/v3_i1.py" | cut -c1-16); v3leaf sha256 $(sha256sum "$CODE/src/porygon/v3leaf.py" | cut -c1-16)"
  echo "netv2 src: $NV (commit $(cat "$NV/COMMIT" 2>/dev/null || echo unknown))"
  echo "net: $NET sha256 $(sha256sum "$NET" | cut -c1-64)"
  echo "roots: $ROOTS sha256 $(sha256sum "$ROOTS" | cut -c1-16)"
  echo "engine: $(sha256sum "$G"/.engine-gen9/poke_engine/*.so | cut -c1-16)"
  echo "python: $PY ($("$PY" -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>&1))"
} > "$RUN/manifest.txt"
cat "$RUN/manifest.txt"
SITE=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
( cd "$SITE" && timeout 1800 find torch nvidia triton -name '*.so*' -size +1M -exec cat {} + > /dev/null )
echo "$(date -u +%T) pre-read done"
cd "$CODE"
$PY scripts/fastenv/v3_i1.py i1 --roots "$ROOTS" --net "$NET" --out "$RUN/i1" > "$RUN/i1.log" 2>&1
echo "$(date -u +%T) i1 rc $?: $(tail -1 "$RUN/i1.log")"
echo "$(date -u +%T) done $RUN"
