#!/bin/bash
# V3 gates P1 + P2 (.orchestra/fastenv/V3-GATES-PREREG.md): one helper job, no games. P1 (CPU) replays Stage 1's
# search-label records and checks the leaf encoder; P2 runs 12 tree workers (3 battles x 4 determinizations) on
# the T4 for 1000 ms per decision. The code is copied into the run dir first.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch scripts/fastenv/v3_gates.sbatch
# Output: runs/v3gates-<date>-<job>/ (manifest.txt, p1/p1.json, p2/p2.json, *.log).
#SBATCH --job-name=ws-fastenv-v3gates
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
C=$PWD
G=/vol/bitbucket/fr1226/gloria
D=/vol/bitbucket/fr1226/porygon-ws-data
NV=$C/frozen/netv2-fba02fb-20261110
NET=$C/nets/g9p-cd-104.71M.pt
PY=$G/.venv/bin/python
RUN=$C/runs/v3gates-$(date -u +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-0}
mkdir -p "$RUN/code"
cp -r src scripts "$RUN/code/"
cp COMMIT "$RUN/code/" 2>/dev/null || true
CODE=$RUN/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PORYGON_GEN=9 NETV2_SRC=$NV/src
export PYTHONPATH=$G/.engine-gen9:$CODE/src
{
  echo "kind: v3 gates P1 + P2 (prereg .orchestra/fastenv/V3-GATES-PREREG.md, md5 at submission in HANDOFF)"
  echo "time: $(date -u +%FT%TZ) node $(hostname) SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?} cpuset $(taskset -pc $$ | sed 's/.*: //')"
  nvidia-smi -L 2>&1 | head -2
  echo "fastenv commit: $(cat "$CODE/COMMIT" 2>/dev/null || echo unknown); v3leaf sha256 $(sha256sum "$CODE/src/porygon/v3leaf.py" | cut -c1-16); v3_gates sha256 $(sha256sum "$CODE/scripts/fastenv/v3_gates.py" | cut -c1-16)"
  echo "netv2 src: $NV (commit $(cat "$NV/COMMIT" 2>/dev/null || echo unknown))"
  echo "net: $NET sha256 $(sha256sum "$NET" | cut -c1-64)"
  echo "engine: $(sha256sum "$G"/.engine-gen9/poke_engine/*.so | cut -c1-16)"
  echo "python: $PY ($("$PY" -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>&1))"
} > "$RUN/manifest.txt"
cat "$RUN/manifest.txt"
# pre-read the venv's big libraries (slow shared disk)
SITE=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
( cd "$SITE" && timeout 1200 find torch nvidia -name '*.so*' -size +1M -exec cat {} + > /dev/null )
echo "$(date -u +%T) pre-read done"
SL="$D/datasets/searchlabels/records/gen9randombattle/part-0000[67].jsonl.gz"
cd "$CODE"
$PY scripts/fastenv/v3_gates.py p1 --records "$SL" --source 6771931 --net "$NET" --out "$RUN/p1" \
  > "$RUN/p1.log" 2>&1
echo "$(date -u +%T) p1 rc $?: $(tail -1 "$RUN/p1.log")"
$PY scripts/fastenv/v3_gates.py p2 --roots "$RUN/p1/roots.pkl" --net "$NET" --out "$RUN/p2" > "$RUN/p2.log" 2>&1
echo "$(date -u +%T) p2 rc $?: $(tail -1 "$RUN/p2.log")"
echo "$(date -u +%T) done $RUN"
