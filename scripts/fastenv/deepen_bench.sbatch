#!/bin/bash
# porygon.deepen microbench (docs/netsearch/SCALE-SEARCH-DESIGN.md §6): one small job on fastenv's helper partition,
# not the benchmark slot. Code is frozen into the run dir at submit time by the caller:
#   R=$C/runs/dbench-<date>; mkdir -p $R/code && rsync -a src scripts $R/code/ && echo <commit> > $R/code/COMMIT
#   cd $C && sbatch scripts/fastenv/deepen_bench.sbatch $R
# Stages: smoke (4 positions, budgets 0 and 1024; b0 must equal lookahead2) -> bench (120 positions of S2 + S1,
# budgets 0 / 1024 / 2048 / 4096 / 8192 with the priors backup, the Nash backup at 2048 and 8192, fp16 at 8192). Stage files $R/{STATUS,RESULT,FAILED}.
#SBATCH --job-name=ws-fastenv-dbench
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
R=$1
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
D=/vol/bitbucket/fr1226/porygon-ws-data
PY=$G/.venv/bin/python
CODE=$R/code
NET=$C/nets/g9q-cd-805.03M.pt
FROZEN=$C/frozen/netv2-fba02fb-20261110
I=$C/netsearch/study-20261109
SL="$D/datasets/searchlabels/records/gen9randombattle/part-0000[67].jsonl.gz"
export PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src NETV2_SRC=$FROZEN/src OMP_NUM_THREADS=1
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$R/STATUS"; }
fail() { st "FAILED at $1: $2"; echo "$(now) $1: $2 (job ${SLURM_JOB_ID:-?})" > "$R/FAILED"; exit 1; }
trap 'fail signal "TERM/INT"' TERM INT
cd "$C" || exit 1
[ ! -e "$R/RESULT" ] && [ ! -e "$R/FAILED" ] || { echo "$R already finished" >&2; exit 1; }
NCPU=${SLURM_CPUS_ON_NODE:-0}
st "start: job ${SLURM_JOB_ID:-?} on $(hostname), cpus $NCPU, gpu $(nvidia-smi -L 2>&1 | head -1), code $(cat "$CODE/COMMIT")"
[ "$NCPU" -ge 8 ] || fail checks "only $NCPU CPUs"
W=$((NCPU - 2)); [ "$W" -le 10 ] || W=10
SITE=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
( cd "$SITE" && timeout 1200 find torch triton nvidia -name '*.so*' -size +1M -exec cat {} + > /dev/null ) &
B=$CODE/scripts/fastenv/deepen_bench.py
$PY "$B" run --records "$SL" --source 6771931 --rows "$I/s2-sl-ARP.jsonl" --net "$NET" --out "$R/smoke" \
  --positions 4 --budgets 0,1024 --half-budgets "" --nash-budgets 1024 --workers "$W" > "$R/smoke.log" 2>&1 || fail smoke "see smoke.log"
grep -q "PARITY b0 vs la" "$R/smoke/summary.txt" || fail smoke "no parity line"
st "smoke OK: $(grep -o 'PARITY.*' "$R/smoke/summary.txt")"
cat "$I/s2-sl-ARP.jsonl" "$I/s1-sl-AB.jsonl" > "$R/rows.jsonl"
$PY "$B" run --records "$SL" --source 6771931 --rows "$R/rows.jsonl" --net "$NET" --out "$R/bench" \
  --positions 120 --workers "$W" > "$R/bench.log" 2>&1 || fail bench "see bench.log"
cp "$R/bench/summary.txt" "$R/RESULT"
st "bench done"
