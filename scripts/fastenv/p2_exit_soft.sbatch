#!/bin/bash
# Council test P2: ExIt with a soft completed-Q target, offline on the ns-27 data
# (docs/netsearch/P2-EXIT-SOFT-PREREG.md = council/p2-exit-soft/PREREG.txt, r--, md5 in PREREG.md5).
# fastenv's one helper job (training partition, 12 CPUs, 1 GPU, nice 500); frozen code council/p2-exit-soft/code:
#   cd $C && sbatch --nice=500 council/p2-exit-soft/code/scripts/fastenv/p2_exit_soft.sbatch
# checks -> calibrate (must reproduce the PREREG's beta) -> shards (ns-27's E shards, target replaced by pi_imp)
# -> fine-tune E' (ns-27's E recipe) -> load check -> information check -> T1 E' vs C, T2 E' vs S (4,000 duplicate
# pairs each, ns-27's in-process harness) -> VERDICT.txt (r--). Every stage resumes from finished outputs; a failed
# stage writes FAILED and stops.
#SBATCH --job-name=ws-fastenv-p2soft
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16000
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
PY=$G/.venv/bin/python
D=$C/council/p2-exit-soft
CODE=$D/code
P=$CODE/scripts/fastenv/p2_exit_soft.py
R27=$C/runs/exit-la2                  # ns-27: labels, shards/E, ft-C (ns-29's control), ft-E
EX=$R27/code                          # ns-27's frozen code (1197016): eval harness and load check
FROZEN=$C/frozen/netv2-fba02fb-20261110
NET=$C/nets/g9q-cd-805.03M.pt; NET_SHA=e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f
NC=$R27/ft-C/final.pt; NC_MD5=6767993369b66e7031722d8479020806
NE=$R27/ft-E/final.pt; NE_MD5=8097eafd1e9fed73dbdfe85b4efaa78b
BETA=35.9003
ROWS=119967
SEED1=11240221; SEED2=11240222; GAMES=8000
PS_EVAL=$C/ps-sets-pinned             # October sets, as ns-27's T1-T3
cd "$C" || exit 1
J=${SLURM_JOB_ID:-none}
export PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$EX/src NETV2_SRC=$FROZEN/src OMP_NUM_THREADS=1
export PORYGON_NODE=$G/.node/bin/node PATH=$G/.node/bin:$PATH
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$D/STATUS"; }
fail() { st "FAILED at stage $1: $2"; echo "$(now) stage $1: $2 (job $J)" >> "$D/FAILED"; exit 1; }
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
md5of() { md5sum < "$1" | cut -c1-32; }
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT")"
[ ! -e "$D/FAILED" ] || fail start "$D/FAILED exists (read it, fix, move it aside)"
[ ! -e "$D/VERDICT.txt" ] || fail start "VERDICT.txt exists"

# ---- checks
[ -f "$D/PREREG.txt" ] && [ ! -w "$D/PREREG.txt" ] || fail checks "PREREG.txt missing or writable"
[ "$(md5of "$D/PREREG.txt")" = "$(cut -c1-32 "$D/PREREG.md5")" ] || fail checks "PREREG md5 != PREREG.md5"
grep -q "β = $BETA\b" "$D/PREREG.txt" || fail checks "PREREG does not record β = $BETA"
[ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "S sha256 mismatch"
[ "$(md5of "$NC")" = "$NC_MD5" ] || fail checks "C (ns-29's control) md5 mismatch"
[ "$(md5of "$NE")" = "$NE_MD5" ] || fail checks "E (ns-27) md5 mismatch"
[ "$(ls "$R27"/labels/label-*.npz | wc -l)" = 60 ] && [ "$(ls "$R27"/shards/E/shard-*.npz | wc -l)" = 60 ] \
  || fail checks "want 60 label parts and 60 ns-27 E shards"
[ "${SLURM_CPUS_ON_NODE:-0}" -ge 12 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 12)"
nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
[ -d "$PS_EVAL/dist/sim" ] && [ -d "$FROZEN/src" ] && [ -f "$EX/scripts/fastenv/exit_la2.py" ] \
  || fail checks "missing $PS_EVAL/dist/sim, $FROZEN/src or $EX"
st "checks OK: PREREG md5 $(md5of "$D/PREREG.txt"), S sha256 ok, C md5 ok, E md5 ok, 60 parts, β $BETA"

# ---- calibrate: the data check and β, which must reproduce the PREREG's value
"$PY" -u "$P" calibrate --labels "$R27/labels" --json "$D/calibration.json" --expect "$BETA" > "$D/calibrate.log" 2>&1 \
  || fail calibrate "$(tail -1 "$D/calibrate.log")"
st "calibrate: $(tail -1 "$D/calibrate.log")"

# ---- shards (skipped once the fine-tune and the information check are done: the scratch shards are deleted then)
if [ ! -s "$D/ft-Ep/final.pt" ] || [ ! -e "$D/check.done" ]; then
  if ! grep -q "\"rows\": $ROWS," "$D/shards/Ep/SHARDS.json" 2>/dev/null; then
    rm -rf "$D/shards"
    "$PY" -u "$P" shards --labels "$R27/labels" --e-shards "$R27/shards/E" --beta "$BETA" --out "$D/shards/Ep" \
      > "$D/shards.log" 2>&1 || fail shards "$(tail -1 "$D/shards.log")"
    grep -q "\"rows\": $ROWS," "$D/shards/Ep/SHARDS.json" || fail shards "SHARDS.json: $(cat "$D/shards/Ep/SHARDS.json")"
  fi
  st "shards: $(cat "$D/shards/Ep/SHARDS.json")"
fi

# ---- fine-tune E': ns-27's E arguments (exit_pilot.sbatch ft), only the shard dir differs
if [ ! -s "$D/ft-Ep/final.pt" ]; then
  rm -rf "$D/ft-Ep"
  PYTHONPATH=$FROZEN/src OMP_NUM_THREADS=4 "$PY" -u -m porygon.v2.pretrain --shards "$D/shards/Ep/shard-*.npz" --gen 9 \
    --out "$D/ft-Ep" --init "$NET" --epochs 2 --batch-size 512 --lr 1e-5 --warmup 50 --weight-decay 0.01 \
    --value-coef 0.5 --aux-coef 0 --val-frac 0.05 --seed 0 --no-bf16 > "$D/ft-Ep.log" 2>&1 \
    || fail ft "fine-tune E' ($D/ft-Ep.log)"
  [ -s "$D/ft-Ep/final.pt" ] || fail ft "no final.pt"
fi
same=$("$PY" - "$R27/ft-E/pretrain.log" "$D/ft-Ep/pretrain.log" <<'PYEOF'
import json, sys
a, b = (json.loads(open(p).readline()) for p in sys.argv[1:3])
keys = ("decisions", "train", "val", "games", "shards", "spec")
diff = [k for k in keys if a.get(k) != b.get(k)]
arg = [k for k in a["args"] if k not in ("shards", "out") and a["args"][k] != b["args"].get(k)]
print("SAME" if not diff and not arg else f"DIFF data {diff} args {arg}", {k: b.get(k) for k in keys[:4]})
PYEOF
)
case $same in SAME*) ;; *) fail ft "E' data or arguments differ from ns-27's E: $same" ;; esac
l=$("$PY" - "$D/ft-Ep/final.pt" <<'PYEOF' 2>&1 | tail -1
import sys
sys.path.insert(0, "/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/exit-la2/code/scripts/fastenv")
import netsearch_study as ST
ST.setup("gen9randombattle")
import torch
from porygon.v2.model import load, obs_to_tensors
from porygon.v2.encoder import empty
n = load(sys.argv[1], "cpu")
o = n(obs_to_tensors([empty(9)], "cpu"), aux=False)
print("LOAD", "OK" if int(n.gen) == 9 and torch.isfinite(o["logits"]).all() else "BAD", int(n.gen))
PYEOF
)
case $l in "LOAD OK"*) ;; *) fail load "ft-Ep: $l" ;; esac
rm -f "$D/ft-Ep/best.pt" "$D/ft-Ep/last.pt"  # final.pt is the net (no selection)
st "fine-tune E' done: $same; load OK; final.pt md5 $(md5of "$D/ft-Ep/final.pt")"

# ---- information check on the held-out games (not part of the rule; a failure is noted, not FAILED)
if [ ! -e "$D/check.done" ]; then
  if "$PY" -u "$P" check --shards "$D/shards/Ep" --start "$NET" --net Ep="$D/ft-Ep/final.pt" --net E="$NE" \
      --net C="$NC" --json "$D/check.json" > "$D/check.log" 2>&1; then
    st "information check: $(tr -d '\n ' < "$D/check.json" | cut -c1-400)"
  else
    st "information check FAILED (information only; chain continues): $(tail -1 "$D/check.log")"
  fi
  touch "$D/check.done"
fi
rm -rf "$D/shards"  # scratch (rebuilt bit-identically by the shards stage from the labels and β)

# ---- T1 E' vs C, T2 E' vs S: ns-27's harness and settings, fresh seeds, 8,000 games = 4,000 pairs each
ev() {  # NAME A B SEED
  [ -s "$D/$1.json" ] && return 0
  PORYGON_PS=$PS_EVAL "$PY" -u "$EX/scripts/fastenv/exit_la2.py" eval --a "$2" --b "$3" --games "$GAMES" --seed "$4" \
    --workers 10 --battles 32 --json "$D/$1.json.tmp" > "$D/$1.log" 2>&1 && mv "$D/$1.json.tmp" "$D/$1.json"
}
ev T1 "$D/ft-Ep/final.pt" "$NC" "$SEED1" || fail eval "T1 ($D/T1.log)"; st "T1 E' vs C: $(tail -1 "$D/T1.log")"
ev T2 "$D/ft-Ep/final.pt" "$NET" "$SEED2" || fail eval "T2 ($D/T2.log)"; st "T2 E' vs S: $(tail -1 "$D/T2.log")"

# ---- verdict (PREREG rule)
"$PY" -u "$P" verdict --t1 "$D/T1.json" --t2 "$D/T2.json" --calibration "$D/calibration.json" --check "$D/check.json" \
  --out "$D/VERDICT.txt.tmp" > "$D/verdict.log" 2>&1 || fail verdict "verdict ($D/verdict.log)"
mv -f "$D/VERDICT.txt.tmp" "$D/VERDICT.txt" && chmod a-w "$D/VERDICT.txt"
st "VERDICT: $(head -3 "$D/VERDICT.txt" | tr '\n' ' ')"
trap - TERM INT
exit 0
