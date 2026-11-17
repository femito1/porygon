#!/bin/bash
# Equivalence + resume test of pretrain_resumable.py (CPU, tiny Net; P4 shard-00000 + shard-val-0000 symlinked with
# meta.json = 23,996 training + 23,976 val decisions; 282 steps over 6 epochs, the resume point inside epoch 5).
# A = porygon.v2.pretrain itself; B = the wrapper, uninterrupted; C = the wrapper stopped at step 200 (exit 75) and
# restarted. PASS iff the final.pt state dicts of A, B, C are bit-identical and their val lines match.
#   bash check_resume.sh SRC OUTDIR [SHARD_DIR]     (SRC = netv2 src tree with porygon.v2, e.g. $C/src-p2b)
set -uo pipefail
SRC=${1:?src}; O=${2:?outdir}
SH=${3:-/vol/bitbucket/fr1226/porygon-ws-data/datasets/frozen/p4-gen9-20261116-1502/v2pre-human-shuf}
PY=${PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SRC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
rm -rf "$O"; mkdir -p "$O/shards"
for f in shard-00000.npz shard-val-0000.npz meta.json; do ln -s "$SH/$f" "$O/shards/$f"; done
MJ='{"d_model":64,"n_layers":1,"n_heads":2,"ffn_mult":1,"value_hidden":64,"pair_hidden":8}'
COMMON=(--shards "$O/shards/shard-*.npz" --gen 9 --model-json "$MJ" --epochs 6 --seed 2 --value-sample one-per-game
        --val-frac 0.06 --no-bf16 --stream-gb 0.05 --stream-parts shards --eval-every 100 --device cpu)
t0=$(date +%s)
$PY -u -m porygon.v2.pretrain --out "$O/A" "${COMMON[@]}" > "$O/A.log" 2>&1; echo "A rc $? ($(( $(date +%s) - t0 )) s)"
$PY -u "$HERE/pretrain_resumable.py" --out "$O/B" "${COMMON[@]}" --state-every 200 > "$O/B.log" 2>&1; echo "B rc $?"
$PY -u "$HERE/pretrain_resumable.py" --out "$O/C" "${COMMON[@]}" --state-every 200 --stop-after 200 > "$O/C1.log" 2>&1
echo "C first part rc $? (75 expected)"
$PY -u "$HERE/pretrain_resumable.py" --out "$O/C" "${COMMON[@]}" --state-every 200 > "$O/C2.log" 2>&1; echo "C resume rc $?"
$PY - "$O" <<'EOF'
import json, sys, torch
o = sys.argv[1]
sd = {k: torch.load(f"{o}/{k}/final.pt", map_location="cpu", weights_only=False) for k in "ABC"}
steps = {k: v["pretrain_step"] for k, v in sd.items()}
def same(a, b):
    x, y = sd[a]["state_dict"], sd[b]["state_dict"]
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)
val = {k: [l for l in open(f"{o}/{k}/pretrain.log") if '"event": "val"' in l] for k in "ABC"}
res = {"steps": steps, "A==B": same("A", "B"), "B==C": same("B", "C"), "valAB": val["A"] == val["B"], "valBC": val["B"] == val["C"], "n_val_lines": len(val["A"]),
       "resume_line": any('"resume"' in l for l in open(f"{o}/C/pretrain.log"))}
res["PASS"] = all([res["A==B"], res["B==C"], res["valAB"], res["valBC"], res["n_val_lines"] > 0, res["resume_line"], len(set(steps.values())) == 1])
print(json.dumps(res))
open(f"{o}/RESULT.json", "w").write(json.dumps(res) + "\n")
EOF
