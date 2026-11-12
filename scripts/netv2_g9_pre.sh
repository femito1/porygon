#!/bin/bash
# G9 pre-step (PRE_CMD in netv2.env, once per run, inside the G9 job before training): refit P1's value head on its
# frozen trunk (policy bit-identical; P1's own value head overfits after ~2k steps), then write
# runs/$RUN_NAME/init.txt = the RL init: the refit if its best held-out value BCE <= 0.60 (pre-registered), else P1
# itself (the run's --critic-warmup-steps fits the value in RL instead).
set -uo pipefail
ROOT=${ROOT:-$(pwd)}; RUN_NAME=${RUN_NAME:-g9}
PY=${PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}
P1=${G9_P1:-$ROOT/runs/pretrain/P1-g9h-dmg-s2.pt}
OUT=$ROOT/runs/pretrain/g9h-dmg-s2-vrefit
SHARDS=${G9_REFIT_SHARDS:-/vol/bitbucket/fr1226/porygon-ws-data/datasets/frozen/p1-gen9-human/v2pre/shard-*.npz}
mkdir -p "$ROOT/runs/$RUN_NAME"
if [ ! -f "$OUT/final.pt" ]; then
  echo "$(date -u +%T) value refit of $P1 -> $OUT"
  OMP_NUM_THREADS=4 "$PY" -u -m porygon.v2.pretrain --shards "$SHARDS" --gen 9 --out "$OUT" --init "$P1" \
    --train-only value_head --reinit value_head --select value --aux-coef 0 --value-coef 1 --epochs 2 \
    --batch-size 1024 --lr 3e-4 --warmup 200 --eval-every 250 --val-frac 0.06 --no-bf16 --seed 2 \
    2>&1 | grep -v '"event": "train"' | cut -c1-600 | tail -n 30
fi
"$PY" - "$OUT" "$P1" "$ROOT/runs/$RUN_NAME/init.txt" <<'PYEOF'
import json, pathlib, sys
out, p1, dst = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
log = out / "pretrain.log"
vals = [json.loads(l) for l in open(log) if '"event": "val"' in l] if log.exists() else []
best = min((v["value_bce"] for v in vals if v.get("step", 0) > 0), default=float("inf"))
ok = best <= 0.60 and (out / "best.pt").exists()
choice = str(out / "best.pt") if ok else p1
dst.write_text(choice + "\n")
print(f"value refit: best held-out value BCE {best:.4f} (pass <= 0.60: {ok}) -> RL init {choice}", flush=True)
PYEOF
