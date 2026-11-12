#!/bin/bash
# P2 / value-retrain pretraining arms inside data's job 677193 (orchestrator 2026-11-09: its A16 is idle; <= 2 CPUs,
# <= 14 GB, streaming loader). Launch from the login node (like data's own steps):
#   nohup srun --jobid=677193 --overlap --ntasks=1 -c 2 --gres=gpu:1 --mem=14G --job-name=nv2-p2 \
#     bash $C/scripts/netv2_p2.sh ARM... > $C/logs/p2-$(date +%Y%m%d-%H%M%S).out 2>&1 &
# ARMs (pre-registered in .orchestra/netv2/STATUS.md, 2026-11-09 04:33), run in the order given; an arm whose
# final.pt exists is skipped:
#   p2ft   P1 fine-tuned on the gen9 search labels (human held-out rows reported, not trained on)
#   c005   VR-c005: P1's recipe, value coef 0.05            1pg   VR-1pg: value loss on one position per game
#   p2mix  human + search labels (search rows carry 1/4 of the policy loss), value recipe from the VR rule
#   ctrl   VR-ctrl: P1's recipe at fp32 + streaming (only if both VR arms miss the policy-CE bar)
# Code: $C/src-p2 (a snapshot of ws/netv2; the training jobs' $C/src is not touched).
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
export PYTHONPATH=$C/src-p2 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$C"
H="$C/datasets/p1s2-human/shard-*.npz"          # P1-s2's exact 42 human shards (999,118 decisions)
S="$C/datasets/g9sl-0425/shard-*.npz"           # gen9 search labels as of 2026-11-09 04:25 (116,810 decisions)
P1=$C/runs/pretrain/g9h-dmg-s2/final.pt
CFG='{"n_layers":4,"ffn_mult":1}'
COMMON=(--gen 9 --val-frac 0.06 --no-bf16 --stream-gb 3)
echo "p2 step on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader), code $(cat $C/src-p2/GIT_COMMIT 2>/dev/null)"

run() {  # NAME pretrain-args...
  local name=$1; shift
  local out=$C/runs/pretrain/$name
  if [ -f "$out/final.pt" ]; then echo "skip $name (final.pt exists)"; return 0; fi
  # one runner per arm (a second step started in parallel skips it); a stale lock after a crash: rmdir it
  mkdir -p "$C/runs/pretrain"
  if ! mkdir "$out.lock" 2>/dev/null; then echo "skip $name (locked: $out.lock)"; return 0; fi
  echo "== $(date -u +%H:%M:%S) $name: $*"
  $PY -u -m porygon.v2.pretrain --out "$out" "${COMMON[@]}" "$@" 2>&1 | grep -vE '"event": "train"' | cut -c1-900
  echo "== $(date -u +%H:%M:%S) $name exit ${PIPESTATUS[0]}"
  rmdir "$out.lock"
}

# VR rule (pre-registered 02:05, interpretation fixed 04:33): qualifying = final held-out value BCE <= 0.62 and final
# policy CE <= 1.126; the qualifying arm with the lower final value BCE sets P2's value recipe; no arm reaches 0.62 ->
# value_trunk_grad 0; an arm reaches 0.62 but misses the CE bar -> the lower-BCE such arm, flagged (ctrl tells
# whether fp32 / streaming moved the CE).
vr_choice() {
  $PY -I - "$C/runs/pretrain" <<'EOF'
import json, sys
root = sys.argv[1]
res = {}
for arm in ("c005", "1pg"):
    try:
        vals = [json.loads(x) for x in open(f"{root}/g9h-vr-{arm}-s2/pretrain.log") if '"event": "val"' in x]
        res[arm] = (vals[-1]["value_bce"], vals[-1]["policy_ce"])
    except (OSError, IndexError, KeyError):
        pass
ok = {a: v for a, v in res.items() if v[0] <= 0.62 and v[1] <= 1.126}
near = {a: v for a, v in res.items() if v[0] <= 0.62}
pick = min(ok, key=lambda a: ok[a][0]) if ok else (min(near, key=lambda a: near[a][0]) + "-flag" if near else "vtg0")
print(pick, json.dumps(res), file=sys.stderr)
print(pick)
EOF
}

for arm in "$@"; do
  case $arm in
    p2ft)
      run g9-p2ft-s2 --shards "$S" --mix "$H:0" --init "$P1" --epochs 4 --lr 1e-4 --warmup 100 --value-coef 0.05 \
        --aux-coef 0.1 --select policy --eval-every 100 --seed 2 ;;
    c005)
      run g9h-vr-c005-s2 --shards "$H" --model-json "$CFG" --epochs 6 --seed 2 --value-coef 0.05 ;;
    1pg)
      run g9h-vr-1pg-s2 --shards "$H" --model-json "$CFG" --epochs 6 --seed 2 --value-sample one-per-game ;;
    ctrl)
      run g9h-vr-ctrl-s2 --shards "$H" --model-json "$CFG" --epochs 6 --seed 2 ;;
    p2mix)
      if [ ! -f "$C/runs/pretrain/g9h-vr-c005-s2/final.pt" ] || [ ! -f "$C/runs/pretrain/g9h-vr-1pg-s2/final.pt" ]; then
        echo "p2mix skipped: both VR arms must be finished first"; continue
      fi
      # W = N_human_train / (3 N_search_train) over rows with policy-target mass on a legal action
      W=$($PY -I - "$H" "$S" <<'EOF'
import glob, sys, zlib
import numpy as np
def n_train(g):
    n = 0
    for p in sorted(glob.glob(g)):
        with np.load(p) as z:
            game, t, m = z["game"], z["target"], z["obs/action_mask"]
            val = np.array([zlib.crc32(int(x).to_bytes(8, "little", signed=True)) % 10_000 < 600 for x in game])
            n += int(((t * (m > 0)).sum(-1) > 0)[~val].sum())
    return n
nh, ns = n_train(sys.argv[1]), n_train(sys.argv[2])
print(f"{nh / (3 * ns):.4f}")
print(f"p2mix weight: human {nh} search {ns}", file=sys.stderr)
EOF
)
      pick=$(vr_choice)
      echo "p2mix: search weight $W, value recipe from the VR rule: $pick"
      case $pick in
        c005*) VR=(--value-coef 0.05); MJ=$CFG ;;
        1pg*) VR=(--value-sample one-per-game); MJ=$CFG ;;
        *) VR=(); MJ='{"n_layers":4,"ffn_mult":1,"value_trunk_grad":0.0}' ;;
      esac
      run g9-p2mix-s2 --shards "$H" --mix "$S:$W" --model-json "$MJ" --epochs 6 --seed 2 "${VR[@]}" ;;
    *) echo "unknown arm $arm" ;;
  esac
done
echo "p2 step done $(date -u +%H:%M:%S)"
