#!/bin/bash
# Freeze the live gen9 human (+ v1, search-label) shard sets for a netv2 pretraining round, then write the
# pre-shuffled copy of the human set (generalises freeze_p2.sh; P3 onwards). One step inside a data job:
#   srun --jobid=J --overlap --ntasks=1 -c 1 --mem=6G env FREEZE_COMMIT=<ws/data commit> \
#     bash scripts/ops/freeze_human.sh NAME [SEED=20261109] [GROUP=12]      -> $D/datasets/frozen/NAME-gen9-<UTC stamp>/
# 1. hard links (never refreshed) under the convert lock, so no refresh round swaps a set mid-link;
# 2. summary.json: per set decisions, games, held-out (netv2 is_val 0.06) counts, rating 5/50/95%; per set
#    heldout-games-0.06.txt; "checks": P1's and P2's held-out games are still held out and none is a training game;
# 3. scripts/ops/shuffle_shards.py build -> v2pre-human-shuf/ (verify.json; published only when every check passes).
# Memory: --group 12 x 24k rows x 13.5 KB = 3.9 GB of output buffers + one decoded source shard (~4.7 GB peak).
set -uo pipefail
D=/vol/bitbucket/fr1226/porygon-ws-data
NAME=${1:?usage: freeze_human.sh NAME [SEED] [GROUP]}; SEED=${2:-20261109}; GROUP=${3:-12}
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
P1LIST=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/g9h-heldout-games.txt
P2=$D/datasets/frozen/p2-gen9-20261109-0645
T=$D/datasets/frozen/$NAME-gen9-$(date -u +%Y%m%d-%H%M)
[ -e "$T" ] && { echo "exists: $T"; exit 1; }
echo "$(date -u +%T) freeze $NAME -> $T (seed $SEED, group $GROUP, code ${FREEZE_COMMIT:-unknown})"
exec 9> "$D/datasets/.convert.lock"; flock 9   # wait for any refresh round to finish; hold it while linking
mkdir -p "$T"
for pair in "v2pre-human:$D/datasets/v2pre/gen9-human" "v1-human:$D/datasets/human/encoded/gen9randombattle/v1" \
            "v2pre-searchlabels:$D/datasets/v2pre/gen9-searchlabels"; do
  name=${pair%%:*}; src=$(readlink -f "${pair#*:}")
  mkdir -p "$T/$name"
  ln "$src"/shard-*.npz "$T/$name/" || { echo "link failed: $src"; exit 1; }
  cp "$src/meta.json" "$T/$name/meta-source.json"
  echo "$name <- $src" >> "$T/SOURCES.txt"
done
exec 9>&-
echo "$(date -u +%T) linked: $(ls "$T/v2pre-human" | grep -c '^shard-') human shards"
"$PY" -I - "$T" "$P2" "$P1LIST" <<'PY' || { echo "summary failed"; exit 1; }
import glob, json, sys, zlib
import numpy as np
T, P2, P1LIST = sys.argv[1:4]
def is_val(g):  # netv2 porygon.v2.pretrain.is_val(game, 0.06)
    h = np.array([zlib.crc32(int(x).to_bytes(8, "little", signed=True)) for x in g], dtype=np.uint64)
    return (h % 10_000) < 0.06 * 10_000
def ids(path):
    return {int(x) for x in open(path).read().splitlines() if x.strip() and not x.startswith("#")}
def scan(d):
    n = nv = 0; games = set(); vg = set(); r = []
    for p in sorted(glob.glob(f"{d}/shard-*.npz")):
        with np.load(p) as z:
            g = z["game"]; m = is_val(g); n += len(g); nv += int(m.sum())
            games.update(np.unique(g).tolist()); vg.update(np.unique(g[m]).tolist())
            if "rating" in z.files: r.append(z["rating"])
    rr = np.concatenate(r) if r else np.array([])
    pct = [int(x) for x in np.percentile(rr[rr > 0], [5, 50, 95])] if (rr > 0).any() else None
    return n, nv, games, vg, pct
out = {}
for name in ("v2pre-human", "v2pre-searchlabels"):
    n, nv, games, vg, pct = scan(f"{T}/{name}")
    out[name] = {"decisions": n, "games": len(games), "val_decisions_0.06": nv, "val_games_0.06": len(vg),
                 "rating_pct_5_50_95": pct}
    with open(f"{T}/{name}/heldout-games-0.06.txt", "w") as f:
        f.write("# netv2 porygon.v2.pretrain.is_val(game, 0.06); superset-consistent with g9h-heldout-games.txt\n")
        f.write("\n".join(str(x) for x in sorted(vg)) + "\n")
    if name == "v2pre-human":
        hg, hv = games, vg
p2_games = scan(f"{P2}/v2pre-human")[2]
p2_val, p1_val = ids(f"{P2}/v2pre-human/heldout-games-0.06.txt"), ids(P1LIST)
c = {"p2_games": len(p2_games), "p2_games_missing_from_p3": len(p2_games - hg),
     "p2_heldout": len(p2_val), "p2_heldout_not_heldout_in_p3": len((p2_val & hg) - hv),
     "p2_heldout_missing_from_p3": len(p2_val - hg), "p2_training_games_heldout_in_p3": len((p2_games - p2_val) & hv),
     "p1_heldout": len(p1_val), "p1_heldout_in_p3_training": len(p1_val & (hg - hv)),
     "p1_heldout_missing_from_p3": len(p1_val - hg)}
c["ok"] = (c["p2_heldout_not_heldout_in_p3"] == 0 and c["p2_training_games_heldout_in_p3"] == 0
           and c["p1_heldout_in_p3_training"] == 0)
out["checks"] = c
json.dump(out, open(f"{T}/summary.json", "w"), indent=1)
print(json.dumps(out))
sys.exit(0 if c["ok"] else 3)
PY
echo "$(date -u +%T) summary written; shuffling"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -I "$D/code/scripts/ops/shuffle_shards.py" build \
  "$T/v2pre-human" "$T/v2pre-human-shuf" --seed "$SEED" --rows 24000 --group "$GROUP" --spot 200 \
  --commit "${FREEZE_COMMIT:-unknown}"
rc=$?
echo "$(date -u +%T) freeze $NAME done: shuffle exit $rc"
exit $rc
