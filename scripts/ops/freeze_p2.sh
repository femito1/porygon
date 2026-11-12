#!/bin/bash
# Overnight: freeze a larger gen9 v2 shard set for netv2's next pretraining round (hard links, after the 06:20
# refresh round), with counts and netv2's held-out rule. Runs as a step inside 677193:
#   srun --jobid=677193 --overlap --ntasks=1 -c 1 --mem=4G bash logs/freeze_p2.sh EPOCH_START
set -uo pipefail
D=/vol/bitbucket/fr1226/porygon-ws-data
START=${1:-0}; now=$(date +%s); [ "$START" -gt "$now" ] && sleep $(( START - now ))
T=$D/datasets/frozen/p2-gen9-$(date +%Y%m%d-%H%M)
exec 9> "$D/datasets/.convert.lock"; flock 9   # wait for any refresh round to finish; hold it while linking
mkdir -p "$T"
for pair in "v2pre-human:$D/datasets/v2pre/gen9-human" "v1-human:$D/datasets/human/encoded/gen9randombattle/v1" \
            "v2pre-searchlabels:$D/datasets/v2pre/gen9-searchlabels"; do
  name=${pair%%:*}; src=$(readlink -f "${pair#*:}")
  mkdir -p "$T/$name"; ln "$src"/shard-*.npz "$T/$name/"; cp "$src/meta.json" "$T/$name/meta-source.json"
  echo "$name <- $src" >> "$T/SOURCES.txt"
done
exec 9>&-
/vol/bitbucket/fr1226/gloria/.venv/bin/python -I - "$T" <<'PY'
import glob, json, sys, zlib
import numpy as np
T = sys.argv[1]; out = {}
def is_val(g):
    h = np.array([zlib.crc32(int(x).to_bytes(8, "little", signed=True)) for x in g], dtype=np.uint64)
    return (h % 10_000) < 600
for name in ("v2pre-human", "v2pre-searchlabels"):
    n = nv = 0; games = set(); vg = set(); r = []
    for p in sorted(glob.glob(f"{T}/{name}/shard-*.npz")):
        with np.load(p) as z:
            g = z["game"]; m = is_val(g); n += len(g); nv += int(m.sum())
            games.update(np.unique(g).tolist()); vg.update(np.unique(g[m]).tolist())
            if "rating" in z.files: r.append(z["rating"])
    rr = np.concatenate(r) if r else np.array([])
    out[name] = {"decisions": n, "games": len(games), "val_decisions_0.06": nv, "val_games_0.06": len(vg),
                 "rating_pct_5_50_95": [int(x) for x in np.percentile(rr[rr > 0], [5, 50, 95])] if (rr > 0).any() else None}
    with open(f"{T}/{name}/heldout-games-0.06.txt", "w") as f:
        f.write("# netv2 porygon.v2.pretrain.is_val(game, 0.06); superset-consistent with g9h-heldout-games.txt\n")
        f.write("\n".join(str(x) for x in sorted(vg)) + "\n")
json.dump(out, open(f"{T}/summary.json", "w"), indent=1)
print(json.dumps(out))
PY
echo "FROZEN $T"
