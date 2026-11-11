#!/bin/bash
# U7 (BC size-vs-quality test) shard set, as steps inside a data job:
#   bash scripts/ops/u7_build.sh select T      hard-link the live gen9 human v2 set into T/source (convert lock) and
#                                              write T/select/{keep.npy,train-games.txt,selection.json} (u7_select.py)
#   bash scripts/ops/u7_build.sh build T [GROUP=36]
#                                              shuffled training shards of the kept rows (shuffle_shards.py --keep,
#                                              verified), P3's shard-val-* hard-linked, held-out checks, summary.json,
#                                              MD5SUMS, then published as T/v2pre-human-shuf and made read-only
# Memory (build): GROUP x 24k rows x 13.4 KB (36 -> 11.6 GB) + one decoded source shard.
set -uo pipefail
D=/vol/bitbucket/fr1226/porygon-ws-data
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
P3=$D/datasets/frozen/p3-gen9-20261109-1735
P2=$D/datasets/frozen/p2-gen9-20261109-0645
P1LIST=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/g9h-heldout-games.txt
SEED=20261111
CMD=${1:?select|build}; T=${2:?target dir}
say() { echo "$(date -u +%T) $*"; }
case "$CMD" in
select)
  [ -e "$T" ] && { say "exists: $T"; exit 1; }
  exec 9> "$D/datasets/.convert.lock"; flock 9
  src=$(readlink -f "$D/datasets/v2pre/gen9-human")
  mkdir -p "$T/source"
  ln "$src"/shard-*.npz "$T/source/" || { say "link failed: $src"; exit 1; }
  cp "$src/meta.json" "$T/source/meta-source.json"
  echo "source <- $src (hard links, $(date -u +%FT%TZ))" > "$T/SOURCES.txt"
  echo "heldout <- $P3/v2pre-human-shuf/shard-val-*.npz (hard links)" >> "$T/SOURCES.txt"
  exec 9>&-
  say "linked $(ls "$T/source" | grep -c '^shard-') source shards from $src"
  "$PY" -I "$D/code/scripts/ops/u7_select.py" "$T/source" "$P3/v2pre-human" "$D/raw/corpus/gen9randombattle/index.jsonl.gz" \
    "$T/select" --seed "$SEED"
  rc=$?; say "select exit $rc"; exit $rc ;;
build)
  GROUP=${3:-36}
  S=$T/stage-shuf
  [ -e "$T/v2pre-human-shuf" ] && { say "exists: $T/v2pre-human-shuf"; exit 1; }
  say "build: group $GROUP, seed $SEED, code ${U7_COMMIT:-unknown}"
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -I "$D/code/scripts/ops/shuffle_shards.py" build \
    "$T/source" "$S" --keep "$T/select/keep.npy" --seed "$SEED" --rows 24000 --group "$GROUP" --spot 200 \
    --commit "${U7_COMMIT:-unknown}" || { say "shuffle build failed"; exit 2; }
  ln "$P3"/v2pre-human-shuf/shard-val-*.npz "$S/" || { say "val link failed"; exit 3; }
  cp "$P3/v2pre-human-shuf/heldout-games-0.06.txt" "$S/"
  "$PY" -I - "$T" "$S" "$P3" "$P2" "$P1LIST" <<'PY' || { say "finalize failed"; exit 4; }
import glob, hashlib, json, os, sys, time, zlib
from pathlib import Path
import numpy as np
T, S, P3, P2, P1LIST = map(Path, sys.argv[1:6])
def is_val(g):  # netv2 porygon.v2.pretrain.is_val(game, 0.06)
    h = np.array([zlib.crc32(int(x).to_bytes(8, "little", signed=True)) for x in g], dtype=np.uint64)
    return (h % 10_000) < 0.06 * 10_000
def ids(p):
    return {int(x) for x in open(p).read().splitlines() if x.strip() and not x.startswith("#")}
meta = json.loads((S / "meta.json").read_text())
p3m = json.loads((P3 / "v2pre-human-shuf/meta.json").read_text())
c = {}
# held-out shards: P3's files, same inodes, all rows held out, games = P3's held-out list
vs = sorted(p.name for p in S.glob("shard-val-*.npz"))
c["val_shards"] = len(vs); c["val_shards_equal_p3_list"] = vs == p3m["val_shards"]
c["val_same_inode_as_p3"] = all(os.stat(S / n).st_ino == os.stat(P3 / "v2pre-human-shuf" / n).st_ino for n in vs)
vrows, vgames, vbad = 0, set(), 0
for n in vs:
    with np.load(S / n) as z:
        g = z["game"]
    vrows += len(g); vgames |= set(np.unique(g).tolist()); vbad += int((~is_val(g)).sum())
c["val_rows"] = vrows; c["val_rows_equal_p3"] = vrows == p3m["val_rows"]; c["val_rows_not_is_val"] = vbad
p3_val = ids(P3 / "v2pre-human-shuf/heldout-games-0.06.txt")
c["val_games"] = len(vgames); c["val_games_equal_p3_heldout_list"] = vgames == p3_val
# training shards: none held out by the rule, none in P1/P2/P3 held-out lists
trows, tg, tval, rat = 0, [], 0, []
for n in meta["train_shards"]:
    with np.load(S / n) as z:
        g = z["game"]; rat.append(z["rating"].astype(np.int32))
    trows += len(g); tg.append(np.unique(g)); tval += int(is_val(g).sum())
tgs = set(np.unique(np.concatenate(tg)).tolist()); rat = np.concatenate(rat)
c["train_rows"] = trows; c["train_games"] = len(tgs); c["train_rows_is_val"] = tval
sel = json.loads((T / "select/selection.json").read_text())
c["train_rows_equal_selection"] = trows == sel["train_rows"]
c["train_games_equal_selection"] = tgs == ids(T / "select/train-games.txt")
c["train_rows_own_rating_lt1500_or_unknown"] = int((rat < 1500).sum())
for nm, lst in (("p3", p3_val), ("p2", ids(P2 / "v2pre-human/heldout-games-0.06.txt")), ("p1", ids(P1LIST))):
    c[f"{nm}_heldout_in_u7_training"] = len(lst & tgs)
c["ok"] = bool(c["val_shards_equal_p3_list"] and c["val_same_inode_as_p3"] and c["val_rows_equal_p3"]
               and c["val_rows_not_is_val"] == 0 and c["val_games_equal_p3_heldout_list"] and c["train_rows_is_val"] == 0
               and c["train_rows_equal_selection"] and c["train_games_equal_selection"]
               and c["train_rows_own_rating_lt1500_or_unknown"] == 0
               and c["p3_heldout_in_u7_training"] == c["p2_heldout_in_u7_training"] == c["p1_heldout_in_u7_training"] == 0)
ver = json.loads((S / "verify.json").read_text())
(S / "verify.json").write_text(json.dumps({"ok": bool(ver["ok"] and c["ok"]), "train_shuffle_verify": ver,
                                           "heldout_and_split_checks": c,
                                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
meta.update({"val_shards": vs, "val_rows": vrows, "rows": trows + vrows, "train_rows": trows,
             "heldout": "P3's shard-val-* (hard links of " + str(P3 / "v2pre-human-shuf") + "), not rebuilt",
             "selection": str(T / "select/selection.json")})
(S / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
# all-rows rating percentiles, as P3's summary.json (rating > 0 over training + held-out rows)
vr = []
for n in vs:
    with np.load(S / n) as z:
        vr.append(z["rating"].astype(np.int32))
allr = np.concatenate([rat] + vr)
summ = {"v2pre-human-shuf": {"decisions": trows + vrows, "games": len(tgs | vgames), "train_decisions": trows,
                             "train_games": len(tgs), "val_decisions_0.06": vrows, "val_games_0.06": len(vgames),
                             "rating_pct_5_50_95": [int(x) for x in np.percentile(allr[allr > 0], [5, 50, 95])],
                             "train_rating_pct_5_50_95": [int(x) for x in np.percentile(rat[rat > 0], [5, 50, 95])],
                             "train_share_own_rating_ge1900": round(float((rat >= 1900).mean()), 4)},
        "p3_reference": {"decisions": 7046891, "train_decisions": 6620882, "val_decisions_0.06": 426009,
                         "rating_pct_5_50_95": [1926, 2066, 2322],
                         "train_own_rating": sel["p3_train"]},
        "u7_train_own_rating": sel["u7_train"], "seed": sel["seed"], "window_utc": sel["window_utc"],
        "frame": sel["frame"], "checks": c}
(T / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
print(json.dumps({"ok": bool(ver["ok"] and c["ok"]), "checks": c, "summary": summ["v2pre-human-shuf"]}))
sys.exit(0 if (ver["ok"] and c["ok"]) else 5)
PY
  mv "$S" "$T/v2pre-human-shuf" || exit 6
  (cd "$T" && find v2pre-human-shuf select summary.json SOURCES.txt README.txt -type f | sort | xargs md5sum > MD5SUMS) || exit 7
  chmod -R a-w "$T/v2pre-human-shuf" "$T/select" "$T/summary.json" "$T/SOURCES.txt" "$T/README.txt" "$T/MD5SUMS"
  say "published $T/v2pre-human-shuf ($(ls "$T/v2pre-human-shuf" | grep -c '^shard-') shards); MD5SUMS written" ;;
*) echo "usage: u7_build.sh select|build T"; exit 1 ;;
esac
