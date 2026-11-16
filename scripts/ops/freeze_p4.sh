#!/bin/bash
# P4: the next BC init set (freeze_human.sh's procedure, plus a row filter and an extra hold-out). One step inside
# the data host, run from a private copy:
#   srun --jobid=J --overlap --ntasks=1 -c 2 --mem=16G env FREEZE_COMMIT=<ws/data commit> \
#     bash freeze_p4.sh [SEED=20261116] [GROUP=36]          -> $D/datasets/frozen/p4-gen9-<UTC stamp>/
# 1. hard links of the live v2pre/gen9-human under the convert lock (no refresh round swaps it mid-link);
# 2. p4_select.py: keep rows whose deciding side's own rating >= 1900, minus every game of the human records
#    part-0070[0-3] (NNUE gate 1 set H = Step 2's 3,000-position set): those games are in neither split;
# 3. shuffle_shards.py build --keep (training rows permuted, held-out rows = netv2 is_val(game, 0.06) -> shard-val-*;
#    verified: bijection, per-row hashes, split);
# 4. checks: every output row >= 1900 and not a set-H game; no training game in P1/P2/P3 (= U7's) held-out lists;
#    counts equal the selection; summary.json, MD5SUMS (re-checked), read-only; READY written last, only if all pass.
# A failed stage writes FAILED (stage + exit code) and stops. Memory: GROUP x 24k rows x 13.5 KB (36 -> 11.6 GB).
set -uo pipefail
# P4_D / P4_PY / P4_OPS / P4_P1LIST override the paths for a local smoke test only.
D=${P4_D:-/vol/bitbucket/fr1226/porygon-ws-data}
SEED=${1:-20261116}; GROUP=${2:-36}; MIN=1900
PY=${P4_PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}
OPS=${P4_OPS:-$D/code/scripts/ops}
P1LIST=${P4_P1LIST:-/vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/g9h-heldout-games.txt}
P2=$D/datasets/frozen/p2-gen9-20261109-0645
P3=$D/datasets/frozen/p3-gen9-20261109-1735
T=$D/datasets/frozen/p4-gen9-$(date -u +%Y%m%d-%H%M)
say() { echo "$(date -u +%FT%TZ) $*"; }
fail() { say "FAILED: $1 (exit $2)"; echo "$(date -u +%FT%TZ) stage $1 exit $2" > "$T/FAILED"; exit 1; }
[ -e "$T" ] && { say "exists: $T"; exit 1; }
mkdir -p "$T" || exit 1
echo "RUNNING since $(date -u +%FT%TZ) job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?} host $(hostname)" > "$T/STATUS"
say "freeze p4 -> $T (min own rating $MIN, seed $SEED, group $GROUP, code ${FREEZE_COMMIT:-unknown})"
# 1. link
exec 9> "$D/datasets/.convert.lock"; flock 9
src=$(readlink -f "$D/datasets/v2pre/gen9-human"); rec=$(readlink -f "$D/datasets/human/records/gen9randombattle")
mkdir -p "$T/v2pre-human"
ln "$src"/shard-*.npz "$T/v2pre-human/" || fail link 1
cp "$src/meta.json" "$T/v2pre-human/meta-source.json" || fail link 2
mkdir -p "$T/setH-records"
ln "$rec"/part-0070[0-3].jsonl.gz "$T/setH-records/" || fail link 3
exec 9>&-
{ echo "v2pre-human <- $src (hard links, $(date -u +%FT%TZ))"
  echo "setH-records <- $rec/part-0070[0-3].jsonl.gz (hard links; their games are excluded from P4)"; } > "$T/SOURCES.txt"
say "linked $(ls "$T/v2pre-human" | grep -c '^shard-') human shards, $(ls "$T/setH-records" | wc -l) set-H record files"
[ "$(ls "$T/setH-records" | wc -l)" = 4 ] || fail link 4
# 2. select
"$PY" -I "$OPS/p4_select.py" "$T/v2pre-human" "$T/select" --min-rating "$MIN" \
  --exclude-records "$T"/setH-records/part-0070[0-3].jsonl.gz || fail select $?
say "selection written; shuffling"
# 3. shuffle (published to stage-shuf only on verify OK)
S=$T/stage-shuf
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -I "$OPS/shuffle_shards.py" build \
  "$T/v2pre-human" "$S" --keep "$T/select/keep.npy" --seed "$SEED" --rows 24000 --group "$GROUP" --spot 200 \
  --commit "${FREEZE_COMMIT:-unknown}" || fail shuffle $?
cp "$T/select/heldout-games-0.06.txt" "$S/" || fail shuffle 1
# 4. checks + summary
"$PY" -I - "$T" "$S" "$P3" "$P2" "$P1LIST" "$MIN" "$SEED" <<'PY' || fail checks $?
import json, sys, time, zlib
from pathlib import Path
import numpy as np
T, S, P3, P2, P1LIST = map(Path, sys.argv[1:6]); MIN, SEED = int(sys.argv[6]), int(sys.argv[7])
def is_val(g):  # netv2 porygon.v2.pretrain.is_val(game, 0.06)
    h = np.array([zlib.crc32(int(x).to_bytes(8, "little", signed=True)) for x in g], dtype=np.uint64)
    return (h % 10_000) < 0.06 * 10_000
def ids(p):
    return {int(x) for x in open(p).read().splitlines() if x.strip() and not x.startswith("#")}
meta = json.loads((S / "meta.json").read_text()); sel = json.loads((T / "select/selection.json").read_text())
H = ids(T / "select/excluded-games.txt")
def scan(names):
    rows, games, rat, nval, nlow, nh = 0, [], [], 0, 0, 0
    for n in names:
        with np.load(S / n) as z:
            g = z["game"]; r = z["rating"].astype(np.int32)
        rows += len(g); games.append(np.unique(g)); rat.append(r); nval += int(is_val(g).sum())
        nlow += int((r < MIN).sum()); nh += int(np.isin(g, list(H)).sum())
    return rows, set(np.unique(np.concatenate(games)).tolist()), np.concatenate(rat), nval, nlow, nh
tr, tg, trat, tval, tlow, th = scan(meta["train_shards"])
vr, vg, vrat, vval, vlow, vh = scan(meta["val_shards"])
k = sel["kept"]
c = {"train_rows": tr, "train_games": len(tg), "val_rows": vr, "val_games": len(vg),
     "train_rows_equal_selection": tr == k["train_decisions"], "val_rows_equal_selection": vr == k["val_decisions_0.06"],
     "train_games_equal_selection": len(tg) == k["train_games"], "val_games_equal_selection": len(vg) == k["val_games_0.06"],
     "train_rows_is_val": tval, "val_rows_not_is_val": vr - vval, "rows_own_rating_lt_min": tlow + vlow,
     "setH_rows": th + vh, "setH_games_in_p4": len(H & (tg | vg)), "train_val_game_overlap": len(tg & vg),
     "val_games_equal_heldout_list": vg == ids(S / "heldout-games-0.06.txt")}
for nm, lst in (("p3_u7", ids(P3 / "v2pre-human/heldout-games-0.06.txt")),
                ("p2", ids(P2 / "v2pre-human/heldout-games-0.06.txt")), ("p1", ids(P1LIST))):
    c[f"{nm}_heldout"] = len(lst); c[f"{nm}_heldout_in_p4_training"] = len(lst & tg)
    c[f"{nm}_heldout_in_p4_val"] = len(lst & vg)
c["p3_val_shards_heldout_in_p4_training"] = 0
for n in json.loads((P3 / "v2pre-human-shuf/meta.json").read_text())["val_shards"]:
    with np.load(P3 / "v2pre-human-shuf" / n) as z:
        c["p3_val_shards_heldout_in_p4_training"] += len(set(np.unique(z["game"]).tolist()) & tg)
c["ok"] = bool(all(c[x] for x in ("train_rows_equal_selection", "val_rows_equal_selection", "train_games_equal_selection",
                                   "val_games_equal_selection", "val_games_equal_heldout_list"))
               and c["train_rows_is_val"] == c["val_rows_not_is_val"] == c["rows_own_rating_lt_min"] == 0
               and c["setH_rows"] == c["setH_games_in_p4"] == c["train_val_game_overlap"] == 0
               and c["p3_u7_heldout_in_p4_training"] == c["p2_heldout_in_p4_training"] == c["p1_heldout_in_p4_training"] == 0
               and c["p3_val_shards_heldout_in_p4_training"] == 0)
ver = json.loads((S / "verify.json").read_text())
ok = bool(ver["ok"] and c["ok"])
(S / "verify.json").write_text(json.dumps({"ok": ok, "train_shuffle_verify": ver, "p4_checks": c,
                                           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
allr = np.concatenate([trat, vrat])
p = lambda x: [int(v) for v in np.percentile(x[x > 0], [5, 50, 95])]
summ = {"v2pre-human-shuf": {"decisions": tr + vr, "games": len(tg | vg), "train_decisions": tr, "train_games": len(tg),
                             "val_decisions_0.06": vr, "val_games_0.06": len(vg), "rating_pct_5_50_95": p(allr),
                             "train_rating_pct_5_50_95": p(trat), "share_own_rating_ge1900": float((allr >= 1900).mean()),
                             "share_own_rating_ge2100": round(float((allr >= 2100).mean()), 4),
                             "train_shards": len(meta["train_shards"]), "val_shards": len(meta["val_shards"]),
                             "seed": SEED},
        "rule": f"own rating >= {MIN}; minus every game of human records part-0070[0-3] (NNUE gate 1 set H); "
                "held out = netv2 is_val(game, 0.06); no Jaxcalibur game (excluded at crawl and convert)",
        "live_at_freeze": sel["live"], "excluded_setH": {k2: sel[k2] for k2 in
            ("excluded_records", "excluded_games", "excluded_games_in_live", "excluded_decisions_in_live",
             "excluded_decisions_ge_min")},
        "p3_reference": {"decisions": 7046891, "games": 127888, "train_decisions": 6620882, "val_decisions_0.06": 426009,
                         "rating_pct_5_50_95": [1926, 2066, 2322]},
        "checks": c, "verify_ok": ok}
(T / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
print(json.dumps({"ok": ok, "summary": summ["v2pre-human-shuf"], "checks": c}), flush=True)
sys.exit(0 if ok else 5)
PY
mv "$S" "$T/v2pre-human-shuf" || fail publish $?
cat > "$T/README.txt" <<EOF
P4 gen9 human BC set, frozen $(date -u +%FT%TZ) by scripts/ops/freeze_p4.sh (ws/data ${FREEZE_COMMIT:-unknown}).
Rows: deciding side's own rating >= $MIN from the live v2pre/gen9-human at freeze time (v2pre-human/, hard links).
Excluded entirely: every game of human records part-0070[0-3] (NNUE gate 1 set H; setH-records/, select/excluded-games.txt).
Held out: netv2 is_val(game, 0.06) -> v2pre-human-shuf/shard-val-*; so P1/P2/P3/U7 held-out games stay held out.
Training shards: shuffled, seed $SEED, ~24k rows each. Counts and checks: summary.json; v2pre-human-shuf/verify.json.
READY is written last, only after every check and the MD5SUMS re-check passed.
EOF
(cd "$T" && find v2pre-human-shuf select setH-records summary.json SOURCES.txt README.txt -type f | sort \
  | xargs md5sum > MD5SUMS) || fail md5 1
(cd "$T" && md5sum --quiet -c MD5SUMS) || fail md5 2
chmod -R a-w "$T/v2pre-human-shuf" "$T/select" "$T/summary.json" "$T/SOURCES.txt" "$T/README.txt" "$T/MD5SUMS" || fail md5 3
echo "DONE $(date -u +%FT%TZ)" > "$T/STATUS"
: > "$T/READY"
say "published $T ($(ls "$T/v2pre-human-shuf" | grep -c '^shard-') shards, $(du -sh "$T/v2pre-human-shuf" | cut -f1)); READY"
