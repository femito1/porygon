"""P4 row selection: the rows of a frozen copy of the live gen9 human v2 set that P4 keeps.

  python -I scripts/ops/p4_select.py SRC_DIR OUT_DIR --min-rating 1900 --exclude-records R1.jsonl.gz [R2 ...]

Kept row = the deciding side's own `rating` >= MIN_RATING and the row's game is not in any --exclude-records file
(game id = crc32 of the record's battle `id`, as convert_replays.py's v2pre `game`). Held-out rows (netv2
`is_val(game, 0.06)`) are kept by the same rule and become P4's shard-val-*; excluded games are in neither split.
Writes OUT_DIR/keep.npy (sorted int64 global source rows, for shuffle_shards.py --keep), excluded-games.txt,
heldout-games-0.06.txt (kept held-out games) and selection.json (live-set totals and bands, the selection counts).
Also writes SRC_DIR/heldout-games-0.06.txt (every held-out game of the source), as freeze_human.sh does.
Reads only the `game` and `rating` members.
"""

import argparse
import glob
import gzip
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np


def is_val(game: np.ndarray, frac: float = 0.06) -> np.ndarray:  # netv2 porygon.v2.pretrain.is_val, verbatim
    h = np.array([zlib.crc32(int(g).to_bytes(8, "little", signed=True)) for g in game], dtype=np.uint64)
    return (h % 10_000) < frac * 10_000


def pct(x: np.ndarray):
    x = x[x > 0]
    return [int(v) for v in np.percentile(x, [5, 50, 95])] if len(x) else None


def record_games(paths: list[str]) -> tuple[set[int], dict]:
    games, info = set(), {}
    for p in paths:
        n, g = 0, set()
        with gzip.open(p, "rt") as f:
            for line in f:
                if line.strip():
                    g.add(zlib.crc32(json.loads(line)["id"].encode())); n += 1
        info[str(Path(p).resolve())] = {"records": n, "games": len(g), "md5": hashlib.md5(Path(p).read_bytes()).hexdigest()}
        games |= g
    return games, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--min-rating", type=int, default=1900)
    ap.add_argument("--exclude-records", nargs="+", default=[])
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    paths = sorted(glob.glob(str(src / "shard-*.npz")))
    assert paths, f"no shards in {src}"
    g, r = [], []
    for p in paths:
        with np.load(p) as z:
            g.append(z["game"].astype(np.int64)); r.append(z["rating"].astype(np.int32))
    game, rating = np.concatenate(g), np.concatenate(r)
    val = is_val(game)
    excl, excl_info = record_games(a.exclude_records)
    ex = np.isin(game, np.fromiter(excl, np.int64, len(excl)))
    hi = rating >= a.min_rating
    keep = hi & ~ex
    np.save(out / "keep.npy", np.flatnonzero(keep).astype(np.int64))
    (out / "excluded-games.txt").write_text(
        "# crc32(battle id) of every record in the --exclude-records files (NNUE gate 1 set H, Step 2)\n"
        + "\n".join(str(x) for x in sorted(excl)) + "\n")
    hd = "# netv2 porygon.v2.pretrain.is_val(game, 0.06); superset-consistent with g9h-heldout-games.txt\n"
    (src / "heldout-games-0.06.txt").write_text(hd + "\n".join(str(x) for x in np.unique(game[val])) + "\n")
    (out / "heldout-games-0.06.txt").write_text(hd + "\n".join(str(x) for x in np.unique(game[keep & val])) + "\n")
    bands = {}
    for lo, up, name in ((1900, 10**6, ">=1900"), (1700, 1900, "1700-1899"), (1500, 1700, "1500-1699"),
                         (-10**6, 1500, "<1500 or unknown")):
        m = (rating >= lo) & (rating < up)
        bands[name] = {"decisions": int(m.sum()), "games": int(len(np.unique(game[m])))}
    sel = {
        "source": str(src.resolve()), "source_shards": len(paths), "min_own_rating": a.min_rating,
        "live": {"decisions": int(len(game)), "games": int(len(np.unique(game))),
                 "val_decisions_0.06": int(val.sum()), "val_games_0.06": int(len(np.unique(game[val]))),
                 "rating_pct_5_50_95": pct(rating), "own_rating_bands": bands},
        "excluded_records": excl_info, "excluded_games": len(excl),
        "excluded_games_in_live": int(len(np.unique(game[ex]))), "excluded_decisions_in_live": int(ex.sum()),
        "excluded_decisions_ge_min": int((ex & hi).sum()),
        "kept": {"decisions": int(keep.sum()), "games": int(len(np.unique(game[keep]))),
                 "train_decisions": int((keep & ~val).sum()), "train_games": int(len(np.unique(game[keep & ~val]))),
                 "val_decisions_0.06": int((keep & val).sum()), "val_games_0.06": int(len(np.unique(game[keep & val]))),
                 "rating_pct_5_50_95": pct(rating[keep]), "train_rating_pct_5_50_95": pct(rating[keep & ~val])},
    }
    (out / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    print(json.dumps(sel), flush=True)
    return 0 if sel["kept"]["decisions"] > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
