"""Held-out masks for porygon.bc (--val-mask) that reproduce another dataset's game split on paired shards.

    python scripts/paired_val_mask.py V1_DIR PAIRED_DIR OUT_DIR [--frac 0.06] [--expect N] [--games FILE]

V1_DIR and PAIRED_DIR hold shards of the same converter pass under the same file names (e.g. the v1 and v2pre
encodings of one dataset version). Each pair must agree row by row on turn, action, mask, forced, z and target
(and PAIRED's obs/action_mask, if present, must equal mask); any mismatch aborts. The split is netv2's
porygon.v2.pretrain.is_val on PAIRED's int64 `game`: crc32(game as 8 little-endian signed bytes) % 10000
< frac * 10000. --games FILE (one id per line, # comments): the val games must be exactly the listed ids present
in PAIRED. Writes OUT_DIR/<shard stem>.npy (bool per row) and OUT_DIR/split.json (counts).
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np


def is_val(game: np.ndarray, frac: float) -> np.ndarray:
    """Exactly porygon.v2.pretrain.is_val (netv2)."""
    h = np.array([zlib.crc32(int(g).to_bytes(8, "little", signed=True)) for g in game], dtype=np.uint64)
    return (h % 10_000) < frac * 10_000


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("v1", type=Path)
    ap.add_argument("paired", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frac", type=float, default=0.06)
    ap.add_argument("--expect", type=int, help="abort unless the val set has exactly this many decisions")
    ap.add_argument("--games", type=Path, help="held-out game ids the val set must equal (within PAIRED)")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in a.v1.glob("shard-*.npz"))
    other = sorted(p.name for p in a.paired.glob("shard-*.npz"))
    if names != other:
        sys.exit(f"shard names differ: {sorted(set(names) ^ set(other))[:10]}")
    tot = {"shards": len(names), "decisions": 0, "val_decisions": 0, "val_games": 0, "games": 0}
    all_games, val_games = set(), set()
    for name in names:
        with np.load(a.v1 / name) as x, np.load(a.paired / name) as y:
            n = len(x["z"])
            if len(y["z"]) != n:
                sys.exit(f"{name}: {n} rows vs {len(y['z'])}")
            for k in ("turn", "action", "mask", "forced", "z"):
                if not np.array_equal(x[k], y[k]):
                    sys.exit(f"{name}: `{k}` differs ({int((np.asarray(x[k]) != np.asarray(y[k])).sum())} entries)")
            if not np.array_equal(x["target"].astype(np.float32), y["target"].astype(np.float32)):
                sys.exit(f"{name}: `target` differs")
            if "obs/action_mask" in y.files and not np.array_equal(y["obs/action_mask"] > 0, x["mask"] > 0):
                sys.exit(f"{name}: obs/action_mask differs from mask")
            g2 = y["game"].astype(np.int64)
            # the same games: v1 and paired game ids must map one to one
            pairs = np.unique(np.stack([x["game"].astype(np.int64), g2], 1), axis=0)
            if len(pairs) != len(np.unique(g2)) or len(pairs) != len(np.unique(x["game"])):
                sys.exit(f"{name}: game ids do not map one to one")
            m = is_val(g2, a.frac)
        np.save(a.out / f"{Path(name).stem}.npy", m)
        tot["decisions"] += n
        tot["val_decisions"] += int(m.sum())
        tot["games"] += len(np.unique(g2))
        tot["val_games"] += len(np.unique(g2[m]))
        all_games.update(np.unique(g2).tolist())
        val_games.update(np.unique(g2[m]).tolist())
    if a.games:
        listed = {int(x) for x in a.games.read_text().split("\n") if x.strip() and not x.startswith("#")}
        want = listed & all_games
        if val_games != want:
            sys.exit(f"val games != listed games in the data: {len(val_games - want)} extra, "
                     f"{len(want - val_games)} missing")
        tot |= {"games_file": str(a.games), "listed": len(listed), "listed_in_data": len(want), "games_match": True}
    tot |= {"frac": a.frac, "v1": str(a.v1.resolve()), "paired": str(a.paired.resolve()),
            "rule": "netv2 porygon.v2.pretrain.is_val(paired game int64, frac)"}
    (a.out / "split.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot))
    if a.expect is not None and tot["val_decisions"] != a.expect:
        sys.exit(f"val decisions {tot['val_decisions']} != expected {a.expect}")


if __name__ == "__main__":
    main()
