"""Write the game ids `porygon.v2.pretrain` holds out (crc32 split by game id) for a set of shards.

    python scripts/pretrain_heldout.py 'DIR/shard-*.npz' --val-frac 0.06 --out heldout.txt
"""
import argparse
import glob

import numpy as np

from porygon.v2.pretrain import is_val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shards")
    p.add_argument("--val-frac", type=float, default=0.06)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    games = np.unique(np.concatenate([np.load(f)["game"] for f in sorted(glob.glob(a.shards))]))
    held = games[is_val(games, a.val_frac)]
    with open(a.out, "w") as f:
        f.write(f"# porygon.v2.pretrain held-out games: crc32(int64 little-endian game id) % 10000 < {a.val_frac} * 10000\n")
        f.write("\n".join(str(int(g)) for g in held) + "\n")
    print(f"{len(held)} of {len(games)} games held out -> {a.out}")


if __name__ == "__main__":
    main()
