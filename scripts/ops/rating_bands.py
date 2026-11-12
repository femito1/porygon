"""Rating-band breakdown of a v2 shard set, for cutting or weighting a pretraining freeze by rating.

  python -I scripts/ops/rating_bands.py DIR [--json OUT]

Rows (decisions) are banded by the deciding player's own `rating`; games by the game rating = the higher of the two
players' ratings over the game's rows (`max(rating, opp_rating)`). -1 = unknown. Per band: counts and the rating
5/50/95% within the band. Reads only the `rating`, `opp_rating` and `game` members of each shard.
"""

import argparse
import glob
import json
import sys

import numpy as np

BANDS = [(1900, 10**6, ">=1900"), (1700, 1900, "1700-1899"), (1500, 1700, "1500-1699"), (0, 1500, "<1500")]


def pct(x: np.ndarray):
    return [int(v) for v in np.percentile(x, [5, 50, 95])] if len(x) else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--json")
    a = p.parse_args()
    paths = sorted(glob.glob(f"{a.dir}/shard-*.npz"))
    assert paths, f"no shards in {a.dir}"
    r, o, g = [], [], []
    for q in paths:
        with np.load(q) as z:
            r.append(z["rating"].astype(np.int32)); o.append(z["opp_rating"].astype(np.int32)); g.append(z["game"])
    r, o, g = np.concatenate(r), np.concatenate(o), np.concatenate(g)
    codes, inv = np.unique(g, return_inverse=True)
    gr = np.full(len(codes), -1, np.int32)
    np.maximum.at(gr, inv, np.maximum(r, o))
    res = {"dir": a.dir, "shards": len(paths), "rows": int(len(r)), "games": int(len(codes)),
           "rows_rule": "deciding player's own rating", "games_rule": "max(rating, opp_rating) over the game's rows",
           "bands": {}}
    for lo, hi, name in BANDS + [(None, None, "unknown")]:
        rm = (r < 0) if lo is None else (r >= lo) & (r < hi)
        gm = (gr < 0) if lo is None else (gr >= lo) & (gr < hi)
        res["bands"][name] = {"decisions": int(rm.sum()), "decisions_pct": round(100 * float(rm.mean()), 1),
                              "own_rating_5_50_95": pct(r[rm]), "games": int(gm.sum()),
                              "games_pct": round(100 * float(gm.mean()), 1),
                              "decisions_in_those_games": int(gm[inv].sum()), "game_rating_5_50_95": pct(gr[gm])}
    print(f"{a.dir}: {res['shards']} shards, {res['rows']} decisions, {res['games']} games")
    print(f"{'band':>10} | {'decisions (own rating)':>24} {'Elo 5/50/95':>16} | {'games (game rating)':>21} "
          f"{'their decisions':>15} {'Elo 5/50/95':>16}")
    for name, b in res["bands"].items():
        e1 = "/".join(map(str, b["own_rating_5_50_95"])) if b["own_rating_5_50_95"] else "-"
        e2 = "/".join(map(str, b["game_rating_5_50_95"])) if b["game_rating_5_50_95"] else "-"
        print(f"{name:>10} | {b['decisions']:>14} ({b['decisions_pct']:>5}%) {e1:>16} | {b['games']:>12} "
              f"({b['games_pct']:>5}%) {b['decisions_in_those_games']:>15} {e2:>16}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
