"""U7 row selection (BC size-vs-quality test): a uniform random sample of whole games from the gen9 human v2 pool
rated >= 1500, inside P3's battle-date window, with as many training decisions as P3 (6,620,882).

  python -I scripts/ops/u7_select.py LIVE_DIR P3_DIR INDEX_JSONL_GZ OUT_DIR [--seed 20261111] [--target N]

Frame (one unit = one game id = crc32(battle id), as porygon.replay.seed_of):
  - the battle's uploadtime is inside [min, max] of P3's games' uploadtimes (P3 = the >= 1900 crawl band, newest
    first, so its window is where that crawl had reached);
  - replay rating >= 1500 (the crawl rule); no player matching foulplay|playingfoul|jaxcalibur|bot<N>;
  - not held out (netv2 is_val(game, 0.06)), so P1/P2/P3 held-out games never become training rows;
  - the game id belongs to exactly one corpus battle (crc32 collisions dropped: their date and rating are ambiguous);
  - rows: the deciding side's own rating >= 1500 (unknown / < 1500 sides dropped; probable-bot sides are already
    absent from the pool).
Games are permuted with numpy default_rng(seed) and taken in order until the cumulative decisions are closest to
--target. Writes OUT_DIR/keep.npy (sorted global row ids over the sorted LIVE shards, for shuffle_shards.py --keep),
train-games.txt, selection.json (frame counts, seed, U7 vs P3 rating / date distributions).
"""

import argparse
import glob
import gzip
import hashlib
import json
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

EXCL = re.compile(r"foulplay|playingfoul|jaxcalibur|bot\d*$")
P3_TRAIN_ROWS = 6_620_882


def is_val(game: np.ndarray, frac: float = 0.06) -> np.ndarray:  # netv2 porygon.v2.pretrain.is_val, verbatim
    h = np.array([zlib.crc32(int(g).to_bytes(8, "little", signed=True)) for g in game], dtype=np.uint64)
    return (h % 10_000) < frac * 10_000


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def scan(d, members):
    paths = sorted(str(p) for p in Path(d).glob("shard-*.npz"))
    out = {m: [] for m in members}
    for p in paths:
        with np.load(p) as z:
            for m in members:
                out[m].append(z[m])
    return paths, {m: np.concatenate(v) for m, v in out.items()}


def pct(x, q=(1, 5, 25, 50, 75, 95, 99)):
    return {str(k): int(v) for k, v in zip(q, np.percentile(x, q))} if len(x) else None


def dpct(t, q=(0, 5, 25, 50, 75, 95, 100)):
    return {str(k): time.strftime("%Y-%m-%d", time.gmtime(int(v))) for k, v in zip(q, np.percentile(t, q))}


def dist(rating, game_rr, game_up, rows_game):
    """rating distribution of rows (own rating), games (replay rating) and battle dates."""
    r = rating[rating > 0]
    bands = {">=1900": (1900, 10**6), "1700-1899": (1700, 1900), "1500-1699": (1500, 1700), "<1500": (1, 1500)}
    return {"rows": int(len(rating)), "games": int(len(game_rr)),
            "own_rating_pct": pct(r), "own_rating_mean": round(float(r.mean()), 1),
            "share_rows_own_ge1900": round(float((rating >= 1900).mean()), 4),
            "rows_by_own_rating": {b: int(((rating >= lo) & (rating < hi)).sum()) for b, (lo, hi) in bands.items()},
            "rows_unknown_rating": int((rating <= 0).sum()),
            "game_replay_rating_pct": pct(game_rr[game_rr > 0]),
            "share_games_replay_ge1900": round(float((game_rr >= 1900).mean()), 4),
            "battle_date_pct": dpct(game_up[game_up > 0]),
            "decisions_per_game_mean": round(len(rows_game) / max(1, len(game_rr)), 2)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("live"); p.add_argument("p3"); p.add_argument("index"); p.add_argument("out")
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--target", type=int, default=P3_TRAIN_ROWS)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    lp, L = scan(a.live, ["game", "rating"])
    pp, P = scan(a.p3, ["game", "rating"])
    print(f"live {len(lp)} shards {len(L['game'])} rows; P3 {len(pp)} shards {len(P['game'])} rows", flush=True)
    info = {}
    with gzip.open(a.index, "rt") as f:
        for line in f:
            r = json.loads(line)
            info.setdefault(zlib.crc32(r["id"].encode()), []).append(
                (r["id"], int(r.get("uploadtime") or 0), int(r.get("rating") or -1), r.get("players") or []))
    lg = np.unique(L["game"]); pg = np.unique(P["game"])
    n_entries = np.array([len(info.get(int(c), [])) for c in lg])
    first = [info[int(c)][0] if int(c) in info else ("", 0, -1, []) for c in lg]
    up = np.array([x[1] for x in first], np.int64); rr = np.array([x[2] for x in first], np.int32)
    bad_name = np.array([any(EXCL.search(norm(q)) for q in x[3]) for x in first])
    # P3's battle-date window (P3's own games, unambiguous ids only)
    p3_sel = np.isin(lg, pg) & (n_entries == 1) & (up > 0)
    w0, w1 = int(up[p3_sel].min()), int(up[p3_sel].max())
    val_g = is_val(lg)
    frame = {"games_in_pool": int(len(lg)), "crc_collision_ids": int((n_entries > 1).sum()),
             "no_index_entry": int((n_entries == 0).sum()), "excluded_name": int(bad_name.sum())}
    elig = (n_entries == 1) & (up >= w0) & (up <= w1) & (rr >= 1500) & ~bad_name & ~val_g
    frame.update({"games_in_window": int(((up >= w0) & (up <= w1)).sum()),
                  "games_eligible_before_row_filter": int(elig.sum()),
                  "held_out_games_in_window": int((((up >= w0) & (up <= w1)) & val_g).sum())})
    gi = np.searchsorted(lg, L["game"])
    row_ok = elig[gi] & (L["rating"] >= 1500)
    cnt = np.bincount(gi[row_ok], minlength=len(lg))
    cand = np.flatnonzero(cnt > 0)
    frame.update({"rows_eligible": int(row_ok.sum()), "games_eligible": int(len(cand)),
                  "rows_dropped_own_rating_lt1500_or_unknown": int((elig[gi] & ~(L["rating"] >= 1500)).sum())})
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(cand)
    cum = np.cumsum(cnt[order])
    k = int(np.argmin(np.abs(cum - a.target)))
    chosen = np.sort(order[:k + 1])
    gsel = np.zeros(len(lg), bool); gsel[chosen] = True
    rows = row_ok & gsel[gi]
    keep = np.flatnonzero(rows).astype(np.int64)
    n = int(len(keep))
    assert n == int(cum[k]) and abs(n - a.target) <= 0.005 * a.target, (n, a.target)
    np.save(out / "keep.npy", keep)
    with open(out / "train-games.txt", "w") as f:
        f.write("# U7 training games (int64 game id = crc32 of the battle id), sorted\n")
        f.write("\n".join(str(int(x)) for x in lg[chosen]) + "\n")
    # distributions: U7 training rows vs P3 training rows (and P3 all rows, as P3's summary.json)
    pv = is_val(P["game"])
    pgi = np.searchsorted(lg, P["game"])
    p3_train_games = np.unique(pgi[~pv])
    u7 = dist(L["rating"][keep], rr[chosen], up[chosen], keep)
    p3 = dist(P["rating"][~pv], rr[p3_train_games], up[p3_train_games], np.flatnonzero(~pv))
    p3_all_pct = [int(x) for x in np.percentile(P["rating"][P["rating"] > 0], [5, 50, 95])]
    res = {"what": "U7 = uniform random sample of whole games, gen9 human v2 pool rated >= 1500, inside P3's "
                   "battle-date window, matched to P3's training decisions; held-out = P3's shard-val-* (hard links)",
           "live": str(Path(a.live).resolve()), "p3": str(Path(a.p3).resolve()), "index": a.index,
           "seed": a.seed, "rng": "numpy.random.default_rng(seed).permutation(eligible game positions, sorted ids)",
           "target_train_rows": a.target, "train_rows": n, "train_games": int(len(chosen)),
           "rows_vs_target_pct": round(100 * (n - a.target) / a.target, 4),
           "window_unix": [w0, w1], "window_utc": [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(x)) for x in (w0, w1)],
           "frame": frame, "u7_train": u7, "p3_train": p3, "p3_all_rows_own_rating_pct_5_50_95": p3_all_pct,
           "overlap_u7_train_games_also_p3_train": int(np.isin(lg[chosen], lg[p3_train_games]).sum()),
           "keep_md5": hashlib.md5((out / "keep.npy").read_bytes()).hexdigest(),
           "live_shards": len(lp), "live_rows": int(len(L["game"])), "seconds": round(time.time() - t0, 1)}
    (out / "selection.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
