"""U7 (BC size-vs-quality test) pool analysis: the live gen9 human v2 set vs P3, by rating band and battle date.

  python -I scripts/ops/u7_analyze.py LIVE_DIR P3_DIR INDEX_JSONL_GZ OUT_JSON

Reads only `game`, `rating`, `opp_rating` of every shard and the corpus index (id -> uploadtime, replay rating,
players). game = crc32(battle id) (porygon.replay.seed_of). Reports P3's battle-date window, the live pool's
coverage of the corpus's >= 1500 games inside that window, by band, and any Jaxcalibur / excluded-name game.
"""

import glob
import gzip
import json
import re
import sys
import time
import zlib

import numpy as np

EXCL = re.compile(r"foulplay|playingfoul|jaxcalibur|bot\d*$")


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def scan(d, members):
    out = {m: [] for m in members}
    paths = sorted(glob.glob(f"{d}/shard-*.npz"))
    for p in paths:
        with np.load(p) as z:
            for m in members:
                out[m].append(z[m])
    return paths, {m: np.concatenate(v) for m, v in out.items()}


def day(t):
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(t)))


def main() -> int:
    live, p3, index, out = sys.argv[1:5]
    lp, L = scan(live, ["game", "rating", "opp_rating"])
    pp, P = scan(p3, ["game"])
    lg = np.unique(L["game"]); pg = np.unique(P["game"])
    print(f"live {len(lp)} shards {len(L['game'])} rows {len(lg)} games; P3 {len(pp)} shards {len(P['game'])} rows "
          f"{len(pg)} games; P3 games missing from live: {len(np.setdiff1d(pg, lg))}", flush=True)
    lset = set(lg.tolist())
    info = {}  # crc -> list of (id, uploadtime, rating, players)
    n_idx = 0
    with gzip.open(index, "rt") as f:
        for line in f:
            n_idx += 1
            r = json.loads(line)
            c = zlib.crc32(r["id"].encode())
            info.setdefault(c, []).append((r["id"], r.get("uploadtime") or 0, r.get("rating") or -1,
                                           r.get("players") or [], bool(r.get("suspect_bot"))))
    coll = sum(1 for c in lset if len(info.get(c, [])) > 1)
    miss = sum(1 for c in lset if c not in info)
    print(f"index {n_idx} lines; live games without an index entry {miss}; crc collisions among live games {coll}",
          flush=True)

    def first(c):
        v = info.get(int(c))
        return v[0] if v else None
    up = np.array([(first(c) or (0, 0))[1] for c in lg], dtype=np.int64)
    rr = np.array([(first(c) or (0, 0, -1))[2] for c in lg], dtype=np.int32)
    jax = [first(c)[0] for c in lg if first(c) and any("jaxcalibur" in norm(p) for p in first(c)[3])]
    excl = [first(c)[0] for c in lg if first(c) and any(EXCL.search(norm(p)) for p in first(c)[3])]
    inp3 = np.isin(lg, pg)
    pu = up[inp3 & (up > 0)]
    w0, w1 = int(pu.min()), int(pu.max())
    res = {"live": live, "p3": p3, "live_rows": int(len(L["game"])), "live_games": int(len(lg)),
           "p3_rows": int(len(P["game"])), "p3_games": int(len(pg)),
           "p3_games_missing_from_live": int(len(np.setdiff1d(pg, lg))),
           "live_games_no_index": miss, "crc_collisions_live": coll, "jaxcalibur_games": jax[:20],
           "n_jaxcalibur_games": len(jax), "n_excluded_name_games": len(excl), "excluded_name_games": excl[:20],
           "p3_window": [w0, w1, day(w0), day(w1)],
           "p3_uploadtime_pct_0_1_5_50_95_100": [day(x) for x in np.percentile(pu, [0, 1, 5, 50, 95, 100])],
           "p3_replay_rating_pct_0_5_50_95": [int(x) for x in np.percentile(rr[inp3], [0, 5, 50, 95])]}
    # per-row: own rating band, game in window, game replay rating
    gi = np.searchsorted(lg, L["game"])
    row_up, row_rr = up[gi], rr[gi]
    inw = (row_up >= w0) & (row_up <= w1)
    bands = [(1900, 10**6, ">=1900"), (1700, 1900, "1700-1899"), (1500, 1700, "1500-1699"), (0, 1500, "<1500"),
             (-10**6, 0, "unknown")]
    res["rows_by_own_rating"] = {}
    for lo, hi, name in bands:
        m = (L["rating"] >= lo) & (L["rating"] < hi)
        res["rows_by_own_rating"][name] = {"all": int(m.sum()), "in_p3_window": int((m & inw).sum())}
    res["games_by_replay_rating"] = {}
    for lo, hi, name in bands:
        m = (rr >= lo) & (rr < hi)
        w = (up >= w0) & (up <= w1)
        res["games_by_replay_rating"][name] = {"all": int(m.sum()), "in_p3_window": int((m & w).sum())}
    # corpus coverage inside the window: index games rated >= 1500 (replay rating) vs those in the live pool
    cov = {}
    for c, v in info.items():
        for (bid, t, rat, players, sus) in v:
            if rat is None or rat < 1500 or not (w0 <= t <= w1):
                continue
            b = ">=1900" if rat >= 1900 else "1700-1899" if rat >= 1700 else "1500-1699"
            mo = time.strftime("%Y-%m", time.gmtime(t))
            k = cov.setdefault(b, {}).setdefault(mo, [0, 0])
            k[0] += 1; k[1] += int(c in lset)
    res["window_coverage_index_vs_live"] = cov
    # monthly live games by band (all dates)
    mon = {}
    for t, rat in zip(up, rr):
        if t <= 0:
            continue
        b = ">=1900" if rat >= 1900 else "1700-1899" if rat >= 1700 else "1500-1699" if rat >= 1500 else "<1500"
        mo = time.strftime("%Y-%m", time.gmtime(int(t)))
        mon.setdefault(mo, {}).setdefault(b, 0); mon[mo][b] += 1
    res["live_games_by_month_band"] = dict(sorted(mon.items()))
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("live_games_by_month_band",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
