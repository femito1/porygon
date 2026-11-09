"""Pre-shuffled copy of a v2 supervised shard set (CONTRACTS.md "v2 supervised shards"): the same rows, arrays and
dtypes; the training rows (netv2 `is_val(game, VAL_FRAC)` false) globally permuted with a fixed seed and cut into
`shard-NNNNN.npz` of ~ROWS rows; the held-out rows (is_val true, source order) in `shard-val-NNNN.npz`. Every
training shard is then a uniform random sample of the training rows, so a streaming loader can build each part from
whole shards (each shard decompressed once per epoch). The source set is only read.

  python -I scripts/ops/shuffle_shards.py build  SRC_DIR OUT_DIR [--seed 20261109] [--rows 24000] [--group 10]
  python -I scripts/ops/shuffle_shards.py verify SRC_DIR OUT_DIR [--spot 200]

build writes OUT_DIR.tmp-<pid>/, runs verify on it and renames it to OUT_DIR only if every check passes (OUT_DIR
must not exist). Memory ~ group x rows x row bytes (10 x 24k x 13.5 KB = 3.2 GB) + one decoded source shard.
OUT_DIR/rowmap.npz: per output shard `<name>/src_shard` (index into meta.json `source_shards`) and `<name>/src_row`.

verify reads every source and output file once, independently of how OUT_DIR was built:
  - total rows equal, and rowmap is a bijection onto the source rows;
  - the multiset of game ids equal;
  - held-out: no training shard holds an is_val row, every shard-val row is is_val, the held-out count is equal;
  - every member: same names, dtype and per-row shape in every shard;
  - every row: blake2b-64 over all its members' bytes (sorted member order) equals its source row's;
  - SPOT random rows: every member byte-equal to the source row.
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np


def is_val(game: np.ndarray, frac: float) -> np.ndarray:  # netv2 porygon.v2.pretrain.is_val, verbatim
    h = np.array([zlib.crc32(int(g).to_bytes(8, "little", signed=True)) for g in game], dtype=np.uint64)
    return (h % 10_000) < frac * 10_000


def say(*a) -> None:
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def src_shards(src: Path) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(src / "shard-*.npz")))
    assert paths, f"no shards in {src}"
    return paths


def decode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def row_hashes(d: dict[str, np.ndarray]) -> np.ndarray:
    """blake2b-64 of each row over all members' bytes, members in sorted order."""
    n = len(d["game"])
    buf = np.concatenate([np.ascontiguousarray(d[k]).view(np.uint8).reshape(n, -1) for k in sorted(d)], axis=1)
    out = np.empty(n, np.uint64)
    for i in range(n):
        out[i] = int.from_bytes(hashlib.blake2b(buf[i], digest_size=8).digest(), "little")
    return out


def build(a) -> int:
    src, out = Path(a.src), Path(a.out)
    assert not out.exists(), f"{out} exists"
    paths = src_shards(src)
    games, n_of = [], []
    for p in paths:  # only the game member is decompressed here
        with np.load(p) as z:
            g = z["game"]
        games.append(g); n_of.append(len(g))
    sid = np.concatenate([np.full(n, i, np.int16) for i, n in enumerate(n_of)])
    srow = np.concatenate([np.arange(n, dtype=np.int32) for n in n_of])
    game = np.concatenate(games)
    val = is_val(game, a.val_frac)
    n_total, n_val = len(game), int(val.sum())
    train_idx = np.flatnonzero(~val)
    val_idx = np.flatnonzero(val)  # source order
    perm = np.random.default_rng(a.seed).permutation(train_idx)
    k_train = -(-len(perm) // a.rows)
    k_val = -(-len(val_idx) // a.rows)
    plan = [(f"shard-{i:05d}.npz", c) for i, c in enumerate(np.array_split(perm, k_train))]
    plan += [(f"shard-val-{i:04d}.npz", c) for i, c in enumerate(np.array_split(val_idx, k_val))]
    say(f"{len(paths)} source shards, {n_total} rows ({n_val} held out at {a.val_frac}); "
        f"{k_train} training + {k_val} held-out output shards, groups of {a.group}")
    first = decode(paths[0])
    spec = {k: (v.dtype, v.shape[1:]) for k, v in first.items()}
    del first
    tmp = out.parent / f"{out.name}.tmp-{os.getpid()}"
    tmp.mkdir(parents=True)
    rowmap = {}
    for g0 in range(0, len(plan), a.group):
        grp = plan[g0:g0 + a.group]
        bufs = [{k: np.empty((len(c), *shp), dt) for k, (dt, shp) in spec.items()} for _, c in grp]
        # every global row of the group -> (output shard in group, position), grouped by source shard
        rows = np.concatenate([c for _, c in grp])
        o_of = np.concatenate([np.full(len(c), j, np.int32) for j, (_, c) in enumerate(grp)])
        pos = np.concatenate([np.arange(len(c), dtype=np.int64) for _, c in grp])
        order = np.argsort(sid[rows], kind="stable")
        s_sorted = sid[rows][order]
        bounds = np.flatnonzero(np.diff(s_sorted)) + 1
        for chunk in np.split(order, bounds):
            s = int(sid[rows[chunk[0]]])
            d = decode(paths[s])
            assert {k: (v.dtype, v.shape[1:]) for k, v in d.items()} == spec, f"member spec differs in {paths[s]}"
            r = srow[rows[chunk]]
            oo, pp = o_of[chunk], pos[chunk]
            for j in np.unique(oo):
                m = oo == j
                for k in spec:
                    bufs[j][k][pp[m]] = d[k][r[m]]
            del d
        for (name, c), b in zip(grp, bufs):
            np.savez_compressed(tmp / name, **b)
            rowmap[f"{name}/src_shard"] = sid[c]
            rowmap[f"{name}/src_row"] = srow[c]
        del bufs
        say(f"group {g0 // a.group + 1}/{-(-len(plan) // a.group)}: wrote {', '.join(n for n, _ in grp[:2])}"
            f"{' ...' if len(grp) > 2 else ''}")
    np.savez_compressed(tmp / "rowmap.npz", **rowmap)
    for extra in ("heldout-games-0.06.txt",):
        if (src / extra).exists():
            shutil.copy2(src / extra, tmp / extra)
    commit = a.commit
    if commit is None:
        try:
            commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True, timeout=30).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
    src_meta = None
    for mname in ("meta.json", "meta-source.json"):
        if (src / mname).exists():
            src_meta = json.loads((src / mname).read_text()); break
    meta = {"layout": "shuffled", "seed": a.seed, "val_frac": a.val_frac,
            "val_rule": "netv2 porygon.v2.pretrain.is_val: crc32(int64 game, 8 bytes little-endian) % 10000 < frac*10000",
            "rows": n_total, "train_rows": int(len(perm)), "val_rows": n_val, "rows_per_shard_target": a.rows,
            "train_shards": [n for n, _ in plan[:k_train]], "val_shards": [n for n, _ in plan[k_train:]],
            "source": str(src.resolve()), "source_shards": [p.name for p in paths], "source_meta": src_meta,
            "members": {k: [str(dt), list(shp)] for k, (dt, shp) in spec.items()},
            "note": "rows bit-identical to the source; training shards are uniform random samples of the training "
                    "rows (whole-shard streaming parts are unbiased); rowmap.npz maps every row to its source row",
            "code": {"script": "scripts/ops/shuffle_shards.py", "commit": commit},
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    say(f"built {tmp}; verifying")
    rc = verify_dirs(src, tmp, a.spot, a.val_frac)
    if rc != 0:
        say(f"VERIFY FAILED: left {tmp} in place, {out} not created")
        return rc
    os.rename(tmp, out)
    say(f"published {out}")
    return 0


def verify_dirs(src: Path, out: Path, spot: int, val_frac: float) -> int:
    meta = json.loads((out / "meta.json").read_text())
    paths = src_shards(src)
    fails = []
    def check(ok: bool, msg: str) -> None:
        if not ok:
            fails.append(msg); say("FAIL", msg)
    check([p.name for p in paths] == meta["source_shards"], "source shard list differs from meta.json")
    with np.load(out / "rowmap.npz") as z:
        rm = {k: z[k] for k in z.files}
    names = meta["train_shards"] + meta["val_shards"]
    check(sorted(glob.glob(str(out / "shard-*.npz"))) == sorted(str(out / n) for n in names),
          "output shard files differ from meta.json")
    # source pass: spec, per-row hashes, games, held-out flags; stash the spot rows
    rng = np.random.default_rng(12345)
    spot_rows = {}  # (src shard, src row) -> None until filled
    picks = []
    for _ in range(spot):
        n = names[int(rng.integers(len(names)))]
        i = int(rng.integers(len(rm[f"{n}/src_row"])))
        key = (int(rm[f"{n}/src_shard"][i]), int(rm[f"{n}/src_row"][i]))
        picks.append((n, i, key)); spot_rows[key] = None
    spec, src_hash, src_games, n_src, n_src_val = None, [], [], 0, 0
    for s, p in enumerate(paths):
        d = decode(p)
        sp = {k: (str(v.dtype), v.shape[1:]) for k, v in d.items()}
        spec = spec or sp
        check(sp == spec, f"source member spec differs in {p.name}")
        src_hash.append(row_hashes(d)); src_games.append(d["game"])
        n_src += len(d["game"]); n_src_val += int(is_val(d["game"], val_frac).sum())
        for (ss, rr) in spot_rows:
            if ss == s:
                spot_rows[(ss, rr)] = {k: v[rr].copy() for k, v in d.items()}
        del d
    say(f"source pass: {n_src} rows, {n_src_val} held out, {len(spec)} members")
    # bijection
    offsets = np.cumsum([0] + [len(h) for h in src_hash])
    seen = np.zeros(n_src, np.int8)
    for n in names:
        gi = offsets[rm[f"{n}/src_shard"].astype(np.int64)] + rm[f"{n}/src_row"]
        np.add.at(seen, gi, 1)
    check(bool((seen == 1).all()), f"rowmap is not a bijection: {int((seen == 0).sum())} unmapped, "
                                   f"{int((seen > 1).sum())} mapped more than once")
    # output pass
    all_hash = np.concatenate(src_hash)
    n_out, n_out_val, out_games, bad_hash = 0, 0, [], 0
    for n in names:
        d = decode(out / n)
        sp = {k: (str(v.dtype), v.shape[1:]) for k, v in d.items()}
        check(sp == spec, f"member spec differs in {n}")
        rows = len(d["game"])
        check(rows == len(rm[f"{n}/src_row"]), f"{n}: {rows} rows vs {len(rm[f'{n}/src_row'])} in rowmap")
        v = is_val(d["game"], val_frac)
        if n.startswith("shard-val-"):
            check(bool(v.all()), f"{n}: {int((~v).sum())} training rows in a held-out shard")
        else:
            check(not v.any(), f"{n}: {int(v.sum())} held-out rows in a training shard")
        n_out += rows; n_out_val += int(v.sum()); out_games.append(d["game"])
        h = row_hashes(d)
        ref = all_hash[offsets[rm[f"{n}/src_shard"].astype(np.int64)] + rm[f"{n}/src_row"]]
        bad_hash += int((h != ref).sum())
        for (pn, i, key) in picks:
            if pn == n:
                ok = set(d) == set(spot_rows[key]) and all(d[k][i].tobytes() == spot_rows[key][k].tobytes() for k in d)
                check(ok, f"spot row {n}[{i}] != source {paths[key[0]].name}[{key[1]}]")
        del d
    check(n_out == n_src, f"rows: {n_out} out vs {n_src} source")
    check(n_out == meta["rows"], f"rows: {n_out} out vs meta {meta['rows']}")
    check(n_out_val == n_src_val == meta["val_rows"], f"held out: {n_out_val} out vs {n_src_val} source vs meta {meta['val_rows']}")
    check(bad_hash == 0, f"{bad_hash} rows differ from their source row (blake2b-64)")
    check(np.array_equal(np.sort(np.concatenate(out_games)), np.sort(np.concatenate(src_games))),
          "multiset of game ids differs")
    held = out / "heldout-games-0.06.txt"
    if held.exists() and abs(val_frac - 0.06) < 1e-12:
        listed = {int(x) for x in held.read_text().splitlines() if x.strip() and not x.startswith("#")}
        got = {int(g) for gg in out_games for g in np.unique(gg[is_val(gg, val_frac)])}
        check(listed == got, f"held-out game list differs: {len(listed)} listed vs {len(got)} in shards")
    summary = {"rows": n_out, "val_rows": n_out_val, "members": len(spec), "shards": len(names),
               "hash_mismatches": bad_hash, "spot_rows": len(picks), "failures": fails}
    say("VERIFY", "OK" if not fails else "FAILED", json.dumps(summary))
    (out / "verify.json").write_text(json.dumps({**summary, "ok": not fails,
                                                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    return 0 if not fails else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["build", "verify"])
    p.add_argument("src"); p.add_argument("out")
    p.add_argument("--seed", type=int, default=20261109)
    p.add_argument("--rows", type=int, default=24000)
    p.add_argument("--group", type=int, default=10)
    p.add_argument("--spot", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.06)
    p.add_argument("--commit", default=None, help="code commit recorded in meta.json (default: git, if a checkout)")
    a = p.parse_args()
    if a.cmd == "build":
        return build(a)
    return verify_dirs(Path(a.src), Path(a.out), a.spot, a.val_frac)


if __name__ == "__main__":
    sys.exit(main())
