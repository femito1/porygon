"""Freeze the gen9 tracker positions in one file that v2diag.load_pos reads directly (DIR/rows.pkl).

The gen9 tracker read every searched row of data's gen9-searchlabels shards on every call (v2diag.load_npz_pos, which
indexes NpzFile members inside its row loop and so re-decompresses whole arrays once per row: ~25 min per call), then
sampled from the full list (v2diag_decay: 8,000 rows by rng(0); v2diag_switch: candidates and donors). The sample
therefore depended on the shard set, which data refreshes (shard 6 was added at 08:16 on 2026-11-09). This script
copies a pinned shard set, rebuilds exactly the row list load_npz_pos returns for it (same order, same keys and values;
gid keeps the path the tracker used), checks it against load_npz_pos on one shard, and pickles the full list, so the
unchanged tracker scripts select the same positions as before, quickly and independently of later refreshes.

    python scripts/wsp/v2diag_cache9.py --src DATA_VERSION_DIR --as-path ORIGINAL_DIR --out v2diag/pos9
"""

import argparse
import glob
import hashlib
import json
import math
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from v2diag import load_npz_pos, load_pos


def rows_from_shards(files, as_names, searched_only=True):
    """load_npz_pos with each NpzFile member read once per shard (identical output)."""
    rows = []
    for f, name in zip(files, as_names):
        z = np.load(f)
        cols = {k[4:]: z[k] for k in z.files if k.startswith("obs/")}
        zz, game, target = z["z"], z["game"], z["target"]
        rv = z["root_value"] if "root_value" in z.files else None
        sel = np.where(z["searched"] > 0)[0] if (searched_only and "searched" in z.files) else np.arange(len(zz))
        for i in sel:
            rows.append({"gid": (name, int(game[i])), "obs": {k: v[i] for k, v in cols.items()},
                         "v1_probs": target[i].astype(np.float32), "v1_value": float(rv[i]) if rv is not None else 0.0,
                         "won": bool(zz[i] > 0)})
    return rows


def same_rows(a, b):
    if len(a) != len(b):
        return f"length {len(a)} vs {len(b)}"
    for j, (x, y) in enumerate(zip(a, b)):
        same_value = x["v1_value"] == y["v1_value"] or (math.isnan(x["v1_value"]) and math.isnan(y["v1_value"]))
        if x.keys() != y.keys() or x["gid"] != y["gid"] or x["won"] != y["won"] or not same_value:  # root_value: NaN = absent
            return f"row {j}: scalar fields differ"
        if not (x["v1_probs"].dtype == y["v1_probs"].dtype and np.array_equal(x["v1_probs"], y["v1_probs"])):
            return f"row {j}: v1_probs differ"
        if list(x["obs"]) != list(y["obs"]):
            return f"row {j}: obs keys differ"
        for k in x["obs"]:
            u, v = x["obs"][k], y["obs"][k]
            if u.dtype != v.dtype or u.shape != v.shape or not np.array_equal(u, v):
                return f"row {j}: obs/{k} differs"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pinned shard dir (a data version dir, not the moving symlink)")
    ap.add_argument("--as-path", required=True, help="dir name the tracker globbed (kept in each row's gid)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-shard", default="shard-00001.npz", help="shard compared against load_npz_pos")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(f"{a.src}/shard-*.npz"))
    assert files, a.src
    manifest = {"src": os.path.realpath(a.src), "as_path": a.as_path, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "shards": []}
    local = []
    for f in files:
        dst = out / "shards" / Path(f).name
        shutil.copyfile(f, dst)
        manifest["shards"].append({"name": Path(f).name, "bytes": dst.stat().st_size,
                                   "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()[:16]})
        local.append(str(dst))
    names = [f"{a.as_path}/{Path(f).name}" for f in files]
    t = time.time()
    rows = rows_from_shards(local, names)
    print(f"{len(rows)} rows from {len(files)} shards in {time.time() - t:.1f} s (fast loader)", flush=True)
    # the fast loader against the tracker's own loader, on one shard read from the tracker's path
    k = [Path(f).name for f in files].index(a.check_shard)
    t = time.time()
    ref = load_npz_pos(f"{a.as_path}/{a.check_shard}")
    t_ref = time.time() - t
    lo = sum(1 for r in rows if r["gid"][0] < names[k])
    mine = [r for r in rows if r["gid"][0] == names[k]]
    err = same_rows(mine, ref)
    print(f"check vs load_npz_pos on {a.check_shard}: {len(ref)} rows, {'IDENTICAL' if err is None else 'MISMATCH ' + err} "
          f"(load_npz_pos {t_ref:.1f} s for this shard; rows {lo}..{lo + len(mine) - 1} of the list)", flush=True)
    assert err is None
    with open(out / "rows.pkl.tmp", "wb") as fh:
        pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(out / "rows.pkl.tmp", out / "rows.pkl")
    t = time.time()
    back = load_pos(str(out))
    err = same_rows(back, rows)
    print(f"rows.pkl {os.path.getsize(out / 'rows.pkl') / 1e6:.0f} MB; load_pos({out}) {time.time() - t:.1f} s, "
          f"{'IDENTICAL' if err is None else 'MISMATCH ' + err}", flush=True)
    assert err is None
    manifest.update({"rows": len(rows), "rows_pkl_bytes": os.path.getsize(out / "rows.pkl"),
                     "check": f"{a.check_shard}: identical to load_npz_pos ({len(ref)} rows)"})
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
