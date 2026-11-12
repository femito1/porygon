"""Add target_argmax / target_t05 to every shard of a v2pre search-label dataset version, as a new version dir
(same naming as convert_replays: <final>.v<stamp>-<pid>), then swap the <final> symlink by rename. Shard by shard."""
import json, os, sys, time
from pathlib import Path
import numpy as np
final = Path(sys.argv[1])
src = final.resolve()
new = final.parent / f"{final.name}.v{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
new.mkdir()
for p in sorted(src.glob("shard-*.npz")):
    with np.load(p) as z:
        a = {k: z[k] for k in z.files}
    if "target_argmax" not in a and a.get("searched") is not None and a["searched"].any():
        t = np.clip(a["target"].astype(np.float32), 0, None); has = t.sum(1) > 0
        am = np.zeros_like(t); am[np.flatnonzero(has), t[has].argmax(1)] = 1
        sq = t ** 2; s = sq.sum(1, keepdims=True); t05 = np.divide(sq, s, out=np.zeros_like(sq), where=s > 0)
        a["target_argmax"], a["target_t05"] = am.astype(np.float16), t05.astype(np.float16)
    tmp = new / (p.stem + ".tmp.npz"); np.savez_compressed(tmp, **a); os.replace(tmp, new / p.name)
meta = json.loads((src / "meta.json").read_text())
meta["extras"] = meta.get("extras", "") + "; target_argmax f16[N,A], target_t05 f16[N,A] (added 2026-11-09 by add_sharp.py)"
meta["derived_from"] = str(src)
(new / "meta.json").write_text(json.dumps(meta, indent=1))
link = final.parent / (final.name + ".tmp-link"); link.unlink(missing_ok=True); link.symlink_to(new.name); os.replace(link, final)
print("swapped", final, "->", new.name, len(list(new.glob("shard-*.npz"))), "shards")
