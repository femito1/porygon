"""Copy spd_h2h.py shard records with the sides' results swapped (winner A <-> B, score -> 1 - score), so wsp's
seq_monitor tests "B beats A" with the same one-sided design on the same pairs. Meta files are copied unchanged.
    python scripts/netv2_cmp_flip.py SRC_DIR DST_DIR"""

import json
import shutil
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
SWAP = {"A": "B", "B": "A"}
for f in sorted(src.glob("shard-*.jsonl")):
    if f.name.endswith(".part.jsonl"):
        continue
    out = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for g in r.get("games", []):
            g["winner"] = SWAP.get(g.get("winner"), g.get("winner"))
        if "score" in r:
            r["score"] = 1.0 - float(r["score"])
        out.append(json.dumps(r))
    tmp = dst / (f.name + ".tmp")
    tmp.write_text("\n".join(out) + ("\n" if out else ""))
    tmp.replace(dst / f.name)
for m in sorted(src.glob("shard-*.meta.json")):
    shutil.copyfile(m, dst / m.name)
