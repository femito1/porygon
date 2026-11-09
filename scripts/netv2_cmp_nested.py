"""Merge spd_h2h.py shard records (shard-KKKK.jsonl, global pair ids) into ONE shard in pair-id order, so wsp's
seq_monitor evaluates nested looks (look j = pairs 0 .. j*k - 1) when the shards were played one after another.
Its default ordering interleaves shards round-robin (right for shards played in parallel); with one shard per look
played in sequence that re-draws the data of every earlier look each time a shard is added. Incomplete (.part)
shards are skipped; pair ids must be contiguous from 0.
    python scripts/netv2_cmp_nested.py SRC_DIR DST_DIR   # then: seq_monitor.py DST_DIR --design ... --shard-pairs N"""

import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
recs = {}
for f in sorted(src.glob("shard-*.jsonl")):
    if f.name.endswith(".part.jsonl"):
        continue
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["pair"] in recs:
                raise SystemExit(f"duplicate pair id {r['pair']} in {f}")
            r["shard"] = 0
            recs[r["pair"]] = r
ids = sorted(recs)
if ids != list(range(len(ids))):
    raise SystemExit(f"pair ids not contiguous from 0: {ids[:3]} .. {ids[-3:]} ({len(ids)} pairs)")
dst.mkdir(parents=True, exist_ok=True)
tmp = dst / "shard-0000.jsonl.tmp"
tmp.write_text("".join(json.dumps(recs[i]) + "\n" for i in ids))
tmp.replace(dst / "shard-0000.jsonl")
print(f"{len(ids)} pairs -> {dst / 'shard-0000.jsonl'}")
