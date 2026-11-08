"""eds: debug the replay phase parse + read rest_turns / sleep_turns in the logged oracle states (asleep ST rows)."""
import json, re
from collections import Counter
from pathlib import Path
import numpy as np
from beh_common import GLORIA, load_local

OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
H = ph["slp"] & ph["st_legal"].any(1)
rows = np.flatnonzero(H)
orc = {}
want = set(rows.tolist())
with open("/vol/bitbucket/fr1226/porygon-ws-wsp/data/oracle.jsonl") as fh:
    for i, line in enumerate(fh):
        if i in want:
            orc[i] = json.loads(line)
ex = None
tab = Counter()
for i in rows:
    o = orc.get(i)
    if not o:
        tab[("none",)] += 1; continue
    s = o["state"]
    if ex is None:
        ex = s[:1500]
    mons = [m.split(",") for m in s.split("=") if len(m.split(",")) > 21][:6]  # side one = our side
    sl = [m for m in mons if m[18].lower() == "sleep"]
    rt = tuple((m[19], m[20]) for m in sl)
    tab[(str(ph["origin"][i])[:5], int(ph["slept"][i]), int(ph["cnt"][i]), rt)] += 1

for k, v in sorted(tab.items(), key=lambda kv: str(kv[0])):
    print(k, v)
# examples of odd rows
odd = rows[(ph["slept"][rows] == 0) & (ph["cnt"][rows] >= 4)][:3]
for i in odd:
    gm = d["games"][int(d["game"][i])]; t = int(d["turn"][i])
    lines = Path(GLORIA, "runs", gm["run"], "replays", gm["tag"] + ".log").read_text().splitlines()
    print("\n== row", i, gm["tag"], gm["run"][-6:], "turn", t, "player", gm["player"], "cnt", ph["cnt"][i], "woke", ph["woke"][i])
    cur = 0
    for ln in lines:
        if ln.startswith("|turn|"):
            cur = int(ln.split("|")[2])
        if t - 4 <= cur <= t and re.search(r"\|(move|cant|-status|-curestatus|switch|drag|turn|faint)\|", ln):
            print("  ", ln[:110])
