"""eds: corrected oracle labels for Rest sleepers.

The logged runs (309116/309120, 2026-11-03) converted a Rest sleeper with rest_turns = max(1, 3 - poke-env counter);
poke-env's counter adds 3 per Sleep Talk turn, so after one Sleep Talk the search and the logged oracle state
believed the mon wakes next turn (fixed in porygon.search 006ff76, 2026-11-07). This relabels the asleep, Sleep-Talk-
legal, Rest-origin rows whose logged rest_turns differs from the truth (3 - attempts so far, from the replay; see
eds_phase.py), with the same oracle as r2-300k (MCTS 300k iterations, eval_r1), plus control rows relabeled
unmodified (reproducibility vs r2).
    python scripts/wsp/eds_relabel.py WORKER NWORKERS [--limit K]  -> out/eds/relabel-<worker>.jsonl
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/vol/bitbucket/fr1226/porygon-ws-wsp/scripts")
from oracle_labels import label  # noqa: E402
from porygon.search import DEFAULT_EVAL  # noqa: E402

W, NW = int(sys.argv[1]), int(sys.argv[2])
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds")
ph = np.load(OUT / "phase.npz", allow_pickle=True)
H = ph["slp"] & ph["st_legal"].any(1) & (ph["origin"] == "rest") & (ph["slept"] >= 0)
cand = np.flatnonzero(H)
want = {}
with open("/vol/bitbucket/fr1226/porygon-ws-wsp/data/oracle.jsonl") as fh:
    for i, line in enumerate(fh):
        if H[i]:
            want[i] = json.loads(line)
jobs = []
ctrl = 0
for i in cand:
    o = want.get(int(i))
    if not o or not o.get("state"):
        continue
    ch = o["state"].split("=")
    act = int(ch[6])
    mon = ch[act].split(",")
    if mon[18] != "SLEEP":
        continue
    truth = 3 - int(ph["slept"][i])
    if int(mon[19]) != truth:
        mon[19] = str(truth)
        ch[act] = ",".join(mon)
        jobs.append((int(i), "corrected", int(o and 0) or truth, "=".join(ch), o))
    elif ctrl < 24 and ph["slept"][i] == 2:
        ctrl += 1
        jobs.append((int(i), "control", truth, o["state"], o))
jobs = jobs[W::NW][:limit]
cfg = json.loads(Path(DEFAULT_EVAL).read_text())
out = OUT / f"relabel-{W}.jsonl"
done = {json.loads(l)["row"] for l in open(out)} if out.exists() else set()
with open(out, "a") as sink:
    for i, kind, truth, state, o in jobs:
        if i in done:
            continue
        t0 = time.time()
        rec = dict(o, state=state)
        try:
            lab = label({"oracle": rec}, 300_000, cfg)
        except BaseException as e:  # engine panics
            lab = {"error": f"{type(e).__name__}: {e}"[:200]}
        sink.write(json.dumps({"row": i, "kind": kind, "rest_turns": truth, "sec": round(time.time() - t0, 1)} | (lab or {})) + "\n")
        sink.flush()
print("done", len(jobs))
