"""eds: what the corrected oracle labels say (relabel-*.jsonl vs r2-300k), by phase and counter.
    python scripts/wsp/eds_relabel_an.py -> out/eds/relabel.txt, relabel_q.npy (row-aligned corrected Q, NaN elsewhere)
"""
import json
from pathlib import Path

import numpy as np

from beh_common import load_local, ratio_boot

OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N)
lab = np.load("/vol/bitbucket/fr1226/porygon-ws-wsp/out/beh/derived_r2-300k.npz", allow_pickle=True)["lab_q"]
Qc = np.full((N, 10), np.nan); kind = np.full(N, "", dtype=object); err = 0
for f in sorted(OUT.glob("relabel-*.jsonl")):
    for line in open(f):
        x = json.loads(line)
        if "q" not in x:
            err += 1; continue
        Qc[x["row"]] = [np.nan if q is None else q for q in x["q"]]; kind[x["row"]] = x["kind"]
np.save(OUT / "relabel_q.npy", Qc)
st = ph["st_legal"]; j = st.argmax(1)
txt = []


def P(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); txt.append(s)


def gap(Q):
    qm = np.nanmax(np.where(np.isfinite(Q), Q, -9), 1)
    return Q[r, 6 + j] - qm


g_r2, g_c = gap(lab), gap(Qc)
ctrl = kind == "control"; cor = kind == "corrected"
P(f"relabeled rows: corrected {cor.sum()}, control {ctrl.sum()}, errors {err}")
dq = (Qc - lab)[ctrl]
P(f"control (unmodified state, relabeled): Q(ST)-maxQ r2 {np.nanmean(g_r2[ctrl]):+.3f} vs relabel {np.nanmean(g_c[ctrl]):+.3f}; "
  f"mean |ΔQ| over visited actions {np.nanmean(np.abs(dq)):.4f}")
P("\nRest sleeper, ST legal | n | Q(ST)−maxQ r2 (logged, buggy rest_turns) | corrected rest_turns | share ST corrected oracle")
for s in (0, 1):
    for name, m in (("all counters", np.ones(N, bool)), ("counter ≤ 3", ph["cnt"] <= 3), ("counter ≥ 4", ph["cnt"] >= 4)):
        q = cor & (ph["slept"] == s) & m
        if q.sum() == 0:
            continue
        a, b = ratio_boot(d["game"][q], g_r2[q]), ratio_boot(d["game"][q], g_c[q])
        P(f"slept {s}, {name} | {q.sum()} | {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}] | {b[0]:+.3f} [{b[1]:+.3f},{b[2]:+.3f}]")
(OUT / "relabel.txt").write_text("\n".join(txt) + "\n")
