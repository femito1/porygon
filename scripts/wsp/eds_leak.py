"""eds: where the L2 gate leaks — non-Sleep-Talk tokens with gate pre-activation > 0 (move names, r_ST / r_cond).
    python scripts/wsp/eds_leak.py 500M 4
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB

tag, TH = sys.argv[1], int(sys.argv[2])
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds"); LOC = Path("/tmp/wsp-674320/eds")
d = load_local(); N = len(d["mask"]); r = np.arange(N)
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
names = MOVES[mons[r, own_a, 3:7].astype(np.int64)]
G = np.load(OUT / f"gate_{tag}_c{TH}.npz"); f2 = np.load(LOC / f"{tag}_f2.npy", mmap_mode="r")
hits = []
for s in range(0, N, 4096):
    F = np.asarray(f2[s:s + 4096])
    pre = F @ G["w"] + G["b"]; rs = F @ G["W_s"] + G["b_s"]; rc = F @ G["W_c"] + G["b_c"]
    for i, t in zip(*np.nonzero(pre > 0)):
        nm = names[s + i, t - 13] if t >= 13 else f"token{t}"
        if nm != "sleeptalk":
            hits.append((s + i, t, nm, float(pre[i, t]), float(rs[i, t]), float(rc[i, t]), bool(mons[s + i, own_a[s + i], NUM_OFF + 63] > .5)))
print(f"non-ST tokens with pre > 0: {len(hits)} in {len(set(h[0] for h in hits))} rows")
print("by name:", Counter(h[2] for h in hits).most_common(15))
print("asleep fraction:", np.mean([h[6] for h in hits]) if hits else 0)
for h in sorted(hits, key=lambda h: -h[3])[:12]:
    print(f"  row {h[0]} tok {h[1]} {h[2]}: pre {h[3]:+.2f} r_ST {h[4]:+.2f} r_cond {h[5]:+.2f} asleep {h[6]}")

# ---- design grid on the cached readouts (no heads): leak vs target activation as M grows
from beh_common import NUM_OFF as _N  # noqa: E402,F401
ph = np.load(OUT / "phase.npz", allow_pickle=True)
is_st = names == "sleeptalk"; jst = is_st.argmax(1); has = is_st.any(1)
H = ph["slp"] & ph["st_legal"].any(1); cnt = ph["cnt"]
RS = np.zeros((N, 17), np.float32); RC = np.zeros((N, 17), np.float32)
for s in range(0, N, 4096):
    F = np.asarray(f2[s:s + 4096]); RS[s:s + 4096] = F @ G["W_s"] + G["b_s"]; RC[s:s + 4096] = F @ G["W_c"] + G["b_c"]
stt = np.zeros((N, 17), bool); stt[np.flatnonzero(has), 13 + jst[has]] = True
gelu = lambda x: 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))  # noqa: E731
K, th = float(G["K"]), float(G["theta"])
print("\nM | non-ST tokens pre>0 | pre>-3 | rows | W: mean hid at ST token | K rows (counter<TH): mean hid | ST awake: max pre")
for M in (12, 16, 20, 24, 30, 40):
    pre = K * (RC - th) + M * (RS - 1)
    off = ~stt
    W = H & (cnt >= TH); Kr = H & (cnt < TH)
    print(f"{M} | {np.sum(off & (pre > 0))} | {np.sum(off & (pre > -3))} | {np.sum((off & (pre > 0)).any(1))} | "
          f"{gelu(pre[W, 13 + jst[W]]).mean():.2f} | {gelu(pre[Kr, 13 + jst[Kr]]).mean():.3f} | {pre[has & ~ph['slp'], 13 + jst[has & ~ph['slp']]].max():+.2f}")
cs = float(np.median(RS[H, 13 + jst[H]]))
print(f"\nre-centred ST term: M·(r_ST − {cs:.3f}) (median r_ST over asleep ST tokens)")
print("M | non-ST tokens pre>0 | pre>-3 | rows | W: mean hid | K rows: mean hid | ST awake: max pre | W rows with hid<0.5")
for M in (12, 20, 30, 40):
    pre = K * (RC - th) + M * (RS - cs)
    off = ~stt; W = H & (cnt >= TH); Kr = H & (cnt < TH)
    hw = gelu(pre[W, 13 + jst[W]])
    print(f"{M} | {np.sum(off & (pre > 0))} | {np.sum(off & (pre > -3))} | {np.sum((off & (pre > 0)).any(1))} | {hw.mean():.2f} | "
          f"{gelu(pre[Kr, 13 + jst[Kr]]).mean():.3f} | {pre[has & ~ph['slp'], 13 + jst[has & ~ph['slp']]].max():+.2f} | {np.mean(hw < 0.5):.3f}")
