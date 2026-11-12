"""beh case study: Sleep Talk while about to wake (a narrow, precise failure mode).
Observational: net P(sleeptalk) vs search share by the sleep counter, oracle Q gaps (r2-300k), Rest vs other sleep.
Causal: set the active's status counter to each value 0..0.6 on asleep positions with Sleep Talk legal and read
the net's P(sleeptalk); also which components carry the counter (patch the counter-changed token stream).
    python scripts/wsp/beh_sleep.py 500M -> out/beh/sleep_<tag>.txt
"""
import sys

import numpy as np
import torch

from beh_common import OUT, NUM_OFF, STATUS, decode, features, load_local, ratio_boot
from wsp_common import Intervention, Net, load
from beh_common import FIELD_DIM, MON_DIM

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
d = load_local(); f = features(d)
der = np.load(OUT / "derived_r2-300k.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N)
nums = decode(d["obs"])["nums"]
A = nums[r, f["own_a"]]
slp = A[:, STATUS + 4] > .5
cnt = np.round(A[:, 65] * 10).astype(int)
st = (f["move_names"] == "sleeptalk") & (d["mask"][:, 6:] > 0)
has = st.any(1) & slp
rest = (f["move_names"] == "rest").any(1)
lq = der["lab_q"]; qmax = np.nanmax(np.where(np.isfinite(lq), lq, -1), 1)
j = st.argmax(1)
q_st = lq[r, 6 + j] - qmax
pn = (d["probs"][:, 6:] * st).sum(1); ps = (d["shares"][:, 6:] * st).sum(1)
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


P(f"asleep with Sleep Talk legal: {has.sum()} decisions in {len(np.unique(d['game'][has]))} games; has Rest {np.mean(rest[has]):.2f}")
P("sleep counter (turns asleep) | n | net P(sleeptalk) | search share | oracle Q(sleeptalk) - max Q | net expected Q loss from sleeptalk mass")
for c in range(0, 7):
    s = has & (cnt == c)
    if s.sum() < 5:
        continue
    e = ratio_boot(d["game"][s], q_st[s])
    P(f"{c} | {s.sum()} | {pn[s].mean():.3f} | {ps[s].mean():.3f} | {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}] | {np.mean(pn[s]*q_st[s]):+.4f}")
for name, m in (("with Rest in moveset", rest), ("without Rest", ~rest)):
    s = has & m & (cnt >= 2)
    P(f"counter>=2, {name}: n {s.sum()}, net {pn[s].mean():.3f}, search {ps[s].mean():.3f}, Q gap {np.nanmean(q_st[s]):+.3f}")

# causal: counter sweep
net = Net(load(tag))
rows = np.flatnonzero(has)
X = d["obs"][rows].astype(np.float32); MK = d["mask"][rows]
n = len(rows); ar = np.arange(n)
oa = f["own_a"][rows]
jj = j[rows]


def run(X, iv=None):
    with torch.no_grad():
        lg, v = net.forward(torch.as_tensor(X), torch.as_tensor(MK), iv)
    p = lg.double().softmax(-1).numpy()
    return p, lg.double().numpy(), v.numpy()


P("\ncausal sweep: set the active's sleep counter to c on all these positions")
P("c | net P(sleeptalk) | Δ logit(sleeptalk) vs c=0 | Δ mean logit(other legal moves)")
base_p, base_lg, _ = run(X)
for c in range(0, 8):
    Xc = X.copy(); m = Xc[:, FIELD_DIM:].reshape(n, 12, MON_DIM)
    m[ar, oa, NUM_OFF + 65] = c / 10
    p, lg, _ = run(Xc)
    if c == 0:
        lg0 = lg
    oth = (MK[:, 6:] > 0) & ~st[rows]
    dl_o = np.nanmean(np.where(oth, lg[:, 6:] - lg0[:, 6:], np.nan), 1)
    P(f"{c} | {p[ar, 6 + jj].mean():.3f} | {np.mean(lg[ar, 6+jj] - lg0[ar, 6+jj]):+.3f} | {np.nanmean(dl_o):+.3f}")
# where does the counter enter: patch token inputs/residuals from counter=4 into counter=0 runs
X0 = X.copy(); m0 = X0[:, FIELD_DIM:].reshape(n, 12, MON_DIM); m0[ar, oa, NUM_OFF + 65] = 0.0
X4 = X.copy(); m4 = X4[:, FIELD_DIM:].reshape(n, 12, MON_DIM); m4[ar, oa, NUM_OFF + 65] = 0.4
keep = ("L-1.out", "L0.out", "L1.out", "L2.out")
with torch.no_grad():
    c0, c4 = {}, {}
    l0, _ = net.forward(torch.as_tensor(X0), torch.as_tensor(MK), None, c0)
    l4, _ = net.forward(torch.as_tensor(X4), torch.as_tensor(MK), None, c4)
D = (l4 - l0).double().numpy()[ar, 6 + jj]
P(f"\nactivation patching counter 0.4 -> 0.0 runs; total Δ logit(sleeptalk) {D.mean():+.3f}; recovered fraction by (layer, token)")
for li, key in zip((-1, 0, 1, 2), keep):
    row = []
    for tname, toks in (("own active", None), ("move tokens", [13, 14, 15, 16]), ("field", [0]), ("others", "rest")):
        delta = torch.zeros_like(c0[key])
        if toks is None:
            delta[torch.arange(n), torch.as_tensor(1 + oa)] = (c4[key] - c0[key])[torch.arange(n), torch.as_tensor(1 + oa)]
        elif toks == "rest":
            msk = torch.ones(n, 17, dtype=torch.bool); msk[:, 0] = False; msk[:, 13:] = False
            msk[torch.arange(n), torch.as_tensor(1 + oa)] = False
            delta[msk] = (c4[key] - c0[key])[msk]
        else:
            delta[:, toks] = (c4[key] - c0[key])[:, toks]
        with torch.no_grad():
            lp, _ = net.forward(torch.as_tensor(X0), torch.as_tensor(MK), Intervention(resid_add={(li, None): delta}))
        rec = (lp - l0).double().numpy()[ar, 6 + jj]
        row.append(f"{tname} {rec.mean() / D.mean():.2f}")
    P(f"{key}: " + ", ".join(row))
(OUT / f"sleep_{tag}.txt").write_text("\n".join(txt) + "\n")
