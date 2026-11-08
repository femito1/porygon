"""eds edit v2: an exact input-level gate in mon_proj that "exaggerates the counter" of our own asleep Sleep Talk user.

Program (one repurposed hidden unit u of mon_proj = Linear(573→256) · GELU · Linear(256→256)):
  mon_proj[2].bias       += W2_old[:, u] · mean(hid_u)       (mean-replacement of the old unit)
  mon_proj[0].weight[u]   = Kf·(e_slp + e_active + e_mine) + Kc·e_counter + Ks·Σ_slots u_ST   (u_ST in each of the
                            4 move_proj slots: the Sleep Talk readout, eds_flag.py)
  mon_proj[0].bias[u]     = −3·Kf − Kc·τ − Ks·(1 − 4·b_ST)
  mon_proj[2].weight[:, u]= v / pre(counter 4)     v = token(counter 7) − token(counter 4), mean over target rows
so on every mon token  hid_u = GELU(pre) ≈ 0 unless (asleep ∧ active ∧ ours ∧ Sleep Talk in moveset ∧ counter > τ),
and then the own-active token moves as if its counter were ≈ 7; the network's own circuit does the rest.
Each evaluation over the dataset (full forward, all 48,518 rows) is a check pass, logged to out/eds/checklog.jsonl.
    python scripts/wsp/eds_edit2.py TAG LABEL TAU [--kf 200 --kc 100 --ks 90] [--save PATH] [--why "..."]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from beh_common import NUM_OFF, load_local, ratio_boot
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import Net, load, run_batched

torch.set_num_threads(2)
tag, label, tau = sys.argv[1], sys.argv[2], float(sys.argv[3])
arg = lambda k, dflt: type(dflt)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else dflt  # noqa: E731
Kf, Kc, Ks = arg("--kf", 200.0), arg("--kc", 100.0), arg("--ks", 90.0)
c_emul = arg("--emul", 0.7)
save = arg("--save", ""); why = arg("--why", "")
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds"); LOC = Path("/tmp/wsp-674320/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N); game = d["game"]
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
ids = mons[r, own_a, 3:7].astype(np.int64)
is_st = ids == int(np.flatnonzero(MOVES == "sleeptalk")[0]); jst = is_st.argmax(1)
has_rest = (ids == int(np.flatnonzero(MOVES == "rest")[0])).any(1)
slp, cnt, origin, slept = ph["slp"], ph["cnt"], ph["origin"], ph["slept"]
H = slp & ph["st_legal"].any(1)
FL = np.load(OUT / f"flag_{tag}.npz")
u = int(FL["unit"]); mean_u = float(FL["mean"][u]); u_st = torch.as_tensor(FL["u_st"], dtype=torch.float32); b_st = float(FL["b_st"])
pol0 = load(tag); pol = load(tag)
fe = pol.features_extractor; mp0, mp2 = fe.mon_proj[0], fe.mon_proj[2]
NOFF = 64 + 32 + 32 + 4 * 64 + 64   # numerics offset inside the mon_proj input
with torch.no_grad():
    W2old = mp2.weight[:, u].clone()
    row = torch.zeros(mp0.in_features)
    row[NOFF + 63] = Kf; row[NOFF + 115] = Kf; row[NOFF + 116] = Kf; row[NOFF + 65] = Kc
    for k in range(4):
        row[128 + 64 * k: 128 + 64 * (k + 1)] = Ks * u_st
    bias = -3 * Kf - Kc * tau - Ks * (1 - 4 * b_st)
    pre4 = Kc * (0.4 - tau)                       # pre-activation of a target mon at counter 4 (ST readout = 1)
    v = torch.as_tensor(FL["v"], dtype=torch.float32) * ((c_emul - 0.4) / 0.3)   # v was measured for counter 4 → 7
    mp2.bias += W2old * mean_u
    mp0.weight[u] = row; mp0.bias[u] = bias
    mp2.weight[:, u] = v / torch.nn.functional.gelu(torch.tensor(pre4))
n_changed = mp0.in_features + 1 + 2 * mp2.out_features
# ---- one check pass: full forward of stock and edited nets over all rows
t0 = time.time()
net0, net1 = Net(pol0), Net(pol)
o0 = run_batched(net0, d["obs"], d["mask"])
o1 = run_batched(net1, d["obs"], d["mask"])
P0, P1, V0, V1 = o0["probs"].astype(np.float64), o1["probs"].astype(np.float64), o0["value"], o1["value"]
# gate activation on every mon token (to verify it is off elsewhere)
G = np.zeros((N, 12), np.float32)
with torch.no_grad():
    for s in range(0, N, 4096):
        c = {}
        net1.embed(torch.as_tensor(d["obs"][s:s + 4096], dtype=torch.float32), None, c)
        G[s:s + 4096] = (c["mon_in"] @ mp0.weight[u] + mp0.bias[u]).numpy()
ST = np.where(is_st & (d["mask"][:, 6:] > 0), 1.0, 0.0)
pst0 = (P0[:, 6:] * ST).sum(1); pst1 = (P1[:, 6:] * ST).sum(1)
TV = 0.5 * np.abs(P1 - P0).sum(1); dV = np.abs(V1 - V0)
lab = np.load("/vol/bitbucket/fr1226/porygon-ws-wsp/out/beh/derived_r2-300k.npz", allow_pickle=True)["lab_q"]
Qc = np.load(OUT / "relabel_q.npy"); fix = np.isfinite(Qc).any(1); Qb = np.where(fix[:, None], Qc, lab)


def eq(Pm, Q):
    ok = np.isfinite(Q) & (d["mask"] > 0)
    w = np.where(ok, Pm, 0); w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    out = np.where(ok, w * np.nan_to_num(Q), 0).sum(1) - np.nanmax(np.where(ok, Q, -9), 1)
    return np.where(ok.any(1), out, np.nan)


txt = []


def Pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


def ci(m, y):
    m = m & np.isfinite(y)
    e = ratio_boot(game[m], y[m]); return f"{e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}] (n {m.sum()}, games {len(np.unique(game[m]))})"


rest = origin == "rest"
TH = int(round(tau * 10 + 0.5))
Wset = H & (cnt >= TH); Kset = H & (cnt < TH); Wk = H & rest & (slept == 2); As = H & rest & (slept >= 0) & (slept <= 1)
O = ~H; res = {}
Pr(f"== edit {label}: {tag}; mon_proj unit {u}; τ {tau} (gate counter ≥ {TH}); Kf {Kf} Kc {Kc} Ks {Ks}; emulated counter "
   f"{c_emul * 10:.0f}; weights changed {n_changed}; pass took {time.time() - t0:.0f}s  ({why})")
on = G > 0
own_tok = np.zeros((N, 12), bool); own_tok[r, own_a] = True
Pr(f"gate on (pre > 0): {on.sum()} mon tokens; all are the own active: {bool((on <= own_tok).all())}; rows with gate on {on.any(1).sum()}, "
   f"of which in W {np.sum(on.any(1) & Wset)}; gate on outside W: {np.sum(on.any(1) & ~Wset)} rows; max pre outside W {G[~Wset].max():+.2f}")
for name, m in ((f"C1 W (asleep, ST legal, counter ≥ {TH})", Wset), (f"C2 K (asleep, ST legal, counter < {TH})", Kset),
                ("C3 Wk (Rest, wakes this turn)", Wk), ("    As (Rest, asleep this turn)", As), ("    As ∩ W", As & Wset),
                ("C3 As ∩ K", As & Kset), ("    natural sleep (other origin)", H & np.char.startswith(origin.astype(str), "other"))):
    if m.sum():
        Pr(f"{name}: P(ST) {pst0[m].mean():.3f} -> {pst1[m].mean():.3f}; ΔP {ci(m, pst1 - pst0)}")
        res[name.strip()] = dict(n=int(m.sum()), p0=float(pst0[m].mean()), p1=float(pst1[m].mean()))
for c in range(0, 8):
    m = H & (cnt == c)
    if m.sum():
        Pr(f"    counter {c}: n {m.sum()}, P(ST) {pst0[m].mean():.3f} -> {pst1[m].mean():.3f}")
Pr(f"C4 TV on O (outside asleep∧ST legal, n {O.sum()}): mean {TV[O].mean():.2e}, p99 {np.percentile(TV[O], 99):.2e}, max {TV[O].max():.2e}")
Pr(f"   TV on W: mean {TV[Wset].mean():.3f} (P(ST) mass moved to: switch {np.mean((P1 - P0)[Wset, :6].sum(1)):+.3f}, other moves "
   f"{np.mean(((P1 - P0)[Wset, 6:] * (1 - ST[Wset])).sum(1)):+.3f})")
Pr(f"C5 |ΔV| all rows: mean {dV.mean():.2e} (p99 {np.percentile(dV, 99):.2e}); on W: mean {dV[Wset].mean():.3f}, signed {np.mean((V1 - V0)[Wset]):+.3f}")
m7 = slp & ~has_rest
Pr(f"C7 asleep without Rest in the moveset (n {m7.sum()}): max TV {TV[m7].max():.2e}, mean {TV[m7].mean():.2e}")
m8 = slp & has_rest & ~is_st.any(1)
Pr(f"    asleep, Rest but no Sleep Talk (n {m8.sum()}): max TV {TV[m8].max():.2e}")
q0, q1, qb0, qb1 = eq(P0, lab), eq(P1, lab), eq(P0, Qb), eq(P1, Qb)
Pr("C6 expected oracle Q − max Q (policy-weighted over visited legal actions):")
for name, m, a0, a1 in (("Wk, r2 labels (valid there)", Wk, q0, q1), ("As ∩ W, corrected labels", As & Wset, qb0, qb1),
                        ("As, corrected labels", As, qb0, qb1), ("all Rest-origin asleep-ST rows, best labels", H & rest, qb0, qb1),
                        ("all asleep-ST rows, best labels", H, qb0, qb1), ("all asleep-ST rows, r2 labels (as briefed)", H, q0, q1)):
    ok = m & np.isfinite(a0) & np.isfinite(a1)
    Pr(f"  {name}: {np.mean(a0[ok]):+.4f} -> {np.mean(a1[ok]):+.4f}; Δ {ci(ok, a1 - a0)}")
    res["C6 " + name] = float(np.mean((a1 - a0)[ok]))
res.update(TV_O_mean=float(TV[O].mean()), TV_O_p99=float(np.percentile(TV[O], 99)), dV_mean=float(dV.mean()), C7_maxTV=float(TV[m7].max()))
with open(OUT / "checklog.jsonl", "a") as fh:
    fh.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), label=label, tag=tag, kind="check pass (full forward, all rows)",
                             tau=tau, Kf=Kf, Kc=Kc, Ks=Ks, emul=c_emul, unit=u, why=why, results=res)) + "\n")
np.savez(OUT / f"edit2_{label}_{tag}.npz", pst0=pst0, pst1=pst1, TV=TV, V0=V0, V1=V1, P1=P1.astype(np.float32))
if save:
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    pol.save(save)
    Pr(f"saved {save}")
(OUT / f"edit2_{label}_{tag}.txt").write_text("\n".join(txt) + "\n")
