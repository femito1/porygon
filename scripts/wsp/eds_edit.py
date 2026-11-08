"""eds step 2b/3/4: write the edit into layer 2's MLP unit u, evaluate the pre-registered checks, optionally save.

Program (one repurposed unit u of encoder layer 2's MLP):
  linear2.bias      += W2_old[:, u] · mean(hid_u)          (the old unit's mean output: = mean-ablating it)
  linear1.weight[u]  = gate direction  (eds_gate.py: K·r_c + M·r_ST + M·r_slp, in f2 = norm2(L2.mid) coordinates)
  linear1.bias[u]    = gate offset
  linear2.weight[:,u]= −s · d̂,  d = ∂score_ST/∂h (h = L2.out at the ST token, through the final LayerNorm and the
                       ActionHead MLP), averaged over the gate's target rows, made orthogonal to ∂V/∂h
Every evaluation of the edited network over the dataset is a check pass, appended to out/eds/checklog.jsonl.
Evaluation is exact: f2 and L2.out are cached (eds_cache.py), the unit only changes L2.out, so
L2.out' = L2.out + W2_old[:,u]·(m_u − hid_u) + W2_new[:,u]·GELU(f2·w + b), then final LN + heads.
    python scripts/wsp/eds_edit.py TAG GATE_TH UNIT SCALE LABEL [--save PATH] [--why "reason"]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from beh_common import NUM_OFF, load_local, ratio_boot
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import load

torch.set_num_threads(2)
tag, TH, u, scale, label = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
why = sys.argv[sys.argv.index("--why") + 1] if "--why" in sys.argv else ""
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
pol = load(tag); fe = pol.features_extractor; L2 = fe.encoder.layers[2]
G = np.load(OUT / f"gate_{tag}_c{TH}.npz")
U = np.load(OUT / f"{tag}_units.npz")
f2 = np.load(LOC / f"{tag}_f2.npy", mmap_mode="r"); o2 = np.load(LOC / f"{tag}_out2.npy", mmap_mode="r")
W1u, b1u = L2.linear1.weight[u].detach().clone(), float(L2.linear1.bias[u])
W2u = L2.linear2.weight[:, u].detach().clone(); m_u = float(U["mean"][u])
w_g, b_g = torch.as_tensor(G["w"]), float(G["b"])
MK = torch.as_tensor(d["mask"])


def heads(o, mk):
    lat = fe.norm(o); flat = lat.flatten(1)
    lg = pol.action_net(flat).masked_fill(mk == 0, torch.finfo(torch.float32).min)
    return lg, pol.value_net(pol.mlp_extractor.forward_critic(flat)).squeeze(-1)


# ---- output direction: −∂score_ST/∂h at the ST token, ⟂ ∂V/∂h, averaged over target rows (reading: gradients)
T = np.flatnonzero(H & (cnt >= TH))
o = torch.as_tensor(np.asarray(o2[T])).requires_grad_(True)
lg, v = heads(o, MK[T])
gs = torch.autograd.grad(lg[torch.arange(len(T)), 6 + torch.as_tensor(jst[T])].sum(), o)[0][torch.arange(len(T)), 13 + torch.as_tensor(jst[T])]
o = torch.as_tensor(np.asarray(o2[T])).requires_grad_(True)
lg, v = heads(o, MK[T])
gv = torch.autograd.grad(v.sum(), o)[0][torch.arange(len(T)), 13 + torch.as_tensor(jst[T])]
gs_m = Fn.normalize(gs, dim=1).mean(0); gv_m = Fn.normalize(gv, dim=1).mean(0)
gvh = gv_m / gv_m.norm()
dvec = gs_m - (gs_m @ gvh) * gvh
dhat = dvec / dvec.norm()
cons = float((Fn.normalize(gs, dim=1) @ (gs_m / gs_m.norm())).mean())
# scale rule (a priori, no outcome): SCALE < 0 means "aim for a first-order Δ score_ST of SCALE at the reference
# activation" — the gate's pre-activation for a typical counter-TH row, K·(1 − θ) (r_cond = 1, r_ST = 1)
gproj = float((gs @ dhat).mean())
hid_ref = float(Fn.gelu(torch.tensor(float(G["K"]) * (1 - float(G["theta"])))))
if scale < 0:
    scale = -scale / (hid_ref * -gproj) if gproj < 0 else -scale / (hid_ref * gproj)
W2new = (-scale * dhat).detach()

# ---- evaluate (one check pass over all rows)
t0 = time.time()
P0 = np.zeros((N, 10)); P1 = np.zeros((N, 10)); PA = np.zeros((N, 10)); V0 = np.zeros(N); V1 = np.zeros(N); HID = np.zeros(N, np.float32)
with torch.no_grad():
    for s in range(0, N, 2048):
        fb = torch.as_tensor(np.asarray(f2[s:s + 2048])); ob = torch.as_tensor(np.asarray(o2[s:s + 2048]))
        mk = MK[s:s + 2048]
        hid_old = Fn.gelu(fb @ W1u + b1u)
        hid_new = Fn.gelu(fb @ w_g + b_g)
        o_new = ob + (m_u - hid_old).unsqueeze(-1) * W2u + hid_new.unsqueeze(-1) * W2new
        l0, v0 = heads(ob, mk); l1, v1 = heads(o_new, mk)
        la, _ = heads(ob + (m_u - hid_old).unsqueeze(-1) * W2u, mk)
        PA[s:s + 2048] = la.double().softmax(-1).numpy()
        P0[s:s + 2048] = l0.double().softmax(-1).numpy(); P1[s:s + 2048] = l1.double().softmax(-1).numpy()
        V0[s:s + 2048] = v0.numpy(); V1[s:s + 2048] = v1.numpy()
        sl = np.arange(s, min(s + 2048, N))
        HID[sl] = hid_new[np.arange(len(sl)), 13 + jst[sl]].numpy()
ST = np.where(is_st & (d["mask"][:, 6:] > 0), 1.0, 0.0)
pst0 = (P0[:, 6:] * ST).sum(1); pst1 = (P1[:, 6:] * ST).sum(1)
TV = 0.5 * np.abs(P1 - P0).sum(1); dV = np.abs(V1 - V0)
# oracle Q: r2 labels, replaced by the corrected relabels where the logged rest_turns was wrong
lab = np.load("/vol/bitbucket/fr1226/porygon-ws-wsp/out/beh/derived_r2-300k.npz", allow_pickle=True)["lab_q"]
Qc = np.load(OUT / "relabel_q.npy"); fix = np.isfinite(Qc).any(1)
Qb = np.where(fix[:, None], Qc, lab)


def eq(Pm, Q):
    ok = np.isfinite(Q) & (d["mask"] > 0)
    w = np.where(ok, Pm, 0); w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    out = np.where(ok, w * np.nan_to_num(Q), 0).sum(1) - np.nanmax(np.where(ok, Q, -9), 1)
    return np.where(ok.any(1), out, np.nan)   # rows without oracle labels: excluded


txt = []


def Pr(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


def ci(m, y):
    e = ratio_boot(game[m], y[m]); return f"{e[0]:+.4f} [{e[1]:+.4f}, {e[2]:+.4f}] (n {m.sum()}, games {len(np.unique(game[m]))})"


rest = origin == "rest"
Wset = H & (cnt >= TH); Kset = H & (cnt < TH); Wk = H & rest & (slept == 2); As = H & rest & (slept >= 0) & (slept <= 1)
O = ~H
res = {}
Pr(f"== edit {label}: {tag}, gate counter >= {TH}, unit {u}, scale {scale}  ({why})")
Pr(f"output direction: cos(mean ∂score_ST/∂h, ∂V/∂h) {float(gs_m @ gvh / gs_m.norm()):+.3f}; per-row gradient consistency "
   f"(mean cos with the mean) {cons:.3f}; target rows {len(T)}; mean ∂score/∂h·d̂ {gproj:+.3f}; reference hid {hid_ref:.2f}; "
   f"|W2 column| {scale:.2f}")
Pr(f"baseline check: max |P0 - stock probs| {np.abs(P0 - np.load(LOC / f'{tag}_probs.npy')).max():.1e}")
for name, m in (("W (gate set: asleep, ST legal, counter ≥ %d)" % TH, Wset), ("K (asleep, ST legal, counter < %d)" % TH, Kset),
                ("Wk (Rest, wakes this turn)", Wk), ("As (Rest, asleep this turn)", As), ("As ∩ W", As & Wset),
                ("As ∩ K", As & Kset), ("natural sleep (other origin)", H & np.char.startswith(origin.astype(str), "other"))):
    if m.sum() == 0:
        continue
    Pr(f"{name}: P(ST) {pst0[m].mean():.3f} -> {pst1[m].mean():.3f}; ΔP {ci(m, pst1 - pst0)}; gate hid mean {HID[m].mean():.2f}")
    res[name] = dict(n=int(m.sum()), p0=float(pst0[m].mean()), p1=float(pst1[m].mean()))
for c in range(0, 8):
    m = H & (cnt == c)
    if m.sum():
        Pr(f"  counter {c}: n {m.sum()}, P(ST) {pst0[m].mean():.3f} -> {pst1[m].mean():.3f}")
Pr(f"C4 TV on O (rows outside asleep∧ST legal, n {O.sum()}): mean {TV[O].mean():.2e}, p99 {np.percentile(TV[O], 99):.2e}, max {TV[O].max():.2e}")
Pr(f"   TV from the mean-replacement of unit {u} alone is the same as above on O (the gate is ≈ 0 there): gate hid max on O "
   f"(ST token) {np.abs(HID[O & is_st.any(1)]).max() if (O & is_st.any(1)).any() else 0:.1e}")
Pr(f"C5 |ΔV| all rows: mean {dV.mean():.2e} (p99 {np.percentile(dV, 99):.2e}); on W: mean {dV[Wset].mean():.3f}, signed {np.mean((V1 - V0)[Wset]):+.3f}")
TVA = 0.5 * np.abs(PA - P0).sum(1)
Pr(f"   of which the mean-replacement of unit {u} alone: TV on O mean {TVA[O].mean():.2e}, p99 {np.percentile(TVA[O], 99):.2e}, max {TVA[O].max():.2e}")
top = np.argsort(-TV * O)[:5]
for i in top:
    Pr(f"   top-TV row {i}: TV {TV[i]:.3f} (unit replacement alone {TVA[i]:.3f}); asleep {bool(slp[i])}, ST in moveset {bool(is_st[i].any())}, "
       f"ST legal {bool(ph['st_legal'][i].any())}, gate hid at ST token {HID[i]:+.2f}; argmax {P0[i].argmax()}→{P1[i].argmax()}, max P {P0[i].max():.3f}")
m7 = slp & ~has_rest
Pr(f"C7 asleep without Rest in the moveset (n {m7.sum()}): max TV {TV[m7].max():.2e}, mean {TV[m7].mean():.2e}; "
   f"gate hid max there {np.abs(HID[m7]).max():.1e}; unit replacement alone: max TV {TVA[m7].max():.2e}")
q0, q1 = eq(P0, lab), eq(P1, lab)
qb0, qb1 = eq(P0, Qb), eq(P1, Qb)
Pr(f"C6 expected oracle Q − max Q (policy-weighted, visited legal actions):")
for name, m, a0, a1 in (("Wk, r2 labels (valid there)", Wk, q0, q1), ("Wk, best labels", Wk, qb0, qb1),
                        ("As ∩ W, corrected labels", As & Wset, qb0, qb1), ("As, corrected labels", As, qb0, qb1),
                        ("all Rest-origin asleep-ST rows, best labels", H & rest, qb0, qb1),
                        ("W, r2 labels (as briefed)", Wset, q0, q1), ("all asleep-ST rows, r2 labels", H, q0, q1)):
    ok = m & np.isfinite(a0) & np.isfinite(a1)
    Pr(f"  {name}: {np.mean(a0[ok]):+.4f} -> {np.mean(a1[ok]):+.4f}; Δ {ci(ok, a1 - a0)}")
    res["C6 " + name] = float(np.nanmean((a1 - a0)[ok]))
res.update(TV_O_mean=float(TV[O].mean()), TV_O_p99=float(np.percentile(TV[O], 99)), dV_mean=float(dV.mean()),
           C7_maxTV=float(TV[m7].max()))
with open(OUT / "checklog.jsonl", "a") as fh:
    fh.write(json.dumps(dict(time=time.strftime("%H:%M:%S"), label=label, tag=tag, TH=TH, unit=u, scale=scale, why=why,
                             gate=dict(K=float(G["K"]), M=float(G["M"]), theta=float(G["theta"])), rows=N, results=res)) + "\n")
(OUT / f"edit_{label}_{tag}.txt").write_text("\n".join(txt) + "\n")
np.savez(OUT / f"edit_{label}_{tag}.npz", pst0=pst0, pst1=pst1, TV=TV, V0=V0, V1=V1, P1=P1.astype(np.float32))

if save:
    with torch.no_grad():
        L2.linear2.bias += W2u * m_u
        L2.linear1.weight[u] = w_g
        L2.linear1.bias[u] = b_g
        L2.linear2.weight[:, u] = W2new
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    pol.save(save)
    Pr(f"saved {save}: weights changed = linear1 row {W1u.numel()} + linear1 bias 1 + linear2 column {W2u.numel()} + "
       f"linear2 bias {W2u.numel()} = {2 * W1u.numel() + 1 + W2u.numel()}")
    (OUT / f"edit_{label}_{tag}.txt").write_text("\n".join(txt) + "\n")
