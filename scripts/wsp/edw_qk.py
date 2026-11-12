"""edw design v3 (source edit): scale the opponent-type × candidate-type interaction where it is computed — the
layer-1 attention keys (edw_target2 §4: L1 h4's QK type map is the gen4 type chart at r −0.79; h0/h7 +0.42/+0.67).

Program: in layer 1, for the heads H whose QK type map tracks the chart (|r| ≥ 0.4: h0, h2, h3, h4, h7), make the KEY
partially blind to the opponent's types:   W_K,h ← W_K,h (I − (1−λ) T_oᵀ R_oᵀ)
where T_o [18,256] are the opponent-active token's type directions at the L1 attention input (regressed on type flags
+ base stats + HP, train rows) and R_o [256,18] reads the type flags back (least squares), so T_oᵀR_oᵀa is the
type-explained part of a key input a. Queries (the candidate's own types → its static vulnerability) are untouched.
The swap measures exactly the dynamic part (the candidate's types do not change), so λ is calibrated on the TRAIN
swaps: one check at λ = 0.5, linear interpolation to the target ratio ρ* of the swap dfn coefficient
(dyn dfn target / net, edw_decomp: voluntary 1.174/1.853 = 0.63, forced 1.245/1.652 = 0.75, weighted 0.65).
Strengths qk1/qk2/qk3: λ_k = 1 − (k/3)(1 − λ*). For each: save the .pt, cache final latents (all rows, both swap
halves) for edw_eval.py.
    python scripts/wsp/edw_qk.py 500M
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from beh_common import BASE, HP, NUM_OFF, TYP, load_local
from edw_common import FIN_TOK, LOCAL, OUT, log_check, matchup, split
from porygon.encoder import FIELD_DIM, MON_DIM
from porygon.model import load_policy
from wsp_common import CKPTS, Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
HEADS = [0, 2, 3, 4, 7]
RHO = 0.65
d = load_local()
N = len(d["mask"]); mask = d["mask"] > 0
tr = split(d["game"])
f = matchup(d["obs"])
txt, res = [], {"heads": HEADS, "rho_target": RHO}


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


base = load(CKPTS.get(tag, tag))
net0 = Net(base)
# ---- type directions at the L1 attention input, opponent-active token
rr = np.flatnonzero(tr & f["has_opp"])
rr = np.random.default_rng(3).choice(rr, 6000, replace=False)
oa = f["opp_a"][rr]
with torch.no_grad():
    c = {}
    A = []
    for s in range(0, len(rr), 1000):
        c = {}
        net0.forward(torch.as_tensor(d["obs"][rr[s:s + 1000]].astype(np.float32)), torch.as_tensor(d["mask"][rr[s:s + 1000]]), None, c)
        A.append(net0.layers[1].norm1(c["L0.out"][np.arange(len(c["L0.out"])), 1 + oa[s:s + 1000]]).double().numpy())
A = np.concatenate(A)
nums = d["obs"][rr, FIELD_DIM:].reshape(len(rr), 12, MON_DIM)[np.arange(len(rr)), oa, NUM_OFF:NUM_OFF + 125].astype(np.float64)
T = nums[:, TYP:TYP + 18]
X = np.c_[T, nums[:, BASE:BASE + 6], nums[:, HP], np.ones(len(rr))]
To = np.linalg.lstsq(X, A, rcond=None)[0][:18]                     # [18,256]
Ac = A - A.mean(0)
G = Ac.T @ Ac
Ro = np.linalg.solve(G + 1e-3 * np.trace(G) / 256 * np.eye(256), Ac.T @ (T - T.mean(0)))   # [256,18]
pred = Ac @ Ro + T.mean(0)
present = T.sum(0) > 30
acc = np.mean((pred[:, present] > 0.5) == (T[:, present] > 0.5))
Pt = To.T @ Ro.T                                                   # [256,256]: a -> type-explained part
P(f"opp-active type directions at L1 input: {present.sum()} types present; type flags read back from a with accuracy {acc:.4f}; "
  f"‖type part‖/‖a − mean‖ = {np.linalg.norm(Ac @ Pt.T) / np.linalg.norm(Ac):.3f}")


def edited(lam):
    p = load(CKPTS.get(tag, tag))
    L = p.features_extractor.encoder.layers[1].self_attn
    with torch.no_grad():
        Wk = L.in_proj_weight[256:512].double()
        E = torch.eye(256, dtype=torch.float64) - (1 - lam) * torch.as_tensor(Pt)
        for h in HEADS:
            sl = slice(32 * h, 32 * h + 32)
            L.in_proj_weight[256 + 32 * h:256 + 32 * h + 32] = (Wk[sl] @ E).to(L.in_proj_weight.dtype)
    return p


def fins(p, X, MK, bs=1000):
    n = Net(p); F_, V_ = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            c = {}
            _, v = n.forward(torch.as_tensor(X[s:s + bs], dtype=torch.float32), torch.as_tensor(MK[s:s + bs]), None, c)
            F_.append(c["final"][:, FIN_TOK].half().numpy()); V_.append(v.numpy())
    return np.concatenate(F_), np.concatenate(V_)


def swap_coef(p, half, heads_only=False):
    z = np.load(LOCAL / f"{tag}_swap_{half}.npz")
    from edw_common import head_logits, head_weights
    W = head_weights(p)
    MK = d["mask"][z["rows"]]
    f0_, v0_ = fins(p, z["obs0"].astype(np.float32), MK); ff_, vf_ = fins(p, z["obsf"].astype(np.float32), MK)
    l0, l1 = head_logits(f0_, W), head_logits(ff_, W)
    leg = MK[:, :6] > 0
    a0, a1 = matchup(z["obs0"].astype(np.float32)), matchup(z["obsf"].astype(np.float32))
    cen = lambda a: (lambda b: b - np.nanmean(b, 1, keepdims=True))(np.where(leg, a, np.nan))  # noqa: E731
    dl = cen(l1[:, :6] - l0[:, :6]); xd = cen(a1["dfn"] - a0["dfn"]); xo = cen(a1["off"] - a0["off"])
    ok = np.isfinite(dl)
    return np.linalg.lstsq(np.c_[xd[ok], xo[ok]], dl[ok], rcond=None)[0], (f0_, ff_)


c1, _ = swap_coef(base, "train")
p05 = edited(0.5)
c05, _ = swap_coef(p05, "train")
log_check("qk calibration λ=0.5: train swaps forward", 8000, "TRAIN swaps", coef=c05.tolist())
r05 = c05[0] / c1[0]
lam_star = float(np.clip(1 - (1 - RHO) * 0.5 / max(1 - r05, 1e-6), 0.0, 1.0))
P(f"train swaps: original dfn {c1[0]:+.3f}, off {c1[1]:+.3f}; λ=0.5 → dfn {c05[0]:+.3f}, off {c05[1]:+.3f} (ratio {r05:.3f}); "
  f"target ratio {RHO} → λ* = {lam_star:.3f}")
res.update(orig=c1.tolist(), lam05=c05.tolist(), lam_star=lam_star)
out_dir = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/ckpt/edits")
obs_all = d["obs"]
_, v_orig = fins(base, obs_all[:2000].astype(np.float32), d["mask"][:2000])
for k in (1, 2, 3):
    lam = 1 - k / 3 * (1 - lam_star)
    nm = f"qk{k}"
    p = edited(lam)
    path = out_dir / f"{tag}_switch_{nm}.pt"
    p.save(str(path))
    q = load_policy(path)
    fa, va = fins(q, obs_all, d["mask"])
    np.savez(LOCAL / f"{tag}_full_{nm}.npz", fin=fa, value=va)
    for half in ("train", "test"):
        cc, (f0_, ff_) = swap_coef(q, half)
        np.savez(LOCAL / f"{tag}_swap_{half}_{nm}.npz", fin_0=f0_, fin_f=ff_)
        P(f"{nm} (λ {lam:.3f}) {half} swaps: dfn {cc[0]:+.3f}, off {cc[1]:+.3f}")
        res.setdefault(nm, {})[half] = cc.tolist()
    sd0, sd1 = base.state_dict(), q.state_dict()
    nch = sum(int((sd1[k_] != sd0[k_]).sum()) for k_ in sd1 if not k_.startswith(("pi_features", "vf_features")))
    res[nm].update(lam=lam, weights_changed=nch, path=str(path))
    P(f"{nm}: saved {path}; weights changed {nch}")
    log_check(f"qk build {nm}: forward all rows + both swap halves", N + 16000, "all rows (cache for eval)", lam=lam)
json.dump(res, open(OUT / f"qk_{tag}.json", "w"), indent=1, default=float)
(OUT / f"qk_{tag}.txt").write_text("\n".join(txt) + "\n")
