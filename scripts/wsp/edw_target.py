"""edw step 2: what SHOULD the switch logit depend on? TRAIN games only (reading: regressions on logged data).

(a) Observational regressions of the log share of each legal switch candidate relative to the best legal move
    (voluntary decisions), for the search (log(share+ε)) and for the net (raw logits, and log(p+ε) with the same ε),
    on the candidate's matchup features (dfn = incoming STAB threat, off = best own effectiveness, spd, hp) and the
    active mon's (dfn_act, off_act). Forced switches: the same within the candidates (centred).
(b) The additive switch-logit correction that best matches the search's shares: conditional logit
    θ(a) = τ·logit_net(a) + [a is a switch]·(β·φ(a) + β0), fit by maximum likelihood to the search's shares
    (τ = 1 fixed: exactly the edit's form; τ free: the part not explained by temperature), with φ the TRUE features
    and with φ̂ the final-latent probe readouts the edit can implement.
    python scripts/wsp/edw_target.py 500M
"""
import json
import sys

import numpy as np
import torch

from beh_common import load_local
from edw_common import LOCAL, OUT, matchup, split

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
d = load_local()
N = len(d["mask"]); mask = d["mask"] > 0
tr = split(d["game"])
f = matchup(d["obs"])
full = np.load(LOCAL / f"{tag}_full.npz")
lg = np.where(mask, full["logits"], -np.inf)
p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
sh = d["shares"].astype(np.float64)
pr = np.load(OUT / f"probes_{tag}.npz")
fin = full["fin"]
txt, res = [], {}


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


hat = {}
for k in ("dfn", "off", "spd", "hp"):
    hat[k] = fin[:, 1:7].astype(np.float32) @ pr[f"fin:{k}:w"].astype(np.float32) + float(pr[f"fin:{k}:b"])

legsw, legmv = mask[:, :6], mask[:, 6:]
vol = legsw.any(1) & legmv.any(1) & f["has_opp"] & f["has_act"]
forced = legsw.any(1) & ~legmv.any(1) & f["has_opp"]
P(f"# edw target ({tag}); TRAIN games {len(np.unique(d['game'][tr]))}, rows {tr.sum()}; voluntary {(vol & tr).sum()}, "
  f"forced (opp active present) {(forced & tr).sum()}")
z = sh[vol & tr][mask[vol & tr]]
P(f"search shares on legal actions: zero {np.mean(z == 0):.3f}, < 0.01 {np.mean(z < 0.01):.3f}; net p < 0.01 "
  f"{np.mean(p[vol & tr][mask[vol & tr]] < 0.01):.3f}")

# ------------------------------------------------------------------ (a) OLS
CAND = ["dfn", "off", "spd", "hp"]
ROW = ["dfn_act", "off_act"]


def ols(y, X, g, nb=200):
    A = np.c_[X, np.ones(len(X))]
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    ug, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(0); bs = []
    idx_by = [np.flatnonzero(inv == k) for k in range(len(ug))]
    for _ in range(nb):
        ii = np.concatenate([idx_by[k] for k in rng.integers(0, len(ug), len(ug))])
        bs.append(np.linalg.lstsq(A[ii], y[ii], rcond=None)[0])
    bs = np.array(bs)
    return c, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


rows = np.flatnonzero(vol & tr)
ri, si = np.nonzero(legsw[rows]); ri = rows[ri]
X = np.c_[[f[k][ri, si] for k in CAND] + [f[k][ri] for k in ROW]].T
g = d["game"][ri]
bm_net = np.where(legmv[ri], lg[ri, 6:], -np.inf).max(1)
ys = {"net logit": lg[ri, si] - bm_net}
for eps in (1e-2, 1e-3):
    ys[f"net log(p+{eps:g})"] = np.log(p[ri, si] + eps) - np.log(np.where(legmv[ri], p[ri, 6:], 0).max(1) + eps)
    ys[f"search log(s+{eps:g})"] = np.log(sh[ri, si] + eps) - np.log(np.where(legmv[ri], sh[ri, 6:], 0).max(1) + eps)
P(f"\n## (a) voluntary: y = log share of switch candidate − log share of best legal move; {len(ri)} candidates, "
  f"{len(rows)} decisions (95% CI: bootstrap over train games)")
names = CAND + ROW + ["const"]
for k, y in ys.items():
    c, lo, hi = ols(y, X, g)
    P(f"{k:22s}: " + ", ".join(f"{n} {c[i]:+.2f} [{lo[i]:+.2f},{hi[i]:+.2f}]" for i, n in enumerate(names)))
    res.setdefault("ols_vol", {})[k] = dict(zip(names, map(float, c)))

rows = np.flatnonzero(forced & tr & (legsw.sum(1) >= 2))
ri, si = np.nonzero(legsw[rows]); ri = rows[ri]
def cen_rows(v):
    s = np.zeros(N); c = np.zeros(N); np.add.at(s, ri, v); np.add.at(c, ri, 1)
    return v - s[ri] / c[ri]
Xf = np.c_[[cen_rows(f[k][ri, si]) for k in CAND]].T
g = d["game"][ri]
P(f"\n## (a) forced switches (≥2 candidates): y centred within the decision; {len(ri)} candidates, {len(rows)} decisions")
for k, y in {"net logit": lg[ri, si], "net log(p+0.01)": np.log(p[ri, si] + 1e-2),
             "search log(s+0.01)": np.log(sh[ri, si] + 1e-2), "net log(p+0.001)": np.log(p[ri, si] + 1e-3),
             "search log(s+0.001)": np.log(sh[ri, si] + 1e-3)}.items():
    c, lo, hi = ols(cen_rows(y), Xf, g)
    P(f"{k:22s}: " + ", ".join(f"{n} {c[i]:+.2f} [{lo[i]:+.2f},{hi[i]:+.2f}]" for i, n in enumerate(CAND)))
    res.setdefault("ols_forced", {})[k] = dict(zip(CAND, map(float, c[:4])))

# ------------------------------------------------------------------ (b) conditional logit
P("\n## (b) conditional logit: θ = τ·logit_net + switch·(β·φ + β0), ML fit to the search's shares (train)")
rows = np.flatnonzero(tr & legsw.any(1) & f["has_opp"])
L0 = torch.as_tensor(np.where(mask[rows], np.clip(full["logits"][rows], -60, 60), -1e9))
S = torch.as_tensor(sh[rows] * mask[rows]); S = S / S.sum(1, keepdim=True)
MKt = torch.as_tensor(mask[rows])


def fit(feats, free_tau, use_hat, rowfeats=()):
    Fc = [torch.as_tensor((hat[k][rows] if use_hat and k in hat else f[k][rows]).astype(np.float64)) for k in feats]
    Fr = [torch.as_tensor(f[k][rows].astype(np.float64))[:, None].expand(-1, 6) for k in rowfeats]
    Phi = torch.stack(Fc + Fr + [torch.ones(len(rows), 6, dtype=torch.float64)], -1)   # [n,6,K]
    beta = torch.zeros(Phi.shape[-1], dtype=torch.float64, requires_grad=True)
    lt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta, lt] if free_tau else [beta], max_iter=200, line_search_fn="strong_wolfe")
    def theta():
        tau = torch.exp(lt) if free_tau else 1.0
        th = L0 * tau
        th = torch.cat([th[:, :6] + Phi @ beta, th[:, 6:]], 1)
        return th.masked_fill(~MKt, -1e9)
    def closure():
        opt.zero_grad()
        loss = -(S * torch.log_softmax(theta(), 1)).sum(1).mean()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        ce = -(S * torch.log_softmax(theta(), 1)).sum(1).mean().item()
        L0m = L0.masked_fill(~MKt, -1e9)
        ce0 = -(S * torch.log_softmax(L0m, 1)).sum(1).mean().item()
    return beta.detach().numpy(), float(torch.exp(lt)) if free_tau else 1.0, ce, ce0


for feats, rowf in ((["dfn", "off"], ()), (["dfn", "off", "spd", "hp"], ()), (["dfn", "off"], ("dfn_act", "off_act")),
                    (["dfn", "off", "spd", "hp"], ("dfn_act", "off_act"))):
    for use_hat in (False, True):
        for free_tau in (False, True):
            b, tau, ce, ce0 = fit(feats, free_tau, use_hat, rowf)
            nm = feats + list(rowf) + ["const"]
            key = f"{'+'.join(nm)}|{'hat' if use_hat else 'true'}|{'tau' if free_tau else 'tau1'}"
            P(f"{'φ̂' if use_hat else 'φ '} τ={'free' if free_tau else '1   '} " + ", ".join(f"{n} {v:+.3f}" for n, v in zip(nm, b))
              + f" | τ {tau:.3f} | CE {ce:.4f} (net alone {ce0:.4f})")
            res.setdefault("clogit", {})[key] = dict(beta=dict(zip(nm, map(float, b))), tau=tau, ce=ce, ce0=ce0)

json.dump(res, open(OUT / f"target_{tag}.json", "w"), indent=1, default=float)
(OUT / f"target_{tag}.txt").write_text("\n".join(txt) + "\n")
