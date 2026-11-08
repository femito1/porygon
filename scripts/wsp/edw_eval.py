"""edw step 4: held-out evaluation of edited ActionHeads on the TEST games (each edit = one logged check pass
over the test rows + test swaps). Bootstrap over test games (1,000 resamples), paired (edit − original).

    python scripts/wsp/edw_eval.py 500M s1 s2 s3      (W files LOCAL/W_<tag>_<name>.pt from edw_design.py)
"""
import json
import sys

import numpy as np
import torch

from beh_common import load_local
from edw_common import LOCAL, OUT, head_logits, head_weights, log_check, matchup, softmax, split
from wsp_common import CKPTS, load

torch.set_num_threads(2)
HALF = "test"
if sys.argv[1] == "--train":   # in-sample sanity check of a design on the TRAIN games (not a reported result)
    HALF = "train"; sys.argv.pop(1)
tag = sys.argv[1]
names = sys.argv[2:]
d = load_local()
N = len(d["mask"]); mask = d["mask"] > 0
tr = split(d["game"])
te = ~tr if HALF == "test" else tr
f = matchup(d["obs"])
full = np.load(LOCAL / f"{tag}_full.npz")
fin = full["fin"]
W0 = head_weights(load(CKPTS.get(tag, tag)))
sh = d["shares"].astype(np.float64)
Q = np.load("/vol/bitbucket/fr1226/porygon-ws-wsp/out/beh/derived_r2-300k.npz", allow_pickle=True)["lab_q"]
rows = np.flatnonzero(te)
game = d["game"][rows]
MK = mask[rows]; SH = sh[rows]; QQ = np.where(MK, Q[rows], np.nan)
legsw, legmv = MK[:, :6].any(1), MK[:, 6:].any(1)
V, F, NS = legsw & legmv, legsw & ~legmv, legmv & ~legsw
has_q = np.isfinite(QQ).any(1)
qmax = np.nanmax(np.where(has_q[:, None], QQ, -1), 1)
s_arg = SH.argmax(1)
txt, res = [], {}


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


def per_row(p):
    """per-row metric arrays (NaN where not applicable)."""
    a = p.argmax(1); r = np.arange(len(p))
    qa = QQ[r, a] - qmax
    qs = np.nansum(p * np.nan_to_num(QQ, nan=0.0), 1) / np.where(np.isfinite(QQ), p, 0).sum(1) - qmax
    qa = np.where(has_q, qa, np.nan); qs = np.where(has_q, qs, np.nan)
    m = {"top1=search": (a == s_arg).astype(float), "E[share|argmax]": SH[r, a], "E[share|sample]": (p * SH).sum(1),
         "entropy": -(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0)).sum(1)}
    for nm, sel in (("voluntary", V), ("forced", F)):
        m[f"top1=search | {nm}"] = np.where(sel, m["top1=search"], np.nan)
        m[f"E[share|sample] | {nm}"] = np.where(sel, m["E[share|sample]"], np.nan)
    for nm, sel in (("all", np.ones(len(p), bool)), ("voluntary", V), ("forced", F), ("no switch legal", NS),
                    ("vol, search argmax switch", V & (s_arg < 6)), ("vol, search argmax move", V & (s_arg >= 6)),
                    ("vol, net argmax switch", V & (a < 6))):
        m[f"Qgap argmax | {nm}"] = np.where(sel, qa, np.nan)
        m[f"Qgap sample | {nm}"] = np.where(sel, qs, np.nan)
    psw = p[:, :6].sum(1)
    m["P(switch) | vol"] = np.where(V, psw, np.nan)
    for nm, sel in (("opp STAB SE on our active", f["dfn_act"][rows] >= 1), ("not SE", f["dfn_act"][rows] < 1),
                    ("our best move SE on them", f["off_act"][rows] >= 1), ("we are faster", f["faster_act"][rows] > 0.5),
                    ("we are slower", f["faster_act"][rows] < 0.5), ("own HP < 50%", f["hp_act"][rows] < 0.5)):
        m[f"P(switch) | vol, {nm}"] = np.where(V & sel, psw, np.nan)
    return m


def boot(Ms, n_boot=1000, seed=0):
    """Ms: list of dicts of per-row arrays (same keys). Returns per key: estimates and paired diffs vs Ms[0]."""
    ug, inv = np.unique(game, return_inverse=True)
    G = len(ug)
    rng = np.random.default_rng(seed)
    C = rng.multinomial(G, np.full(G, 1 / G), size=n_boot).astype(float)
    out = {}
    for k in Ms[0]:
        S, Cn = [], []
        for M in Ms:
            y = M[k]; ok = np.isfinite(y)
            S.append(np.bincount(inv, weights=np.where(ok, y, 0), minlength=G)); Cn.append(np.bincount(inv, weights=ok, minlength=G))
        est = [s.sum() / max(c.sum(), 1) for s, c in zip(S, Cn)]
        bs = [(C @ s) / np.maximum(C @ c, 1) for s, c in zip(S, Cn)]
        out[k] = {"est": est, "n": int(Cn[0].sum()),
                  "diff": [(est[i] - est[0], np.percentile(bs[i] - bs[0], 2.5), np.percentile(bs[i] - bs[0], 97.5)) for i in range(len(Ms))]}
    return out


Ws = {"orig": W0}
FINS = {"orig": fin}
VALS = {"orig": full["value"]}
for nm in names:
    trunk = LOCAL / f"{tag}_full_{nm}.npz"
    if trunk.exists():          # trunk edit (edw_qk.py): its own final latents, original head
        zz = np.load(trunk); Ws[nm] = W0; FINS[nm] = zz["fin"]; VALS[nm] = zz["value"]
    else:                       # head edit: original latents, edited head
        Ws[nm] = torch.load(LOCAL / f"W_{tag}_{nm}.pt"); FINS[nm] = fin; VALS[nm] = full["value"]
lgs = {nm: head_logits(FINS[nm][rows], W, MK) for nm, W in Ws.items()}
ps = {nm: softmax(lg) for nm, lg in lgs.items()}
Ms = [per_row(ps[nm]) for nm in Ws]
search_M = {k: v for k, v in per_row(np.where(MK, SH, 0) / np.where(MK, SH, 0).sum(1, keepdims=True)).items()}
B = boot(Ms)
Bs = boot([search_M])
P(f"# edw evaluation on the {HALF.upper()} half ({tag}); games {len(np.unique(game))}, rows {len(rows)} (voluntary {V.sum()}, forced {F.sum()}, "
  f"no switch legal {NS.sum()}; oracle-labelled {has_q.sum()})")
hdr = " | ".join(["metric", "n", "search", "orig"] + [f"{nm} (Δ vs orig [95%])" for nm in names])
P("\n" + hdr); P("|".join(["---"] * (4 + len(names))))
for k, v in B.items():
    srch = Bs[k]["est"][0] if not k.startswith("entropy") else Bs[k]["est"][0]
    cells = [k, str(v["n"]), f"{srch:.4f}", f"{v['est'][0]:.4f}"]
    for i in range(1, len(Ws)):
        e, lo, hi = v["diff"][i]
        cells.append(f"{v['est'][i]:.4f} ({e:+.4f} [{lo:+.4f}, {hi:+.4f}])")
    P(" | ".join(cells))
    res[k] = v
res["search"] = {k: v["est"][0] for k, v in Bs.items()}

# TV to the original
P("\nTV distance to the original policy (mean over test rows; max):")
for nm in names:
    tv = 0.5 * np.abs(ps[nm] - ps["orig"]).sum(1)
    P(f"{nm}: all {tv.mean():.4f} | voluntary {tv[V].mean():.4f} | forced {tv[F].mean():.4f} | no switch legal "
      f"{tv[NS].mean():.2e} (max {tv[NS].max():.2e}) | argmax changed {np.mean(ps[nm].argmax(1) != ps['orig'].argmax(1)):.4f} | "
      f"mean |ΔV| {np.abs(VALS[nm][rows] - VALS['orig'][rows]).mean():.4f}")
    res.setdefault("TV", {})[nm] = dict(all=float(tv.mean()), vol=float(tv[V].mean()), forced=float(tv[F].mean()),
                                         nosw=float(tv[NS].mean()), nosw_max=float(tv[NS].max()))

# swap sensitivity on the TEST swaps
P("\nswap sensitivity (TEST swaps): centred Δ switch logit on centred Δdfn, Δoff (per doubling)")
z = np.load(LOCAL / f"{tag}_swap_{HALF}.npz")
srows = z["rows"]; SM = mask[srows]; leg = SM[:, :6]
f0, ff = matchup(z["obs0"].astype(np.float32)), matchup(z["obsf"].astype(np.float32))
forced_s = ~SM[:, 6:].any(1)
def cen(a):
    a = np.where(leg, a, np.nan); return a - np.nanmean(a, 1, keepdims=True)
Xd, Xo = cen(ff["dfn"] - f0["dfn"]), cen(ff["off"] - f0["off"])
for nm, W in Ws.items():
    zs = LOCAL / f"{tag}_swap_{HALF}_{nm}.npz"
    zf = np.load(zs) if zs.exists() else z
    l0 = head_logits(zf["fin_0"], W); l1 = head_logits(zf["fin_f"], W)
    dl = cen(l1[:, :6] - l0[:, :6])
    line = []
    for part, sel in (("all", np.ones(len(srows), bool)), ("voluntary", ~forced_s), ("forced", forced_s)):
        ok = np.isfinite(dl) & sel[:, None]
        c = np.linalg.lstsq(np.c_[Xd[ok], Xo[ok]], dl[ok], rcond=None)[0]
        line.append(f"{part}: dfn {c[0]:+.3f}, off {c[1]:+.3f}")
        res.setdefault("swap", {}).setdefault(nm, {})[part] = c.tolist()
    P(f"{nm}: " + " | ".join(line))

for nm in names:
    log_check(f"eval {nm} ({tag}) on {HALF}", len(rows), f"{HALF.upper()} rows + 4000 {HALF.upper()} swaps")
sfx = "" if HALF == "test" else "_TRAIN"
json.dump(res, open(OUT / f"eval{sfx}_{tag}_{'_'.join(names)}.json", "w"), indent=1, default=float)
(OUT / f"eval{sfx}_{tag}_{'_'.join(names)}.txt").write_text("\n".join(txt) + "\n")
