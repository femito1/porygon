"""Steering control at equal norm: is the large effect of the difference-of-means (DoM) direction specific, or
just the effect of any big push? For each spec: σ along probe / DoM / random directions, cos(DoM, probe),
fraction of DoM in the top-10 principal components, and the action effect of vectors with the SAME norm
(2·σ_DoM) along the probe, the DoM and 16 random directions.

    python scripts/wsp/res_steer2.py 500M
"""

import json
import sys

import numpy as np
import torch

from res_lib import OUT, Intervention, data, forward
from wsp_common import Net, load


def main():
    tag = sys.argv[1]
    d = data()
    T = d["truth"]
    N = len(d["obs"])
    net = Net(load(tag))
    dirs = np.load(OUT / f"probe_dirs_{tag}.npz")
    rng = np.random.default_rng(0)
    xt = torch.as_tensor(d["obs"], dtype=torch.float32)
    mt = torch.as_tensor(d["mask"])
    s = d["split"]
    r = np.arange(N)
    oa = np.maximum(d["own_act"], 0)
    mv = s["move_nums"][r, oa]
    legal = d["mask"][:, 6:10] > 0
    damaging = mv[:, :, 7] < 0.5
    A = forward(net, d, keep=("L1.out", "L2.out"))
    layer_of = {"L1.out": 1, "L2.out": 2}

    @torch.no_grad()
    def run(rows, iv):
        lg, _ = net.forward(xt[rows], mt[rows], iv)
        return lg.double().numpy()

    out = {}
    specs = [("move", "our_pko", "L1.out"), ("move", "our_pko", "L2.out"), ("move", "our_frac", "L1.out"),
             ("bench", "bench_pko", "L2.out"), ("state@field", "thr_pko", "L2.out"),
             ("state@own_active", "thr_pko", "L1.out")]
    for fam, tgt, st in specs:
        li = layer_of[st]
        if fam == "move":
            cand = np.flatnonzero((legal & damaging).sum(1) >= 2)
            rows = rng.choice(cand, 3000, replace=False)
            tok = np.array([13 + rng.choice(np.flatnonzero(legal[i] & damaging[i])) for i in rows])
            act = tok - 13 + 6
        elif fam == "bench":
            cand = np.flatnonzero(np.isfinite(T["bench_frac"]).any(1) & (d["mask"][:, :6].sum(1) > 0))
            rows = rng.choice(cand, 3000, replace=False)
            ks = [np.flatnonzero(np.isfinite(T["bench_frac"][i]) & (d["mask"][i, :6] > 0)) for i in rows]
            keep = np.array([len(k) > 0 for k in ks])
            rows = rows[keep]
            tok = np.array([1 + rng.choice(ks[q]) for q in np.flatnonzero(keep)])
            act = tok - 1
        else:
            cand = np.flatnonzero((d["mask"][:, :6].sum(1) > 0) & (d["mask"][:, 6:10].sum(1) > 0) & (d["own_act"] >= 0))
            rows = rng.choice(cand, 3000, replace=False)
            tok = np.zeros(len(rows), int) if fam == "state@field" else 1 + d["own_act"][rows]
            act = None
        m = d["mask"][rows] > 0

        def eff(lg):
            p = np.exp(lg - lg.max(1, keepdims=True)) * m
            p /= p.sum(1, keepdims=True)
            if act is None:
                return np.log(p[:, :6].sum(1) + 1e-9) - np.log(p[:, 6:10].sum(1) + 1e-9)
            oth = np.where(m, lg, np.nan)
            oth[np.arange(len(rows)), act] = np.nan
            return lg[np.arange(len(rows)), act] - np.nanmean(oth, 1)

        def steer(vec):
            lg = np.zeros((len(rows), 10))
            for t in np.unique(tok):
                k = tok == t
                lg[k] = run(rows[k], Intervention(resid_add={(li, int(t)): torch.as_tensor(vec, dtype=torch.float32)}))
            return eff(lg)
        base = steer(np.zeros(256))
        X = A[st][rows, tok].astype(np.float64)
        Xc = X - X.mean(0)
        w = dirs[f"{fam}|{tgt}|{st}|w"].astype(np.float64)
        dom = dirs[f"{fam}|{tgt}|{st}|dom"].astype(np.float64)
        uw, ud = w / np.linalg.norm(w), dom / np.linalg.norm(dom)
        _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        sig = {"probe": float((Xc @ uw).std()), "dom": float((Xc @ ud).std())}
        rs = []
        for q in range(16):
            v = rng.standard_normal(256)
            rs.append(v / np.linalg.norm(v))
        sig["rand_mean"] = float(np.mean([(Xc @ v).std() for v in rs]))
        norm = 2 * sig["dom"]
        res = {"n": int(len(rows)), "sigma": sig, "cos(dom,probe)": float(ud @ uw),
               "dom_frac_in_top10_PCs": float(np.sum((Vt[:10] @ ud) ** 2)),
               "probe_frac_in_top10_PCs": float(np.sum((Vt[:10] @ uw) ** 2)),
               "push_norm": norm, "stream_norm_mean": float(np.linalg.norm(X, axis=1).mean())}
        for name, u in (("probe", uw), ("dom", ud)):
            for sgn in (+1, -1):
                res[f"{name}{'+' if sgn > 0 else '-'}"] = float(np.mean(steer(sgn * norm * u) - base))
        rnd = [float(np.mean(steer(norm * v) - base)) for v in rs]
        res["rand_mean"] = float(np.mean(rnd))
        res["rand_abs_mean"] = float(np.mean(np.abs(rnd)))
        res["rand_abs_max"] = float(np.max(np.abs(rnd)))
        out[f"{fam}|{tgt}|{st}"] = res
        print(fam, tgt, st, json.dumps(res), flush=True)
    (OUT / f"steer2_{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
