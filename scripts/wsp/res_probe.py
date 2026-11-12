"""Linear (and MLP) probes per layer for engine ground truth, vs shuffled labels and raw-observation baselines.

Token per target family: move token 13+j for our-move targets; field (0), own active (1+a), opp active (7+b)
for state targets; own bench token 1+k for switch-in safety. Ridge (least squares) probes, 5-fold CV grouped by
game, lambda picked on an inner game split; R² (continuous) or AUC (binary) on the out-of-fold predictions with
95% game-bootstrap intervals. Baselines: same probe with shuffled labels (L2.out), linear and MLP probes on the
raw observation numerics of the relevant mons/moves/field. Saves full-data probe directions for steering.

    python scripts/wsp/res_probe.py 500M [--no-raw]
"""

import json
import sys

import numpy as np
import torch
from torch import nn

from res_lib import OUT, STAGES, auc, boot_games, cv_ridge, data, forward, gfolds, r2, raw_features, ridge_fit
from wsp_common import Net, load

BINARY = {"first", "first_vis", "won"}


def mlp_cv(X, Y, games, k=5, epochs=12, seed=0):
    """Multi-output MLP probe (NaN targets masked), game-grouped CV -> out-of-fold predictions."""
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)
    folds = gfolds(games, k, seed)
    pred = torch.zeros_like(Y)
    for f in range(k):
        tr = torch.as_tensor(folds != f)
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = torch.where(sd < 1e-3, torch.ones_like(sd), sd)  # near-constant inputs: no blow-up on test folds
        ym = torch.nanmean(Y[tr], 0)
        ys = torch.as_tensor(np.nanstd(Y[tr].numpy(), 0) + 1e-4)
        Xn, Yn = ((X - mu) / sd).clamp(-10, 10), (Y - ym) / ys
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                            nn.Linear(256, Y.shape[1]))
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        idx = torch.nonzero(tr).squeeze(1)
        for ep in range(epochs):
            perm = idx[torch.randperm(len(idx))]
            for s in range(0, len(perm), 512):
                b = perm[s:s + 512]
                yb = Yn[b]
                m = ~torch.isnan(yb)
                loss = ((net(Xn[b]) - torch.nan_to_num(yb)) ** 2 * m).sum() / m.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            te = torch.as_tensor(folds == f)
            pred[te] = net(Xn[te]) * ys + ym
    return pred.numpy()


def score(name, y, p, games):
    ok = np.isfinite(y)
    y, p, g = y[ok], p[ok], games[ok]
    if name in BINARY:
        keep = y != 0.5
        y, p, g = y[keep], p[keep], g[keep]
        v = auc(y, p)
        ci = boot_games(lambda a, b: auc(a, b), g, y, p, n=100)
        return {"metric": "AUC", "v": float(v), "ci": [float(c) for c in ci], "n": int(len(y))}
    v = r2(y, p)
    ci = boot_games(r2, g, y, p, n=100)
    out = {"metric": "R2", "v": float(v), "ci": [float(c) for c in ci], "n": int(len(y))}
    if name.endswith("pko"):
        yb = y > 0.5
        out["auc_ko"] = float(auc(yb, p))
    return out


def main():
    tag = sys.argv[1]
    do_raw = "--no-raw" not in sys.argv
    only_raw_mlp = "--raw-mlp-only" in sys.argv
    d = data()
    T = d["truth"]
    N = len(d["obs"])
    games = d["game"]
    net = Net(load(tag))
    acts = forward(net, d, keep=STAGES)
    print("forward done", flush=True)
    r = np.arange(N)

    # ---- row sets
    mi, mj = np.nonzero((d["mask"][:, 6:10] > 0) & np.isfinite(T["our_frac"]))
    eff = T["our_eff"][mi, mj]
    move_t = {"our_frac": T["our_frac"][mi, mj], "our_pko": T["our_pko"][mi, mj],
              "our_log2eff": np.where(eff > 0, np.log2(np.where(eff > 0, eff, 1)), np.nan)}
    si = np.flatnonzero((d["own_act"] >= 0) & (d["opp_act"] >= 0))
    st = {k: T[k][si] for k in ("thr_frac", "thr_pko", "thr_frac_vis", "thr_pko_vis", "first", "first_vis",
                                "spe_ratio", "hp_own", "hp_opp", "left_own", "left_opp")}
    st["stab_log2eff"] = np.log2(np.maximum(T["stab_eff"][si], 0.125))
    st["won"] = d["won"][si]
    bi, bk = np.nonzero(np.isfinite(T["bench_frac"]))
    bench_t = {"bench_frac": T["bench_frac"][bi, bk], "bench_pko": T["bench_pko"][bi, bk]}
    families = {
        "move": (mi, 13 + mj, move_t),
        "state@field": (si, np.zeros(len(si), int), st),
        "state@own_active": (si, 1 + d["own_act"][si], st),
        "state@opp_active": (si, 7 + d["opp_act"][si], st),
        "bench": (bi, 1 + bk, bench_t),
    }
    raws = {}
    if do_raw:
        Xm = np.zeros((len(mi), 44 + 125 * 2 + d["split"]["field"].shape[1]), np.float32)
        for j in range(4):
            sel = mj == j
            Xm[sel] = raw_features(d, "move", j)[mi[sel]]
        raws["move"] = Xm
        raws["state"] = raw_features(d, "state")[si]
        Xb = np.zeros((len(bi), 125 + 176 + raws["state"].shape[1]), np.float32)
        for k in range(6):
            sel = bk == k
            Xb[sel] = raw_features(d, "bench", np.full(N, k))[bi[sel]]
        raws["bench"] = Xb
    res, dirs = {}, {}
    for fam, (rows, tok, tg) in families.items():
        g = games[rows]
        res[fam] = {}
        srcs = {s: acts[s][rows, tok].astype(np.float32) for s in STAGES}
        for name, y in ({} if only_raw_mlp else tg).items():
            ok = np.isfinite(y)
            out = {}
            for s, X in srcs.items():
                p = np.full(len(y), np.nan)
                p[ok] = cv_ridge(X[ok], y[ok], g[ok])
                out[s] = score(name, y, p, g)
                if s != "final":
                    m = ridge_fit(X[ok].astype(np.float64), y[ok].astype(np.float64), 1e-3)
                    dirs[f"{fam}|{name}|{s}|w"] = (m[2] / m[1]).astype(np.float32)
                    dirs[f"{fam}|{name}|{s}|mu"] = m[0].astype(np.float32)
                    # difference of means between top and bottom quartile
                    hi, lo = y[ok] >= np.quantile(y[ok], 0.75), y[ok] <= np.quantile(y[ok], 0.25)
                    if hi.any() and lo.any() and hi.sum() < ok.sum() and (hi != lo).any():
                        dirs[f"{fam}|{name}|{s}|dom"] = (X[ok][hi].mean(0) - X[ok][lo].mean(0)).astype(np.float32)
            rng = np.random.default_rng(1)
            ysh = y.copy()
            ysh[ok] = rng.permutation(y[ok])
            p = np.full(len(y), np.nan)
            p[ok] = cv_ridge(srcs["L2.out"][ok], ysh[ok], g[ok])
            out["shuffled@L2"] = score(name, ysh, p, g)
            if do_raw:
                Xr = raws["move" if fam == "move" else "bench" if fam == "bench" else "state"]
                p = np.full(len(y), np.nan)
                p[ok] = cv_ridge(Xr[ok], y[ok], g[ok])
                out["raw_linear"] = score(name, y, p, g)
            res[fam][name] = out
            print(fam, name, {k: round(v["v"], 3) for k, v in out.items()}, flush=True)
        # MLP probes (multi-output): raw numerics (ceiling from the inputs), residual at L-1.out and final
        names = list(tg)
        Y = np.stack([tg[n] for n in names], 1).astype(np.float32)
        Y[:, [n in BINARY for n in names]] = np.where(Y[:, [n in BINARY for n in names]] == 0.5, np.nan,
                                                       Y[:, [n in BINARY for n in names]])
        if only_raw_mlp:
            for n in names:
                res[fam][n] = {}
        mlp_src = {"mlp@L-1.out": srcs["L-1.out"], "mlp@final": srcs["final"]}
        if do_raw:
            mlp_src["raw_mlp"] = raws["move" if fam == "move" else "bench" if fam == "bench" else "state"]
        if fam in ("state@own_active", "state@opp_active"):
            mlp_src.pop("raw_mlp", None)
        for sname, X in mlp_src.items():
            P = mlp_cv(X, Y, g)
            for c, n in enumerate(names):
                res[fam][n][sname] = score(n, tg[n], P[:, c], g)
            print(fam, sname, {n: round(res[fam][n][sname]["v"], 3) for n in names}, flush=True)
    if only_raw_mlp:
        (OUT / f"probes_mlp_{tag}.json").write_text(json.dumps(res, indent=1))
        print("saved")
        return
    (OUT / f"probes_{tag}.json").write_text(json.dumps(res, indent=1))
    np.savez(OUT / f"probe_dirs_{tag}.npz", **dirs)
    print("saved")


if __name__ == "__main__":
    main()
