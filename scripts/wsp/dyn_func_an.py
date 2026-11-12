"""dyn: functional change over training on the logged real positions, from dyn_func.py outputs.

Per checkpoint (95% intervals: bootstrap over GAMES, 2000 replicates, same replicate weights for every
checkpoint so differences are paired):
  agree_search  P(argmax π = argmax search shares)
  share_top     search share of the network's argmax  (Σ shares·1[argmax π])
  exp_share     Σ π·shares
  ce_search     -Σ shares·log π
  kl_600        KL(π_600M || π_k);  agree_600  P(argmax π_k = argmax π_600M);  entropy H(π_k)
  value: Q_k = V_k + Φ with Φ = logged value − V_500M (V+Φ space, where the logged values live);
         mse_z  mean (Q_k − z)², z = 2·won − 1;  corr_z  corr(Q_k, z);  v_rms_600  RMS(V_k − V_600M)
Also: KL between 5M checkpoints of 300-600M at lags 1..60 (does the function random-walk or stay put?),
trend slopes of agree_search over 300-600M, and the arms (600M -> 640M).

    python scripts/wsp/dyn_func_an.py OUT_DIR DATA_DIR
"""
import glob
import json
import os
import sys

import numpy as np

EPS = 1e-8


def main():
    out, data = sys.argv[1], sys.argv[2]
    d = np.load(f"{data}/decisions.npz")
    shares, mask, game, won, logged_v = d["shares"].astype(np.float64), d["mask"], d["game"], d["won"], d["value"]
    N = len(game)
    ok = shares.sum(1) > 0.5
    a_star = shares.argmax(1)
    z = 2 * won - 1
    gids, ginv = np.unique(game, return_inverse=True)
    G = len(gids)
    rng = np.random.default_rng(0)
    B = 2000
    W = rng.multinomial(G, np.ones(G) / G, size=B).astype(np.float64)  # [B, G] game weights

    def gsum(x, sel=None):
        x = np.where(sel, x, 0.0) if sel is not None else x
        return np.bincount(ginv, weights=x, minlength=G)

    n_all, n_ok = gsum(np.ones(N)), gsum(np.ones(N), ok)

    def boot_mean(x, sel=None):
        s = gsum(x, sel)
        n = n_ok if sel is not None else n_all
        return float(s.sum() / n.sum()), (W @ s) / (W @ n)

    files = {}
    for f in glob.glob(f"{out}/func/*.npz"):
        b = os.path.basename(f)[:-4]
        files[b] = f
    lin = sorted((int(k), f) for k, f in files.items() if k.isdigit())
    arm = sorted(((k.split("_")[0], k.split("_")[1]), f) for k, f in files.items() if not k.isdigit())
    load = lambda f: (np.load(f)["probs"].astype(np.float64), np.load(f)["value"].astype(np.float64))
    P600, V600 = load(dict(lin)[600002347])
    P500, V500 = load(dict(lin)[500036720])
    phi = logged_v - V500
    res_ceiling = float(shares.max(1).mean())
    print(f"mean max search share (ceiling of share_top) = {res_ceiling:.4f}")
    print(f"N={N} games={G} rows with shares={ok.sum()}  max|π500 − logged π| = {np.abs(P500 - d['probs']).max():.4f}")
    print(f"Φ = logged − V500: mean {phi.mean():+.3f} sd {phi.std():.3f}  (corr with turn {np.corrcoef(phi, d['turn'])[0,1]:+.2f})")
    top600 = P600.argmax(1)

    def metrics(P, V):
        P = np.clip(P, EPS, 1)
        top = P.argmax(1)
        Q = V + phi
        m = {}
        m["agree_search"] = boot_mean((top == a_star).astype(float), ok)
        m["share_top"] = boot_mean(shares[np.arange(N), top], ok)
        m["exp_share"] = boot_mean((P * shares).sum(1), ok)
        m["ce_search"] = boot_mean(-(shares * np.log(P)).sum(1), ok)
        m["kl_600"] = boot_mean((np.clip(P600, EPS, 1) * (np.log(np.clip(P600, EPS, 1)) - np.log(P))).sum(1))
        m["agree_600"] = boot_mean((top == top600).astype(float))
        m["entropy"] = boot_mean(-(P * np.log(P)).sum(1))
        m["mse_z"] = boot_mean((Q - z) ** 2)
        m["v_rms_600"] = (float(np.sqrt(((V - V600) ** 2).mean())), np.sqrt((W @ gsum((V - V600) ** 2)) / (W @ n_all)))
        # correlation with outcome per replicate
        s = {k: gsum(v) for k, v in dict(q=Q, q2=Q * Q, qz=Q * z, z=z, z2=z * z).items()}
        def corr(w):
            n = w @ n_all
            mq, mz = (w @ s["q"]) / n, (w @ s["z"]) / n
            cov = (w @ s["qz"]) / n - mq * mz
            return cov / np.sqrt(((w @ s["q2"]) / n - mq ** 2) * ((w @ s["z2"]) / n - mz ** 2))
        m["corr_z"] = (float(corr(np.ones(G))), corr(W))
        return m

    res = {"lineage": {}, "arms": {}}
    reps = {}
    for s, f in lin:
        P, V = load(f)
        m = metrics(P, V)
        reps[s] = m
        res["lineage"][s] = {k: [v[0], float(np.percentile(v[1], 2.5)), float(np.percentile(v[1], 97.5))] for k, v in m.items()}
    for (a, s), f in arm:
        P, V = load(f)
        m = metrics(P, V)
        reps[(a, s)] = m
        res["arms"][f"{a}_{s}"] = {k: [v[0], float(np.percentile(v[1], 2.5)), float(np.percentile(v[1], 97.5))] for k, v in m.items()}
    keys = ["agree_search", "share_top", "exp_share", "ce_search", "kl_600", "agree_600", "entropy", "mse_z", "corr_z", "v_rms_600"]
    print("\nsteps(M)  " + "  ".join(f"{k:>22s}" for k in keys))
    for s in [x for x, _ in lin]:
        r = res["lineage"][s]
        print(f"{s/1e6:7.1f}  " + "  ".join(f"{r[k][0]:.4f} [{r[k][1]:.4f},{r[k][2]:.4f}]".rjust(22) for k in keys))
    for k in res["arms"]:
        r = res["arms"][k]
        print(f"{k:>14s}  " + "  ".join(f"{r[kk][0]:.4f} [{r[kk][1]:.4f},{r[kk][2]:.4f}]".rjust(22) for kk in keys))
    # paired differences
    def diff(a, b, k):
        x = reps[a][k][1] - reps[b][k][1]
        return [reps[a][k][0] - reps[b][k][0], float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    pairs = [(400029852, 300028848), (500036720, 400029852), (600002347, 500036720), (600002347, 300028848),
             (300028848, 195948732), (195948732, 100949682), (100949682, 21261312), (21261312, 2499840)]
    pairs += [(k, 600002347) for k in reps if isinstance(k, tuple)]
    pairs += [(("x30", "640025994"), ("ctrl", "640032682")), (("x300", "640025414"), ("ctrl", "640032682"))]
    pairs += [(("swa", w), b) for w in ("550-600", "500-600", "400-600", "300-600") for b in (600002347, 500036720)]
    pairs += [(("swa", "450-500"), 500036720)]
    res["diffs"] = {}
    print("\npaired differences (a − b) [95% CI over games]")
    for a, b in pairs:
        if a in reps and b in reps:
            dd = {k: diff(a, b, k) for k in ("agree_search", "share_top", "exp_share", "ce_search", "mse_z", "corr_z")}
            res["diffs"][f"{a}-{b}"] = dd
            print(f"{str(a):>22s} − {str(b):<22s} " + "  ".join(f"{k} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]" for k, v in dd.items()))
    # trend over 300-600M: slope of agree_search and share_top per 100M, bootstrap
    late = [s for s, _ in lin if 300_000_000 <= s <= 600_002_347]
    xs = np.array(late) / 1e8
    res["trend"] = {}
    for k in ("agree_search", "share_top", "exp_share", "ce_search", "mse_z"):
        Y = np.stack([reps[s][k][1] for s in late], 1)  # [B, n]
        y0 = np.array([reps[s][k][0] for s in late])
        X = np.vstack([xs, np.ones_like(xs)]).T
        sl = np.linalg.lstsq(X, Y.T, rcond=None)[0][0]
        s0 = np.linalg.lstsq(X, y0, rcond=None)[0][0]
        res["trend"][k] = [float(s0), float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]
        print(f"trend 300-600M {k}: {s0:+.5f} per 100M [{np.percentile(sl, 2.5):+.5f}, {np.percentile(sl, 97.5):+.5f}]")
    # functional random walk: KL(π_i || π_j) at lags over the 5M grid in 300-600M
    late5 = [s for s in late if s % 5_000_000 in (28848, 267524, 36720, 2347)]
    Ps = {s: np.clip(load(dict(lin)[s])[0], EPS, 1) for s in late}
    lp = {s: np.log(p) for s, p in Ps.items()}
    res["kl_lag"] = {}
    print("\nKL(π_i || π_j) and top-1 disagreement by step distance over 300-600M checkpoints (all pairs, binned to 10M)")
    bins = {}
    for i, a in enumerate(late):
        for b in late[i + 1:]:
            k = int(round((b - a) / 1e7))
            if k == 0 or len(bins.get(k, [])) >= 40:
                continue
            bins.setdefault(k, []).append((float((Ps[a] * (lp[a] - lp[b])).sum(1).mean()),
                                           float((Ps[a].argmax(1) != Ps[b].argmax(1)).mean())))
    for k in sorted(bins):
        v = np.array(bins[k])
        res["kl_lag"][k * 10] = [float(v[:, 0].mean()), float(v[:, 1].mean()), len(v)]
        if k in (1, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30):
            print(f"Δ ≈ {k*10:3d}M: KL {v[:, 0].mean():.4f}  top-1 disagreement {v[:, 1].mean():.4f}  (pairs {len(v)})")
    # the same at other learning rates: consecutive checkpoints ~5M apart anywhere in the lineage
    steps_all = [s for s, _ in lin]
    res["kl_5m"] = []
    print("\nKL between checkpoints 4-6M apart, anywhere (lr from the Adam state of the earlier one)")
    for i, a in enumerate(steps_all):
        for b in steps_all[i + 1:i + 3]:
            if 4e6 <= b - a <= 6e6 or 9e6 <= b - a <= 11e6:
                pa, pb = np.clip(load(dict(lin)[a])[0], EPS, 1), np.clip(load(dict(lin)[b])[0], EPS, 1)
                kl = float((pa * (np.log(pa) - np.log(pb))).sum(1).mean())
                dis = float((pa.argmax(1) != pb.argmax(1)).mean())
                res["kl_5m"].append([a, b, kl, dis])
                print(f"{a/1e6:6.1f} → {b/1e6:6.1f}M: KL {kl:.4f} disagreement {dis:.4f}")
    # consecutive 5M changes: is each step toward the search? sign consistency of Δagree_search
    da = np.diff([reps[s]["agree_search"][0] for s in late])
    res["step_changes"] = dict(mean=float(da.mean()), sd=float(da.std()), frac_pos=float((da > 0).mean()),
                               lag1_autocorr=float(np.corrcoef(da[:-1], da[1:])[0, 1]))
    print("Δagree_search per checkpoint step:", res["step_changes"])
    json.dump(res, open(f"{out}/func_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
