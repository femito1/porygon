"""dyn: did the embedding-lr arms (600M -> 640M, x1 / x30 / x300 embedding Adam steps) give the embeddings
structure? Ridge probes (5-fold CV R², features z-scored, best λ of a grid) on the table at 640M, on the
movement Δ = E640 − E600 and, for reference, on the lineage's own movement E600 − E2.5M, vs random codes of the
same size and shuffled labels; λ chosen as the best of a grid for every feature set alike. Targets as scripts/probe_embeddings.py (species: base stats, types, level;
moves: base power, status, priority, Fire type). Also cos(Δ_arm, Δ_ctrl) per table.

    python scripts/wsp/dyn_arms_probe.py OUT_DIR
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/vol/bitbucket/fr1226/porygon-ws-wsp/scripts")
from probe_embeddings import targets  # noqa: E402

from porygon.encoder import INDEX  # noqa: E402
from dyn_ckpts import arms, lineage, read  # noqa: E402


def cv_r2(X, y, k=5):
    """Best 5-fold CV R² over λ ∈ {0.1, 1, 10, 100, 1000}·n_features (the max-over-λ bias is shared by the
    random-code and shuffled-label baselines, which go through the same function)."""
    return max(_cv_r2(X, y, k, lam * X.shape[1]) for lam in (0.1, 1, 10, 100, 1000))


def _cv_r2(X, y, k, lam):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    idx = np.random.default_rng(0).permutation(len(y))
    ss = st = 0.0
    for f in np.array_split(idx, k):
        tr = np.setdiff1d(idx, f)
        mu, ym = X[tr].mean(0), y[tr].mean()
        A = X[tr] - mu
        w = np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ (y[tr] - ym))
        e = (X[f] - mu) @ w + ym - y[f]
        ss += e @ e
        st += ((y[f] - ym) ** 2).sum()
    return 1 - ss / st


def table(path, name):
    sd, _, _ = read(path, optim=False)
    return sd[f"features_extractor.{name}.weight"].double().numpy()


def main():
    out = sys.argv[1]
    lin = dict(lineage())
    A = {a: dict(v) for a, v in arms().items()}
    ends = {"ctrl": A["ctrl"][640032682], "x30": A["x30"][640025994], "x300": A["x300"][640025414]}
    tg = targets()
    rng = np.random.default_rng(1)
    res = {}
    for k, (ids, ys) in tg.items():
        rows = [INDEX[k][i] for i in ids]
        E0 = table(lin[2499840], k)[rows]
        E600 = table(lin[600002347], k)[rows]
        feats = {"E600": E600, "lineage Δ 600M−2.5M": E600 - E0}
        D = {}
        for a, p in ends.items():
            E = table(p, k)[rows]
            feats[f"{a} E640"] = E
            feats[f"{a} Δ 640−600"] = D[a] = E - E600
        feats["random code"] = rng.standard_normal(E600.shape)
        r = {}
        for fn, X in feats.items():
            r[fn] = {n: float(cv_r2(X, y)) for n, y in ys.items()}
        r["shuffled labels (E600)"] = {n: float(np.mean([cv_r2(E600, rng.permutation(y)) for _ in range(5)])) for n, y in ys.items()}
        r["rand-code sd over 10"] = {n: float(np.std([cv_r2(rng.standard_normal(E600.shape), y) for _ in range(10)])) for n, y in ys.items()}
        r["cos(Δ_arm, Δ_ctrl)"] = {a: float((D[a] * D["ctrl"]).sum() / np.linalg.norm(D[a]) / np.linalg.norm(D["ctrl"])) for a in ("x30", "x300")}
        r["|Δ|/|E600| (rows in probe)"] = {a: float(np.linalg.norm(D[a]) / np.linalg.norm(E600)) for a in D}
        res[k] = r
        print(f"\n== {k} ({len(rows)} rows)")
        names = list(ys)
        print(f"{'features':28s} " + " ".join(f"{n[:12]:>12s}" for n in names) + "   mean")
        for fn in list(feats) + ["shuffled labels (E600)", "rand-code sd over 10"]:
            v = [r[fn][n] for n in names]
            print(f"{fn:28s} " + " ".join(f"{x:12.3f}" for x in v) + f" {np.mean(v):7.3f}")
        print("cos(Δ_arm, Δ_ctrl):", r["cos(Δ_arm, Δ_ctrl)"], " |Δ|/|E|:", r["|Δ|/|E600| (rows in probe)"])
    json.dump(res, open(f"{out}/arms_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
