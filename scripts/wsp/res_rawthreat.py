"""Fairer raw-observation baseline for the opponent-threat targets: the 'state' raw features plus the opponent
active's (revealed) move numerics, which res_probe.py's state baseline omitted. Linear ridge + MLP, game CV.

    python scripts/wsp/res_rawthreat.py
"""

import json

import numpy as np

from res_lib import OUT, cv_ridge, data, raw_features
from res_probe import mlp_cv, score


def main():
    d = data()
    T = d["truth"]
    N = len(d["obs"])
    s = d["split"]
    r = np.arange(N)
    si = np.flatnonzero((d["own_act"] >= 0) & (d["opp_act"] >= 0))
    pa = d["opp_act"]
    X = np.concatenate([raw_features(d, "state"), s["move_nums"][r, 6 + np.maximum(pa, 0)].reshape(N, -1)
                        .astype(np.float32)], 1)[si]
    g = d["game"][si]
    tg = {k: T[k][si] for k in ("thr_frac", "thr_pko", "thr_frac_vis", "thr_pko_vis")}
    tg["stab_log2eff"] = np.log2(np.maximum(T["stab_eff"][si], 0.125))
    res = {}
    for n, y in tg.items():
        ok = np.isfinite(y)
        p = np.full(len(y), np.nan)
        p[ok] = cv_ridge(X[ok], y[ok], g[ok])
        res[n] = {"raw+oppmoves_linear": score(n, y, p, g)}
    names = list(tg)
    P = mlp_cv(X, np.stack([tg[n] for n in names], 1).astype(np.float32), g)
    for c, n in enumerate(names):
        res[n]["raw+oppmoves_mlp"] = score(n, tg[n], P[:, c], g)
    print(json.dumps({n: {k: round(v["v"], 3) for k, v in o.items()} for n, o in res.items()}))
    (OUT / "rawthreat.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
