"""Helpers for the res (residual stream / MLP / probes) study: data + engine truth, gathered tokens, game-grouped
CV ridge probes with game-bootstrap intervals, raw-observation baselines."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from porygon.encoder import ACTIVE_FLAG, FIELD_DIM, MON_NUM_DIM, MOVE_DIM
from wsp_common import Intervention, Net, load, load_data, split_obs  # noqa: F401

WSP = Path("/vol/bitbucket/fr1226/porygon-ws-wsp")
LOCAL = Path("/tmp/wsp-674320/res")
OUT = WSP / "out/res"
STAGES = ["L-1.out", "L0.out", "L1.out", "L2.out", "final"]
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))

# move numerics layout (encoder.encode_move)
MV_BP, MV_ACC, MV_PRIO, MV_PP, MV_AVAIL = 0, 1, 2, 3, 4
MV_CAT = 5            # physical, special, status
MV_TYPE = 8           # 18 types (TYPES)
N_TYPES = MOVE_DIM - 26
MV_STAB = MV_TYPE + N_TYPES
MV_MULT = MV_STAB + 1  # log2(mult)/2, -1.5 for immune
MV_IMM = MV_STAB + 2


def data():
    d = load_data(LOCAL if (LOCAL / "decisions.npz").exists() else WSP / "data")
    t = np.load(OUT / "truth.npz")
    d["truth"] = {k: t[k] for k in t.files}
    s = split_obs(d["obs"])
    nums = s["nums"].astype(np.float32)
    d["own_act"] = np.where(nums[:, :6, ACTIVE_FLAG].max(1) > 0.5, nums[:, :6, ACTIVE_FLAG].argmax(1), -1)
    f = nums[:, 6:, MON_NUM_DIM - 2]
    d["opp_act"] = np.where(f.max(1) > 0.5, f.argmax(1), -1)
    d["split"] = s
    return d


def raw_features(d, kind: str, j=None) -> np.ndarray:
    """Raw-observation baseline inputs. kind 'move': move j numerics + own/opp active numerics + field;
    'state': field + own/opp active numerics + own active's 4 move numerics; 'bench' (j = own slot array)."""
    s = d["split"]
    N = len(d["obs"])
    r = np.arange(N)
    oa, pa = np.maximum(d["own_act"], 0), np.maximum(d["opp_act"], 0)
    own = s["nums"][r, oa].astype(np.float32)
    opp = s["nums"][r, 6 + pa].astype(np.float32)
    field = s["field"].astype(np.float32)
    amoves = s["move_nums"][r, oa].reshape(N, -1).astype(np.float32)
    if kind == "state":
        return np.concatenate([field, own, opp, amoves], 1)
    if kind == "move":
        mj = s["move_nums"][r, oa, j].astype(np.float32)
        return np.concatenate([mj, field, own, opp], 1)
    if kind == "bench":
        b = s["nums"][r, j].astype(np.float32)
        bm = s["move_nums"][r, j].reshape(N, -1).astype(np.float32)
        return np.concatenate([b, bm, field, own, opp, amoves], 1)
    raise ValueError(kind)


@torch.no_grad()
def forward(net: Net, d, keep=STAGES, idx=None, bs=1024, iv=None, extra=None):
    """Forward over rows idx; returns probs, logits, value and the kept cache entries as float16 [n,17,D]."""
    idx = np.arange(len(d["obs"])) if idx is None else idx
    out = {"probs": [], "logits": [], "value": []} | {k: [] for k in keep}
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        x = torch.as_tensor(d["obs"][b], dtype=torch.float32)
        m = torch.as_tensor(d["mask"][b])
        cache = {}
        lg, v = net.forward(x, m, iv, cache)
        out["probs"].append(lg.softmax(-1).numpy())
        out["logits"].append(lg.numpy())
        out["value"].append(v.numpy())
        for k in keep:
            out[k].append(cache[k].half().numpy())
        if extra:
            extra(cache, b)
    return {k: np.concatenate(v) for k, v in out.items()}


def gfolds(games: np.ndarray, k=5, seed=0):
    ug = np.unique(games)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ug)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    return np.array([fold_of[g] for g in games])


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = y.mean()
    A = Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - ym))
    return mu, sd, w, ym


def ridge_pred(model, X):
    mu, sd, w, ym = model
    return ((X - mu) / sd) @ w + ym


def cv_ridge(X, y, games, k=5, lams=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1), seed=0, return_lam=False):
    """Game-grouped k-fold CV ridge via per-fold Gram matrices (global standardisation). Lambda (relative to n)
    is the one with the lowest pooled out-of-fold MSE. Returns out-of-fold predictions."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    folds = gfolds(games, k, seed)
    G = [Z[folds == f].T @ Z[folds == f] for f in range(k)]
    b = [Z[folds == f].T @ y[folds == f] for f in range(k)]
    Gt, bt = sum(G), sum(b)
    best, best_pred = None, None
    D = np.eye(Z.shape[1])
    D[-1, -1] = 0  # no penalty on the intercept
    for lam in lams:
        pred = np.zeros_like(y)
        for f in range(k):
            n_tr = (folds != f).sum()
            w = np.linalg.solve(Gt - G[f] + lam * n_tr * D, bt - b[f])
            pred[folds == f] = Z[folds == f] @ w
        mse = np.mean((pred - y) ** 2)
        if best is None or mse < best[0]:
            best, best_pred = (mse, lam), pred
    return (best_pred, best[1]) if return_lam else best_pred


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def auc(y, s):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = y.sum()
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def boot_games(stat, games, *arrs, n=200, seed=0):
    """Bootstrap over games: stat(*arrs[sel]) for resampled game sets; returns (2.5%, 97.5%)."""
    ug, inv = np.unique(games, return_inverse=True)
    rows = [np.flatnonzero(inv == g) for g in range(len(ug))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug))
        sel = np.concatenate([rows[g] for g in pick])
        vals.append(stat(*[a[sel] for a in arrs]))
    return np.nanpercentile(vals, [2.5, 97.5])


def fmt(x, ci=None, nd=3):
    if ci is None:
        return f"{x:.{nd}f}"
    return f"{x:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
