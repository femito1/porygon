"""Fit poke-engine's tunable leaf evaluation to engine self-play outcomes (`porygon.selfplay data`).

Model: P(side one wins | s) = sigmoid(w . f(s)), f = pe.eval_features (antisymmetric, so no intercept).
f[0] is the stock evaluate() / 100, the other features are corrections, L2-penalised towards 0, so where the data
says little the fit stays at a rescaled stock evaluation.

Target: the luck-adjusted outcome  y_t = z - sum_{u >= t} luck_u  (selfplay's docstring): E[y_t | s_t] = E[z | s_t]
but with the engine's chance (rolls, crits, misses, secondary effects) mostly removed. Loss: squared error of the
probability (Texel's), which stays a proper regression loss for targets outside [0, 1]. Because
E[(y - p)^2] = E[(y - m)^2] + E[(m - p)^2] with m = E[y | s] = E[z | s], held-out MSE on y ranks models exactly
like MSE on raw outcomes, with less noise; both are reported, plus log loss on z.

Splits are by team pair (both games of a pair are on the same side of every split).

Models:
  stock_engine  the stock evaluation at the engine's own leaf scale, sigmoid(0.0125 * eval) (no fit)
  stock_k       the stock evaluation with a fitted scale (Texel's K)
  full          all features, L2 on the corrections (strength by cross-validation over --l2)
  full_fe       as full plus a fixed effect per team pair (+alpha for swap 0, -alpha for swap 1): the weights are
                then identified only by variation within a team pair, which is what a search compares (all its
                leaves share the teams); the team pair's strength would cancel in the root-relative leaf value

Writes <out>/<model>.json (pe.set_eval configs: root-relative, scale 1, weights in logits) and report.json.

    python scripts/fit_eval.py runs/selfplay/r1 --out runs/evalfit/r1
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

N = 64


def load(dirs: list[str], max_games: int | None = None):
    from porygon.selfplay import read_games
    feats, y, z, game, pair, swap, turn = [], [], [], [], [], [], []
    g = 0
    for d in dirs:
        for rec in read_games(Path(d)):
            if "winner" not in rec or not rec.get("features"):
                continue
            zz = 1.0 if rec["winner"] == 1 else 0.0
            luck = np.asarray(rec["luck"], dtype=np.float64)
            after = np.cumsum(luck[::-1])[::-1]  # sum_{u >= t} luck_u
            for t, fd in enumerate(rec["features"]):
                row = np.zeros(N, dtype=np.float32)
                for k, v in fd.items():
                    row[int(k)] = v
                feats.append(row)
                y.append(zz - after[t])
                z.append(zz)
                game.append(g)
                pair.append(hash((d, rec["pair"])) & 0x7FFFFFFF)
                swap.append(rec["swap"])
                turn.append(t)
            g += 1
            if max_games and g >= max_games:
                break
    pair = np.asarray(pair)
    _, pair = np.unique(pair, return_inverse=True)
    return (np.stack(feats), np.asarray(y), np.asarray(z), np.asarray(game), pair, np.asarray(swap),
            np.asarray(turn), g)


def fit(X, y, l2: float, w0: np.ndarray | None = None, mask: np.ndarray | None = None,
        pair: np.ndarray | None = None, swap: np.ndarray | None = None, l2_fe: float = 1e-3, iters: int = 300):
    """Minimise mean (sigmoid(Xw [+ fe]) - y)^2 + l2 * |w[1:]|^2 with L-BFGS. `mask`: features allowed to move."""
    Xt, yt = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    w = torch.tensor(w0 if w0 is not None else np.eye(N)[0] * 1.25, dtype=torch.float64, requires_grad=True)
    m = torch.tensor(mask if mask is not None else np.ones(N), dtype=torch.float64)
    base = w.detach().clone()
    params = [w]
    if pair is not None:
        alpha = torch.zeros(int(pair.max()) + 1, dtype=torch.float64, requires_grad=True)
        pt = torch.tensor(pair)
        sg = torch.tensor(1.0 - 2.0 * swap, dtype=torch.float64)
        params.append(alpha)
    opt = torch.optim.LBFGS(params, lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        ww = base + m * (w - base)
        logit = Xt @ ww
        if pair is not None:
            logit = logit + sg * alpha[pt]
        loss = ((torch.sigmoid(logit) - yt) ** 2).mean() + l2 * (ww[1:] ** 2).sum()
        if pair is not None:
            loss = loss + l2_fe * (alpha ** 2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    ww = (base + m * (w - base)).detach().numpy()
    return ww


def metrics(X, y, z, w) -> dict[str, float]:
    p = 1 / (1 + np.exp(-(X @ w)))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {"mse_y": float(np.mean((p - y) ** 2)), "mse_z": float(np.mean((p - z) ** 2)),
            "logloss_z": float(-np.mean(z * np.log(pc) + (1 - z) * np.log(1 - pc)))}


def folds(pair: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(pair.max()) + 1)
    return perm[pair] % k


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--l2", type=float, nargs="+", default=[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-share", type=float, default=0.2, help="team pairs held out for the final report")
    p.add_argument("--max-games", type=int)
    p.add_argument("--target", choices=["luck", "raw"], default="luck",
                   help="fit to the luck-adjusted outcome (default) or the raw win / loss (ablation); evaluation "
                        "always reports both")
    p.add_argument("--scale", type=float, default=1.0, help="leaf scale written into the configs (search used 2)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import poke_engine as pe
    names = pe.eval_feature_names()
    assert len(names) == N

    X, y, z, game, pair, swap, turn, n_games = load(args.data, args.max_games)
    t = z if args.target == "raw" else y  # the training target; held-out scoring stays on y (less noisy, same ranking)
    print(f"{n_games} games, {len(X)} positions, {int(pair.max()) + 1} team pairs; "
          f"var(z) {z.var():.4f}, var(y) {y.var():.4f}", flush=True)
    test = folds(pair, round(1 / args.test_share), seed=1) == 0
    tr, te = ~test, test
    report: dict = {"target": args.target, "games": n_games, "positions": len(X), "var_z": float(z.var()), "var_y": float(y.var())}

    stock_mask = np.eye(N)[0]
    w_engine = np.eye(N)[0] * 1.25  # sigmoid(0.0125 * eval) = sigmoid(1.25 * f0)
    w_k = fit(X[tr], t[tr], 0.0, w_engine, mask=stock_mask)
    print(f"stock K: {w_k[0] / 100:.5f} per eval point (engine leaf uses 0.0125)", flush=True)

    # cross-validated L2 for the full model, on the training pairs
    fid = folds(pair[tr], args.folds)
    cv = {}
    for l2 in args.l2:
        losses = []
        for k in range(args.folds):
            a, b = fid != k, fid == k
            w = fit(X[tr][a], t[tr][a], l2, w_k)
            losses.append(metrics(X[tr][b], y[tr][b], z[tr][b], w)["mse_y"])
        cv[l2] = float(np.mean(losses))
        print(f"l2 {l2:g}: cv mse_y {cv[l2]:.6f}", flush=True)
    best = min(cv, key=cv.get)
    w_full = fit(X[tr], t[tr], best, w_k)
    w_fe = fit(X[tr], t[tr], best, w_k, pair=pair[tr], swap=swap[tr])
    report |= {"stock_k": float(w_k[0] / 100), "cv": {str(k): v for k, v in cv.items()}, "l2": best}

    models = {"stock_engine": w_engine, "stock_k": w_k, "full": w_full, "full_fe": w_fe}
    report["test"] = {m: metrics(X[te], y[te], z[te], w) for m, w in models.items()}
    # by game phase: early / mid / late decisions
    for lo, hi in ((0, 10), (10, 30), (30, 10**9)):
        sel = te & (turn >= lo) & (turn < hi)
        report["test"][f"turn_{lo}_{hi}"] = {m: metrics(X[sel], y[sel], z[sel], w)["mse_y"] for m, w in models.items()}
    for m, r in report["test"].items():
        print(m, json.dumps({k: round(v, 6) for k, v in r.items()}), flush=True)
    report["weights"] = {m: {n: round(float(v), 5) for n, v in zip(names, w)} for m, w in models.items()}
    sd = X[tr].std(0)
    report["weight_x_sd"] = {m: {n: round(float(v * s), 4) for n, v, s in zip(names, w, sd)}
                             for m, w in models.items() if m in ("full", "full_fe")}
    for m, w in models.items():
        (out / f"{m}.json").write_text(json.dumps({"weights": [float(v) for v in w], "scale": args.scale,
                                                   "root_weight": 1.0}))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("feature  full  full_fe  (logit weight; stock feature = eval/100)")
    for n, a, b in zip(names, w_full, w_fe):
        print(f"{n:16s} {a:9.4f} {b:9.4f}")


if __name__ == "__main__":
    main()
