"""Logit lens / tuned lens and residual-stream geometry.

Lens: the final LayerNorm + ActionHead (and the value MLP) applied to the residual stream after each layer
(L-1 = token inputs + pos). Tuned lens: one affine map per stage (shared by all 17 tokens) fitted by least squares
to predict L2.out from L{i}.out on 80% of games, evaluated on the other 20%. Agreement with the final policy
(top-1, KL), with the search (top-1 of the visit shares), value correlation and outcome AUC.
Geometry: norms per token type, per-sublayer write sizes, cross-token cosine similarity (raw and after removing
the per-position dataset mean), effective rank per token type.

    python scripts/wsp/res_lens.py 500M
"""

import json
import sys

import numpy as np
import torch

from res_lib import OUT, auc, boot_games, data, forward, gfolds
from wsp_common import Net, load

ST = ["L-1.out", "L0.out", "L1.out", "L2.out"]


def pr(X):
    X = X - X.mean(0)
    ev = np.linalg.eigvalsh(X.T @ X / len(X))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    tag = sys.argv[1]
    out_name = tag.rsplit("/", 1)[-1].replace(".zip", "") if "/" in tag else tag
    d = data()
    N = len(d["obs"])
    pol = load(tag)
    net = Net(pol)
    fe = pol.features_extractor
    keep = ST + [f"L{i}.attn" for i in range(3)] + [f"L{i}.mlp" for i in range(3)]
    A = forward(net, d, keep=keep)
    mask = torch.as_tensor(d["mask"])
    legal = d["mask"] > 0
    fin_logits = A["logits"]
    fin_p = A["probs"]
    fin_arg = fin_p.argmax(1)
    search_arg = d["shares"].argmax(1)
    phi = d["value"] - A["value"] if tag == "500M" else None  # logged value = V_500M + Φ
    games = d["game"]
    test = gfolds(games, 5, 7) == 0
    res = {"lens": {}, "tuned_lens": {}, "geometry": {}}

    @torch.no_grad()
    def head(h):
        out_p, out_v = [], []
        for s in range(0, N, 2048):
            lat = fe.norm(torch.as_tensor(h[s:s + 2048], dtype=torch.float32))
            lg = net.policy_from_latent(lat, mask[s:s + 2048])
            out_p.append(lg.softmax(-1).numpy())
            out_v.append(pol.value_net(pol.mlp_extractor.forward_critic(lat.flatten(1))).squeeze(-1).numpy())
        return np.concatenate(out_p), np.concatenate(out_v)

    def metrics(p, v, rows):
        arg = p[rows].argmax(1)
        kl = np.sum(np.where(legal[rows], fin_p[rows] * (np.log(fin_p[rows] + 1e-12) - np.log(p[rows] + 1e-12)), 0), 1)
        g = games[rows]
        top1 = (arg == fin_arg[rows]).astype(float)
        srch = (arg == search_arg[rows]).astype(float)
        out = {"agree_final": float(top1.mean()),
               "agree_final_ci": boot_games(np.mean, g, top1, n=200).tolist(),
               "KL_final||lens": float(kl.mean()),
               "agree_search": float(srch.mean()),
               "agree_search_ci": boot_games(np.mean, g, srch, n=200).tolist(),
               "agree_played": float((arg == d["action"][rows]).mean()),
               "p_on_final_argmax": float(p[rows, fin_arg[rows]].mean()),
               "entropy": float(-np.sum(np.where(legal[rows], p[rows] * np.log(p[rows] + 1e-12), 0), 1).mean()),
               "value_corr_final": float(np.corrcoef(v[rows], A["value"][rows])[0, 1])}
        if phi is not None:
            ok = np.isfinite(d["won"][rows]) & (d["won"][rows] != 0.5) & np.isfinite(phi[rows])
            out["value+phi_auc_won"] = float(auc(d["won"][rows][ok] > 0.5, (v[rows] + phi[rows])[ok]))
        # moves-vs-switch decision
        sw = p[rows, :6].sum(1) > 0.5
        out["agree_final_switch_vs_move"] = float((sw == (fin_p[rows, :6].sum(1) > 0.5)).mean())
        return out

    allrows = np.arange(N)
    ref = metrics(fin_p, A["value"], allrows)
    res["final"] = ref
    ref_t = metrics(fin_p, A["value"], np.flatnonzero(test))
    res["final_test"] = ref_t
    L2 = A["L2.out"].astype(np.float32)
    for s in ST:
        h = A[s].astype(np.float32)
        p, v = head(h)
        res["lens"][s] = metrics(p, v, allrows)
        # tuned lens: affine map h -> L2.out, shared across tokens, fit on train games
        if s != "L2.out":
            Xtr = h[~test].reshape(-1, h.shape[-1])
            Ytr = L2[~test].reshape(-1, h.shape[-1])
            pick = np.random.default_rng(0).choice(len(Xtr), min(len(Xtr), 250000), replace=False)
            Xtr, Ytr = Xtr[pick], Ytr[pick]
            Xa = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1).astype(np.float64)
            W = np.linalg.solve(Xa.T @ Xa + 1e-3 * np.eye(Xa.shape[1]) * len(Xa), Xa.T @ Ytr.astype(np.float64))
            hp = np.concatenate([h, np.ones(h.shape[:2] + (1,), np.float32)], -1) @ W.astype(np.float32)
            p2, v2 = head(hp)
            res["tuned_lens"][s] = metrics(p2, v2, np.flatnonzero(test))
            # per-token tuned-lens fit quality
            res["tuned_lens"][s]["fit_R2"] = float(1 - ((hp[test] - L2[test]) ** 2).sum()
                                                   / ((L2[test] - L2[test].mean((0, 1))) ** 2).sum())
        print(s, res["lens"][s]["agree_final"], res["lens"][s]["agree_search"],
              res["tuned_lens"].get(s, {}).get("agree_final"), flush=True)
    # ---- geometry
    nums = d["split"]["nums"]
    r = np.arange(N)
    tt = {"field": [np.zeros(N, int)]}
    alive_own = nums[:, :6, 1] < 0.5
    tt_rows = {
        "field": (r, np.zeros(N, int)),
        "own_active": (r[d["own_act"] >= 0], 1 + d["own_act"][d["own_act"] >= 0]),
        "opp_active": (r[d["opp_act"] >= 0], 7 + d["opp_act"][d["opp_act"] >= 0]),
        "move(legal)": np.nonzero(d["mask"][:, 6:10] > 0),
    }
    tt_rows["move(legal)"] = (tt_rows["move(legal)"][0], 13 + tt_rows["move(legal)"][1])
    ob = np.nonzero(alive_own & (nums[:, :6, 115] < 0.5))
    tt_rows["own_bench(alive)"] = (ob[0], 1 + ob[1])
    seen = (nums[:, 6:, -1] < 0.5) & (nums[:, 6:, 1] < 0.5) & (nums[:, 6:, 123] < 0.5)
    os_ = np.nonzero(seen)
    tt_rows["opp_bench(seen,alive)"] = (os_[0], 7 + os_[1])
    un = np.nonzero(nums[:, 6:, -1] > 0.5)
    tt_rows["opp_unseen"] = (un[0], 7 + un[1])
    fo = np.nonzero(nums[:, :6, 1] > 0.5)
    tt_rows["own_fainted"] = (fo[0], 1 + fo[1])
    geo = {}
    for name, (rows, pos) in tt_rows.items():
        g = {"n": int(len(rows))}
        for s in ST:
            g[f"|{s}|"] = float(np.linalg.norm(A[s][rows, pos].astype(np.float32), axis=1).mean())
        for i in range(3):
            g[f"|L{i}.attn|"] = float(np.linalg.norm(A[f"L{i}.attn"][rows, pos].astype(np.float32), axis=1).mean())
            g[f"|L{i}.mlp|"] = float(np.linalg.norm(A[f"L{i}.mlp"][rows, pos].astype(np.float32), axis=1).mean())
        sub = np.random.default_rng(0).choice(len(rows), min(len(rows), 20000), replace=False)
        for s in ST:
            g[f"PR {s}"] = pr(A[s][rows[sub], pos[sub]].astype(np.float64))
        geo[name] = g
    res["geometry"]["per_token_type"] = geo
    # cross-token cosine similarity within a decision
    mean_pos = {s: A[s].astype(np.float32).mean(0) for s in ST}

    def pair_cos(s, slots_mask, offset, centered):
        X = A[s][:, offset:offset + 6].astype(np.float32)
        if centered:
            X = X - mean_pos[s][offset:offset + 6]
        Xn = X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)
        C = np.einsum("nid,njd->nij", Xn, Xn)
        m = slots_mask[:, :, None] & slots_mask[:, None, :] & ~np.eye(6, dtype=bool)[None]
        return float(C[m].mean())
    bench_mask = alive_own & (nums[:, :6, 115] < 0.5)
    opp_mask = seen
    cos = {}
    for s in ST:
        cos[s] = {"own_bench_raw": pair_cos(s, bench_mask, 1, False),
                  "own_bench_centered": pair_cos(s, bench_mask, 1, True),
                  "opp_seen_bench_raw": pair_cos(s, opp_mask, 7, False),
                  "opp_seen_bench_centered": pair_cos(s, opp_mask, 7, True)}
        # within-decision spread vs total spread for own bench tokens
        X = A[s][:, 1:7].astype(np.float32)
        w, tot = [], []
        for n_ in range(0, N, 4096):
            Xb, mb = X[n_:n_ + 4096], bench_mask[n_:n_ + 4096]
            cnt = mb.sum(1)
            ok = cnt >= 2
            mu = (Xb * mb[..., None]).sum(1) / np.maximum(cnt, 1)[:, None]
            w.append((((Xb - mu[:, None]) ** 2).sum(-1) * mb)[ok].sum() / max(1, 1))
            tot.append(cnt[ok].sum())
        allX = X[bench_mask]
        within = np.sum(w) / np.sum(tot)
        total = ((allX - allX.mean(0)) ** 2).sum(-1).mean()
        cos[s]["own_bench_within_decision_var_frac"] = float(within / total)
        # move tokens likewise
        Xm = A[s][:, 13:17].astype(np.float32)
        mm = d["mask"][:, 6:10] > 0
        cnt = mm.sum(1)
        ok = cnt >= 2
        mu = (Xm * mm[..., None]).sum(1) / np.maximum(cnt, 1)[:, None]
        within = ((((Xm - mu[:, None]) ** 2).sum(-1) * mm)[ok]).sum() / cnt[ok].sum()
        allM = Xm[mm]
        cos[s]["move_within_decision_var_frac"] = float(within / ((allM - allM.mean(0)) ** 2).sum(-1).mean())
    res["geometry"]["cross_token"] = cos
    (OUT / f"lens_{out_name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["geometry"]["cross_token"], indent=0))


if __name__ == "__main__":
    main()
