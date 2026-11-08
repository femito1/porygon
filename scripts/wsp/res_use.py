"""Use, not just presence.

(a) Within-decision regressions: the net's move logit (centred over the decision's legal damaging moves) vs the
    move's true P(KO) / damage fraction, controlling for what the observation states directly
    (power × effectiveness × STAB, accuracy, priority, and in a richer control all 44 move numerics), as partial
    correlations with game-bootstrap 95% intervals; the search's visit shares as the reference "player that
    computes damage". Same for the switch decision vs the opponent's KO threat.
(b) Activation steering: add α·σ·u (u = unit probe direction, σ = std of the stream's projection on u) at one
    token after the layer where the target is decoded best; effect on that action's logit relative to the other
    legal actions; 8 random directions with the same α·σ as the baseline.
(c) Head-read alignment: cosine between the probe direction and the mean gradient of the action's logit w.r.t.
    the stream at that token.

    python scripts/wsp/res_use.py 500M
"""

import json
import sys

import numpy as np
import torch

from res_lib import MV_ACC, MV_BP, MV_CAT, MV_IMM, MV_MULT, MV_PRIO, MV_STAB, OUT, Intervention, boot_games, data, forward
from wsp_common import Net, load


def resid(y, C):
    C1 = np.concatenate([C, np.ones((len(C), 1))], 1)
    w, *_ = np.linalg.lstsq(C1, y, rcond=None)
    return y - C1 @ w


def partial(y, x, C):
    a, b = resid(y, C), resid(x, C)
    return float(np.corrcoef(a, b)[0, 1])


def r2_of(y, C):
    e = resid(y, C)
    return 1 - e.var() / y.var()


def centre(v, grp):
    s = np.bincount(grp, v)
    c = np.bincount(grp)
    return v - (s / c)[grp]


def main():
    tag = sys.argv[1]
    d = data()
    T = d["truth"]
    N = len(d["obs"])
    pol = load(tag)
    net = Net(pol)
    A = forward(net, d, keep=())
    logits = A["logits"].astype(np.float64)
    s = d["split"]
    r = np.arange(N)
    oa = np.maximum(d["own_act"], 0)
    mv = s["move_nums"][r, oa].astype(np.float64)  # N,4,44
    legal = d["mask"][:, 6:10] > 0
    damaging = mv[:, :, MV_CAT + 2] < 0.5
    eff = np.where(mv[:, :, MV_IMM] > 0.5, 0.0, 2.0 ** (2 * mv[:, :, MV_MULT]))
    pes = mv[:, :, MV_BP] * eff * (1 + 0.5 * mv[:, :, MV_STAB])
    sel = legal & damaging & np.isfinite(T["our_pko"])
    multi = sel.sum(1) >= 2
    sel &= multi[:, None]
    ii, jj = np.nonzero(sel)
    grp = np.unique(ii, return_inverse=True)[1]
    games = d["game"][ii]
    lg = logits[ii, 6 + jj]
    sh = np.log(d["shares"][ii, 6 + jj] + 0.01)
    pr_logged = np.log(d["probs"][ii, 6 + jj] + 1e-6)
    pko, frac = T["our_pko"][ii, jj].astype(np.float64), T["our_frac"][ii, jj].astype(np.float64)
    ctrlA = np.stack([pes[ii, jj], mv[ii, jj, MV_ACC], pes[ii, jj] * mv[ii, jj, MV_ACC], mv[ii, jj, MV_PRIO]], 1)
    ctrlB = np.concatenate([ctrlA, mv[ii, jj]], 1)
    Y = {"net_logit": centre(lg, grp), "search_logshare": centre(sh, grp)}
    X = {"pko": centre(pko, grp), "frac": centre(frac, grp)}
    CA = np.stack([centre(c, grp) for c in ctrlA.T], 1)
    CB = np.stack([centre(c, grp) for c in ctrlB.T], 1)
    res = {"a_moves": {"n_rows": int(len(ii)), "n_decisions": int(multi.sum())}}
    idx = np.arange(len(ii))
    for yn, y in Y.items():
        out = {}
        out["R2_ctrlA"] = float(r2_of(y, CA))
        out["R2_ctrlB"] = float(r2_of(y, CB))
        out["R2_ctrlB+truth"] = float(r2_of(y, np.concatenate([CB, X["pko"][:, None], X["frac"][:, None]], 1)))
        for xn, x in X.items():
            out[f"raw_corr_{xn}"] = float(np.corrcoef(y, x)[0, 1])
            for cn, C in (("ctrlA", CA), ("ctrlB", CB)):
                v = partial(y, x, C)
                ci = boot_games(lambda k: partial(y[k], x[k], C[k]), games, idx, n=200)
                out[f"partial_{xn}|{cn}"] = [v, ci.tolist()]
        res["a_moves"][yn] = out
        print(yn, json.dumps(out), flush=True)
    # top-1 sanity: how often does each pick the max-P(KO) move when one move KOs (pko>0.8) and another doesn't
    # (decisions where exactly one legal damaging move has pko > 0.8 and the max pko among the rest < 0.2)
    pk = np.where(sel, T["our_pko"], np.nan)
    ko_dec = []
    for i in np.flatnonzero(multi):
        v = pk[i]
        ok = np.isfinite(v)
        if (v[ok] > 0.8).sum() == 1 and np.sort(v[ok])[-2] < 0.2:
            ko_dec.append(i)
    ko_dec = np.array(ko_dec)
    if len(ko_dec):
        kj = np.nanargmax(pk[ko_dec], 1)
        p_net = A["probs"][ko_dec, 6 + kj]
        p_moves = A["probs"][ko_dec, 6:10].sum(1)
        res["a_ko_choice"] = {
            "n": int(len(ko_dec)),
            "net_p_ko_move": float(p_net.mean()),
            "net_p_ko_move_given_attacks": float((p_net / np.maximum(p_moves, 1e-9)).mean()),
            "net_argmax_is_ko_move": float((A["probs"][ko_dec].argmax(1) == 6 + kj).mean()),
            "search_share_ko_move": float(d["shares"][ko_dec, 6 + kj].mean()),
            "search_argmax_is_ko_move": float((d["shares"][ko_dec].argmax(1) == 6 + kj).mean()),
            "net_ko_move_is_max_pes": float((np.nanargmax(np.where(sel[ko_dec], pes[ko_dec], -1), 1) == kj).mean()),
        }
        print(res["a_ko_choice"], flush=True)
    # switch vs opponent's KO threat
    sw_ok = (d["mask"][:, :6].sum(1) > 0) & (d["mask"][:, 6:10].sum(1) > 0) & (d["own_act"] >= 0) & (d["opp_act"] >= 0)
    rows = np.flatnonzero(sw_ok)
    P = A["probs"][rows]
    y_net = np.log(P[:, :6].sum(1) + 1e-6) - np.log(P[:, 6:10].sum(1) + 1e-6)
    S = d["shares"][rows]
    y_s = np.log(S[:, :6].sum(1) + 0.01) - np.log(S[:, 6:10].sum(1) + 0.01)
    own_n = s["nums"][rows, d["own_act"][rows]].astype(np.float64)
    opp_n = s["nums"][rows, 6 + d["opp_act"][rows]].astype(np.float64)
    best_pes = np.where(legal[rows] & damaging[rows], pes[rows], 0).max(1)
    C = np.stack([own_n[:, 0], opp_n[:, 0], np.log((own_n[:, 13] + 1e-3) / (opp_n[:, 13] + 1e-3)), best_pes,
                  np.log2(np.maximum(T["stab_eff"][rows], 0.125)), d["split"]["field"][rows, -2].astype(float)], 1)
    g = d["game"][rows]
    sw = {"n": int(len(rows))}
    for yn, y in (("net_log_switch_odds", y_net), ("search_log_switch_odds", y_s)):
        o = {}
        for xn in ("thr_pko", "thr_frac", "thr_pko_vis", "first"):
            x = T[xn][rows].astype(np.float64)
            ok = np.isfinite(x) & (x != 0.5 if xn == "first" else True)
            k = np.flatnonzero(ok)
            v = partial(y[k], x[k], C[k])
            ci = boot_games(lambda kk: partial(y[kk], x[kk], C[kk]), g[k], k, n=200)
            o[f"partial_{xn}|ctrl"] = [v, ci.tolist()]
            o[f"raw_corr_{xn}"] = float(np.corrcoef(y[k], x[k])[0, 1])
        sw[yn] = o
        print(yn, o, flush=True)
    res["a_switch"] = sw

    # ---------------- (b) steering ----------------
    probes = json.loads((OUT / f"probes_{tag}.json").read_text())
    dirs = np.load(OUT / f"probe_dirs_{tag}.npz")
    stages = ["L-1.out", "L0.out", "L1.out", "L2.out"]
    layer_of = {"L-1.out": -1, "L0.out": 0, "L1.out": 1, "L2.out": 2}
    rng = np.random.default_rng(0)
    xt = torch.as_tensor(d["obs"], dtype=torch.float32)
    mt = torch.as_tensor(d["mask"])

    @torch.no_grad()
    def run(rows, iv):
        lg_, _ = net.forward(xt[rows], mt[rows], iv)
        return lg_.double().numpy()

    keepA = forward(net, d, keep=tuple(stages), idx=np.arange(N))
    steer = {}
    specs = [("move", "our_pko"), ("move", "our_frac"), ("state@field", "thr_pko"),
             ("state@own_active", "thr_pko"), ("bench", "bench_pko")]
    for fam, tgt in specs:
        best = max(stages[1:], key=lambda st: probes[fam][tgt][st]["v"])
        for st in sorted({best, "L2.out", "L0.out"}):
            li = layer_of[st]
            # rows and token per spec
            if fam == "move":
                cand = np.flatnonzero((legal & damaging).sum(1) >= 2)
                rows_ = rng.choice(cand, min(4000, len(cand)), replace=False)
                tok = np.array([13 + rng.choice(np.flatnonzero(legal[i] & damaging[i])) for i in rows_])
                act = tok - 13 + 6
            elif fam == "bench":
                cand = np.flatnonzero(np.isfinite(T["bench_frac"]).any(1) & (d["mask"][:, :6].sum(1) > 0))
                rows_ = rng.choice(cand, min(4000, len(cand)), replace=False)
                ks = [np.flatnonzero(np.isfinite(T["bench_frac"][i]) & (d["mask"][i, :6] > 0)) for i in rows_]
                keep_ = np.array([len(k) > 0 for k in ks])
                rows_ = rows_[keep_]
                tok = np.array([1 + rng.choice(ks[q]) for q in np.flatnonzero(keep_)])
                act = tok - 1
            else:
                rows_ = rng.choice(rows, min(4000, len(rows)), replace=False)
                tok = np.zeros(len(rows_), int) if fam == "state@field" else 1 + d["own_act"][rows_]
                act = None  # measure log switch odds
            w = dirs[f"{fam}|{tgt}|{st}|w"].astype(np.float64)
            dom = dirs.get(f"{fam}|{tgt}|{st}|dom")
            Xs = keepA[st][rows_, tok].astype(np.float64)
            dir_list = {"probe": w / np.linalg.norm(w)}
            if dom is not None:
                dir_list["dom"] = dom.astype(np.float64) / np.linalg.norm(dom)
            for q in range(8):
                v = rng.standard_normal(256)
                dir_list[f"rand{q}"] = v / np.linalg.norm(v)
            base = np.zeros((len(rows_), 10))
            for t in np.unique(tok):
                kk = tok == t
                base[kk] = run(rows_[kk], None)

            def effect(lgv):
                if act is not None:
                    m = d["mask"][rows_] > 0
                    oth = np.where(m, lgv, np.nan)
                    oth[np.arange(len(rows_)), act] = np.nan
                    rel = lgv[np.arange(len(rows_)), act] - np.nanmean(oth, 1)
                    pp = np.exp(lgv - lgv.max(1, keepdims=True)) * m
                    pp /= pp.sum(1, keepdims=True)
                    return rel, pp[np.arange(len(rows_)), act]
                m = d["mask"][rows_] > 0
                pp = np.exp(lgv - lgv.max(1, keepdims=True)) * m
                pp /= pp.sum(1, keepdims=True)
                return np.log(pp[:, :6].sum(1) + 1e-9) - np.log(pp[:, 6:10].sum(1) + 1e-9), pp[:, :6].sum(1)
            b_rel, b_p = effect(base)
            out = {"stage": st, "probe_R2_or_AUC": probes[fam][tgt][st]["v"], "n": int(len(rows_)),
                   "target_sd": float(np.nanstd(T[tgt][rows_] if T[tgt].ndim == 1 else
                                                (T[tgt][rows_, tok - 13] if fam == "move" else T[tgt][rows_, tok - 1])))}
            for dn, u in dir_list.items():
                sig = float((Xs @ u).std())
                for a in (-2.0, -1.0, 1.0, 2.0):
                    if dn.startswith("rand") and abs(a) != 2.0:
                        continue
                    vec = torch.as_tensor(a * sig * u, dtype=torch.float32)
                    lgv = np.zeros_like(base)
                    for t in np.unique(tok):
                        kk = tok == t
                        lgv[kk] = run(rows_[kk], Intervention(resid_add={(li, int(t)): vec}))
                    rel, p = effect(lgv)
                    out[f"{dn}@{a:+.0f}"] = {"d_rel_logit": float(np.mean(rel - b_rel)),
                                             "d_rel_logit_absmean": float(np.mean(np.abs(rel - b_rel))),
                                             "d_p": float(np.mean(p - b_p))}
                if dn == "probe":
                    # implied change in the target per unit α (probe slope): α·σ·|w|
                    out["target_shift_per_alpha"] = float(sig * np.linalg.norm(w))
            rnd = [out[f"rand{q}@+2"]["d_rel_logit_absmean"] for q in range(8)]
            out["rand@+2_absmean_mean"] = float(np.mean(rnd))
            steer[f"{fam}|{tgt}|{st}"] = out
            print(fam, tgt, st, {k: (v["d_rel_logit"] if isinstance(v, dict) else v) for k, v in out.items()
                                 if not k.startswith("rand") or k.startswith("rand@")}, flush=True)
    res["b_steering"] = steer

    # ---------------- (c) head-read alignment at L2.out ----------------
    al = {}
    cand = np.flatnonzero(legal.any(1))
    rows_ = rng.choice(cand, 3000, replace=False)
    for fam, tgt, tokf in (("move", "our_pko", None), ("move", "our_frac", None)):
        grads = []
        for i in rows_[:1500]:
            j = rng.choice(np.flatnonzero(legal[i]))
            cache = {}
            x1 = xt[i:i + 1]
            h = None
            # recompute with grad on L2.out
            with torch.enable_grad():
                hh = net.embed(x1)
                for li in range(3):
                    hh = net.layer(li, hh, None, None)
                hh = hh.detach().requires_grad_(True)
                lat = net.fe.norm(hh)
                lgv = net.policy_from_latent(lat, mt[i:i + 1])
                lgv[0, 6 + j].backward()
            grads.append(hh.grad[0, 13 + j].double().numpy())
        G = np.stack(grads)
        w = dirs[f"{fam}|{tgt}|L2.out|w"].astype(np.float64)
        mg = G.mean(0)
        cosm = float(mg @ w / np.linalg.norm(mg) / np.linalg.norm(w))
        cosi = float(np.mean(G @ w / np.linalg.norm(G, axis=1) / np.linalg.norm(w)))
        rnd = []
        for q in range(50):
            v = rng.standard_normal(256)
            rnd.append(abs(mg @ v / np.linalg.norm(mg) / np.linalg.norm(v)))
        al[f"{fam}|{tgt}"] = {"cos(mean grad, probe w)": cosm, "mean cos(grad_i, w)": cosi,
                              "random |cos| mean": float(np.mean(rnd))}
    res["c_head_alignment"] = al
    print(al)
    (OUT / f"use_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
