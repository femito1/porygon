"""Offline checks of porygon.lookahead2 on logged positions (F2 records replayed by netsearch_study's replayer); no
games are played.

  bench    per position: the net's T-1 probabilities / P(win) and the lookahead's statistics for runs A (K worlds),
           optional retest B (K fresh worlds) and judge R (K_R fresh worlds): rows, x, Q, value, leaves, timing.
           Positions = a study rows file (--rows: the same (record, side, decision) as netsearch_study) or sampled.
  outcome  Step 2: at consecutive decision pairs (t, t+1) of held-out records, the net's P(win) and poke-engine's
           stock evaluate() (mean over K determinizations, porygon.search.convert) with the game's result.
  analyze-bench / analyze-outcome   summaries of those JSONL files (pure; laptop or cluster).

    cd $C && PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$C/src NETV2_SRC=$C/frozen/netv2-fba02fb-20261110/src \\
      $PY scripts/fastenv/lookahead2_study.py bench --records "$SL" --source 6771931 --rows ROWS --net NET \\
        --out OUT.jsonl --worlds 4 --retest --ref 16 --workers 11 --concurrency 3
    ... lookahead2_study.py outcome --records "$HU" --net NET --out OUT.jsonl --pairs 1000 --per-record 2
    ... lookahead2_study.py analyze-bench OUT.jsonl [--json S.json];  analyze-outcome OUT.jsonl [--json S.json]
"""
import argparse
import asyncio
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
T_NET = 0.5
DELTAS = [-math.inf, 0.0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05,
          0.06, 0.075, 0.1, 0.15, math.inf]
PRECISION = 0.9  # the margin rule's target: share of overrides a fresh, 4x larger lookahead (judge R) confirms


def sharpen(p: np.ndarray, t: float) -> np.ndarray:
    z = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)) / t, -np.inf)
    e = np.exp(z - z[np.isfinite(z)].max())
    return e / e.sum()


# ---------------------------------------------------------------- bench

def run_row(res, t: float) -> dict:
    return {"rows": res.rows, "x": [round(float(v), 5) for v in res.x], "q": [round(float(v), 6) for v in res.q],
            "value": round(res.value, 6), "expl": round(res.expl, 6), "worlds": res.worlds,
            "failed_worlds": res.failed_worlds, "leaves": res.leaves, "terminal": res.terminal, "errors": res.errors,
            "cols": round(res.cols, 2), "t": round(t, 3), "t_expand": round(res.t_expand, 3),
            "t_score": round(res.t_score, 3), "t_fwd": round(res.t_fwd, 3), "t_solve": round(res.t_solve, 3),
            "cpu_expand": round(res.cpu_expand, 3)}


async def bench(a) -> int:
    import netsearch_study as ST
    NS, S, R = ST.setup(a.format)
    import torch
    from porygon import lookahead2 as LA
    torch.set_num_threads(1)
    net = ST.load_net(a.net, "cpu")
    leaf = ST.load_net(a.net, a.leaf_device) if a.leaf_device != "cpu" else net
    pool = LA.LookaheadPool(a.workers)
    gen = 9
    if a.rows:
        rows = [json.loads(x) for x in open(a.rows) if x.strip()]
        want: dict[tuple, set] = {}
        for r in rows:
            want.setdefault((r["rec"], r.get("side")), set()).add(r["idx"])
        recs = [r for r in ST.records(a.records, a.source, a.seed) if (r["id"], r.get("side")) in want]
        chosen = {(r["id"], r.get("side")): want[(r["id"], r.get("side"))] for r in recs}
    else:
        recs = ST.records(a.records, a.source, a.seed)[:max(1, 3 * a.positions // max(a.per_record, 1))]
        chosen = {(r["id"], r.get("side")): ST.targets(r, a.per_record, a.seed) for r in recs}
    out = open(a.out, "a")
    sem = asyncio.Semaphore(a.concurrency)
    done = {"n": 0, "fail": 0}
    t_start = time.time()
    runs = [("A", a.worlds)] + ([("B", a.worlds)] if a.retest else []) + ([("R", a.ref)] if a.ref else [])

    async def one(rec):
        async with sem:
            key = (rec["id"], rec.get("side"))
            async for d, battle, first, logged in ST.replay_targets(rec, R, NS, gen, chosen[key]):
                if done["n"] >= a.positions:
                    return
                probs, mask, win = NS.net_forward(net, battle)
                if mask.sum() < 2:
                    continue
                row = {"rec": rec["id"], "side": rec.get("side"), "idx": d.msg, "turn": d.turn, "kind": d.kind,
                       "n_legal": int(mask.sum()), "result": rec.get("result"), "rating": rec.get("rating"),
                       "mask": (mask > 0).astype(int).tolist(), "probs": [round(float(p), 6) for p in probs],
                       "win": round(win, 5), "action": (logged or {}).get("a")}
                try:
                    for name, k in runs:
                        rng = random.Random(f"{a.seed}:{rec['id']}:{rec.get('side')}:{d.msg}:{name}")
                        t0 = time.perf_counter()
                        res = await LA.lookahead(pool, leaf, battle, first, rng, mask, worlds=k,
                                                 min_prob=a.min_prob, iters=a.iters, batch=a.batch)
                        row[name] = run_row(res, time.perf_counter() - t0)
                except Exception as e:
                    done["fail"] += 1
                    row["error"] = f"{type(e).__name__}: {e}"
                out.write(json.dumps(row) + "\n")
                out.flush()
                done["n"] += 1
                if done["n"] % 25 == 0:
                    print(f"{done['n']} positions ({done['fail']} failed), {time.time() - t_start:.0f}s", flush=True)

    await asyncio.gather(*(one(r) for r in recs))
    out.close()
    await pool.close()
    print(f"done: {done['n']} positions ({done['fail']} failed) in {time.time() - t_start:.0f}s -> {a.out}")
    return 0 if done["n"] and done["fail"] <= 0.02 * done["n"] else 1


def _q_of(run: dict, a: int) -> float:
    return run["q"][run["rows"].index(a)] if a in run["rows"] else math.nan


def _pick(run: dict) -> int:
    return run["rows"][int(np.argmax(run["x"]))]


def analyze_bench(path: str) -> dict:
    rows = [json.loads(x) for x in open(path) if x.strip()]
    ok = [r for r in rows if "A" in r and "error" not in r]
    out = {"positions": len(rows), "usable": len(ok), "failed": len(rows) - len(ok)}
    q = lambda xs, p: float(np.quantile(xs, p)) if len(xs) else math.nan  # noqa: E731
    for name in ("A", "B", "R"):
        rs = [r[name] for r in ok if name in r]
        if not rs:
            continue
        lv = [x["leaves"] for x in rs]
        tt = [x["t"] for x in rs]
        cpu = [x["cpu_expand"] for x in rs]
        out[f"run_{name}"] = {"n": len(rs), "worlds": rs[0]["worlds"] if rs else None,
                              "leaves_median": q(lv, .5), "leaves_q10": q(lv, .1), "leaves_q90": q(lv, .9),
                              "leaves_max": max(lv), "seconds_median": q(tt, .5), "seconds_q90": q(tt, .9),
                              "seconds_max": max(tt), "cpu_median": q(cpu, .5), "cpu_q90": q(cpu, .9),
                              "t_score_median": q([x["t_score"] for x in rs], .5),
                              "t_fwd_median": q([x.get("t_fwd", math.nan) for x in rs], .5),
                              "t_expand_median": q([x["t_expand"] for x in rs], .5),
                              "t_solve_median": q([x["t_solve"] for x in rs], .5),
                              "expl_median": q([x["expl"] for x in rs], .5), "expl_q90": q([x["expl"] for x in rs], .9),
                              "failed_worlds": int(sum(x["failed_worlds"] for x in rs)),
                              "errors": int(sum(x["errors"] for x in rs)), "cols_mean": float(np.mean([x["cols"] for x in rs])),
                              "rows_mean": float(np.mean([len(x["rows"]) for x in rs]))}
    # agreement and the T0.5 draw
    agree_argmax, stab, draws = [], [], []
    for r in ok:
        p = np.array(r["probs"])
        A = r["A"]
        a_l = _pick(A)
        agree_argmax.append(a_l == int(np.argmax(p)))
        if "B" in r:
            stab.append(a_l == _pick(r["B"]))
        draws.append(sharpen(p, T_NET))
    out["agree_pick_eq_net_argmax"] = float(np.mean(agree_argmax)) if agree_argmax else None
    out["agree_pick_A_eq_B"] = float(np.mean(stab)) if stab else None
    # gaps D(a) = Q_A(a_L) - Q_A(a) over the T0.5 draw; override and judge curves per margin
    curves = []
    for delta in DELTAS:
        ov = vB = vR = vR_raw = vR_pure = vB_raw = vB_pure = 0.0
        conf = {"B": [0.0, 0.0], "R": [0.0, 0.0]}  # override weight confirmed by the judge (its Q(a_L) > Q(a_net)), total
        nB = nR = 0
        undefined = 0.0
        for r, w in zip(ok, draws):
            A = r["A"]
            a_l = _pick(A)
            qa_l = _q_of(A, a_l)
            for a in np.flatnonzero(w > 0):
                pa = float(w[a])
                qa = _q_of(A, int(a))
                if not math.isfinite(qa):
                    undefined += pa
                    over = delta == -math.inf
                else:
                    over = a_l != a and (delta == -math.inf or (qa_l - qa) > delta)
                play = a_l if over else int(a)
                ov += pa * over
                for name, acc in (("B", "B"), ("R", "R")):
                    if name in r:
                        J = r[name]
                        qj_play, qj_net, qj_pure = _q_of(J, play), _q_of(J, int(a)), _q_of(J, a_l)
                        if all(math.isfinite(v) for v in (qj_play, qj_net, qj_pure)):
                            if over and a_l != a:
                                conf[name][0] += pa * (qj_pure > qj_net)
                                conf[name][1] += pa
                            if name == "B":
                                vB += pa * (qj_play - qj_net)
                                vB_pure += pa * (qj_play - qj_pure)
                            else:
                                vR += pa * (qj_play - qj_net)
                                vR_pure += pa * (qj_play - qj_pure)
        nB = sum(1 for r in ok if "B" in r)
        nR = sum(1 for r in ok if "R" in r)
        curves.append({"delta": delta, "override": ov / max(len(ok), 1), "undefined_q": undefined / max(len(ok), 1),
                       "judgeB_gain_vs_raw": vB / nB if nB else None, "judgeB_vs_pure": vB_pure / nB if nB else None,
                       "judgeR_gain_vs_raw": vR / nR if nR else None, "judgeR_vs_pure": vR_pure / nR if nR else None,
                       "precision_B": conf["B"][0] / conf["B"][1] if conf["B"][1] else None,
                       "precision_R": conf["R"][0] / conf["R"][1] if conf["R"][1] else None})
    out["curves"] = curves
    # the pre-registered margin rule: the smallest grid margin >= 0 from which on (every larger finite grid margin
    # too) at least PRECISION of the overrides are confirmed by judge R; judge B if there is no R
    judge = "R" if any(c["precision_R"] is not None for c in curves) else "B"
    fin = [c for c in curves if math.isfinite(c["delta"]) and c["delta"] >= 0]
    rule = None
    for i, c in enumerate(fin):
        tail = [x[f"precision_{judge}"] for x in fin[i:] if x[f"precision_{judge}"] is not None]
        if tail and all(t >= PRECISION for t in tail):
            rule = c
            break
    out["margin_rule"] = {"judge": judge, "precision": PRECISION, "delta": rule["delta"] if rule else None,
                          "override": rule["override"] if rule else None,
                          f"judge{judge}_gain_vs_raw": rule[f"judge{judge}_gain_vs_raw"] if rule else None,
                          "rule": f"smallest grid margin >= 0 from which on >= {PRECISION:.0%} of the overrides are "
                                  f"confirmed by judge {judge} (its Q of the lookahead's pick > its Q of the net's draw)"}
    # selected-gap inflation: D_A at A's pick vs the same pair of actions judged by B (and R)
    for name in ("B", "R"):
        diffs, ws = [], []
        for r, w in zip(ok, draws):
            if name not in r:
                continue
            A, J = r["A"], r[name]
            a_l = _pick(A)
            for a in np.flatnonzero(w > 0):
                if int(a) == a_l:
                    continue
                dA = _q_of(A, a_l) - _q_of(A, int(a))
                dJ = _q_of(J, a_l) - _q_of(J, int(a))
                if math.isfinite(dA) and math.isfinite(dJ):
                    diffs.append(dA - dJ)
                    ws.append(float(w[a]))
        if diffs:
            d, wt = np.array(diffs), np.array(ws)
            m = float((d * wt).sum() / wt.sum())
            out[f"inflation_vs_{name}"] = {"mean": m, "n_pairs": len(d), "sd": float(np.sqrt(((d - m) ** 2 * wt).sum() / wt.sum()))}
    # the distribution of D for the net's T0.5 draws (weighted)
    gaps, gw = [], []
    for r, w in zip(ok, draws):
        A = r["A"]
        a_l = _pick(A)
        for a in np.flatnonzero(w > 0):
            dA = _q_of(A, a_l) - _q_of(A, int(a))
            if math.isfinite(dA):
                gaps.append(dA)
                gw.append(float(w[a]))
    if gaps:
        g, wt = np.array(gaps), np.array(gw)
        order = np.argsort(g)
        cw = np.cumsum(wt[order]) / wt.sum()
        out["gap_T05_quantiles"] = {f"q{p}": float(g[order][np.searchsorted(cw, p / 100)]) for p in (50, 75, 90, 95, 99)}
        out["gap_T05_share_zero"] = float(wt[g <= 1e-9].sum() / wt.sum())
    return out


def format_bench(s: dict) -> str:
    lines = [f"positions {s['positions']}, usable {s['usable']}, failed {s['failed']}"]
    for name in ("A", "B", "R"):
        if f"run_{name}" in s:
            r = s[f"run_{name}"]
            lines.append(f"  run {name} (worlds {r['worlds']}): leaves median {r['leaves_median']:.0f} [q10 {r['leaves_q10']:.0f},"
                         f" q90 {r['leaves_q90']:.0f}, max {r['leaves_max']}], wall s median {r['seconds_median']:.2f} q90 "
                         f"{r['seconds_q90']:.2f} max {r['seconds_max']:.2f}; worker cpu s median {r['cpu_median']:.2f} q90 "
                         f"{r['cpu_q90']:.2f}; score {r['t_score_median']:.3f} s (forward {r['t_fwd_median']:.3f}), solve "
                         f"{r['t_solve_median']:.3f} s; rows "
                         f"{r['rows_mean']:.1f} x cols {r['cols_mean']:.1f}; expl median {r['expl_median']:.4f}; failed "
                         f"worlds {r['failed_worlds']}, engine errors {r['errors']}")
    lines.append(f"  lookahead pick = net argmax {s['agree_pick_eq_net_argmax']:.3f}; A pick = B pick "
                 f"{s['agree_pick_A_eq_B'] if s['agree_pick_A_eq_B'] is None else round(s['agree_pick_A_eq_B'], 3)}")
    for name in ("B", "R"):
        if f"inflation_vs_{name}" in s:
            x = s[f"inflation_vs_{name}"]
            lines.append(f"  selected-gap inflation vs {name}: {x['mean']:+.4f} (sd {x['sd']:.4f}, {x['n_pairs']} pairs)")
    if "gap_T05_quantiles" in s:
        lines.append(f"  D over the net's T0.5 draws: share 0 {s['gap_T05_share_zero']:.3f}; " +
                     ", ".join(f"{k} {v:.4f}" for k, v in s["gap_T05_quantiles"].items()))
    lines.append("  margin | override | judge B: gain vs raw, vs pure, precision | judge R: gain vs raw, vs pure, precision"
                 "  (P(win) per decision)")
    f = lambda v: "   -   " if v is None else f"{v:+.5f}"  # noqa: E731
    g = lambda v: "  -  " if v is None else f"{v:.3f}"  # noqa: E731
    for c in s["curves"]:
        lines.append(f"  {c['delta']:>7} | {c['override']:.3f} | {f(c['judgeB_gain_vs_raw'])}, {f(c['judgeB_vs_pure'])}, "
                     f"{g(c['precision_B'])} | {f(c['judgeR_gain_vs_raw'])}, {f(c['judgeR_vs_pure'])}, {g(c['precision_R'])}")
    m = s["margin_rule"]
    lines.append(f"  MARGIN RULE ({m['rule']}): delta = {m['delta']} (override {m['override']})")
    return "\n".join(lines)


# ---------------------------------------------------------------- outcome (Step 2)

def pair_targets(rec: dict, per_record: int, seed: int) -> list[tuple[int, int]]:
    """Up to per_record pairs of consecutive decisions with >= 2 legal actions (v3_gates.pair_targets, >= 2 legal)."""
    seq = sorted(d["idx"] for d in rec["decisions"] if d.get("n_legal", 0) >= 2)
    cands = list(range(len(seq) - 1))
    h = int(hashlib.sha1(f"{seed}:{rec['id']}:{rec.get('side')}".encode()).hexdigest()[:12], 16)
    ks = random.Random(h).sample(cands, min(per_record, len(cands)))
    return [(seq[k], seq[k + 1]) for k in sorted(ks)]


async def outcome(a) -> int:
    import netsearch_study as ST
    NS, S, R = ST.setup(a.format)
    import poke_engine as pe
    import torch
    torch.set_num_threads(1)
    pe.set_eval(None, 0.0125, 1.0)  # the stock evaluate(), as the gen9 benchmark search scores leaves
    net = ST.load_net(a.net, a.device)
    recs = ST.records(a.records, a.source, a.seed)
    out = open(a.out, "w")
    n = fails = 0
    t0 = time.time()
    for rec in recs:
        if n >= a.pairs:
            break
        pairs = pair_targets(rec, a.per_record, a.seed)
        want = {t for p in pairs for t in p}
        got = {}
        try:
            async for d, battle, first, logged in ST.replay_targets(rec, R, NS, 9, want):
                probs, mask, win = NS.net_forward(net, battle)
                hist = S.History(battle)
                rng = random.Random(f"{a.seed}:{rec['id']}:{d.msg}")
                evals = []
                for _ in range(a.dets):
                    conv = S.convert(battle, rng, first, hist)
                    evals.append(float(pe.evaluate(pe.State.from_string(conv.state_str))))
                got[d.msg] = {"turn": d.turn, "win": win, "eval": float(np.mean(evals)), "eval_sd": float(np.std(evals)),
                              "n_legal": int(mask.sum())}
        except Exception as e:  # a record that fails to replay or convert
            fails += 1
            print(f"record {rec['id']}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for t, t1 in pairs:
            if t in got and t1 in got and n < a.pairs:
                out.write(json.dumps({"rec": rec["id"], "side": rec.get("side"), "result": rec.get("result"),
                                      "rating": rec.get("rating"), "t": t, "t1": t1, "a": got[t], "b": got[t1]}) + "\n")
                n += 1
        if n and n % 200 == 0:
            print(f"{n} pairs, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    out.close()
    print(f"done: {n} pairs ({fails} failed records) in {time.time() - t0:.0f}s -> {a.out}")
    return 0 if n >= 0.9 * a.pairs else 1


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share their mean rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted 1/2."""
    r = _rankdata(np.asarray(score, float))
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _logit_fit(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    """Logistic regression (Newton) with an intercept; X [n, d]."""
    Z = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ b))
        g = Z.T @ (y - p)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-9 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    return b


def _predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ b)))


def analyze_outcome(path: str, boot: int = 1000, seed: int = 5) -> dict:
    rows = [json.loads(x) for x in open(path) if x.strip()]
    rows = [r for r in rows if r.get("result") in (1, -1)]
    games = np.array([int(hashlib.sha1(str(r["rec"]).encode()).hexdigest()[:8], 16) for r in rows])  # both sides of a battle: one cluster
    y = np.array([(r["result"] + 1) / 2 for r in rows], float)
    # positions: both members of every pair (t and t+1), clustered by game for the intervals
    pn = np.concatenate([[r["a"]["win"] for r in rows], [r["b"]["win"] for r in rows]])
    ev = np.concatenate([[r["a"]["eval"] for r in rows], [r["b"]["eval"] for r in rows]])
    turn = np.concatenate([[r["a"]["turn"] for r in rows], [r["b"]["turn"] for r in rows]])
    yy = np.concatenate([y, y])
    gg = np.concatenate([games, games])
    pe_search = 1 / (1 + np.exp(-0.0125 * ev))  # sigmoid(0.0125 (eval - 0)): relative to an even position
    # the hand eval's best affine calibration, 2-fold cross-fitted by game
    fold = (np.abs(gg) % 2).astype(int)
    pe_fit = np.zeros_like(pe_search)
    for f in (0, 1):
        b = _logit_fit(ev[fold != f, None] / 100, yy[fold != f])
        pe_fit[fold == f] = _predict(b, ev[fold == f, None] / 100)
    pn_fit = np.zeros_like(pn)  # the net recalibrated the same way (fair comparison of log-losses)
    ln = np.log(np.clip(pn, 1e-6, 1 - 1e-6) / np.clip(1 - pn, 1e-6, 1))
    for f in (0, 1):
        b = _logit_fit(ln[fold != f, None], yy[fold != f])
        pn_fit[fold == f] = _predict(b, ln[fold == f, None])

    def metrics(idx: np.ndarray) -> dict:
        return {"auc_net": _auc(pn[idx], yy[idx]), "auc_eval": _auc(ev[idx], yy[idx]),
                "ll_net": _logloss(pn[idx], yy[idx]), "ll_net_cal": _logloss(pn_fit[idx], yy[idx]),
                "ll_eval_search_scale": _logloss(pe_search[idx], yy[idx]), "ll_eval_cal": _logloss(pe_fit[idx], yy[idx]),
                "brier_net": float(((pn[idx] - yy[idx]) ** 2).mean()), "brier_eval_cal": float(((pe_fit[idx] - yy[idx]) ** 2).mean()),
                "ll_const": _logloss(np.full(idx.sum() if idx.dtype == bool else len(idx), yy[idx].mean()), yy[idx])}

    allpos = np.ones(len(yy), bool)
    out = {"pairs": len(rows), "positions": int(len(yy)), "games": int(len(set(games.tolist()))),
           "result_mean": float(y.mean()), "all": metrics(allpos)}
    # game-clustered bootstrap of the net - eval differences
    rng = np.random.default_rng(seed)
    ug = np.array(sorted(set(gg.tolist())))
    members = {g: np.flatnonzero(gg == g) for g in ug}
    d_auc, d_ll = [], []
    for _ in range(boot):
        pick = rng.choice(ug, len(ug))
        idx = np.concatenate([members[g] for g in pick])
        d_auc.append(_auc(pn[idx], yy[idx]) - _auc(ev[idx], yy[idx]))
        d_ll.append(_logloss(pe_fit[idx], yy[idx]) - _logloss(pn_fit[idx], yy[idx]))
    out["boot"] = {"auc_net_minus_eval": [float(np.quantile(d_auc, q)) for q in (0.025, 0.5, 0.975)],
                   "ll_eval_cal_minus_net_cal": [float(np.quantile(d_ll, q)) for q in (0.025, 0.5, 0.975)], "n": boot}
    out["by_turn"] = {}
    for lo, hi in ((0, 5), (6, 15), (16, 30), (31, 1000)):
        idx = (turn >= lo) & (turn <= hi)
        if idx.sum() >= 50 and 0 < yy[idx].mean() < 1:
            out["by_turn"][f"{lo}-{hi}"] = {"n": int(idx.sum()), **metrics(idx)}
    # calibration (deciles of the prediction)
    def calib(p: np.ndarray) -> list:
        qs = np.quantile(p, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, 9)
        return [{"bin": int(k), "mean_pred": float(p[b == k].mean()), "freq": float(yy[b == k].mean()), "n": int((b == k).sum())}
                for k in range(10) if (b == k).any()]
    out["calibration_net"] = calib(pn)
    out["calibration_eval_search_scale"] = calib(pe_search)
    # residual information: does one predictor add to the other? (logistic regression on both, z of each slope)
    X = np.column_stack([ln, ev / 100])
    b = _logit_fit(X, yy)
    p = _predict(b, X)
    Z = np.column_stack([np.ones(len(X)), X])
    cov = np.linalg.inv((Z * (p * (1 - p))[:, None]).T @ Z)
    se = np.sqrt(np.diag(cov))
    out["joint_logit"] = {"coef_net_logit": float(b[1]), "z_net": float(b[1] / se[1]), "coef_eval_per100": float(b[2]),
                          "z_eval": float(b[2] / se[2]), "note": "naive SEs (positions within a game are correlated)"}
    # the leaf question (one turn): with the root's net value fixed, does the scorer's change t -> t+1 carry outcome
    # information? logit(P_net(t)) + delta_scorer, delta in the scorer's own units; z and log-loss gain per position
    la = np.log(np.clip(np.array([r["a"]["win"] for r in rows]), 1e-6, 1 - 1e-6) / np.clip(1 - np.array([r["a"]["win"] for r in rows]), 1e-6, 1))
    lb = np.log(np.clip(np.array([r["b"]["win"] for r in rows]), 1e-6, 1 - 1e-6) / np.clip(1 - np.array([r["b"]["win"] for r in rows]), 1e-6, 1))
    dev = (np.array([r["b"]["eval"] for r in rows]) - np.array([r["a"]["eval"] for r in rows])) / 100
    base = _logit_fit(la[:, None], y)
    ll0 = _logloss(_predict(base, la[:, None]), y)
    leaf = {"pairs": len(rows), "ll_root_only": ll0}
    for name, dlt in (("net_logit", lb - la), ("eval_per100", dev),
                      ("eval_search_scale", 1 / (1 + np.exp(-1.25 * dev)) - 0.5)):
        X2 = np.column_stack([la, dlt])
        b2 = _logit_fit(X2, y)
        p2 = _predict(b2, X2)
        Z2 = np.column_stack([np.ones(len(X2)), X2])
        se2 = np.sqrt(np.diag(np.linalg.inv((Z2 * (p2 * (1 - p2))[:, None]).T @ Z2)))
        leaf[name] = {"coef": float(b2[2]), "z": float(b2[2] / se2[2]), "ll": _logloss(p2, y),
                      "ll_gain": ll0 - _logloss(p2, y), "auc_delta_alone": _auc(dlt, y)}
    out["one_turn_change"] = leaf
    return out


def format_outcome(s: dict) -> str:
    a = s["all"]
    lines = [f"{s['pairs']} decision pairs (t, t+1), {s['positions']} positions, {s['games']} games; result mean {s['result_mean']:.3f}",
             f"  AUC: net {a['auc_net']:.4f}, eval {a['auc_eval']:.4f}  (boot net - eval {s['boot']['auc_net_minus_eval']})",
             f"  log-loss: const {a['ll_const']:.4f} | net raw {a['ll_net']:.4f}, net calibrated {a['ll_net_cal']:.4f} | "
             f"eval at the search's scale sigmoid(0.0125 eval) {a['ll_eval_search_scale']:.4f}, eval calibrated {a['ll_eval_cal']:.4f}"
             f"  (boot eval_cal - net_cal {s['boot']['ll_eval_cal_minus_net_cal']})",
             f"  Brier: net {a['brier_net']:.4f}, eval calibrated {a['brier_eval_cal']:.4f}"]
    for k, v in s["by_turn"].items():
        lines.append(f"  turns {k} (n {v['n']}): AUC net {v['auc_net']:.4f} eval {v['auc_eval']:.4f}; log-loss net cal "
                     f"{v['ll_net_cal']:.4f} eval cal {v['ll_eval_cal']:.4f} const {v['ll_const']:.4f}")
    j = s["joint_logit"]
    lines.append(f"  joint logistic: net logit coef {j['coef_net_logit']:.3f} (z {j['z_net']:.1f}), eval/100 coef "
                 f"{j['coef_eval_per100']:.3f} (z {j['z_eval']:.1f}) [{j['note']}]")
    o = s["one_turn_change"]
    lines.append(f"  one-turn change given logit P_net(t): log-loss root only {o['ll_root_only']:.4f}")
    for k in ("net_logit", "eval_per100", "eval_search_scale"):
        v = o[k]
        lines.append(f"    + delta {k}: coef {v['coef']:.3f} (z {v['z']:.1f}), log-loss gain {v['ll_gain']:.4f}, "
                     f"AUC of delta alone {v['auc_delta_alone']:.4f}")
    lines.append("  calibration (decile: mean predicted -> observed):")
    lines.append("    net:  " + ", ".join(f"{c['mean_pred']:.2f}->{c['freq']:.2f}" for c in s["calibration_net"]))
    lines.append("    eval: " + ", ".join(f"{c['mean_pred']:.2f}->{c['freq']:.2f}" for c in s["calibration_eval_search_scale"]))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("bench", "outcome"):
        b = sub.add_parser(name)
        b.add_argument("--records", nargs="+", required=True)
        b.add_argument("--source")
        b.add_argument("--net", required=True)
        b.add_argument("--out", required=True)
        b.add_argument("--format", default="gen9randombattle")
        b.add_argument("--seed", type=int, default=11)
        b.add_argument("--per-record", type=int, default=2)
        b.add_argument("--device", default="cpu")
        if name == "bench":
            b.add_argument("--rows", help="replay exactly these study rows (rec, side, idx)")
            b.add_argument("--positions", type=int, default=100000)
            b.add_argument("--worlds", type=int, default=4)
            b.add_argument("--retest", action="store_true")
            b.add_argument("--ref", type=int, default=0, help="judge R: this many fresh worlds")
            b.add_argument("--workers", type=int, default=11)
            b.add_argument("--concurrency", type=int, default=3)
            b.add_argument("--leaf-device", default="cuda")
            b.add_argument("--min-prob", type=float, default=0.0)
            b.add_argument("--iters", type=int, default=1000)
            b.add_argument("--batch", type=int, default=2048)
        else:
            b.add_argument("--pairs", type=int, default=1000)
            b.add_argument("--dets", type=int, default=4)
    for name in ("analyze-bench", "analyze-outcome"):
        b = sub.add_parser(name)
        b.add_argument("path")
        b.add_argument("--json")
    a = p.parse_args()
    if a.cmd == "bench":
        return asyncio.run(bench(a))
    if a.cmd == "outcome":
        return asyncio.run(outcome(a))
    s = analyze_bench(a.path) if a.cmd == "analyze-bench" else analyze_outcome(a.path)
    print(format_bench(s) if a.cmd == "analyze-bench" else format_outcome(s))
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
