"""Analysis of netsearch_study.py output (one JSON line per position): how often the margin rule overrides the
network at each margin, how noisy the gap is, how much selecting the search's pick inflates its gap (the optimizer's
curse), and what a held-out judge says each margin is worth. Definitions are porygon.netsearch's (RootStats, gate).

    PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$C/src $PY scripts/fastenv/netsearch_analyze.py OUT.jsonl [...] \
        [--json SUMMARY.json] [--by source|kind]

Per position: a_arg = the network's argmax, p05 = its T-0.5 distribution (the hybrid's a_net is a draw from it),
s_X = search X's pick (A = the decision search; B = retest; R = reference; P = PUCT; L = the original game's logged
search), D_X(a) = Q_X(s_A) - Q_X(a). Expectations over the network's draw are exact (sums over p05), not sampled.
Judges: B (an independent search with A's settings: removes A's selection noise) and R (more samples, longer: less
noise as well). Both share the engine's evaluation, so they favour the search's view; only games measure that bias.
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MARGINS = [-math.inf, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, math.inf]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def stats_of(NS, d):
    if not d:
        return None
    return NS.RootStats(np.array(d["n"], float), np.array(d["w"], float), np.array(d["ok"], bool))


def logged_stats(row, A):
    """The original game's logged search as (q [A], pick) or None."""
    lg = row.get("logged")
    if not lg or lg.get("q") is None or lg.get("shares") is None:
        return None
    q = np.array([np.nan if x is None else x for x in lg["q"]], float)
    sh = np.array(lg["shares"], float)
    mask = np.array(row["mask"]) > 0
    if len(q) != A or not np.isfinite(q[mask]).any():
        return None
    return q, int(np.argmax(np.where(mask, sh, -np.inf)))


def mean_se(xs):
    xs = np.asarray([x for x in xs if x is not None and math.isfinite(x)], float)
    if len(xs) == 0:
        return math.nan, math.nan, 0
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else math.nan, len(xs)


def analyse(NS, rows):
    out = {"positions": len(rows)}
    per = []
    for r in rows:
        mask = np.array(r["mask"]) > 0
        probs = np.array(r["probs"], float)
        p05 = NS.sharpen(probs, 0.5)
        A = stats_of(NS, r.get("A"))
        if A is None or not A.ok.any():
            continue
        X = {k: stats_of(NS, r.get(k)) for k in ("B", "R", "P")}
        per.append(dict(r=r, mask=mask, probs=probs, p05=p05, A=A, sA=NS.search_pick(A, mask), qA=A.q,
                        X={k: v for k, v in X.items() if v is not None and v.ok.any()},
                        L=logged_stats(r, len(mask)), arg=int(np.argmax(np.where(mask, probs, -np.inf)))))
    out["usable"] = len(per)
    if not per:
        return out
    legal_actions = [p["mask"].sum() for p in per]
    out["legal_mean"] = float(np.mean(legal_actions))
    out["kind_forced_share"] = float(np.mean([p["r"]["kind"] == "forced" for p in per]))

    # agreement
    agree_arg = [p["sA"] == p["arg"] for p in per]
    agree_05 = [p["p05"][p["sA"]] for p in per]
    out["agree"] = {"search_pick_eq_net_argmax": float(np.mean(agree_arg)),
                    "search_pick_eq_net_T05_draw (expected)": float(np.mean(agree_05)),
                    "net_top_p05_mean": float(np.mean([p["p05"].max() for p in per])),
                    "search_top_share_mean": float(np.mean([p["A"].share.max() for p in per]))}
    for k in ("B", "R", "P"):
        sub = [p for p in per if k in p["X"]]
        if sub:
            out["agree"][f"search_pick_A_eq_{k}"] = float(np.mean([NS.search_pick(p["X"][k], p["mask"]) == p["sA"]
                                                                   for p in sub]))
            out["agree"][f"pick_{k}_eq_net_argmax"] = float(np.mean([NS.search_pick(p["X"][k], p["mask"]) == p["arg"]
                                                                    for p in sub]))
            out["agree"][f"n_{k}"] = len(sub)
    subL = [p for p in per if p["L"] is not None]
    if subL:
        out["agree"]["search_pick_A_eq_logged"] = float(np.mean([p["L"][1] == p["sA"] for p in subL]))
        out["agree"]["n_logged"] = len(subL)

    # the gap and its noise
    gaps = [p["qA"][p["sA"]] - p["qA"][p["arg"]] for p in per if p["sA"] != p["arg"]]
    gaps = [g for g in gaps if math.isfinite(g)]
    if gaps:
        out["gap_A_when_disagree_argmax"] = {f"q{int(100 * q)}": float(np.quantile(gaps, q))
                                             for q in (0.1, 0.25, 0.5, 0.75, 0.9)} | {"n": len(gaps)}
    noise = {}
    for k in ("B", "R"):
        sub = [p for p in per if k in p["X"]]
        if not sub:
            continue
        dq = []  # single-action Q, A vs judge (common legal actions)
        bias, dd = [], []  # selected gap re-estimated by the judge: D_A(s_A, arg) - D_k(s_A, arg)
        for p in sub:
            qk = p["X"][k].q
            m = p["mask"] & np.isfinite(p["qA"]) & np.isfinite(qk)
            dq += list((p["qA"] - qk)[m])
            if p["sA"] != p["arg"]:
                ga = p["qA"][p["sA"]] - p["qA"][p["arg"]]
                gk = qk[p["sA"]] - qk[p["arg"]]
                if math.isfinite(ga) and math.isfinite(gk):
                    bias.append(ga - gk)
                    dd.append((ga, gk))
        noise[k] = {"n_positions": len(sub), "sd_Q_A_minus_judge": float(np.std(dq)) if dq else math.nan,
                    "selected_gap_inflation_mean": mean_se(bias)[0], "selected_gap_inflation_se": mean_se(bias)[1],
                    "n_disagree": len(bias),
                    "sd_gap_A_minus_judge": float(np.std([a - b for a, b in dd])) if dd else math.nan,
                    "share_judge_gap_le_0": float(np.mean([b <= 0 for a, b in dd])) if dd else math.nan}
    # per-position noise of the gap from the determinizations (K=4): t-statistics
    ts = []
    for p in per:
        if p["sA"] != p["arg"]:
            m, se, c = NS.gap_t(p["A"], p["sA"], p["arg"])
            if c > 1 and se and math.isfinite(se) and se > 0:
                ts.append(m / se)
    if ts:
        noise["gap_t_A"] = {f"q{int(100 * q)}": float(np.quantile(ts, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    out["noise"] = noise

    # the margin grid: override rates, stability, judged value (exact expectation over the T-0.5 draw)
    table = []
    for delta in MARGINS:
        row = {"margin": delta}
        ov05, ovarg = [], []
        for p in per:
            e = 0.0
            for a in np.flatnonzero(p["mask"]):
                e += p["p05"][a] * NS.gate(int(a), p["A"], p["mask"], delta).override
            ov05.append(e)
            ovarg.append(NS.gate(p["arg"], p["A"], p["mask"], delta).override)
        row["override_T05"] = float(np.mean(ov05))
        row["override_argmax"] = float(np.mean(ovarg))
        for k in ("B", "R"):
            sub = [p for p in per if k in p["X"]]
            if not sub:
                continue
            g_net, g_search, stable = [], [], []
            for p in sub:
                qk, mask = p["X"][k].q, p["mask"]
                if not np.isfinite(qk[mask]).all():
                    continue
                vn = vh = 0.0
                for a in np.flatnonzero(mask):
                    c = NS.gate(int(a), p["A"], mask, delta).action
                    vh += p["p05"][a] * qk[c]
                    vn += p["p05"][a] * qk[a]
                g_net.append(vh - vn)  # hybrid minus raw network, judged by k
                g_search.append(vh - qk[p["sA"]])  # hybrid minus pure search
                if k == "B":
                    stable.append(NS.gate(p["arg"], p["A"], mask, delta).override
                                  == NS.gate(p["arg"], p["X"]["B"], mask, delta).override)
            m1, s1, n1 = mean_se(g_net)
            m2, s2, _ = mean_se(g_search)
            row[f"judge_{k}"] = {"hybrid_minus_net": m1, "se1": s1, "hybrid_minus_search": m2, "se2": s2, "n": n1}
            if stable:
                row["override_argmax_same_in_B"] = float(np.mean(stable))
        table.append(row)
    out["margins"] = table

    # PUCT arm: how the network's priors move the search
    sub = [p for p in per if "P" in p["X"]]
    if sub:
        out["puct"] = {"n": len(sub),
                       "pick_P_eq_net_argmax": float(np.mean([NS.search_pick(p["X"]["P"], p["mask"]) == p["arg"]
                                                              for p in sub])),
                       "pick_P_eq_pick_A": float(np.mean([NS.search_pick(p["X"]["P"], p["mask"]) == p["sA"]
                                                          for p in sub])),
                       "visits_P_over_A": float(np.mean([p["X"]["P"].visits / max(p["A"].visits, 1) for p in sub]))}
        if any("R" in p["X"] for p in sub):
            sR = [p for p in sub if "R" in p["X"]]
            out["puct"]["judge_R_pick_P_minus_pick_A"] = mean_se(
                [p["X"]["R"].q[NS.search_pick(p["X"]["P"], p["mask"])] - p["X"]["R"].q[p["sA"]] for p in sR])

    # value head vs result, search root value
    res = [(p["r"]["win"], p["r"]["result"]) for p in per if p["r"].get("result") in (1, -1)]
    if res:
        w = np.clip(np.array([x for x, _ in res]), 1e-6, 1 - 1e-6)
        y = np.array([(z + 1) / 2 for _, z in res])
        out["value_head"] = {"n": len(res), "bce": float(-np.mean(y * np.log(w) + (1 - y) * np.log(1 - w))),
                             "brier": float(np.mean((w - y) ** 2)), "mean_win": float(w.mean()),
                             "result_mean": float(y.mean()),
                             "corr_with_search_root": float(np.corrcoef(
                                 [p["r"]["win"] for p in per], [p["A"].root for p in per])[0, 1])}
    # visits: how much evidence each Q rests on
    vis = np.concatenate([p["A"].n[p["A"].ok].sum(1) for p in per])
    logged_vis = [p["r"]["logged"]["visits"] for p in per if (p["r"].get("logged") or {}).get("visits")]
    min_legal = [p["A"].n[p["A"].ok][:, p["mask"]].min() for p in per]
    out["visits"] = {"per_det_median": float(np.median(vis)), "per_det_p10": float(np.quantile(vis, 0.1)),
                     "per_det_p90": float(np.quantile(vis, 0.9)),
                     "logged_per_det_median": float(np.median(logged_vis)) if logged_vis else math.nan,
                     "all_legal_have_q": float(np.mean([np.isfinite(p["qA"][p["mask"]]).all() for p in per])),
                     "min_legal_visits_median": float(np.median(min_legal)),
                     "min_legal_visits_p10": float(np.quantile(min_legal, 0.1))}
    # timing
    out["timing"] = {"t_net_ms": mean_se([p["r"]["t_net_ms"] for p in per])[0],
                     "t_search_A_s": mean_se([p["r"]["A"]["t"] for p in per])[0],
                     "visits_A_per_det": mean_se([p["A"].visits for p in per])[0],
                     "failed_dets_A": int(sum((~p["A"].ok).sum() for p in per)),
                     "mask_mismatch": int(sum(not p["r"].get("mask_ok", True) for p in per))}
    return out


def fmt(x):
    if isinstance(x, float):
        return "inf" if x == math.inf else "-inf" if x == -math.inf else f"{x:.4f}"
    return str(x)


def report(name, s):
    print(f"== {name}: {s.get('positions')} positions, {s.get('usable')} usable, legal {fmt(s.get('legal_mean', 0))}, "
          f"forced {fmt(s.get('kind_forced_share', 0))}")
    for k in ("agree", "gap_A_when_disagree_argmax", "noise", "puct", "value_head", "visits", "timing"):
        if k in s:
            print(f"  {k}: " + json.dumps(s[k], default=fmt))
    if "margins" in s:
        print("  margin | override T0.5 / argmax | stable(B) | judge B: hyb-net, hyb-search | judge R: hyb-net, hyb-search")
        for r in s["margins"]:
            jb, jr = r.get("judge_B", {}), r.get("judge_R", {})
            print(f"  {fmt(r['margin']):>6} | {r['override_T05']:.3f} / {r['override_argmax']:.3f} | "
                  f"{fmt(r.get('override_argmax_same_in_B', math.nan))} | "
                  f"{fmt(jb.get('hybrid_minus_net', math.nan))} ± {fmt(jb.get('se1', math.nan))}, "
                  f"{fmt(jb.get('hybrid_minus_search', math.nan))} ± {fmt(jb.get('se2', math.nan))} | "
                  f"{fmt(jr.get('hybrid_minus_net', math.nan))} ± {fmt(jr.get('se1', math.nan))}, "
                  f"{fmt(jr.get('hybrid_minus_search', math.nan))} ± {fmt(jr.get('se2', math.nan))}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--json")
    p.add_argument("--by", choices=["source", "kind"])
    a = p.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from porygon import netsearch as NS
    rows = load(a.files)
    groups = {"all": rows}
    if a.by:
        g = defaultdict(list)
        for r in rows:
            g[str(r.get(a.by)).split("/")[0]].append(r)
        groups |= dict(g)
    res = {}
    for name, rs in groups.items():
        res[name] = analyse(NS, rs)
        report(name, res[name])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=fmt))


if __name__ == "__main__":
    main()
