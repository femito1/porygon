"""Fidelity statistics of the spike: Showdown (ps_games.js) vs poke-engine (engine_games.py) on identical team pairs
with identical policies. For every (gen, policy) cell:
  turns, first faint   mean, median, 10/90% quantiles in each sim; the paired mean difference (engine - Showdown,
                       per-pair means, 95% interval over pairs)
  side-one win rate    overall in each sim (ties 1/2) and the paired difference with a 95% interval; per-pair
                       win-rate differences (quantiles) and, net of binomial noise, the RMS of the per-pair true
                       win-probability gap and the correlation of the two sims' per-pair win probabilities
  failures             engine panics / errors and games hitting the decision cap
The native (pure-Rust) random games are compared with the Python-driven ones as a check of the native bench.

    python compare.py --dir out [--json summary.json]
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.startswith("{\"pair\"")]


def q(xs: list[float], f: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))]


def mean_ci(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(v / n)


def by_pair(games: list[dict]) -> dict[int, list[dict]]:
    d = defaultdict(list)
    for g in games:
        d[g["pair"]].append(g)
    return d


def ok(g: dict) -> bool:
    return g["winner"] >= 0


def score(g: dict) -> float:
    return 1.0 if g["winner"] == 1 else 0.5 if g["winner"] == 0 else 0.0


def compare(a: list[dict], b: list[dict], name_a: str, name_b: str) -> dict:
    out: dict = {}
    for name, games in ((name_a, a), (name_b, b)):
        good = [g for g in games if ok(g)]
        t = [g["turns"] for g in good]
        ff = [g["first_faint"] for g in good if g["first_faint"] >= 0]
        out[name] = {"games": len(games), "panic_or_error": sum(g["winner"] == -1 for g in games) / len(games),
                     "too_long": sum(g["winner"] == -2 for g in games) / len(games),
                     "ties": sum(g["winner"] == 0 for g in good) / len(good),
                     "turns_mean": sum(t) / len(t), "turns_median": q(t, 0.5), "turns_q10": q(t, 0.1),
                     "turns_q90": q(t, 0.9), "first_faint_mean": sum(ff) / len(ff),
                     "first_faint_median": q(ff, 0.5), "p1_win": sum(score(g) for g in good) / len(good)}
    pa, pb = by_pair(a), by_pair(b)
    common = sorted(k for k in pa if k in pb)
    dturn, dff, dwin, wa, wb, va, vb = [], [], [], [], [], [], []
    for k in common:
        ga, gb = [g for g in pa[k] if ok(g)], [g for g in pb[k] if ok(g)]
        if len(ga) < 2 or len(gb) < 2:
            continue
        dturn.append(sum(g["turns"] for g in gb) / len(gb) - sum(g["turns"] for g in ga) / len(ga))
        fa = [g["first_faint"] for g in ga if g["first_faint"] >= 0]
        fb = [g["first_faint"] for g in gb if g["first_faint"] >= 0]
        if fa and fb:
            dff.append(sum(fb) / len(fb) - sum(fa) / len(fa))
        sa, sb = [score(g) for g in ga], [score(g) for g in gb]
        ma, mb = sum(sa) / len(sa), sum(sb) / len(sb)
        wa.append(ma); wb.append(mb); dwin.append(mb - ma)
        # unbiased estimates of the sampling variance of each per-pair mean
        va.append(sum((x - ma) ** 2 for x in sa) / (len(sa) - 1) / len(sa))
        vb.append(sum((x - mb) ** 2 for x in sb) / (len(sb) - 1) / len(sb))
    n = len(dwin)

    def gap_stats(idx):
        m = len(idx)
        msq = sum(dwin[i] ** 2 - va[i] - vb[i] for i in idx) / m
        ma_, mb_ = sum(wa[i] for i in idx) / m, sum(wb[i] for i in idx) / m
        var_a = sum((wa[i] - ma_) ** 2 for i in idx) / (m - 1) - sum(va[i] for i in idx) / m
        var_b = sum((wb[i] - mb_) ** 2 for i in idx) / (m - 1) - sum(vb[i] for i in idx) / m
        cov = sum((wa[i] - ma_) * (wb[i] - mb_) for i in idx) / (m - 1)
        corr = cov / math.sqrt(var_a * var_b) if var_a > 0 and var_b > 0 else float("nan")
        return math.sqrt(max(msq, 0.0)), var_a, var_b, corr

    rms, var_a, var_b, corr = gap_stats(range(n))
    rng = random.Random(0)
    boot = sorted(gap_stats([rng.randrange(n) for _ in range(n)])[::3] for _ in range(200))
    rms_b = sorted(b[0] for b in boot)
    corr_b = sorted(b[1] for b in boot)
    out["paired"] = {
        "pairs": n,
        "turns_diff": mean_ci(dturn), "first_faint_diff": mean_ci(dff), "p1_win_diff": mean_ci(dwin),
        "per_pair_win_diff_q": [q(dwin, f) for f in (0.05, 0.25, 0.5, 0.75, 0.95)],
        "per_pair_true_gap_rms": rms, "per_pair_true_gap_rms_95": [rms_b[4], rms_b[194]],
        "per_pair_true_sd": [math.sqrt(max(var_a, 0)), math.sqrt(max(var_b, 0))],
        "per_pair_true_corr": corr, "per_pair_true_corr_95": [corr_b[4], corr_b[194]],
    }
    return out


def fmt(c: dict, a: str, b: str) -> str:
    A, B, P = c[a], c[b], c["paired"]
    lines = [f"  {'':<26}{a:>14}{b:>14}"]
    for k in ("games", "panic_or_error", "too_long", "ties", "turns_mean", "turns_median", "turns_q10", "turns_q90",
              "first_faint_mean", "first_faint_median", "p1_win"):
        lines.append(f"  {k:<26}{A[k]:>14.4g}{B[k]:>14.4g}")
    lines.append(f"  paired over {P['pairs']} pairs ({b} - {a}):")
    for k in ("turns_diff", "first_faint_diff", "p1_win_diff"):
        lines.append(f"    {k:<24}{P[k][0]:+.3f} ± {P[k][1]:.3f}")
    lines.append(f"    per-pair win diff 5/25/50/75/95%: {', '.join(f'{x:+.2f}' for x in P['per_pair_win_diff_q'])}")
    lo, hi = P["per_pair_true_gap_rms_95"]
    clo, chi = P["per_pair_true_corr_95"]
    lines.append(f"    per-pair true win-prob gap RMS {P['per_pair_true_gap_rms']:.3f} [{lo:.3f}, {hi:.3f}]; true per-pair"
                 f" sd {P['per_pair_true_sd'][0]:.3f}/{P['per_pair_true_sd'][1]:.3f}; corr"
                 f" {P['per_pair_true_corr']:.3f} [{clo:.3f}, {chi:.3f}]")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    d = Path(a.dir)
    summary = {}
    for g in (9, 4):
        for pol in ("random", "greedy"):
            ps, en = d / f"ps{g}-{pol}.jsonl", d / f"eng{g}-{pol}.jsonl"
            if not (ps.exists() and en.exists()):
                continue
            c = compare(load(ps), load(en), "showdown", "engine")
            summary[f"gen{g}-{pol}"] = c
            print(f"gen{g} {pol}\n{fmt(c, 'showdown', 'engine')}")
        nat, en = d / f"native{g}-random.jsonl", d / f"eng{g}-random.jsonl"
        if nat.exists() and en.exists():
            c = compare(load(en), load(nat), "engine", "native")
            summary[f"gen{g}-native-check"] = c
            P = c["paired"]
            print(f"gen{g} native-vs-python check: turns {c['engine']['turns_mean']:.2f} vs {c['native']['turns_mean']:.2f},"
                  f" p1 win diff {P['p1_win_diff'][0]:+.3f} ± {P['p1_win_diff'][1]:.3f},"
                  f" native panics {c['native']['panic_or_error']:.4f}")
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
