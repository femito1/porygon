"""Self-tests of the wsp seq plan tools (seq_design.py, seq_monitor.py), and the backtest on finished runs.

    python scripts/wsp/seq_selftest.py                      # numerical engine, simulation, input formats
    python scripts/wsp/seq_selftest.py --backtest DIR ...   # + when each finished run would have stopped
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seq_design as sd  # noqa: E402
import seq_monitor as sm  # noqa: E402
from seq_design import brentq, norm  # noqa: E402  (scipy, or the numpy fallback without it)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")
    return ok


def test_engine() -> bool:
    print("numerical engine vs published constants")
    K = 5
    t = np.arange(1, K + 1) / K
    I = t * 100
    ok = True
    f = lambda c: sum(map(np.sum, sd.exit_probs(I, -c * np.ones(K), c * np.ones(K), 0.0))) - 0.05
    ok &= check("Pocock K=5 two-sided .05 = 2.413", abs(brentq(f, 1.5, 4) - 2.413) < 5e-4)
    sq = np.sqrt(K / np.arange(1, K + 1))
    g = lambda C: sum(map(np.sum, sd.exit_probs(I, -C * sq, C * sq, 0.0))) - 0.05
    ok &= check("O'Brien-Fleming K=5 two-sided .05 C = 2.040", abs(brentq(g, 1.5, 4) - 2.040) < 5e-4)
    b = sd._solve_bounds(t, I, 0.3, 0.025, 0.1, ("obf",), ("hsd", -2.0), False)[1]
    ok &= check("Lan-DeMets OBF-type K=5 one-sided .025", np.allclose(b, [4.8769, 3.3569, 2.6803, 2.2898, 2.0310],
                                                                          atol=2e-3), str(np.round(b, 4)))
    t3 = np.arange(1, 4) / 3
    zf = norm.isf(0.025) + norm.isf(0.1)
    gap = lambda R: (lambda a, b, raw: raw - b[-1])(*sd._solve_bounds(t3, t3 * R * zf ** 2, 1.0, 0.025, 0.1,
                                                                        ("hsd", -4.0), ("hsd", -2.0), False))
    R = brentq(gap, 1.0, 1.5, xtol=1e-8)
    a, b, _ = sd._solve_bounds(t3, t3 * R * zf ** 2, 1.0, 0.025, 0.1, ("hsd", -4.0), ("hsd", -2.0), False)
    ok &= check("gsDesign default (k=3, HSD -4/-2, beta .1): inflation 1.0699, lower -0.2387 0.9411 1.9992",
                abs(R - 1.0699) < 2e-4 and np.allclose(a, [-0.2387, 0.9411, 1.9992], atol=2e-3),
                f"R={R:.4f} a={np.round(a, 4)} b={np.round(b, 4)}")
    return ok


def test_designs(reps: int) -> bool:
    print(f"designs: alpha / power, numerical vs simulated ({reps} sims, empirical mix, plug-in variance)")
    ok = True
    for nm in ("gain", "noninf", "threshold", "margin10"):
        d = sd.solve(sd.preset(nm))
        n0, n1 = sd.oc_numeric(d, d.theta0), sd.oc_numeric(d, d.theta1)
        s0, s1 = sd.simulate(d, d.theta0, reps, 11), sd.simulate(d, d.theta1, reps, 12)
        tol0 = 4 * math.sqrt(d.alpha / reps) + 0.003
        tol1 = 4 * math.sqrt(0.25 / reps) + 0.005
        ok &= check(f"{nm}: alpha <= nominal", s0["p_success"] <= d.alpha + tol0,
                    f"(numerical {n0['p_success']:.4f}, simulated {s0['p_success']:.4f}, nominal {d.alpha})")
        ok &= check(f"{nm}: power >= nominal", s1["p_success"] >= 1 - d.beta - tol1,
                    f"(numerical {n1['p_success']:.4f}, simulated {s1['p_success']:.4f}, nominal {1 - d.beta})")
        ok &= check(f"{nm}: E[pairs] sim ~ numerical", abs(s0["E_pairs"] - n0["E_pairs"]) < 0.03 * n0["E_pairs"] + 5,
                    f"({s0['E_pairs']:.0f} vs {n0['E_pairs']:.0f} at theta0)")
        js = Path(tempfile.mkdtemp()) / "d.json"
        sd.save(d, str(js))
        d2 = sd.load(str(js))
        ok &= check(f"{nm}: JSON round trip", d2.b == d.b and d2.c == d.c and d2.n == d.n)
    return ok


def test_inference(reps: int) -> bool:
    """Coverage of the reported estimates over simulated runs stopped by the monitor's rule (empirical mix,
    plug-in variance): stage-wise 95% CI and repeated 95% CI cover theta; the median-unbiased estimate is above
    theta about half the time; the naive estimate's bias for comparison."""
    print(f"inference at stopping ({reps} simulated runs per row)")
    ok = True
    for nm, th in (("gain", 0.50), ("gain", 0.52), ("gain", 0.55), ("threshold", 0.40), ("margin10", 0.60)):
        d = sd.solve(sd.preset(nm))
        rng = np.random.default_rng(7)
        p = sd.mix_probs(th)
        cov, rcov, above, naive = 0, 0, 0, []
        for _ in range(reps):
            x = sd.VALUES[rng.choice(5, size=d.n_max, p=p)]
            ev = sm.evaluate(d, x)
            r = ev["looks"][-1]
            fi = sd.final_inference(d, ev["look"], r["mean"], r["sd"])
            cov += fi["ci_lo"] <= th <= fi["ci_hi"]
            rcov += fi["rci"][0] <= th <= fi["rci"][1]
            above += fi["mue"] > th
            naive.append(fi["naive"] - th)
        se = math.sqrt(0.05 * 0.95 / reps)
        ok &= check(f"{nm} at {100 * th:.1f}%: stage-wise CI coverage {cov / reps:.3f}, repeated CI {rcov / reps:.3f}, "
                    f"P(MUE > theta) {above / reps:.3f}, naive bias {100 * np.mean(naive):+.2f}pt",
                    abs(cov / reps - 0.95) < 4 * se + 0.01 and rcov / reps >= 0.95 - 4 * se
                    and abs(above / reps - 0.5) < 4 * math.sqrt(0.25 / reps) + 0.02)
    return ok


def _write(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, default=lambda o: o.item()) + "\n" for r in rows))


def test_inputs() -> bool:
    print("monitor input formats and ordering")
    ok = True
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    n = 300
    g = rng.choice([0.0, 0.5, 1.0], size=(n, 2), p=[0.4, 0.05, 0.55])
    name = lambda s: {1.0: "wspa1", 0.0: "wspb1", 0.5: ""}[s]
    games = [{"pair": i, "game": k, "winner": name(g[i, k]), "p1": "wspa1" if k == 0 else "wspb1",
              "p2": "wspb1" if k == 0 else "wspa1"} for i in rng.permutation(n) for k in (1, 0)]
    _write(tmp / "battles.jsonl", games)
    want = g.mean(1)
    x, meta = sm.ordered_scores(sm.read_records([str(tmp / "battles.jsonl")], "wspa"))
    ok &= check("battles.jsonl (shuffled completion order) -> id order", np.allclose(x, want), f"{len(x)} pairs")
    lab = lambda s: {1.0: "A", 0.0: "B", 0.5: None}[s]
    for kind, rec in [("winners A/B", lambda i: {"pair": i, "winners": [lab(g[i, 0]), lab(g[i, 1])]}),
                      ("winners usernames", lambda i: {"pair": i, "winners": [name(g[i, 0]) or None, name(g[i, 1]) or None]}),
                      ("scores", lambda i: {"pair": i, "scores": list(g[i])}),
                      ("a_score", lambda i: {"pair": i, "a_score": float(g[i].mean())}),
                      ("a_wins", lambda i: {"pair": i, "a_wins": float(g[i].sum())})]:
        p = tmp / f"pairs-{kind.replace(' ', '_').replace('/', '')}.jsonl"
        _write(p, [rec(i) for i in range(n)])
        x, _ = sm.ordered_scores(sm.read_records([str(p)], "wspa"))
        ok &= check(f"per-pair records ({kind})", np.allclose(x, want))
    # prefix rule: pair 5 still running -> only pairs 0..4 usable
    _write(tmp / "run.jsonl", [r for r in games if not (r["pair"] == 5 and r["game"] == 1)])
    x, meta = sm.ordered_scores(sm.read_records([str(tmp / "run.jsonl")], "wspa"), stale=400)
    ok &= check("running pair holds back later ids", len(x) == 5 and meta["held_back"] == n - 1 - 5,
                f"(usable {len(x)}, held {meta['held_back']})")
    x, meta = sm.ordered_scores(sm.read_records([str(tmp / "run.jsonl")], "wspa"), stale=200)
    ok &= check("stale incomplete pair dropped after 200 later ids", len(x) == n - 1 and len(meta["dropped"]) == 1)
    # shards: round-robin interleave of id-ordered prefixes
    for s in range(3):
        _write(tmp / f"shard{s}.jsonl", [{"pair": i, "shard": s, "scores": list(g[100 * s + i])} for i in range(100)])
    x, meta = sm.ordered_scores(sm.read_records([str(tmp / f"shard{s}.jsonl") for s in range(3)], "wspa"))
    ok &= check("3 shards interleaved", len(x) == 300 and np.allclose(x[:3], [want[0], want[100], want[200]]))
    _write(tmp / "shard2.jsonl", [{"pair": i, "shard": 2, "scores": list(g[200 + i])} for i in range(40)])
    x, _ = sm.ordered_scores(sm.read_records([str(tmp / f"shard{s}.jsonl") for s in range(3)], "wspa"))
    ok &= check("slow shard limits the interleaved prefix", len(x) == 3 * 40 + 2, f"({len(x)})")
    x, _ = sm.ordered_scores(sm.read_records([str(tmp / f"shard{s}.jsonl") for s in range(3)], "wspa"), final=True)
    ok &= check("--final uses every complete pair", len(x) == 240)
    # spd_h2h.py layout: shard K plays global ids [K*N, (K+1)*N); live part files in completion order
    N = 100
    spd = tmp / "spd"
    spd.mkdir()
    rec = lambda K, i: {"exp": "x", "pair": K * N + i, "shard": K, "worker": i % 7, "score": float(g[K * N + i].mean()),
                        "games": [{"winner": {1.0: "A", 0.0: "B", 0.5: "tie"}[g[K * N + i, j]], "turns": 30,
                                   "void": False, "truncated": False} for j in (0, 1)]}
    _write(spd / "shard-0000.part.jsonl", [rec(0, i) for i in rng.permutation(N)])
    _write(spd / "shard-0001.part.jsonl", [rec(1, i) for i in rng.permutation(N) if i != 3])   # pair 103 running
    try:
        sm.ordered_scores(sm.read_records([str(spd)], "wspa"))
        ok &= check("spd: shard > 0 without meta or --shard-pairs is refused", False)
    except SystemExit:
        ok &= check("spd: shard > 0 without meta or --shard-pairs is refused", True)
    x, meta = sm.ordered_scores(sm.read_records([str(spd)], "wspa"), shard_pairs=N)
    ok &= check("spd: running pair 103 holds shard 1 at 3 pairs -> 7 usable (interleaved)",
                len(x) == 7 and np.allclose(x[:2], [want[0], want[100]]), f"({len(x)})")
    _write(spd / "shard-0001.part.jsonl", [rec(1, i) for i in rng.permutation(N)])
    (spd / "shard-0001.meta.json").write_text(json.dumps({"first": N, "end": 2 * N}))
    _write(spd / "shard-0001.jsonl", [rec(1, i) for i in range(N)])                       # final file, part kept
    x, meta = sm.ordered_scores(sm.read_records([str(spd)], "wspa"))
    ok &= check("spd: meta gives the first id; part + final files of a shard merge (no double count)",
                len(x) == 2 * N and meta["complete_pairs"] == 2 * N and np.allclose(x[1::2], want[100:200]))
    # --order sequential: shards played one after another (one shard per look) -> nested looks
    seq = tmp / "seq"
    seq.mkdir()
    (seq / "shard-0000.meta.json").write_text(json.dumps({"first": 0, "end": N}))
    _write(seq / "shard-0000.jsonl", [rec(0, i) for i in range(N)])
    x0, _ = sm.ordered_scores(sm.read_records([str(seq)], "wspa"), order="sequential")
    (seq / "shard-0001.meta.json").write_text(json.dumps({"first": N, "end": 2 * N}))
    _write(seq / "shard-0001.part.jsonl", [rec(1, i) for i in rng.permutation(N) if i >= 40 or i < 30])  # 130 running
    x1, meta = sm.ordered_scores(sm.read_records([str(seq)], "wspa"), order="sequential")
    xr, _ = sm.ordered_scores(sm.read_records([str(seq)], "wspa"))
    ok &= check("sequential: adding shard 1 leaves the first look's data unchanged (nested); round-robin re-draws it",
                len(x0) == N and np.allclose(x1[:N], x0) and np.allclose(x1, want[:130]) and len(x1) == 130
                and meta["shards"] == 2 and len(xr) == 61 and not np.allclose(xr, x0[:len(xr)]),
                f"({len(x0)}, {len(x1)}, round-robin {len(xr)})")
    x1, meta = sm.ordered_scores(sm.read_records([str(seq)], "wspa"), order="sequential", final=True)
    ok &= check("sequential --final: all complete pairs in global id order", len(x1) == 190
                and np.allclose(x1, np.concatenate([want[:130], want[140:200]])) and len(meta["dropped"]) == 10)
    shutil.move(str(seq / "shard-0000.jsonl"), str(tmp / "held.jsonl"))
    x1, _ = sm.ordered_scores(sm.read_records([str(seq)], "wspa"), order="sequential")
    ok &= check("sequential: a shard with no records yet holds back the later ones (start at --first-id)", len(x1) == 0)
    shutil.move(str(tmp / "held.jsonl"), str(seq / "shard-0000.jsonl"))
    _write(seq / "shard-0002.jsonl", [dict(rec(1, i), shard=2) for i in range(5)])   # ids 100..104 again
    try:
        sm.ordered_scores(sm.read_records([str(seq)], "wspa"), order="sequential", shard_pairs=N)
        ok &= check("sequential: overlapping id ranges are refused", False)
    except SystemExit:
        ok &= check("sequential: overlapping id ranges are refused", True)
    # per-game files restart at pair 0: concatenated in path order
    half = [r for r in games if r["pair"] < 150]
    _write(tmp / "f0.jsonl", half)
    _write(tmp / "f1.jsonl", [dict(r, pair=r["pair"] - 150) for r in games if r["pair"] >= 150])
    x, _ = sm.ordered_scores(sm.read_records([str(tmp / "f0.jsonl"), str(tmp / "f1.jsonl")], "wspa"), order="sequential")
    ok &= check("sequential: per-game files (ids from 0 each) concatenate in path order", len(x) == n and np.allclose(x, want))
    # decisions: an obviously strong A stops early for success; a null A never claims at look 1
    d = sd.solve(sd.preset("gain"))
    strong = np.array([1.0] * 40 + [0.5] * 60)
    ev = sm.evaluate(d, np.tile(strong, 30))
    ok &= check("strong A -> success at look 1", ev["decision"] == "success" and ev["look"] == 0)
    weak = np.array([0.0] * 30 + [0.5] * 60 + [1.0] * 10)
    ev = sm.evaluate(d, np.tile(weak, 30))
    ok &= check("weak A -> futility at look 1", ev["decision"] == "futility" and ev["look"] == 0)
    return ok


def backtest(dirs: list[str], prefix: str, designs: list[str]) -> None:
    print("backtest on finished runs (pair-id order; decision at the first look that crossed)")
    ds = {nm: sd.solve(sd.preset(nm)) for nm in designs}
    print(f"  {'run':42s} {'pairs':>5s} {'fixed-n':>15s} | " + " | ".join(f"{nm:>36s}" for nm in designs))
    for p in dirs:
        x, meta = sm.ordered_scores(sm.read_records([p], prefix), final=True)
        m, s = x.mean(), x.std(ddof=1)
        cols = []
        for nm, d in ds.items():
            ev = sm.evaluate(d, x, final=True)
            if ev["decision"] == "continue":
                cols.append(f"{'continue past ' + str(len(x)):>36s}")
                continue
            j = ev["look"]
            fi = sd.final_inference(d, j, ev["looks"][-1]["mean"], ev["looks"][-1]["sd"])
            cols.append(f"{ev['decision'][:4]} @{d.n[j]:5d}p {100 * fi['mue']:5.1f}% [{100 * fi['ci_lo']:4.1f},{100 * fi['ci_hi']:5.1f}]")
        name = "/".join(Path(p).parts[-2:])
        print(f"  {name[-42:]:42s} {len(x):5d} {100 * m:6.2f}% ±{196 * s / math.sqrt(len(x)):4.2f}  | " + " | ".join(cols))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backtest", nargs="*", default=[])
    ap.add_argument("--a-prefix", default="wspa")
    ap.add_argument("--designs", nargs="+", default=["gain", "noninf"])
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--inference-reps", type=int, default=1000)
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    ok = True
    if not a.skip_tests:
        ok = test_engine() & test_inputs() & test_designs(a.reps)
        if a.inference_reps:
            ok &= test_inference(a.inference_reps)
        print("ALL OK" if ok else "SOME TESTS FAILED")
    if a.backtest:
        backtest(a.backtest, a.a_prefix, a.designs)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
