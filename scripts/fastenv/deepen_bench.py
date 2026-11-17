"""Microbench of porygon.deepen (docs/netsearch/SCALE-SEARCH-DESIGN.md §6) on logged study positions; no games.

Positions are replayed once (netsearch_study's replayer) and copied; then one pass per arm over all positions, with
`--concurrency` decisions in flight as in the slot harness (3 battles):
  la        porygon.lookahead2.lookahead (the reference; K worlds)
  b<B>      porygon.deepen.deepen with extra-leaf budget B (b0 must equal la: same worlds, cells, values, solve)
  h<B>      the same with fp16 leaf forwards (--half-budgets)
Every pass draws the same worlds per position (rng seeded by position). The net's move a_net is drawn once per
position at T 0.5 and the margin gate applied with --margin.

    cd $C && PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src NETV2_SRC=$FROZEN/src \\
      $PY scripts/fastenv/deepen_bench.py run --records "$SL" --source 6771931 --rows $I/s2-sl-ARP.jsonl \\
        --net NET --out DIR --positions 120 --workers 10 --concurrency 3
    ... deepen_bench.py report DIR          (pure: summary.json + summary.txt)
    ... deepen_bench.py delta --records ... --rows ROWS --net NET --out OUT.jsonl --budget B [--ref-worlds 32
        --ref-scale 4]    the lookahead2 δ rule (lookahead2_study.analyze_bench) for deepen at budget B: per position
        A = deepen(K 8, B) and judge R = deepen(K 32, 4B) on fresh worlds, both with the same a_net (the net's T 0.5
        draw, seeded per position); the row's "probs" is that draw one-hot, so the analysis is over the played draw.
"""
import argparse
import asyncio
import copy
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


def q(xs, p):
    return float(np.quantile(xs, p)) if len(xs) else math.nan


async def run(a) -> int:
    import netsearch_study as ST
    NS, S, R = ST.setup(a.format)
    import torch
    from porygon import deepen as D
    from porygon import lookahead2 as LA
    torch.set_num_threads(1)
    net = ST.load_net(a.net, "cpu")
    leaf = ST.load_net(a.net, a.leaf_device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # ---- positions
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    want: dict[tuple, set] = {}
    for r in rows:
        want.setdefault((r["rec"], r.get("side")), set()).add(r["idx"])
    recs = [r for r in ST.records(a.records, a.source, a.seed) if (r["id"], r.get("side")) in want]
    pos = []
    t0 = time.time()
    for rec in recs:
        if len(pos) >= a.positions:
            break
        key = (rec["id"], rec.get("side"))
        async for d, battle, first, logged in ST.replay_targets(rec, R, NS, 9, want[key]):
            probs, mask, win = NS.net_forward(net, battle)
            if mask.sum() < 2:
                continue
            name = f"{rec['id']}:{rec.get('side')}:{d.msg}"
            p = NS.sharpen(probs, 0.5)
            a_net = int(np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)).choice(len(p), p=p / p.sum()))
            pos.append({"name": name, "battle": copy.deepcopy(battle), "first": first, "probs": probs, "mask": mask,
                        "win": win, "a_net": a_net, "n_legal": int(mask.sum()), "turn": d.turn})
            if len(pos) >= a.positions:
                break
    print(f"{len(pos)} positions replayed in {time.time() - t0:.0f}s", flush=True)
    budgets = [int(x) for x in a.budgets.split(",") if x]
    halves = [int(x) for x in a.half_budgets.split(",") if x]
    nashes = [int(x) for x in a.nash_budgets.split(",") if x]
    arms = ([("la", None, False, 0.0)] + [(f"b{b}", b, False, 0.0) for b in budgets]
            + [(f"n{b}", b, False, 1.0) for b in nashes] + [(f"h{b}", b, True, 0.0) for b in halves])
    la_pool = LA.LookaheadPool(a.workers)
    d_pool = D.DeepenPool(a.workers)
    manifest = {"net": a.net, "positions": len(pos), "workers": a.workers, "concurrency": a.concurrency,
                "worlds": a.worlds, "margin": a.margin, "arms": [x[0] for x in arms], "nash_arms": [x[0] for x in arms if x[3] > 0], "cuda": torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None, "cfg": D.Config().__dict__}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    passes = {}
    fh = open(out / "rows.jsonl", "w")
    for arm, b, half, nash in arms:
        sem = asyncio.Semaphore(a.concurrency)
        fails = [0]

        async def one(P):
            async with sem:
                rng = random.Random(f"{a.seed}:{P['name']}:W")
                t1 = time.perf_counter()
                try:
                    if b is None:
                        res = await LA.lookahead(la_pool, leaf, P["battle"], P["first"], rng, P["mask"],
                                                 worlds=a.worlds)
                        extra = {}
                    else:
                        res = await D.deepen(d_pool, leaf, P["battle"], P["first"], rng, P["mask"], P["probs"],
                                             P["a_net"], worlds=a.worlds, cfg=D.Config(budget=b, nash=nash),
                                             half=half)
                        extra = {k: getattr(res, k) for k in ("base_leaves", "extra_leaves", "expansions",
                                                              "failed_expansions", "waves", "depth", "dedup", "cells",
                                                              "branches", "opp_kept")}
                except Exception as e:
                    fails[0] += 1
                    fh.write(json.dumps({"arm": arm, "pos": P["name"], "error": f"{type(e).__name__}: {e}"}) + "\n")
                    return
                t = time.perf_counter() - t1
                g = LA.gate(P["a_net"], res.rows, res.x, res.q, a.margin)
                fh.write(json.dumps({"arm": arm, "pos": P["name"], "n_legal": P["n_legal"], "a_net": P["a_net"],
                                     "rows": res.rows, "x": [round(float(v), 6) for v in res.x],
                                     "q": [float(v) for v in res.q], "pick": g.search, "action": g.action,
                                     "override": g.override, "gap": None if not math.isfinite(g.gap) else g.gap,
                                     "leaves": res.leaves, "t": t, "t_expand": res.t_expand, "t_score": res.t_score,
                                     "t_fwd": res.t_fwd, "t_solve": res.t_solve, "cpu": res.cpu_expand,
                                     "worlds": res.worlds, "errors": res.errors, **extra}) + "\n")
        tw = time.time()
        await asyncio.gather(*(one(P) for P in pos))
        wall = time.time() - tw
        fh.flush()
        passes[arm] = {"wall": wall, "decisions": len(pos) - fails[0], "failed": fails[0],
                       "decisions_per_s": (len(pos) - fails[0]) / wall}
        print(f"pass {arm}: {len(pos) - fails[0]} decisions ({fails[0]} failed) in {wall:.0f}s", flush=True)
        (out / "passes.json").write_text(json.dumps(passes, indent=1))
    fh.close()
    await la_pool.close()
    await d_pool.close()
    s = report(out)
    print(s)
    return 0 if all(p["failed"] <= 0.02 * len(pos) for p in passes.values()) else 1


async def delta(a) -> int:
    import netsearch_study as ST
    NS, S, R = ST.setup(a.format)
    import lookahead2_study as LS
    import torch
    from porygon import deepen as D
    torch.set_num_threads(1)
    net = ST.load_net(a.net, "cpu")
    leaf = ST.load_net(a.net, a.leaf_device)
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    want: dict[tuple, set] = {}
    for r in rows:
        want.setdefault((r["rec"], r.get("side")), set()).add(r["idx"])
    recs = [r for r in ST.records(a.records, a.source, a.seed) if (r["id"], r.get("side")) in want]
    pool = D.DeepenPool(a.workers)
    out = open(a.out, "a")
    sem = asyncio.Semaphore(a.concurrency)
    done = {"n": 0, "fail": 0}
    t_start = time.time()
    runs = [("A", a.worlds, a.budget), ("R", a.ref_worlds, a.budget * a.ref_scale)]

    async def one(rec):
        async with sem:
            key = (rec["id"], rec.get("side"))
            async for d, battle, first, logged in ST.replay_targets(rec, R, NS, 9, want[key]):
                if done["n"] >= a.positions:
                    return
                probs, mask, win = NS.net_forward(net, battle)
                if mask.sum() < 2:
                    continue
                name = f"{rec['id']}:{rec.get('side')}:{d.msg}"
                p = NS.sharpen(probs, 0.5)
                a_net = int(np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:8], 16))
                            .choice(len(p), p=p / p.sum()))
                onehot = [0.0] * len(probs)
                onehot[a_net] = 1.0
                row = {"rec": rec["id"], "side": rec.get("side"), "idx": d.msg, "turn": d.turn, "n_legal": int(mask.sum()),
                       "probs": onehot, "probs_net": [round(float(x), 6) for x in probs], "a_net": a_net,
                       "win": round(win, 5), "budget": a.budget}
                try:
                    for nm, k, b in runs:
                        rng = random.Random(f"{a.seed}:{rec['id']}:{rec.get('side')}:{d.msg}:{nm}")
                        t0 = time.perf_counter()
                        res = await D.deepen(pool, leaf, battle, first, rng, mask, probs, a_net, worlds=k,
                                             cfg=D.Config(budget=b))
                        row[nm] = LS.run_row(res, time.perf_counter() - t0)
                        row[nm].update(extra_leaves=res.extra_leaves, depth=res.depth)
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
    s = LS.analyze_bench(a.out)
    Path(a.out + ".json").write_text(json.dumps(s, indent=1))
    txt = LS.format_bench(s)
    Path(a.out + ".txt").write_text(txt + "\n")
    print(txt)
    return 0 if done["n"] and done["fail"] <= 0.02 * done["n"] else 1


def report(out: Path) -> str:
    out = Path(out)
    rows = [json.loads(x) for x in open(out / "rows.jsonl") if x.strip()]
    passes = json.loads((out / "passes.json").read_text())
    by: dict[str, dict] = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["pos"]] = r
    arms = list(passes)
    summ = {"passes": passes, "arms": {}}
    lines = [f"deepen microbench {out}: arms {arms}"]
    ref = by.get("la", {})
    top = [x for x in arms if x.startswith("b")]
    top = max(top, key=lambda x: int(x[1:])) if top else None
    for arm in arms:
        rs = [r for r in by.get(arm, {}).values() if "error" not in r]
        if not rs:
            continue
        g = lambda k: [r[k] for r in rs if k in r]  # noqa: E731
        s = {"n": len(rs), "errors": sum(1 for r in by.get(arm, {}).values() if "error" in r),
             "leaves_median": q(g("leaves"), .5), "leaves_q90": q(g("leaves"), .9), "leaves_max": max(g("leaves")),
             "t_median": q(g("t"), .5), "t_q90": q(g("t"), .9), "t_q99": q(g("t"), .99), "t_max": max(g("t")),
             "fwd_median": q(g("t_fwd"), .5), "fwd_q90": q(g("t_fwd"), .9), "solve_median": q(g("t_solve"), .5),
             "expand_median": q(g("t_expand"), .5), "cpu_median": q(g("cpu"), .5), "cpu_q90": q(g("cpu"), .9),
             "override": float(np.mean(g("override"))), "decisions_per_s": passes[arm]["decisions_per_s"],
             "leaves_per_s": float(np.sum(g("leaves")) / passes[arm]["wall"])}
        if "extra_leaves" in rs[0]:
            ex = g("extra_leaves")
            s.update(extra_median=q(ex, .5), extra_q10=q(ex, .1), expansions_median=q(g("expansions"), .5),
                     waves_median=q(g("waves"), .5), depth_hist={str(k): int(np.sum(np.array(g("depth")) == k))
                                                                 for k in sorted(set(g("depth")))},
                     dedup_share=float(np.sum(g("dedup")) / max(np.sum(ex) + np.sum(g("dedup")), 1)),
                     leaves_per_cell=float((np.sum(ex) + np.sum(g("dedup"))) / max(np.sum(g("cells")), 1)),
                     failed_expansions=int(np.sum(g("failed_expansions"))),
                     opp_kept=float(np.mean([x for x in g("opp_kept") if x > 0])) if any(x > 0 for x in g("opp_kept"))
                     else None)
        if ref and arm != "la":
            both = [(r, ref[p]) for p, r in by[arm].items() if p in ref and "error" not in r and "error" not in ref[p]]
            s["pick_eq_la"] = float(np.mean([r["pick"] == f["pick"] for r, f in both]))
            qa = lambda r: r["q"][r["rows"].index(r["a_net"])] if r["a_net"] in r["rows"] else math.nan  # noqa: E731
            sh = [qa(r) - qa(f) for r, f in both if math.isfinite(qa(r) - qa(f))]
            s["q_anet_shift_mean"], s["q_anet_shift_median"] = float(np.mean(sh)), float(np.median(sh))
            ql = lambda r: r["q"][r["rows"].index(r["pick"])]  # noqa: E731
            s["q_pick_shift_mean"] = float(np.mean([ql(r) - ql(f) for r, f in both]))
            if arm == "b0":
                dq = [max(abs(a_ - b_) for a_, b_ in zip(r["q"], f["q"])) if r["rows"] == f["rows"] else math.inf
                      for r, f in both]
                dx = [max(abs(a_ - b_) for a_, b_ in zip(r["x"], f["x"])) if r["rows"] == f["rows"] else math.inf
                      for r, f in both]
                s["parity_b0_vs_la"] = {"positions": len(both), "rows_equal": int(sum(r["rows"] == f["rows"] for r, f in both)),
                                        "max_abs_dq": max(dq) if dq else None, "max_abs_dx": max(dx) if dx else None,
                                        "leaves_equal": int(sum(r["leaves"] == f["leaves"] for r, f in both)),
                                        "picks_equal": int(sum(r["pick"] == f["pick"] for r, f in both))}
        if top and arm != top and top in by:
            both = [(r, by[top][p]) for p, r in by[arm].items() if p in by[top] and "error" not in r
                    and "error" not in by[top][p]]
            s[f"pick_eq_{top}"] = float(np.mean([r["pick"] == f["pick"] for r, f in both])) if both else None
        summ["arms"][arm] = s
        line = (f"  {arm:>6}: n {s['n']} (err {s['errors']}), leaves median {s['leaves_median']:.0f} q90 "
                f"{s['leaves_q90']:.0f} max {s['leaves_max']}; wall s median {s['t_median']:.2f} q90 {s['t_q90']:.2f} "
                f"q99 {s['t_q99']:.2f} max {s['t_max']:.2f}; fwd {s['fwd_median']:.3f}, expand {s['expand_median']:.3f}, "
                f"solve {s['solve_median']:.3f} s; worker cpu {s['cpu_median']:.2f} s; {s['decisions_per_s']:.2f} "
                f"decisions/s, {s['leaves_per_s']:.0f} leaves/s; override {s['override']:.3f}")
        if "extra_median" in s:
            line += (f"\n          extra leaves median {s['extra_median']:.0f} (q10 {s['extra_q10']:.0f}), expansions "
                     f"{s['expansions_median']:.0f}, waves {s['waves_median']:.0f}, depth {s['depth_hist']}, dedup "
                     f"{s['dedup_share']:.3f}, leaves/cell {s['leaves_per_cell']:.2f}, failed expansions "
                     f"{s['failed_expansions']}")
        if "pick_eq_la" in s:
            line += (f"; pick = la's {s['pick_eq_la']:.3f}; Q(a_net) - la's: mean {s['q_anet_shift_mean']:+.4f} median "
                     f"{s['q_anet_shift_median']:+.4f}; Q(own pick) - la's Q(its pick) {s['q_pick_shift_mean']:+.4f}")
        if s.get("opp_kept") is not None:
            line += f"; opponent mass kept {s['opp_kept']:.3f}"
        if top and f"pick_eq_{top}" in s and s[f"pick_eq_{top}"] is not None:
            line += f"; pick = {top}'s {s[f'pick_eq_{top}']:.3f}"
        if "parity_b0_vs_la" in s:
            line += f"\n          PARITY b0 vs la: {s['parity_b0_vs_la']}"
        lines.append(line)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    txt = "\n".join(lines)
    (out / "summary.txt").write_text(txt + "\n")
    return txt


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--records", nargs="+", required=True)
    r.add_argument("--source", default=None)
    r.add_argument("--rows", required=True)
    r.add_argument("--net", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--format", default="gen9randombattle")
    r.add_argument("--positions", type=int, default=120)
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--concurrency", type=int, default=3)
    r.add_argument("--worlds", type=int, default=8)
    r.add_argument("--margin", type=float, default=0.005)
    r.add_argument("--budgets", default="0,1024,2048,4096,8192")
    r.add_argument("--half-budgets", default="8192")
    r.add_argument("--nash-budgets", default="2048,8192")
    r.add_argument("--leaf-device", default="cuda")
    r.add_argument("--seed", type=int, default=11)
    p = sub.add_parser("report")
    p.add_argument("out")
    d = sub.add_parser("delta")
    d.add_argument("--records", nargs="+", required=True)
    d.add_argument("--source", default=None)
    d.add_argument("--rows", required=True)
    d.add_argument("--net", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--format", default="gen9randombattle")
    d.add_argument("--budget", type=int, required=True)
    d.add_argument("--worlds", type=int, default=8)
    d.add_argument("--ref-worlds", type=int, default=32)
    d.add_argument("--ref-scale", type=int, default=4)
    d.add_argument("--positions", type=int, default=100000)
    d.add_argument("--workers", type=int, default=10)
    d.add_argument("--concurrency", type=int, default=3)
    d.add_argument("--leaf-device", default="cuda")
    d.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    if a.cmd == "report":
        print(report(Path(a.out)))
        return 0
    if a.cmd == "delta":
        return asyncio.run(delta(a))
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
