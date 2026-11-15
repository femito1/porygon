"""V3 gates P1 (leaf-encoder parity) and P2 (leaf throughput), pre-registered in
`.orchestra/fastenv/V3-GATES-PREREG.md` (ws/fastenv docs/netsearch/V3-GATES-PREREG.md). No games are played.

  p1   replay F2 records (netsearch_study's replayer), at consecutive decision pairs (t, t+1) compare
       porygon.v3leaf's patched observations with the real encoder on b(t+1): test A (root b(t+1), state
       convert(b(t+1))) and test B (root b(t), state convert(b(t+1))); element agreement over the set D and
       |delta P(win)| of the net. Writes p1.json (summary + rule), p1-rows.jsonl, roots.pkl (b(t) roots for p2).
  p2   W worker processes (battles x determinizations) each run a batched decoupled-UCT tree with net P(win)
       leaves on the GPU for `ms` per decision, all at once (a barrier per window); leaves per decision and the
       time split; plus the GPU-only forward rate. Writes p2.json (summary + rule) and p2-rows.jsonl.

    PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$C/src NETV2_SRC=<frozen netv2 src> \
      $PY scripts/fastenv/v3_gates.py p1 --records "$SL" --source 6771931 --net NET --out OUT
    ... v3_gates.py p2 --roots OUT/roots.pkl --net NET --out OUT
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def prep():
    import netsearch_study as ST
    NS, S, R = ST.setup("gen9randombattle")
    from porygon import v3leaf as L
    return ST, NS, S, R, L


def load_net(path: str, device: str):
    import torch
    torch.set_num_threads(1)
    from porygon.v2.model import load
    return load(path, device=device).eval()


def win_probs(net, obs_list, device="cpu", bs=64) -> np.ndarray:
    import torch
    from porygon.v2.model import obs_to_tensors
    out = []
    with torch.inference_mode():
        for i in range(0, len(obs_list), bs):
            o = net(obs_to_tensors(obs_list[i:i + bs], device), aux=False)
            out.append(torch.sigmoid(o["win_logit"].double()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------- P1

def d_selectors(L):
    """The compared set D: (name, field, index function) with the comparison kind."""
    from porygon.v2.spec import STATUSES
    global L_TERA
    C = L.COL
    L_TERA = C["terastallized"]
    cols = (["present", "revealed", "active", "fainted", "hp_frac"] + [f"status_{s}" for s in STATUSES]
            + ["status_counter", "terastallized", "first_turn", "protect_counter", "must_recharge", "preparing",
               "trapped"])
    sel = [("hp", "hp", None, "hp"), ("boosts", "boosts", None, "float")]
    sel += [(f"mon_num.{c}", "mon_num", (slice(None), C[c]), "float") for c in cols]
    sel += [("mon_num.effects", "mon_num", (slice(None), slice(L.EFFECT0, L.EFFECT0 + len(L.E.EFFECTS))), "float")]
    sel += [("move_num.pp_frac", "move_num", (slice(None), slice(None), L.PP), "float"),
            ("move_num.available", "move_num", (slice(None), slice(None), L.AVAIL), "float"),
            ("tera", "tera", None, "exact")]
    fcols = [c for c in L.FIELD_NUM if c.startswith(("weather_", "terrain_", "trickroom"))] + [
        "turn", "opp_unseen", "own_alive", "opp_alive", "force_switch", "trapped", "can_tera", "used_tera",
        "opp_used_tera", "wait", "n_switches", "n_moves"]
    sel += [(f"field_num.{c}", "field_num", (L.FCOL[c],), "float") for c in fcols]
    sel += [("side_num", "side_num", None, "float"), ("action_mask", "action_mask", None, "float")]
    return sel


L_TERA = None  # mon_num column of "terastallized", set by d_selectors


def compare(sel, got: dict, real: dict) -> tuple[int, int, dict]:
    """(agreeing elements, compared elements, {selector name: disagreements}) over D; elements where both are 0
    are not compared; tera_p is compared only on mons the real encoder shows terastallized."""
    agree = total = 0
    bad = {}
    for name, field, idx, kind in sel:
        a = np.asarray(got[field] if idx is None else got[field][idx], dtype=np.float64)
        b = np.asarray(real[field] if idx is None else real[field][idx], dtype=np.float64)
        live = (a != 0) | (b != 0)
        if kind == "hp":
            scale = np.maximum(np.asarray(real["hp"])[:, 1:2], 1.0)
            ok = np.abs(a - b) <= 0.02 * scale
        elif kind == "exact":
            ok = a == b
        else:
            ok = np.abs(a - b) <= 0.02 + 1e-9
        n, k = int(live.sum()), int((live & ok).sum())
        agree, total = agree + k, total + n
        if n - k:
            bad[name] = n - k
    tmask = np.asarray(real["mon_num"])[:, L_TERA] > 0
    if tmask.any():
        a, b = np.asarray(got["tera_p"])[tmask], np.asarray(real["tera_p"])[tmask]
        live = (a != 0) | (b != 0)
        ok = np.abs(a - b) <= 0.02 + 1e-9
        agree, total = agree + int((live & ok).sum()), total + int(live.sum())
        if int((live & ~ok).sum()):
            bad["tera_p"] = int((live & ~ok).sum())
    return agree, total, bad


def revealed_between(o0: dict, o1: dict) -> bool:
    return any(not np.array_equal(o0[k][6:], o1[k][6:]) for k in ("species", "moves", "item", "ability", "tera"))


def pair_targets(rec: dict, per_record: int, seed: int) -> list[tuple[int, int]]:
    seq = sorted(d["idx"] for d in rec["decisions"] if d.get("n_legal", 0) >= 1)
    cands = list(range(len(seq) - 1))
    h = int(hashlib.sha1(f"{seed}:{rec['id']}:{rec.get('side')}".encode()).hexdigest()[:12], 16)
    ks = random.Random(h).sample(cands, min(per_record, len(cands)))
    return [(seq[k], seq[k + 1]) for k in sorted(ks)]


async def replay_pairs(rec, R, NS, pairs):
    """Yield (t_idx, t1_idx, battle at t+1, first seen) after calling `at_t(battle, first)` hooks via a dict."""
    rep = R.StreamReplayer(rec["player"], 9, f"battle-{rec.get('battle') or rec['id']}")
    weather = NS.WeatherSeen()
    want = {t for p in pairs for t in p}
    for msg in R.normalize_stream(rec["stream"]):
        d = rep.feed(msg)
        if d is None:
            continue
        first = weather.first_seen(rep.battle) if d.mask.sum() > 1 else None
        if d.msg in want and not d.retry:
            yield d.msg, rep.battle, first


async def p1(a) -> int:
    import asyncio  # noqa: F401
    ST, NS, S, R, L = prep()
    net = load_net(a.net, "cpu")
    sel = d_selectors(L)
    recs = ST.records(a.records.split(), a.source, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = open(out / "p1-rows.jsonl", "w")
    roots = []
    n_pairs = n_fail = 0
    tA = [0, 0]
    tB = [0, 0]
    dA, dB, dB_rev, dB_norev = [], [], [], []
    tBr = {True: [0, 0], False: [0, 0]}
    badA, badB = {}, {}
    t0 = time.time()
    for rec in recs:
        if n_pairs >= a.pairs:
            break
        pairs = pair_targets(rec, a.per_record, a.seed)
        firsts = {p[0] for p in pairs}
        seconds = {p[1]: p[0] for p in pairs}
        at_t = {}
        try:
            async for idx, battle, first in replay_pairs(rec, R, NS, pairs):
                if idx in seconds and seconds[idx] in at_t and n_pairs < a.pairs:
                    root_t, convs_t, obs_t, turn_t = at_t[seconds[idx]]
                    root1 = L.LeafRoot(battle, first)
                    hist = S.History(battle)
                    real = root1.obs
                    obsA, obsB = [], []
                    for sd in range(a.dets):
                        conv1 = S.convert(battle, random.Random(sd), first, hist)
                        encA = L.LeafEncoder(root1, conv1)
                        obsA.append(encA.encode(encA.state, turns=0))
                        encB = L.LeafEncoder(root_t, convs_t[sd])
                        st1 = L.pe.State.from_string(conv1.state_str)
                        obsB.append(encB.encode(st1, turns=max(battle.turn - turn_t, 0)))
                    pv = win_probs(net, [real] + obsA + obsB)
                    rev = revealed_between(obs_t, real)
                    row = {"rec": rec["id"], "side": rec.get("side"), "t": seconds[idx], "t1": idx,
                           "turn": battle.turn, "revealed": rev, "p_real": float(pv[0]), "A": [], "B": []}
                    for k, (test, lst, tot, dl, bad) in enumerate(
                            (("A", obsA, tA, dA, badA), ("B", obsB, tB, dB, badB))):
                        for sd, o in enumerate(lst):
                            ag, n, bd = compare(sel, o, real)
                            tot[0] += ag
                            tot[1] += n
                            dp = abs(float(pv[1 + k * a.dets + sd]) - float(pv[0]))
                            dl.append(dp)
                            if test == "B":
                                (dB_rev if rev else dB_norev).append(dp)
                                tBr[rev][0] += ag
                                tBr[rev][1] += n
                            for kk, v in bd.items():
                                bad[kk] = bad.get(kk, 0) + v
                            row[test].append({"agree": ag, "n": n, "dp": dp, "bad": bd})
                    rows_f.write(json.dumps(row) + "\n")
                    roots.append((root_t, convs_t))
                    n_pairs += 1
                if idx in firsts:
                    hist = S.History(battle)
                    convs = [S.convert(battle, random.Random(sd), first, hist) for sd in range(4)]
                    root = L.LeafRoot(battle, first)
                    at_t[idx] = (root, convs, root.obs, battle.turn)
        except Exception as e:  # a record that fails to replay or convert counts against the void rule
            n_fail += 1
            print(f"record {rec['id']}: {type(e).__name__}: {e}", file=sys.stderr)
        if n_pairs and n_pairs % 50 == 0:
            print(f"{n_pairs} pairs, {time.time() - t0:.0f} s", file=sys.stderr)
    rows_f.close()

    def q(x):
        return {f"q{p}": float(np.quantile(x, p / 100)) for p in (50, 90, 99)} if x else {}
    agreeA = tA[0] / max(tA[1], 1)
    mA, mB = float(np.mean(dA)) if dA else math.nan, float(np.mean(dB)) if dB else math.nan
    res = {
        "pairs": n_pairs, "target_pairs": a.pairs, "failed_records": n_fail, "dets": a.dets,
        "A": {"agree": agreeA, "elements": tA[1], "mean_abs_dp": mA, **q(dA),
              "bad": dict(sorted(badA.items(), key=lambda kv: -kv[1]))},
        "B": {"agree": tB[0] / max(tB[1], 1), "elements": tB[1], "mean_abs_dp": mB, **q(dB),
              "mean_abs_dp_revealed": float(np.mean(dB_rev)) if dB_rev else None, "n_revealed": len(dB_rev),
              "mean_abs_dp_not_revealed": float(np.mean(dB_norev)) if dB_norev else None,
              "agree_revealed": tBr[True][0] / max(tBr[True][1], 1),
              "agree_not_revealed": tBr[False][0] / max(tBr[False][1], 1),
              "bad": dict(sorted(badB.items(), key=lambda kv: -kv[1]))},
        "rule": "PASS iff A agree >= 0.99 and A mean|dP| <= 0.01 and B mean|dP| <= 0.03",
        "seconds": time.time() - t0,
    }
    res["pass"] = bool(agreeA >= 0.99 and mA <= 0.01 and mB <= 0.03)
    res["void"] = n_pairs < 0.9 * a.pairs
    (out / "p1.json").write_text(json.dumps(res, indent=1))
    with open(out / "roots.pkl", "wb") as fh:
        pickle.dump(roots, fh)
    print(json.dumps({k: res[k] for k in ("pairs", "pass", "void")} | {
        "A": {k: res["A"][k] for k in ("agree", "mean_abs_dp")}, "B": {k: res["B"][k] for k in ("agree", "mean_abs_dp")}}))
    return 0


# ---------------------------------------------------------------- P2: a batched decoupled-UCT leaf workload

class Node:
    __slots__ = ("state", "c1", "a1", "c2", "a2", "n", "n1", "w1", "n2", "w2", "edges", "term", "turns")

    def __init__(self, L, state, tera_ok: bool, turns: int):
        self.state, self.turns = state, turns
        self.term = L.terminal(state)
        s1, s2 = state.side_one, state.side_two
        self.c1, self.a1 = L.side_options(s1, s2, tera_ok)
        self.c2, self.a2 = L.side_options(s2, s1, tera_ok)
        self.n = 0
        self.n1, self.w1 = np.zeros(len(self.c1)), np.zeros(len(self.c1))
        self.n2, self.w2 = np.zeros(len(self.c2)), np.zeros(len(self.c2))
        self.edges = {}  # (i, j) -> [branches, probs, children]


def ucb(n, w, total, c=1.4):
    unv = np.flatnonzero(n == 0)
    if len(unv):
        return int(unv[0])
    return int(np.argmax(w / n + c * np.sqrt(math.log(total) / n)))


def worker(args):
    """One process = one determinization of one battle: for every (root, B) window, wait at the barrier, then
    run the tree for `ms`; returns per-window counts and the time split."""
    idx, battle, det, roots_path, net_path, ms, batches, barrier, n_roots = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    ST, NS, S, R, L = prep()
    from porygon.v2.model import obs_to_tensors
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_net(net_path, dev)
    roots = pickle.load(open(roots_path, "rb"))[:n_roots]
    rng = random.Random(1000 * idx + 7)
    # warm-up (CUDA kernels, caches)
    with torch.inference_mode():
        net(obs_to_tensors([roots[0][0].obs] * 8, dev), aux=False)
    rows = []
    for B in batches:
        for r, (root, convs) in enumerate(roots):
            enc = L.LeafEncoder(root, convs[det])
            tera_ok = not root.used_tera
            barrier.wait()
            t = {"engine": 0.0, "encode": 0.0, "tree": 0.0, "collate": 0.0, "forward": 0.0}
            t_start = time.perf_counter()
            deadline = t_start + ms / 1000
            rootn = Node(L, enc.state, tera_ok, 0)
            t["engine"] += time.perf_counter() - t_start
            leaves = terms = errors = 0
            depth_sum = 0
            while time.perf_counter() < deadline:
                pending = []  # (obs, path)
                while len(pending) < B and time.perf_counter() < deadline:
                    t0 = time.perf_counter()
                    node, path, depth = rootn, [], 0
                    child = None
                    while True:
                        if node.term is not None:
                            child = node
                            break
                        node.n += 1
                        i = ucb(node.n1, node.w1, node.n)
                        j = ucb(node.n2, node.w2, node.n)
                        node.n1[i] += 1
                        node.n2[j] += 1  # virtual loss: the visit counts now, its value at backup
                        path.append((node, i, j))
                        e = node.edges.get((i, j))
                        t["tree"] += time.perf_counter() - t0
                        t0 = time.perf_counter()
                        if e is None:
                            try:
                                br = L.pe.generate_instructions(node.state, L.engine_choice(node.c1[i]),
                                                                L.engine_choice(node.c2[j]))
                            except BaseException as ex:  # an engine panic on an odd option: score it 0.5
                                if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                                    raise
                                br = []
                            if not br:
                                errors += 1
                                e = [[], np.ones(1), [None]]
                            else:
                                p = np.array([b.percentage for b in br], dtype=np.float64)
                                e = [br, p / p.sum(), [None] * len(br)]
                            node.edges[(i, j)] = e
                        t["engine"] += time.perf_counter() - t0
                        t0 = time.perf_counter()
                        if not e[0]:
                            child = None
                            break
                        k = int(np.searchsorted(np.cumsum(e[1]), rng.random() * 0.999999))
                        nxt = e[2][k]
                        depth += 1
                        if nxt is None:
                            t["tree"] += time.perf_counter() - t0
                            t0 = time.perf_counter()
                            st = node.state.apply_instructions(e[0][k])
                            busy = (node.c1[i] != "none") and (node.c2[j] != "none")
                            nxt = Node(L, st, tera_ok, node.turns + int(busy))
                            e[2][k] = nxt
                            t["engine"] += time.perf_counter() - t0
                            child = nxt
                            break
                        node = nxt
                    t["tree"] += time.perf_counter() - t0
                    if child is None or child.term is not None:
                        v = 0.5 if child is None else child.term
                        t0 = time.perf_counter()
                        backup(path, v)
                        t["tree"] += time.perf_counter() - t0
                        terms += 1
                        continue
                    t0 = time.perf_counter()
                    obs = enc.encode(child.state, turns=child.turns)
                    t["encode"] += time.perf_counter() - t0
                    pending.append((obs, path))
                    depth_sum += depth
                if not pending:
                    break
                t0 = time.perf_counter()
                x = obs_to_tensors([o for o, _ in pending], dev)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t["collate"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                with torch.inference_mode():
                    v = torch.sigmoid(net(x, aux=False)["win_logit"].float()).cpu().numpy()
                t1 = time.perf_counter()
                t["forward"] += t1 - t0
                if t1 <= deadline:
                    leaves += len(pending)
                t0 = time.perf_counter()
                for (o, path), vv in zip(pending, v):
                    backup(path, float(vv))
                t["tree"] += time.perf_counter() - t0
            wall = time.perf_counter() - t_start
            rows.append({"worker": idx, "battle": battle, "det": det, "B": B, "root": r, "leaves": leaves,
                         "terminal": terms, "errors": errors, "wall": wall, "t": t,
                         "mean_depth": depth_sum / max(leaves, 1), "root_visits": int(rootn.n)})
    return rows


def backup(path, v: float) -> None:
    for node, i, j in path:
        node.w1[i] += v
        node.w2[j] += 1.0 - v


def gpu_only(args):
    idx, net_path, roots_path, sizes, seconds, barrier = args
    import torch
    torch.set_num_threads(1)
    ST, NS, S, R, L = prep()
    from porygon.v2.model import obs_to_tensors
    net = load_net(net_path, "cuda")
    obs = pickle.load(open(roots_path, "rb"))[0][0].obs
    out = []
    for B in sizes:
        x = obs_to_tensors([obs] * B, "cuda")
        with torch.inference_mode():
            net(x, aux=False)
        torch.cuda.synchronize()
        barrier.wait()
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            with torch.inference_mode():
                net(x, aux=False)["win_logit"].cpu()
            n += B
        out.append({"proc": idx, "B": B, "rate": n / (time.perf_counter() - t0)})
    return out


def p2(a) -> int:
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    batches = [int(x) for x in a.batches.split(",")]
    n_w = a.battles * a.dets
    prep()  # porygon.v2 on the path before unpickling the roots
    roots = pickle.load(open(a.roots, "rb"))
    order = list(range(len(roots)))
    random.Random(a.seed).shuffle(order)
    chosen = [roots[i] for i in order[:a.n_roots]]
    rp = out / "p2-roots.pkl"
    with open(rp, "wb") as fh:
        pickle.dump(chosen, fh)
    t0 = time.time()
    with ctx.Manager() as m:
        barrier = m.Barrier(n_w)
        jobs = [(w, w // a.dets, w % a.dets, str(rp), a.net, a.ms, batches, barrier, len(chosen))
                for w in range(n_w)]
        with ctx.Pool(n_w) as pool:
            res = pool.map(worker, jobs)
        rows = [r for rs in res for r in rs]
        gpu = {}
        import torch
        for P in ((1, n_w) if torch.cuda.is_available() else ()):
            b2 = m.Barrier(P)
            with ctx.Pool(P) as pool:
                g = pool.map(gpu_only, [(i, a.net, str(rp), [64, 256, 1024], 3.0, b2) for i in range(P)])
            gpu[P] = {B: float(sum(x["rate"] for gg in g for x in gg if x["B"] == B)) for B in (64, 256, 1024)}
    with open(out / "p2-rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summ = {}
    for B in batches:
        dec = {}
        split = {k: 0.0 for k in ("engine", "encode", "tree", "collate", "forward")}
        wall = leaves = 0.0
        for r in rows:
            if r["B"] != B:
                continue
            dec[(r["battle"], r["root"])] = dec.get((r["battle"], r["root"]), 0) + r["leaves"]
            for k in split:
                split[k] += r["t"][k]
            wall += r["wall"]
            leaves += r["leaves"]
        vals = sorted(dec.values())
        summ[B] = {"decisions": len(vals), "median_leaves_per_decision": float(np.median(vals)) if vals else 0.0,
                   "q10": float(np.quantile(vals, 0.1)) if vals else 0.0,
                   "q90": float(np.quantile(vals, 0.9)) if vals else 0.0,
                   "us_per_leaf": {k: 1e6 * v / max(leaves, 1) for k, v in split.items()},
                   "share_of_wall": {k: v / max(wall, 1e-9) for k, v in split.items()},
                   "mean_depth": float(np.mean([r["mean_depth"] for r in rows if r["B"] == B])),
                   "errors": int(sum(r["errors"] for r in rows if r["B"] == B))}
    best = max(batches, key=lambda B: summ[B]["median_leaves_per_decision"])
    sb = summ[best]
    share = sb["share_of_wall"]
    largest = max(share, key=share.get)
    per = sb["us_per_leaf"]
    tot = sum(per.values())
    proj_enc20 = sb["median_leaves_per_decision"] * tot / max(tot - per["encode"] * (1 - 1 / 20), 1e-9)
    res = {"batches": summ, "best_B": best, "median_leaves_per_decision": sb["median_leaves_per_decision"],
           "pass": bool(sb["median_leaves_per_decision"] >= 2000),
           "void": sb["decisions"] < 0.9 * a.battles * a.n_roots,
           "largest_share": largest, "projected_with_encode_20x_faster": proj_enc20,
           "gpu_only_leaves_per_s": gpu,
           "projected_shared_server_leaves_per_decision": (gpu[n_w][256] * a.ms / 1000 / a.battles) if gpu else None,
           "rule": "PASS iff median leaves/decision (4 workers x ms) at the best B >= 2000",
           "workers": n_w, "ms": a.ms, "seconds": time.time() - t0}
    (out / "p2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("best_B", "median_leaves_per_decision", "pass", "void", "largest_share",
                                          "projected_with_encode_20x_faster")} | {"share": share, "us": per}))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("p1")
    a1.add_argument("--records", required=True, help="space-separated globs")
    a1.add_argument("--source", default=None)
    a1.add_argument("--net", required=True)
    a1.add_argument("--out", required=True)
    a1.add_argument("--seed", type=int, default=7311)
    a1.add_argument("--pairs", type=int, default=300)
    a1.add_argument("--per-record", type=int, default=3)
    a1.add_argument("--dets", type=int, default=2)
    a2 = sub.add_parser("p2")
    a2.add_argument("--roots", required=True)
    a2.add_argument("--net", required=True)
    a2.add_argument("--out", required=True)
    a2.add_argument("--seed", type=int, default=7312)
    a2.add_argument("--n-roots", type=int, default=30)
    a2.add_argument("--battles", type=int, default=3)
    a2.add_argument("--dets", type=int, default=4)
    a2.add_argument("--ms", type=int, default=1000)
    a2.add_argument("--batches", default="16,32,64")
    a = p.parse_args()
    if a.cmd == "p1":
        import asyncio
        return asyncio.run(p1(a))
    return p2(a)


if __name__ == "__main__":
    sys.exit(main())
