"""Offline checks of the margin rule (porygon.netsearch) on logged positions; no games are played.

F2 records (exact first-person client streams, CONTRACTS.md "Data formats") are replayed into live-identical poke-env
battles with data's StreamReplayer (vendored read-only copy, vendor/SOURCES.txt). At sampled decisions:
  check  correctness: net_forward = porygon.v2.player.policy_probs (T 1 and T 0.5), the v2 mask = poke-env's mask,
         RootStats on real engine results = porygon.search's visit shares / data's logged Q definition, the
         search's pick = SearchPlayer.combine at prior 0, gate endpoints (+inf = network, -inf = search), and
         NetSearchPlayer.choose_move end to end on the replayed battle (no server).
  run    the decision-agreement / noise / timing study: per position the network's T-1 probabilities and P(win), a
         fresh search A, optional retest B (fresh, same settings), reference R (more samples x longer) and PUCT P,
         plus the original game's logged search when the record has one -> one JSON line per position
         (analysis: netsearch_analyze.py).

    cd $C && PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$C/src $PY scripts/fastenv/netsearch_study.py check \
        --records "$D/datasets/searchlabels/records/gen9randombattle/part-*.jsonl.gz" --net NET.pt --positions 12
    ... netsearch_study.py run --records ... --net NET.pt --out OUT.jsonl --positions 600 --per-record 2 \
        [--source 6771931] [--retest] [--ref 16x2000] [--puct 2.0] [--workers 7] [--concurrency 2]
netv2's porygon.v2 comes from NETV2_SRC (default its cluster copy, read-only), appended to porygon.__path__.
"""
import argparse
import asyncio
import glob
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def setup(fmt: str):
    """PORYGON_GEN from the format (before porygon.search is imported), our src first, netv2's porygon.v2 appended,
    data's replayer from the vendored copy. Returns (netsearch module, search module, replay module)."""
    gen = re.match(r"gen(\d+)", fmt).group(1)
    assert os.environ.setdefault("PORYGON_GEN", gen) == gen, f"PORYGON_GEN={os.environ['PORYGON_GEN']} vs {fmt}"
    sys.path.insert(0, str(ROOT / "src"))
    import porygon
    nv = Path(os.environ.get("NETV2_SRC", "/vol/bitbucket/fr1226/porygon-ws-netv2/src")) / "porygon"
    if str(nv) not in porygon.__path__:
        porygon.__path__.append(str(nv))
    spec = importlib.util.spec_from_file_location("data_replay", HERE / "vendor" / "data_replay.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    from porygon import netsearch, search
    return netsearch, search, replay


def records(patterns: list[str], source: str | None, seed: int):
    """F2 records from the files, shuffled (seeded), optionally only those whose source contains `source`."""
    files = sorted(f for p in patterns for f in glob.glob(p))
    assert files, f"no record files match {patterns}"
    recs = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if source is None or source in r.get("source", ""):
                    recs.append(r)
    random.Random(seed).shuffle(recs)
    return recs


def targets(rec: dict, per_record: int, seed: int) -> set[int]:
    """Message indices of up to per_record decisions with >= 2 legal actions, drawn reproducibly per record."""
    cands = [d["idx"] for d in rec["decisions"] if d.get("n_legal", 0) >= 2]
    h = int(hashlib.sha1(f"{seed}:{rec['id']}".encode()).hexdigest()[:12], 16)
    return set(random.Random(h).sample(cands, min(per_record, len(cands))))


async def replay_targets(rec: dict, R, NS, gen: int, chosen: set[int]):
    """Yield (decision, battle, weather first-seen turn, logged decision) at the chosen decisions; the weather
    tracker sees every multi-option decision in order, as the live player."""
    rep = R.StreamReplayer(rec["player"], gen, f"battle-{rec.get('battle') or rec['id']}")
    logged = {d["idx"]: d for d in rec["decisions"]}
    weather = NS.WeatherSeen()
    for msg in R.normalize_stream(rec["stream"]):
        d = rep.feed(msg)
        if d is None:
            continue
        first = weather.first_seen(rep.battle) if d.mask.sum() > 1 else None
        if d.msg in chosen and not d.retry:
            yield d, rep.battle, first, logged.get(d.msg)


def compact(stats, t: float) -> dict:
    return {"n": stats.n.astype(int).tolist(), "w": [[round(x, 2) for x in row] for row in stats.w.tolist()],
            "ok": stats.ok.tolist(), "err": stats.errors[:2], "t": round(t, 3)}


async def close_pool(pool) -> None:
    """Stop the engine workers before the event loop closes (else their transports complain at exit)."""
    while pool.idle is not None and not pool.idle.empty():
        proc = pool.idle.get_nowait()
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


def load_net(path: str, device: str):
    from porygon.v2.model import load
    return load(path, device=device).eval()


# ---------------------------------------------------------------- check

async def check(a, NS, S, R, gen: int) -> int:
    import numpy as np
    import torch
    from poke_env.environment import SinglesEnv
    from porygon.v2.player import policy_probs
    torch.set_num_threads(1)
    net = load_net(a.net, a.device)
    pool = S.engine_pool(a.workers)
    player = NS.NetSearchPlayer(net, battle_format=a.format, start_listening=False, ms=a.ms, samples=a.samples,
                                margin=0.03, seed=a.seed, workers=a.workers)
    combine_holder = type("C", (), {"top_k": None, "root_game": None, "prior_weight": 0.0})()
    fails, n = [], 0
    for rec in records(a.records, a.source, a.seed):
        if n >= a.positions:
            break
        async for d, battle, first, logged in replay_targets(rec, R, NS, gen, targets(rec, 1, a.seed)):
            n += 1
            tag = f"{rec['id']}@{d.msg}"
            probs, mask, win = NS.net_forward(net, battle)
            ref1, m1 = policy_probs(net, battle, 1.0)
            ref05, _ = policy_probs(net, battle, 0.5)
            if np.abs(probs - ref1).max() > 1e-9 or np.abs(NS.sharpen(probs, 0.5) - ref05).max() > 1e-9:
                fails.append(f"{tag}: net_forward/sharpen != policy_probs")
            pe_mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.float32)
            if not np.array_equal(mask > 0, pe_mask > 0) or not np.array_equal(mask > 0, d.mask > 0):
                fails.append(f"{tag}: v2 mask != poke-env mask")
            if not 0.0 <= win <= 1.0:
                fails.append(f"{tag}: P(win) {win}")
            # raw engine results through RootStats, porygon.search's own visit-share path and data's Q definition
            hist = S.History(battle)
            convs = [S.convert(battle, random.Random(n), first, hist) for _ in range(a.samples)]
            results = await asyncio.gather(*(pool.search(c.state_str, a.ms) for c in convs), return_exceptions=True)
            stats = NS.root_stats(convs, results, S.N_ACTIONS)
            if any(isinstance(r, BaseException) for r in results):
                fails.append(f"{tag}: engine errors {[r for r in results if isinstance(r, BaseException)][:1]}")
                continue
            shares = np.mean([S.SearchPlayer._sample_shares(c, r) for c, r in zip(convs, results)], axis=0)
            q_sum, q_n = np.zeros(S.N_ACTIONS), np.zeros(S.N_ACTIONS)
            for conv, r in zip(convs, results):
                for choice, v, w in r:
                    if v and choice != "No Move":
                        q_sum[conv.action(choice)] += w / v
                        q_n[conv.action(choice)] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                q_data = q_sum / q_n
            if not stats.ok.all():
                fails.append(f"{tag}: failed determinizations {stats.errors}")
            if np.abs(stats.share - shares).max() > 1e-12:
                fails.append(f"{tag}: RootStats.share != porygon.search shares")
            if not np.allclose(np.nan_to_num(stats.q, nan=-1), np.nan_to_num(q_data, nan=-1), atol=1e-12):
                fails.append(f"{tag}: RootStats.q != data's logged-q definition")
            pick = NS.search_pick(stats, mask)
            if pick != S.SearchPlayer.combine(combine_holder, shares, probs, mask):
                fails.append(f"{tag}: search_pick != SearchPlayer.combine at prior 0")
            for a_net in np.flatnonzero(mask > 0):
                g_hi, g_lo = NS.gate(int(a_net), stats, mask, math.inf), NS.gate(int(a_net), stats, mask, -math.inf)
                if g_hi.action != a_net or g_lo.action != pick:
                    fails.append(f"{tag}: gate endpoints ({g_hi}, {g_lo})")
            order = await player.choose_move(battle)
            if order is None or not order.message:
                fails.append(f"{tag}: choose_move returned {order!r}")
            print(f"{tag} turn {d.turn} {d.kind} legal {int(mask.sum())}: net {NS.sharpen(probs, .5).round(2).max()}"
                  f" pick {pick} q {np.nanmax(stats.q):.3f} visits {stats.visits:.0f} order {order.message!r}",
                  flush=True)
    await close_pool(pool)
    print(f"check: {n} positions, {len(fails)} failures; player stats {player.stats}")
    for f in fails[:20]:
        print("  FAIL", f)
    return 1 if fails or n == 0 else 0


# ---------------------------------------------------------------- run

async def run(a, NS, S, R, gen: int) -> int:
    import numpy as np
    import torch
    from poke_env.environment import SinglesEnv
    torch.set_num_threads(1)
    net = load_net(a.net, a.device)
    pool = S.engine_pool(a.workers)
    ref = tuple(int(x) for x in a.ref.split("x")) if a.ref else None
    out = open(a.out, "a")
    sem = asyncio.Semaphore(a.concurrency)
    done = {"n": 0, "searches": 0}
    t_start = time.time()

    async def one(rec):
        async with sem:
            if done["n"] >= a.positions:
                return
            rng = random.Random(f"{a.seed}:{rec['id']}")
            async for d, battle, first, logged in replay_targets(rec, R, NS, gen, targets(rec, a.per_record, a.seed)):
                if done["n"] >= a.positions:
                    return
                t0 = time.perf_counter()
                probs, mask, win = NS.net_forward(net, battle)
                t_net = time.perf_counter() - t0
                row = {"rec": rec["id"], "source": rec.get("source"), "side": rec.get("side"), "idx": d.msg,
                       "turn": d.turn, "kind": d.kind, "n_legal": int(mask.sum()), "result": rec.get("result"),
                       "rating": rec.get("rating"), "mask": (mask > 0).astype(int).tolist(),
                       "probs": [round(float(p), 6) for p in probs], "win": round(win, 5),
                       "mask_ok": bool(np.array_equal(mask > 0, np.array(SinglesEnv.get_action_mask(battle)) > 0)),
                       "t_net_ms": round(1000 * t_net, 3), "action": (logged or {}).get("a"),
                       "logged": None}
                if logged and logged.get("search"):
                    s = logged["search"]
                    row["logged"] = {"q": s.get("q"), "shares": s.get("shares"), "visits": s.get("visits")}
                arms = [("A", a.samples, a.ms, 0.0)]
                if a.retest:
                    arms.append(("B", a.samples, a.ms, 0.0))
                if ref:
                    arms.append(("R", ref[0], ref[1], 0.0))
                if a.puct:
                    arms.append(("P", a.samples, a.ms, a.puct))
                for name, k, ms, c in arms:  # one after another: each search has the pool's full width
                    t0 = time.perf_counter()
                    st, _ = await NS.search_root(pool, battle, rng, first, samples=k, ms=ms, probs=probs, mask=mask,
                                                 c_puct=c)
                    row[name] = compact(st, time.perf_counter() - t0)
                    done["searches"] += 1
                out.write(json.dumps(row) + "\n")
                out.flush()
                done["n"] += 1
                if done["n"] % 25 == 0:
                    print(f"{done['n']} positions, {done['searches']} searches, {time.time() - t_start:.0f}s",
                          flush=True)

    recs = records(a.records, a.source, a.seed)
    tasks = [asyncio.create_task(one(r)) for r in recs[:max(1, 3 * a.positions // max(a.per_record, 1))]]
    await asyncio.gather(*tasks)
    out.close()
    await close_pool(pool)
    print(f"done: {done['n']} positions in {time.time() - t_start:.0f}s -> {a.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["check", "run"])
    p.add_argument("--records", nargs="+", required=True, help="F2 record files (globs)")
    p.add_argument("--source", help="only records whose source contains this (e.g. a run's job id)")
    p.add_argument("--net", required=True, help="v2 checkpoint")
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--positions", type=int, default=100)
    p.add_argument("--per-record", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--ms", type=int, default=1000)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--retest", action="store_true", help="a second fresh search B with A's settings")
    p.add_argument("--ref", help="reference search R, SAMPLESxMS (e.g. 16x2000)")
    p.add_argument("--puct", type=float, default=0.0, help="a PUCT search P with this c_puct (network T-1 priors)")
    p.add_argument("--workers", type=int, default=None, help="engine processes (default cores - 1)")
    p.add_argument("--concurrency", type=int, default=2, help="records in flight")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out")
    a = p.parse_args()
    NS, S, R = setup(a.format)
    gen = int(re.match(r"gen(\d+)", a.format).group(1))
    if a.mode == "run":
        assert a.out, "--out"
        return asyncio.run(run(a, NS, S, R, gen))
    return asyncio.run(check(a, NS, S, R, gen))


if __name__ == "__main__":
    sys.exit(main())
