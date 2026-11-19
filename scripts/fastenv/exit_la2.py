"""ExIt-LA2 pilot (docs/netsearch/EXIT-LA2-PREREG.md, design EXIT-LA2-DESIGN.md): distil lookahead2's played policy
into G9q-cd@805M. Subcommands, run in order by scripts/fastenv/exit_pilot.sbatch:

  gen      fastenv self-play of the net vs itself at T 0.5 (host.js --dump-states --dump-streams into one FIFO per
           worker; one reader process per worker keeps only the S lines' prefixes and the G lines); per finished
           game one decision per side, uniform over the side's decisions with >= 2 legal actions (sha1(seed, game,
           side)); one F2-style record per chosen (game, side) with the self-play forward's T-1 probabilities there.
           Ties, turn-cap games and crashes are dropped. gzip'd JSONL parts (rec-a<attempt>-w<w>-NNNN.jsonl.gz, each
           committed by its .json marker). Re-running continues (finished parts count).
  label    replay each record (netsearch_study.replay_targets, data's StreamReplayer), the net's p on the replayed
           battle (parity vs the self-play p: max |dp| <= --parity-tol, same legal set; else dropped), the v2 obs,
           one lookahead2 call (K worlds, the module defaults of LookaheadPlayer) and pi_LA (the exact distribution
           of lookahead2's played move over the net's T 0.5 draw at margin delta). npz parts + meta JSONL;
           dropped rows in drops-*.jsonl. Re-running skips records already labelled or dropped.
  shards   the E (target normalize(sqrt(pi_LA))) and C (target normalize(sqrt(p_0.5)) = p) shard sets in
           porygon.v2.pretrain's format (obs/*, target, z, game) from the same label parts.
  check    manipulation and value checks on the held-out games (pretrain.is_val(game, 0.05)): r = 1 - mean KL(pi_LA ||
           p_0.5 of X) / mean KL(pi_LA || p_0.5 of S), top-1 agreement on overridden positions, value log-loss / AUC.
  eval     porygon.fastenv_eval duplicate pairs between two v2 nets (both T 0.5) -> JSON.
  verdict  stage 1's verdict from the check and eval JSONs and the label counts (PREREG rule).

Environment as lookahead2_study: PORYGON_GEN=9, PYTHONPATH = gen9 engine build + our src, NETV2_SRC = the frozen
netv2 tree's src (porygon.v2 appended to porygon.__path__ by netsearch_study.setup).
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PART_GAMES = 1000  # games per record part
PART_ROWS = 2000  # rows per label part
T_NET = 0.5


# ---------------------------------------------------------------- pure functions (tests/test_exit_la2.py)

def sharpen(p: np.ndarray, t: float) -> np.ndarray:
    """softmax(logits / T) from softmax(logits) (= porygon.netsearch.sharpen; 0 stays 0)."""
    p = np.asarray(p, dtype=np.float64)
    if t == 1.0:
        return p
    with np.errstate(divide="ignore"):
        z = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)) / t, -np.inf)
    if not np.isfinite(z).any():
        return p
    e = np.exp(z - z[np.isfinite(z)].max())
    return e / e.sum()


def pi_la(probs: np.ndarray, rows: list[int], x: np.ndarray, q: np.ndarray, delta: float,
          temperature: float = T_NET) -> tuple[np.ndarray, np.ndarray]:
    """(pi_LA, p_T): pi_LA(a) = sum_a' p_T(a') 1[g(a') = a] with g = porygon.lookahead2.gate's played action for the
    draw a' (a_L iff a_L != a', a' has a row and Q(a_L) - Q(a') > delta; else a')."""
    from porygon.lookahead2 import gate
    p = sharpen(probs, temperature)
    pi = np.zeros_like(p)
    rows = [int(r) for r in rows]
    for a in np.flatnonzero(p > 0):
        pi[gate(int(a), rows, np.asarray(x), np.asarray(q), delta).action] += p[a]
    return pi, p


def sqrt_target(p: np.ndarray) -> np.ndarray:
    """The T-1 target whose T 0.5 sharpening is p: normalize(sqrt(p)) (rows of a 2-D array, or one vector)."""
    s = np.sqrt(np.clip(np.asarray(p, dtype=np.float64), 0, None))
    return (s / s.sum(-1, keepdims=True)).astype(np.float32)


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) per row over p > 0 (q floored at 1e-12)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(q)), 0.0)
    return t.sum(-1)


def is_val(game: np.ndarray, frac: float) -> np.ndarray:
    """porygon.v2.pretrain.is_val (fba02fb), copied: crc32 of the game id, so pretrain and this check agree."""
    import zlib
    h = np.array([zlib.crc32(int(g).to_bytes(8, "little", signed=True)) for g in game], dtype=np.uint64)
    return (h % 10_000) < frac * 10_000


def game_hash(game: str) -> int:
    return int(hashlib.sha1(game.encode()).hexdigest()[:15], 16)


def pick(seed: int, game: str, side: int, n: int) -> int:
    return int(hashlib.sha1(f"{seed}:{game}:{side}".encode()).hexdigest()[:12], 16) % n


def interval(xs) -> tuple[float, float]:
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n < 2:
        return (float(xs.mean()) if n else math.nan), math.nan
    return float(xs.mean()), float(1.96 * xs.std(ddof=1) / math.sqrt(n))


def verdict_line(t1: dict, t2: dict) -> str:
    """PREREG rule on T1 (E vs C) and T2 (E vs S): win rate and ci95 in [0, 1]."""
    lo1, hi1 = t1["win_rate"] - t1["ci95"], t1["win_rate"] + t1["ci95"]
    lo2 = t2["win_rate"] - t2["ci95"]
    if lo1 > 0.5 and lo2 > 0.5:
        return "GO"
    if lo1 > 0.5:
        return "MIXED"
    if hi1 < 0.5:
        return "HARMFUL"
    if hi1 < 0.51:
        return "FLAT"
    return "NOT SHOWN"


# ---------------------------------------------------------------- gen

def reader(w, fifo, tq, out, attempt, counter, errq, seed, fmt) -> None:
    try:
        _reader(w, fifo, tq, out, attempt, counter, seed, fmt)
    except BaseException as e:  # noqa: BLE001  (reported to the main process, which fails the run)
        import traceback
        errq.put(f"reader {w}: {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
        raise


def _reader(w, fifo, tq, out, attempt, counter, seed, fmt) -> None:
    lines: queue.Queue = queue.Queue(maxsize=4096)

    def pump():
        with open(fifo, "r", buffering=1 << 20) as fh:
            for line in fh:
                lines.put(line)
        lines.put(None)
    threading.Thread(target=pump, daemon=True).start()
    msgs: deque = deque()
    stop = False
    pending: dict[str, list] = defaultdict(list)
    buf: list[str] = []
    stats = {"games": 0, "records": 0, "dropped_games": 0, "sides_without_choice": 0}
    n_part = [0]
    games_in_buf = [0]

    def flush():
        if not buf:
            return
        name = f"rec-a{attempt}-w{w}-{n_part[0]:04d}"
        tmp = Path(out) / f".{name}.jsonl.gz"
        with gzip.open(tmp, "wt", compresslevel=6) as fh:
            fh.write("".join(buf))
        os.replace(tmp, Path(out) / f"{name}.jsonl.gz")
        (Path(out) / f"{name}.json").write_text(json.dumps({"games": games_in_buf[0], "records": len(buf),
                                                             "attempt": attempt, "worker": w, **stats}))
        n_part[0] += 1
        buf.clear()
        games_in_buf[0] = 0

    while True:
        line = lines.get()
        if line is None:
            break
        if line.startswith('{"t":"G"'):
            g = json.loads(line)
            decs = pending.pop(g["game"], [])
            if g["crashed"] or g["truncated"] or g["winner"] not in ("p1", "p2") or "streams" not in g:
                stats["dropped_games"] += 1
                continue
            recs = []
            for side in (0, 1):
                cands = [d for d in decs if d["side"] == side and d["n_legal"] >= 2]
                if not cands:
                    stats["sides_without_choice"] += 1
                    continue
                d = cands[pick(seed, g["game"], side, len(cands))]
                recs.append({"id": f"{g['game']}-p{side + 1}", "battle": g["game"].replace(":", "-"), "format": fmt,
                             "side": side, "player": f"p{side + 1}", "stream": g["streams"][side],
                             "decisions": [{"idx": d["e"], "n_legal": d["n_legal"], "turn": d["turn"],
                                            "didx": d["idx"], "legal": d["legal"], "probs": d["probs"]}],
                             "z": 1 if g["winner"] == f"p{side + 1}" else -1, "game": g["game"],
                             "game_id": game_hash(g["game"]), "turns": g["turns"], "n_candidates": len(cands)})
            stats["games"] += 1
            stats["records"] += len(recs)
            games_in_buf[0] += 1
            buf.extend(json.dumps(r) + "\n" for r in recs)
            with counter.get_lock():
                counter.value += 1
            if games_in_buf[0] >= PART_GAMES:
                flush()
            continue
        cut = line.find(',"state":')
        x = json.loads(line[:cut] + "}" if cut > 0 else line)
        while not msgs and not stop:
            m = tq.get()
            if m is None:
                stop = True
            else:
                msgs.extend(m)
        if stop and not msgs:
            break  # the run ended: unfinished games are dropped
        slot, didx, legal, probs = msgs.popleft()
        if slot != x["slot"] or didx != x["idx"]:
            raise RuntimeError(f"join mismatch: row (slot {slot}, decision {didx}) vs S line (slot {x['slot']}, "
                               f"idx {x['idx']}, game {x['game']})")
        pending[x["game"]].append({"side": int(x["side"]), "e": int(x["e"]), "idx": int(x["idx"]),
                                   "turn": int(x["turn"]), "n_legal": len(legal), "legal": legal, "probs": probs})
    flush()


def finished_games(out: Path) -> tuple[int, int]:
    metas = [json.loads(p.read_text()) for p in out.glob("rec-*.json")
             if p.with_name(p.name[:-5] + ".jsonl.gz").exists()]
    return sum(m["games"] for m in metas), 1 + max([m["attempt"] for m in metas], default=-1)


def cmd_gen(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    have, attempt = finished_games(out)
    if have >= a.games:
        print(f"already {have} games (target {a.games})")
        return 0
    seed = a.seed + 100 * attempt
    fdir = Path(a.fifo_dir) / f"exit-gen-{os.getpid()}"
    fdir.mkdir(parents=True, exist_ok=True)
    fifos = []
    for w in range(a.workers):
        f = fdir / f"dump-{w}.fifo"
        if f.exists():
            f.unlink()
        os.mkfifo(f)
        fifos.append(str(f))
    ctx = mp.get_context("spawn")
    counters = [ctx.Value("q", 0) for _ in range(a.workers)]
    tqs = [ctx.Queue() for _ in range(a.workers)]
    errq = ctx.Queue()
    procs = [ctx.Process(target=reader, args=(w, fifos[w], tqs[w], str(out), attempt, counters[w], errq, a.seed,
                                              a.format), daemon=True) for w in range(a.workers)]
    for p in procs:
        p.start()
    sys.path.insert(0, str(HERE))
    import netsearch_study as ST
    ST.setup(a.format)
    import torch
    from porygon.fastenv import FastEnv
    from porygon.v2.model import batch_to_tensors, load, model_keys
    torch.set_num_threads(1)
    net = load(a.net, a.device).eval()
    keys = model_keys(net.gen)
    tgen = torch.Generator(device=a.device).manual_seed(seed)
    manifest = {"net": a.net, "seed": seed, "pick_seed": a.seed, "attempt": attempt, "workers": a.workers,
                "battles": a.battles, "temperature": a.temperature, "start_games": have,
                "PORYGON_PS": os.environ.get("PORYGON_PS"), "host": os.uname().nodename,
                "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"manifest-a{attempt}.json").write_text(json.dumps(manifest, indent=1))
    t0 = last = time.time()
    n_dec = 0
    with FastEnv(a.format, n_workers=a.workers, battles_per_worker=a.battles, seed=seed, encoder="v2",
                 dump_states=str(fdir / "dump-{w}.fifo"), dump_streams=True) as env:
        span = env.span
        didx = np.zeros(a.workers * span, np.int64)
        b = env.reset()
        while True:
            na = b.needs_action
            rows = np.flatnonzero(na)
            acts = np.zeros(len(rows), np.int64)
            p1 = None
            if len(rows):
                obs = {k: v[rows] for k, v in b.obs.items() if k in keys}
                with torch.no_grad():
                    lg = net(batch_to_tensors(obs, a.device), aux=False)["logits"].float()
                    m = torch.as_tensor(b.mask[rows], device=a.device)
                    lt = torch.where(m, lg / a.temperature, torch.finfo(torch.float32).min)
                    acts = torch.multinomial(torch.softmax(lt, -1), 1, generator=tgen).squeeze(1).cpu().numpy()
                    l64 = torch.where(m, lg.double(), -math.inf)
                    p1 = torch.softmax(l64, -1).cpu().numpy()
                mrows = b.mask[rows]
            msgs = defaultdict(list)
            k = 0
            for i in range(len(b)):
                s = int(b.slot[i])
                if b.flags[i] & 2:
                    didx[s] = 0
                if na[i]:
                    legal = np.flatnonzero(mrows[k]).tolist()
                    msgs[s // span].append((s % span, int(didx[s]), legal, [float(p1[k, j]) for j in legal]))
                    didx[s] += 1
                    k += 1
            for w, msg in msgs.items():
                tqs[w].put(msg)
            n_dec += len(rows)
            total = have + sum(c.value for c in counters)
            if not errq.empty():
                raise RuntimeError(errq.get())
            if any(not p.is_alive() for p in procs):
                raise RuntimeError("a reader died")
            now = time.time()
            if now - last > 60:
                last = now
                print(f"{time.strftime('%H:%M:%S')} games {total} ({total - have} new), {n_dec} decisions, "
                      f"{n_dec / (now - t0):.0f} dec/s", flush=True)
            if total >= a.games or now - t0 > a.max_hours * 3600:
                break
            b = env.step(acts)
        for q in tqs:
            q.put(None)
    for p in procs:
        p.join(timeout=900)
    for p in procs:
        if p.is_alive():
            p.terminate()
    if not errq.empty():
        raise RuntimeError(errq.get())
    total, _ = finished_games(out)
    secs = time.time() - t0
    manifest.update({"end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": secs, "decisions": n_dec,
                     "games_total": total})
    (out / f"manifest-a{attempt}.json").write_text(json.dumps(manifest, indent=1))
    for f in fifos:
        Path(f).unlink(missing_ok=True)
    print(f"done: {total} games in parts ({n_dec} decisions this attempt, {secs / 3600:.2f} h)")
    return 0 if total >= a.games else 2


# ---------------------------------------------------------------- label

def iter_records(pattern: str, skip: set[str], limit: int | None):
    files = sorted(glob.glob(pattern))
    assert files, f"no record parts match {pattern}"
    n = 0
    for f in files:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                if limit is not None and n >= limit:
                    return
                r = json.loads(line)
                n += 1
                if r["id"] in skip:
                    continue
                yield r


def labelled_ids(out: Path) -> set[str]:
    ids = set()
    for p in list(out.glob("label-*.meta.jsonl")) + list(out.glob("drops-*.jsonl")):
        if p.name.startswith("label-") and not p.with_name(p.name.replace(".meta.jsonl", ".npz")).exists():
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


async def label(a) -> int:
    sys.path.insert(0, str(HERE))
    import netsearch_study as ST
    NS, S, R = ST.setup(a.format)
    import torch
    from porygon import lookahead2 as LA
    from porygon.v2.encoder import encode
    from porygon.v2.spec import fields
    torch.set_num_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    delta = float(Path(a.delta_file).read_text().split()[0]) if a.delta_file else a.delta
    net = ST.load_net(a.net, "cpu")
    leaf = ST.load_net(a.net, a.leaf_device) if a.leaf_device != "cpu" else net
    pool = LA.LookaheadPool(a.workers)
    skip = labelled_ids(out)
    tag = f"{os.getpid()}-{int(time.time())}"
    fobs = fields(9)
    drops = open(out / f"drops-{tag}.jsonl", "a")
    buf: dict[str, list] = defaultdict(list)
    meta: list[str] = []
    part = [0]
    done = {"n": 0, "drop": 0, "t0": time.time()}
    it = iter_records(a.records, skip, a.limit)

    def flush():
        if not meta:
            return
        name = f"label-{tag}-{part[0]:04d}"
        arr = {f"obs/{k}": np.stack(buf[k]).astype(fobs[k][1]) for k in fobs}
        arr.update(probs=np.stack(buf["probs"]).astype(np.float32), pi=np.stack(buf["pi"]).astype(np.float32),
                   z=np.array(buf["z"], np.float32), game=np.array(buf["game"], np.int64))
        tmp = out / f".{name}.npz"
        np.savez_compressed(tmp, **arr)
        os.replace(tmp, out / f"{name}.npz")
        (out / f"{name}.meta.jsonl").write_text("".join(meta))
        part[0] += 1
        buf.clear()
        meta.clear()

    def drop(rec, reason, **kw):
        drops.write(json.dumps({"id": rec["id"], "reason": reason, **kw}) + "\n")
        drops.flush()
        done["drop"] += 1

    async def one(rec):
        want = rec["decisions"][0]
        seen = False
        async for d, battle, first, logged in ST.replay_targets(rec, R, NS, 9, {want["idx"]}):
            seen = True
            probs, mask, win = NS.net_forward(net, battle)
            legal = np.flatnonzero(mask > 0).tolist()
            if legal != want["legal"]:
                drop(rec, "parity-legal", legal=legal, want=want["legal"])
                return
            dp = float(np.abs(probs[legal] - np.asarray(want["probs"])).max())
            if dp > a.parity_tol:
                drop(rec, "parity-probs", dp=dp)
                return
            obs = encode(battle)
            rng = random.Random(f"{a.seed}:{rec['id']}:{d.msg}")
            t0 = time.perf_counter()
            try:
                res = await LA.lookahead(pool, leaf, battle, first, rng, mask, worlds=a.worlds)
            except Exception as e:  # noqa: BLE001  (a fallback in play; dropped from both arms)
                drop(rec, "lookahead", error=f"{type(e).__name__}: {e}"[:300])
                return
            t = time.perf_counter() - t0
            pi, p05 = pi_la(probs, res.rows, res.x, res.q, delta)
            for k in fobs:
                buf[k].append(np.asarray(obs[k]))
            buf["probs"].append(probs)
            buf["pi"].append(pi)
            buf["z"].append(float(rec["z"]))
            buf["game"].append(int(rec["game_id"]))
            a_l = int(res.rows[int(np.argmax(res.x))])
            meta.append(json.dumps({"id": rec["id"], "idx": d.msg, "turn": d.turn, "kind": d.kind,
                                    "n_legal": len(legal), "dp": dp, "win": round(win, 5), "a_l": a_l,
                                    "rows": [int(r) for r in res.rows], "q": [round(float(v), 6) for v in res.q],
                                    "x": [round(float(v), 6) for v in res.x], "value": round(res.value, 6),
                                    "moved": round(float(np.abs(pi - p05).sum() / 2), 6), "worlds": res.worlds,
                                    "failed_worlds": res.failed_worlds, "leaves": res.leaves, "t": round(t, 3),
                                    "cpu_expand": round(res.cpu_expand, 3)}) + "\n")
            done["n"] += 1
            if len(meta) >= PART_ROWS:
                flush()
            if done["n"] % 500 == 0:
                el = time.time() - done["t0"]
                print(f"{time.strftime('%H:%M:%S')} labelled {done['n']} (+{len(skip)} before), dropped "
                      f"{done['drop']}, {done['n'] / el:.2f} rows/s", flush=True)
            return
        if not seen:
            drop(rec, "not-replayed")

    async def worker():
        for rec in it:
            try:
                await one(rec)
            except Exception as e:  # noqa: BLE001  (a replay failure: dropped, counted)
                drop(rec, "error", error=f"{type(e).__name__}: {e}"[:300])

    await asyncio.gather(*(worker() for _ in range(a.concurrency)))
    flush()
    drops.close()
    await pool.close()
    el = time.time() - done["t0"]
    print(f"done: labelled {done['n']} (+{len(skip)} before), dropped {done['drop']} in {el:.0f}s")
    return 0


# ---------------------------------------------------------------- shards

def label_parts(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("label-*.npz") if p.with_name(p.name[:-4] + ".meta.jsonl").exists())


def cmd_shards(a) -> int:
    src = Path(a.labels)
    for arm in ("E", "C"):
        (Path(a.out) / arm).mkdir(parents=True, exist_ok=True)
    n = 0
    for i, p in enumerate(label_parts(src)):
        z = np.load(p)
        obs = {k: z[k] for k in z.files if k.startswith("obs/")}
        p05 = np.stack([sharpen(r, T_NET) for r in z["probs"]])
        for arm, tgt in (("E", sqrt_target(z["pi"])), ("C", sqrt_target(p05))):
            np.savez(Path(a.out) / arm / f"shard-{i:05d}.npz", **obs, target=tgt, z=z["z"], game=z["game"])
        n += len(z["z"])
    (Path(a.out) / "SHARDS.json").write_text(json.dumps({"rows": n, "parts": i + 1 if n else 0}))
    print(f"shards: {n} rows")
    return 0 if n else 1


# ---------------------------------------------------------------- check

def cmd_check(a) -> int:
    sys.path.insert(0, str(HERE))
    import netsearch_study as ST
    ST.setup(a.format)
    import torch
    from porygon.v2.model import batch_to_tensors, load, model_keys
    parts = label_parts(Path(a.labels))
    data = defaultdict(list)
    for p in parts:
        z = np.load(p)
        v = is_val(z["game"], a.val_frac)
        if not v.any():
            continue
        for k in z.files:
            data[k].append(z[k][v])
    data = {k: np.concatenate(v) for k, v in data.items()}
    n = len(data["z"])
    nets = {"S": a.start, **dict(x.split("=", 1) for x in a.net)}
    pi = data["pi"].astype(np.float64)
    out = {"held_out": n, "nets": nets}
    p05 = {}
    win = {}
    for name, path in nets.items():
        net = load(path, a.device).eval()
        keys = model_keys(net.gen)
        ps, ws = [], []
        for s in range(0, n, 1024):
            obs = {k[4:]: v[s:s + 1024] for k, v in data.items() if k.startswith("obs/") and k[4:] in keys}
            with torch.no_grad():
                o = net(batch_to_tensors(obs, a.device), aux=False)
            lg = o["logits"].double()
            m = torch.as_tensor(data["obs/action_mask"][s:s + 1024] > 0, device=a.device)
            ps.append(torch.softmax(torch.where(m, lg / T_NET, -math.inf), -1).cpu().numpy())
            ws.append(torch.sigmoid(o["win_logit"].double()).cpu().numpy())
        p05[name], win[name] = np.concatenate(ps), np.concatenate(ws)
    base = p05["S"]
    moved = np.abs(pi - base).sum(-1) / 2 > 1e-6
    kl_s = float(kl_rows(pi, base).mean())
    y = (data["z"] > 0).astype(np.float64)
    for name in nets:
        kl = float(kl_rows(pi, p05[name]).mean())
        top = float((p05[name][moved].argmax(-1) == pi[moved].argmax(-1)).mean()) if moved.any() else math.nan
        w = np.clip(win[name], 1e-7, 1 - 1e-7)
        ll = float(-(y * np.log(w) + (1 - y) * np.log(1 - w)).mean())
        out[name] = {"kl": kl, "r": 1 - kl / kl_s if kl_s > 0 else math.nan, "top1_moved": top, "value_logloss": ll,
                     "value_auc": auc(win[name], y)}
    out["moved_positions"] = int(moved.sum())
    out["mean_moved_mass"] = float((np.abs(pi - base).sum(-1) / 2).mean())
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0


def auc(score: np.ndarray, y: np.ndarray) -> float:
    o = np.argsort(score, kind="mergesort")
    r = np.empty(len(score))
    r[o] = np.arange(1, len(score) + 1)
    for v in np.unique(score):  # average ranks of ties
        t = score == v
        if t.sum() > 1:
            r[t] = r[t].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else math.nan


# ---------------------------------------------------------------- eval / verdict

def cmd_eval(a) -> int:
    sys.path.insert(0, str(HERE))
    import netsearch_study as ST
    ST.setup(a.format)
    from porygon.fastenv_eval import evaluate
    r = evaluate(a.format, f"v2:{a.a},temperature={T_NET}", f"v2:{a.b},temperature={T_NET}", a.games, a.workers,
                 a.battles, a.seed, a.device)
    r.update(seed=a.seed, games_asked=a.games, PORYGON_PS=os.environ.get("PORYGON_PS"))
    Path(a.json).write_text(json.dumps(r, indent=1) + "\n")
    print(f"{a.a} vs {a.b}: {100 * r['win_rate']:.2f}% ± {100 * r['ci95']:.2f} ({r['pairs']} pairs, "
          f"{r['seconds']:.0f} s)")
    return 0


def label_counts(d: Path) -> dict:
    rows = sum(len(p.read_text().splitlines()) for p in d.glob("label-*.meta.jsonl")
               if p.with_name(p.name.replace(".meta.jsonl", ".npz")).exists())
    reasons: dict[str, int] = defaultdict(int)
    for p in d.glob("drops-*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                reasons[json.loads(line)["reason"]] += 1
    return {"rows": rows, "drops": dict(reasons)}


def cmd_verdict(a) -> int:
    lc = label_counts(Path(a.labels))
    tot = lc["rows"] + sum(lc["drops"].values())
    parity = lc["drops"].get("parity-legal", 0) + lc["drops"].get("parity-probs", 0)
    fallbacks = lc["drops"].get("lookahead", 0)
    t = {k: json.loads(Path(p).read_text()) for k, p in (("T1", a.t1), ("T2", a.t2), ("T3", a.t3))}
    ck = json.loads(Path(a.check).read_text())
    void = []
    if lc["rows"] < a.min_rows:
        void.append(f"usable labelled rows {lc['rows']} < {a.min_rows}")
    if tot and parity / tot > 0.005:
        void.append(f"parity drops {parity}/{tot} > 0.5%")
    if tot and fallbacks / tot > 0.02:
        void.append(f"lookahead2 fallbacks {fallbacks}/{tot} > 2%")
    line = "VOID" if void else verdict_line(t["T1"], t["T2"])

    def f(r):
        return (f"{100 * r['win_rate']:.2f}% ± {100 * r['ci95']:.2f} [{100 * (r['win_rate'] - r['ci95']):.2f}, "
                f"{100 * (r['win_rate'] + r['ci95']):.2f}] ({r['pairs']} pairs)")
    txt = [f"{line}: T1 E vs C {f(t['T1'])}; T2 E vs S {f(t['T2'])}" + (f" ({'; '.join(void)})" if void else ""),
           f"  T3 C vs S (information) {f(t['T3'])}",
           f"  manipulation r: E {ck['E']['r']:.3f}, C {ck['C']['r']:.3f} (held out {ck['held_out']}; dose retry "
           f"{'RAN' if a.dose_retry else 'not run'}); top-1 on moved positions S {ck['S']['top1_moved']:.3f} "
           f"E {ck['E']['top1_moved']:.3f} C {ck['C']['top1_moved']:.3f}; mean moved mass {ck['mean_moved_mass']:.4f}",
           f"  value log-loss / AUC on z: S {ck['S']['value_logloss']:.4f} / {ck['S']['value_auc']:.4f}, "
           f"E {ck['E']['value_logloss']:.4f} / {ck['E']['value_auc']:.4f}, "
           f"C {ck['C']['value_logloss']:.4f} / {ck['C']['value_auc']:.4f}",
           f"  labels: {lc['rows']} usable, drops {lc['drops']}"]
    Path(a.out).write_text("\n".join(txt) + "\n")
    print("\n".join(txt))
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--net", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--games", type=int, required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--format", default="gen9randombattle")
    g.add_argument("--workers", type=int, default=5)
    g.add_argument("--battles", type=int, default=32)
    g.add_argument("--temperature", type=float, default=T_NET)
    g.add_argument("--device", default="cuda")
    g.add_argument("--fifo-dir", default=os.environ.get("TMPDIR", "/tmp"))
    g.add_argument("--max-hours", type=float, default=6.0)
    lb = sub.add_parser("label")
    lb.add_argument("--records", required=True, help="glob of record parts")
    lb.add_argument("--net", required=True)
    lb.add_argument("--out", required=True)
    lb.add_argument("--delta", type=float, default=0.005)
    lb.add_argument("--delta-file")
    lb.add_argument("--worlds", type=int, default=8)
    lb.add_argument("--workers", type=int, default=11)
    lb.add_argument("--concurrency", type=int, default=4)
    lb.add_argument("--leaf-device", default="cuda")
    lb.add_argument("--parity-tol", type=float, default=1e-5)
    lb.add_argument("--seed", type=int, default=7191102)
    lb.add_argument("--limit", type=int)
    lb.add_argument("--format", default="gen9randombattle")
    sh = sub.add_parser("shards")
    sh.add_argument("--labels", required=True)
    sh.add_argument("--out", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--labels", required=True)
    ck.add_argument("--start", required=True)
    ck.add_argument("--net", action="append", default=[], help="NAME=PATH (E, C)")
    ck.add_argument("--json", required=True)
    ck.add_argument("--val-frac", type=float, default=0.05)
    ck.add_argument("--device", default="cuda")
    ck.add_argument("--format", default="gen9randombattle")
    ev = sub.add_parser("eval")
    ev.add_argument("--a", required=True)
    ev.add_argument("--b", required=True)
    ev.add_argument("--games", type=int, required=True)
    ev.add_argument("--seed", type=int, required=True)
    ev.add_argument("--json", required=True)
    ev.add_argument("--workers", type=int, default=10)
    ev.add_argument("--battles", type=int, default=32)
    ev.add_argument("--device", default="cuda")
    ev.add_argument("--format", default="gen9randombattle")
    vd = sub.add_parser("verdict")
    vd.add_argument("--labels", required=True)
    vd.add_argument("--check", required=True)
    vd.add_argument("--t1", required=True)
    vd.add_argument("--t2", required=True)
    vd.add_argument("--t3", required=True)
    vd.add_argument("--out", required=True)
    vd.add_argument("--min-rows", type=int, default=110_000)
    vd.add_argument("--dose-retry", action="store_true")
    a = p.parse_args()
    if a.cmd == "gen":
        return cmd_gen(a)
    if a.cmd == "label":
        return asyncio.run(label(a))
    return {"shards": cmd_shards, "check": cmd_check, "eval": cmd_eval, "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
