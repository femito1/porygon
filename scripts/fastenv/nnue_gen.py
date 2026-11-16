"""NNUE gate 1 data generator (docs/netsearch/NNUE-GATE1-PREREG.md, "Data"): G9q-cd@800M self-play on fastenv, both
sides the teacher at T 1.0; per decision row the TRUE position's student features, the teacher's P(win) and the
game's result for the side to move.

Main process: FastEnv (encoder v2, host.js --dump-states into one FIFO per worker), one batched teacher forward on
the GPU per lockstep step (P(win) = sigmoid(win_logit); actions sampled from softmax(logits / T) over legal ones);
per worker it sends (local slot, decision index, P(win)) for each decision row, in row order, to that worker's
converter.
Converter process per worker: reads the worker's FIFO (a reader thread drains it, so node never blocks on a full
pipe while the converter waits for P(win)), joins each S line to its row (slot and decision index must match, else
the run fails), builds features(from_string(convert_full(state, seen_hp=True).to_string())) (side two's HP as side
one has seen it: train/serve-consistent inputs) and stock evaluate() of the TRUE State
(from_string(convert_full(state).to_string())), keeps a game's rows until its G line, writes them with the result (ties, turn-cap
games and crashes dropped) into compressed shards; 2% of rows also keep their State string (timing, checks).
Split by game: h = int(sha1(game id)[:8], 16) % 100: < 90 train, 90-91 validation, >= 92 held-out.

    cd $C && PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src NETV2_SRC=$FROZEN/src $PY scripts/fastenv/nnue_gen.py \\
        --net nets/g9q-cd-805.03M.pt --out nnue/gate1/data --target 10000000 --cap 10500000 --workers 5
Re-running on the same --out continues (existing shards count; new seeds per attempt).
"""
import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHARD = 50_000
STATE_RATE = 0.02


def split_of(game: str) -> int:
    h = int(hashlib.sha1(game.encode()).hexdigest()[:8], 16) % 100
    return 0 if h < 90 else 1 if h < 92 else 2


def game_hash(game: str) -> int:
    return int(hashlib.sha1(game.encode()).hexdigest()[:15], 16)


def converter(w: int, fifo: str, tq, out: str, attempt: int, counter, errq) -> None:
    try:
        _converter(w, fifo, tq, out, attempt, counter)
    except BaseException as e:  # noqa: BLE001  (reported to the main process, which fails the run)
        import traceback
        errq.put(f"converter {w}: {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
        raise


def _converter(w, fifo, tq, out, attempt, counter) -> None:
    import poke_engine as pe
    from porygon.nnue.convert_full import convert_full
    from porygon.nnue.features import features
    pe.set_eval(None, 0.0125, 1.0)  # the stock evaluate() (lookahead2_study outcome, Step 2)
    lines: queue.Queue = queue.Queue(maxsize=4096)  # >> one batch's lines (<= 2 x battles): no deadlock, bounded memory

    def reader():
        with open(fifo, "r", buffering=1 << 20) as fh:
            for line in fh:
                lines.put(line)
        lines.put(None)
    threading.Thread(target=reader, daemon=True).start()
    teacher: deque = deque()
    stop = False
    pending: dict[str, list] = defaultdict(list)
    buf: dict[str, list] = defaultdict(list)
    states: list[str] = []
    n_shard = 0
    stats = {"rows": 0, "games": 0, "dropped_games": 0, "dropped_rows": 0}

    def flush():
        nonlocal n_shard
        if not buf["teacher"]:
            return
        name = f"shard-a{attempt}-w{w}-{n_shard:04d}"
        arr = {"ids": np.stack(buf["ids"]).astype(np.int16), "mon": np.stack(buf["mon"]).astype(np.float16),
               "side": np.stack(buf["side"]).astype(np.float16), "field": np.stack(buf["field"]).astype(np.uint8),
               "teacher": np.array(buf["teacher"], np.float32), "outcome": np.array(buf["outcome"], np.uint8),
               "eval": np.array(buf["eval"], np.float32), "turn": np.array(buf["turn"], np.uint16),
               "pside": np.array(buf["pside"], np.uint8), "didx": np.array(buf["didx"], np.uint16),
               "game": np.array(buf["game"], np.int64), "split": np.array(buf["split"], np.uint8)}
        tmp = Path(out) / f".{name}.npz"
        np.savez_compressed(tmp, **arr)
        if states:
            with gzip.open(Path(out) / f"{name}.states.jsonl.gz", "wt") as fh:
                fh.write("".join(states))
        meta = {"n": len(arr["teacher"]), "attempt": attempt, "worker": w, **stats}
        os.replace(tmp, Path(out) / f"{name}.npz")
        (Path(out) / f"{name}.json").write_text(json.dumps(meta))  # the shard's commit marker
        n_shard += 1
        buf.clear()
        states.clear()

    while True:
        line = lines.get()
        if line is None:
            break
        if line.startswith('{"t":"G"'):
            g = json.loads(line)
            rows = pending.pop(g["game"], [])
            if g["crashed"] or g["truncated"] or g["winner"] not in ("p1", "p2"):
                stats["dropped_games"] += 1
                stats["dropped_rows"] += len(rows)
                continue
            stats["games"] += 1
            for r in rows:
                r["outcome"] = int(g["winner"] == f"p{r['pside'] + 1}")
                st = r.pop("state_str", None)
                if st is not None:
                    states.append(json.dumps({"game": g["game"], "side": r["pside"], "didx": r["didx"],
                                              "split": r["split"], "turn": r["turn"], "state": st}) + "\n")
                for k, v in r.items():
                    buf[k].append(v)
            stats["rows"] += len(rows)
            with counter.get_lock():
                counter.value += len(rows)
            if len(buf["teacher"]) >= SHARD:
                flush()
            continue
        x = json.loads(line)
        while not teacher and not stop:
            msg = tq.get()
            if msg is None:
                stop = True
            else:
                teacher.extend(msg)
        if stop and not teacher:
            break  # the run ended: rows of unfinished games are dropped
        slot, didx, pw = teacher.popleft()
        if slot != x["slot"] or didx != x["idx"]:
            raise RuntimeError(f"join mismatch: row (slot {slot}, decision {didx}) vs S line (slot {x['slot']}, "
                               f"idx {x['idx']}, game {x['game']})")
        st = pe.State.from_string(convert_full(x["state"]).to_string())
        f = features(pe.State.from_string(convert_full(x["state"], seen_hp=True).to_string()))
        game = x["game"]
        r = {"ids": f["ids"], "mon": f["mon"], "side": f["side"], "field": f["field"], "teacher": pw,
             "eval": float(pe.evaluate(st)), "turn": int(x["turn"]), "pside": int(x["side"]), "didx": int(x["idx"]),
             "game": game_hash(game), "split": split_of(game)}
        if int(hashlib.sha1(f"{game}:{x['side']}:{x['idx']}".encode()).hexdigest()[:8], 16) % 10000 < STATE_RATE * 10000:
            r["state_str"] = st.to_string()
        pending[game].append(r)
    flush()


def existing(out: Path) -> tuple[int, int]:
    """(positions in finished shards, next attempt number)."""
    metas = [json.loads(p.read_text()) for p in out.glob("shard-*.json") if p.with_suffix(".npz").exists()]
    return sum(m["n"] for m in metas), 1 + max([m["attempt"] for m in metas], default=-1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--target", type=int, default=10_000_000)
    p.add_argument("--cap", type=int, default=10_500_000)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--battles", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1805)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fifo-dir", default=os.environ.get("TMPDIR", "/tmp"))
    p.add_argument("--max-hours", type=float, default=30.0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    have, attempt = existing(out)
    if have >= a.target:
        print(f"already {have} positions (target {a.target})")
        return 0
    seed = a.seed + 100 * attempt
    fdir = Path(a.fifo_dir) / f"nnue-gen-{os.getpid()}"
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
    convs = [ctx.Process(target=converter, args=(w, fifos[w], tqs[w], str(out), attempt, counters[w], errq), daemon=True)
             for w in range(a.workers)]
    for c in convs:
        c.start()

    sys.path.insert(0, str(HERE))
    import netsearch_study as ST  # noqa: F401  (puts netv2's porygon.v2 on porygon.__path__ via NETV2_SRC)
    ST.setup(a.format)
    import torch
    from porygon.fastenv import FastEnv
    from porygon.v2.model import batch_to_tensors, load, model_keys
    torch.set_num_threads(1)
    net = load(a.net, a.device).eval()
    keys = model_keys(net.gen)
    gen = torch.Generator(device=a.device).manual_seed(seed)
    manifest = {"net": a.net, "seed": seed, "attempt": attempt, "workers": a.workers, "battles": a.battles,
                "temperature": a.temperature, "start_positions": have, "host": os.uname().nodename,
                "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"manifest-a{attempt}.json").write_text(json.dumps(manifest, indent=1))
    t0 = last = time.time()
    n_dec = 0
    with FastEnv(a.format, n_workers=a.workers, battles_per_worker=a.battles, seed=seed, encoder="v2",
                 dump_states=str(fdir / "dump-{w}.fifo")) as env:
        span = env.span
        didx = np.zeros(a.workers * span, np.int64)
        b = env.reset()
        missing = [k for k in keys if k not in b.obs]
        assert not missing, f"fastenv's v2 encoder lacks model fields {missing}"
        while True:
            na = b.needs_action
            rows = np.flatnonzero(na)
            acts = np.zeros(len(rows), np.int64)
            pw = np.zeros(len(rows), np.float32)
            if len(rows):
                obs = {k: v[rows] for k, v in b.obs.items() if k in keys}
                with torch.no_grad():
                    o = net(batch_to_tensors(obs, a.device), aux=False)
                    lg = o["logits"].float()
                    m = torch.as_tensor(b.mask[rows], device=a.device)
                    lg = torch.where(m, lg / a.temperature, torch.finfo(torch.float32).min)
                    acts = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).squeeze(1).cpu().numpy()
                    pw = torch.sigmoid(o["win_logit"].float()).cpu().numpy()
            msgs = defaultdict(list)
            k = 0
            for i in range(len(b)):
                s = int(b.slot[i])
                if b.flags[i] & 2:  # DONE: the slot's next battle counts decisions from 0
                    didx[s] = 0
                if na[i]:
                    msgs[s // span].append((s % span, int(didx[s]), float(pw[k])))
                    didx[s] += 1
                    k += 1
            for w, msg in msgs.items():
                tqs[w].put(msg)
            n_dec += len(rows)
            total = have + sum(c.value for c in counters)
            if not errq.empty():
                raise RuntimeError(errq.get())
            if any(not c.is_alive() for c in convs):
                raise RuntimeError("a converter died")
            now = time.time()
            if now - last > 60:
                last = now
                print(f"{time.strftime('%H:%M:%S')} kept {total} positions ({total - have} new), {n_dec} decisions, "
                      f"{n_dec / (now - t0):.0f} dec/s", flush=True)
            if total >= a.target or now - t0 > a.max_hours * 3600:
                break
            b = env.step(acts)
        for q in tqs:
            q.put(None)
    for c in convs:  # converters flush their finished games, then exit (the FIFOs closed with the env)
        c.join(timeout=900)
    for c in convs:
        if c.is_alive():
            c.terminate()
    if not errq.empty():
        raise RuntimeError(errq.get())
    total, _ = existing(out)
    secs = time.time() - t0
    manifest.update({"end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": secs,
                     "decisions": n_dec, "positions_total": total, "decisions_per_s": n_dec / secs})
    (out / f"manifest-a{attempt}.json").write_text(json.dumps(manifest, indent=1))
    for f in fifos:
        Path(f).unlink(missing_ok=True)
    print(f"done: {total} positions in shards ({n_dec} decisions this attempt, {secs / 3600:.2f} h)")
    if total > a.cap:
        print(f"over the cap {a.cap}", file=sys.stderr)
        return 1
    return 0 if total >= a.target else 2


if __name__ == "__main__":
    sys.exit(main())
