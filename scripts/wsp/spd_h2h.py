"""Raw-network head-to-heads A vs B on fastenv's in-process Showdown simulator: duplicate pairs, sharded, per-pair
records. No Showdown server, no poke-env: Node workers (scripts/wsp/spd_host.js on the vendored fastenv JS) run the
battles, the first-person tracker and the observation encoder; this process batches every waiting decision of
every worker into one GPU forward per network.

    python scripts/wsp/spd_h2h.py run --a A --b B --pairs N [--shard K | --first F] --out DIR
           [--seed S] [--format gen4randombattle] [--workers W] [--battles 48] [--device cuda]
    python scripts/wsp/spd_h2h.py merge DIR [DIR ...]          # pooled win rate of A, pair-mean 95% interval

Policies: a v1 checkpoint `PATH[,temperature=T]` (SB3 .zip, or a pool / edit .pt such as ckpt/edits/*.pt; aliases
400M / 500M / 600M), a netv2 net `v2:PATH[,temperature=T]` (needs --v2-src, a netv2 src tree with porygon.v2),
`random`, and for B also the env-side poke-env baseline ports `heuristic` / `max_power`. Actions are sampled from
softmax(logits / T) over the legal actions; T = 1 is exactly the policy's own distribution, as
porygon.env.PolicyPlayer (h2h_dup.py, porygon.tournament) samples it. A v1 and a v2 net cannot meet in one run
(one encoder per env). A is side one (p1) in every game, as in h2h_dup.py.

Pairs and seeds: shard K plays the duplicate pairs [K·N, (K+1)·N) (--first F: [F, F+N)); pair p's battle seed and
both team seeds are a fixed function of (--seed, p) (spd_host.js: a bijective mix of an injective key), so shards
with disjoint ranges never repeat a pair, every pair can be replayed, and two experiments with the same --seed
play the same team pairs (common random numbers). Every pair in the range is played to the end (slots take the next
pair index of their worker when free), so the set of pairs is fixed in advance and does not depend on game lengths.

Output per shard: DIR/shard-KKKK.jsonl (one record per pair, sorted by pair index) and DIR/shard-KKKK.meta.json
(config, versions, node, timing, CPU; written with "status": "running" when the shard starts, so live readers know
its first pair id). Pairs completed so far are appended to shard-KKKK.part.jsonl while the shard runs (completion order
favours short pairs: merge ignores part files unless --partial).
Record: {"exp": id of (A, B, format, seed), "pair": p, "seed": battle seed [4 x u16], "team_seed": [A's team in
game 0, B's team in game 0] (4 x u16 each; game 1 swaps them), "games": [{"winner": "A"|"B"|"tie", "turns": t,
"void": bool, "truncated": bool}, {...}], "score": A's mean over the two games (win 1, tie or void 0.5), "shard",
"worker"}. void = the simulator or tracker crashed (scored as a tie, as porygon.fastenv_eval and h2h_dup.py do).

Env: SPD_FASTENV_JS (vendored fastenv/js, scripts/wsp/spd_vendor.sh), PORYGON_PS (a built pokemon-showdown),
PORYGON_NODE (node binary); scripts/wsp/spd_h2h.sbatch sets them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import select
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOST = Path(__file__).resolve().parent / "spd_host.js"
MAGIC, MAGIC_HELLO = 0x50594746, 0x4c454846
DTYPES = {"f32": "<f4", "i32": "<i4", "u32": "<u4", "i16": "<i2", "u16": "<u2", "u8": "u1", "i8": "i1"}
NEEDS_ACTION, DONE, REJECTED = 1, 2, 8
NO_PAIR = 0xFFFFFFFF
ENV_BOTS = ("heuristic", "max_power")
GLORIA = "/vol/bitbucket/fr1226/gloria"
ALIASES = {"400M": f"{GLORIA}/runs/20261024-135245-300664/final_model.zip",
           "500M": f"{GLORIA}/runs/20261025-195826-301040/final_model.zip",
           "600M": f"{GLORIA}/runs/20261026-082356-301120/final_model.zip"}


# --- policies ------------------------------------------------------------------------------------------------
def parse_spec(spec: str) -> tuple[str, str, float]:
    """'PATH[,temperature=T]' | 'v2:PATH[,...]' | 'random' | bot -> (kind, path or name, temperature)."""
    kind, body = ("v2", spec[3:]) if spec.startswith("v2:") else ("v1", spec)
    path, *opts = body.split(",")
    temp = 1.0
    for o in opts:
        k, _, v = o.partition("=")
        if k not in ("temperature", "t", "T"):
            raise SystemExit(f"unknown policy option {o!r} in {spec!r} (only temperature=T)")
        temp = float(v)
    if not temp > 0:
        raise SystemExit(f"temperature must be > 0 in {spec!r}")
    if kind == "v1" and path in ("random", *ENV_BOTS):
        return path, path, temp
    path = ALIASES.get(path, path)
    if not Path(path).exists():
        raise SystemExit(f"no such checkpoint: {path} (from {spec!r})")
    return kind, os.path.realpath(path), temp


def sample(logits, mask, temp: float):
    """softmax(logits / temp) over legal actions (all actions if a row has none legal, as SB3 would)."""
    import torch
    m = torch.as_tensor(mask, device=logits.device)
    m = m | ~m.any(-1, keepdim=True)
    z = torch.where(m, logits.float() / temp, torch.tensor(float("-inf"), device=logits.device))
    return torch.multinomial(torch.softmax(z, -1), 1).squeeze(1).cpu().numpy()


def make_actor(spec: str, device: str, n_actions: int):
    kind, path, temp = parse_spec(spec)
    if kind == "random":
        rng = np.random.default_rng(12345)
        return lambda obs, mask: (rng.random(mask.shape) * mask).argmax(1)
    import torch
    if kind == "v2":
        from porygon.v2.model import batch_to_tensors, load, model_keys
        net = load(path, device)
        keys = model_keys(net.gen)

        def act_v2(obs, mask):
            with torch.no_grad():
                logits = net(batch_to_tensors({k: obs[k] for k in keys}, device), aux=False)["logits"]
                return sample(logits, mask, temp)
        return act_v2
    from porygon.model import load_policy
    policy = load_policy(path, device)
    assert int(policy.action_space.n) == n_actions, (policy.action_space, n_actions)

    def act_v1(obs, mask):
        x = {"observation": torch.as_tensor(obs, device=device),
             "action_mask": torch.as_tensor(mask.astype(np.int8), device=device)}
        with torch.no_grad():  # the actor path only (no value head): masked logits as SB3's Categorical holds them
            return sample(policy.get_distribution(x).distribution.logits, mask, temp)
    return act_v1


# --- workers (fastenv wire protocol, see spd_host.js) ---------------------------------------------------------
class Worker:
    def __init__(self, index: int, cmd: list[str]):
        self.index = index
        to_child_r, self.to_child = os.pipe()
        self.from_child, from_child_w = os.pipe()
        cmd = cmd + ["--fd-in", str(to_child_r), "--fd-out", str(from_child_w)]
        self.proc = subprocess.Popen(cmd, pass_fds=(to_child_r, from_child_w), stdin=subprocess.DEVNULL)
        os.close(to_child_r)
        os.close(from_child_w)
        self.buf = bytearray(1 << 20)
        self.columns = None

    def _read_into(self, view: memoryview) -> None:
        got = 0
        while got < len(view):
            k = os.readv(self.from_child, [view[got:]])
            if k == 0:
                raise RuntimeError(f"worker {self.index} exited (code {self.proc.poll()})")
            got += k

    def read(self) -> dict[str, np.ndarray]:
        if self.columns is None:
            head = bytearray(8)
            self._read_into(memoryview(head))
            magic, n = struct.unpack("<2I", head)
            assert magic == MAGIC_HELLO, f"worker {self.index}: bad hello {magic:#x}"
            body = bytearray(n)
            self._read_into(memoryview(body))
            self.info = json.loads(body)
            self.columns = [(name, np.dtype(DTYPES[dt]), tuple(shape)) for name, dt, shape in self.info["columns"]]
        head = bytearray(16)
        self._read_into(memoryview(head))
        magic, n, ncol, _ = struct.unpack("<4I", head)
        assert magic == MAGIC and ncol == len(self.columns), f"worker {self.index}: bad batch header"
        sizes = [n * int(np.prod(shape)) * dt.itemsize for _, dt, shape in self.columns]
        total = sum(sizes)
        if len(self.buf) < total:
            self.buf = bytearray(2 * total)
        mv = memoryview(self.buf)[:total]
        self._read_into(mv)
        cols, o = {}, 0
        for (name, dt, shape), size in zip(self.columns, sizes):
            a = np.frombuffer(mv[o:o + size], dtype=dt).reshape(n, *shape).copy()
            cols[name] = a.reshape(n) if shape == (1,) else a
            o += size
        cols["mask"] = cols["mask"].astype(bool)
        return cols

    def send(self, local_slots: np.ndarray, actions: np.ndarray) -> None:
        msg = np.empty((len(local_slots), 2), dtype="<i4")
        msg[:, 0] = local_slots
        msg[:, 1] = actions
        view = memoryview(struct.pack("<2I", 1, len(local_slots)) + msg.tobytes())
        while view:
            view = view[os.write(self.to_child, view):]

    def cpu_seconds(self) -> float:
        try:
            with open(f"/proc/{self.proc.pid}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
        except (FileNotFoundError, IndexError, ValueError):
            return float("nan")

    def close(self) -> None:
        try:
            os.write(self.to_child, struct.pack("<2I", 2, 0))
        except OSError:
            pass
        for fd in (self.from_child, self.to_child):
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def obs_of(cols: dict, rows: np.ndarray):
    if "obs" in cols:
        return cols["obs"][rows]
    return {k[4:]: v[rows] for k, v in cols.items() if k.startswith("obs/")}


def obs_concat(parts: list):
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.concatenate(parts)


# --- one shard ---------------------------------------------------------------------------------------------
def exp_id(a: str, b: str, fmt: str, seed: int) -> str:
    key = json.dumps([parse_spec(a)[1:], parse_spec(b)[1:], fmt, seed])
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def node_info() -> dict:
    info = {"host": socket.gethostname(), "cpus": len(os.sched_getaffinity(0)), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"), "python": platform.python_version()}
    try:
        info["cpu_model"] = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo")
                                 if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return info


def versions() -> dict:
    v = {"spd_host_sha1": hashlib.sha1(HOST.read_bytes()).hexdigest()[:12],
         "spd_h2h_sha1": hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12],
         "fastenv_js": os.environ.get("SPD_FASTENV_JS"), "pokemon_showdown": os.environ.get("PORYGON_PS")}
    for k, f in (("fastenv_commit", Path(os.environ.get("SPD_FASTENV_JS", "/nonexistent")) / "../../VERSION"),):
        try:
            v[k] = f.read_text().strip()
        except OSError:
            pass
    try:
        v["pokemon_showdown_commit"] = subprocess.run(["git", "-C", os.environ["PORYGON_PS"], "rev-parse", "--short", "HEAD"],
                                                      capture_output=True, text=True, timeout=30).stdout.strip()
    except (KeyError, OSError, subprocess.SubprocessError):
        pass
    return v


def run(a) -> None:
    import torch
    torch.set_num_threads(1)
    kinds = [parse_spec(s)[0] for s in (a.a, a.b)]
    if "v2" in kinds and "v1" in kinds:
        raise SystemExit("a v1 (SB3) and a v2 net cannot share an env (one encoder per env)")
    if parse_spec(a.a)[0] in ENV_BOTS:
        raise SystemExit("the env-side bots can only play B (side two)")
    encoder = "v2" if "v2" in kinds else "v1"
    first = a.first if a.first is not None else a.shard * a.pairs
    end = first + a.pairs
    workers_n = a.workers or max(1, len(os.sched_getaffinity(0)) - 1)
    workers_n = max(1, min(workers_n, a.pairs))
    exp = exp_id(a.a, a.b, a.format, a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"shard-{a.shard:04d}"
    if Path(f"{stem}.jsonl").exists() and not a.overwrite:
        raise SystemExit(f"{stem}.jsonl exists (--overwrite to replace)")
    torch.manual_seed(int(hashlib.sha1(f"{exp}/{first}".encode()).hexdigest()[:8], 16))
    gen = int(a.format[3])
    n_actions = 26 if gen == 9 else 10
    actors = [make_actor(a.a, a.device, n_actions)]
    bot = parse_spec(a.b)[0] if parse_spec(a.b)[0] in ENV_BOTS else ""
    actors.append(None if bot else make_actor(a.b, a.device, n_actions))
    node = os.environ.get("PORYGON_NODE", "node")
    quota = [len(range(first + w, end, workers_n)) for w in range(workers_n)]
    workers = []
    for w in range(workers_n):
        cmd = [node, "--max-old-space-size=2048", str(HOST), "--format", a.format, "--battles", str(a.battles),
               "--seed", str((a.shard * 4099 + w) % 2**31), "--encoder", encoder, "--duplicate", "1",
               "--pairs", f"{first + w},{workers_n},{end}", "--pair-seed", str(a.seed), "--max-turns", str(a.max_turns)]
        if bot:
            cmd += ["--bot", bot, "--bot-battles", str(a.battles)]
        workers.append(Worker(w, cmd))
    meta = {"exp": exp, "a": a.a, "b": a.b, "a_resolved": parse_spec(a.a), "b_resolved": parse_spec(a.b),
            "format": a.format, "seed": a.seed, "shard": a.shard, "first": first, "end": end, "pairs": a.pairs,
            "workers": workers_n, "battles_per_worker": a.battles, "encoder": encoder, "device": a.device,
            "max_turns": a.max_turns, "node": node_info(), "versions": versions(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.device.startswith("cuda") and torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    Path(f"{stem}.meta.json").write_text(json.dumps({**meta, "status": "running"}, indent=1))  # live readers: first id
    print(f"spd_h2h {exp}: {a.a} vs {a.b} ({a.format}), pairs [{first}, {end}) seed {a.seed}, "
          f"{workers_n} workers x {a.battles} battles on {meta['node']['host']} ({meta['node']['cpus']} CPUs)", flush=True)

    pending: dict[int, dict] = {}        # pair -> partial record
    done_pairs: dict[int, dict] = {}
    completed = [0] * workers_n
    part = open(f"{stem}.part.jsonl", "w")
    waiting, finished = set(range(workers_n)), set()
    fds = {wk.from_child: wk for wk in workers}
    dec = [0, 0]
    rejected = forwards = fwd_rows = 0
    fwd_secs = 0.0
    t_first = None
    t0, p0 = time.time(), time.process_time()
    last_print = t0
    try:
        while len(finished) < workers_n:
            ready, _, _ = select.select([wk.from_child for wk in workers if wk.index in waiting], [], [], a.stall_timeout)
            if not ready:
                raise RuntimeError(f"no worker produced a batch in {a.stall_timeout} s (stalled)")
            got = []
            for fd in ready:
                wk = fds[fd]
                got.append((wk, wk.read()))
                waiting.discard(wk.index)
            if t_first is None:
                t_first = time.time()
            # terminal rows of side 0 (A's view): one game each
            for wk, c in got:
                rejected += int(((c["flags"] & REJECTED) > 0).sum())
                for i in np.flatnonzero(((c["flags"] & DONE) > 0) & (c["slot"] % 2 == 0)):
                    p = int(c["pair_idx"][i])
                    if p == NO_PAIR:
                        continue
                    info, words = int(c["pair_info"][i]), [int(x) for x in c["pair_seed"][i]]
                    g = info & 1
                    void, trunc = bool(info & 2), bool(info & 4)
                    win = int(c["win"][i])
                    rec = pending.setdefault(p, {"exp": exp, "pair": p, "seed": words[0:4],
                                                 "team_seed": [words[4:8], words[8:12]], "games": [None, None],
                                                 "shard": a.shard, "worker": wk.index})
                    assert rec["seed"] == words[0:4] and rec["games"][g] is None, (p, g, rec, words)
                    rec["games"][g] = {"winner": "A" if win > 0 else "B" if win < 0 else "tie", "turns": int(c["turn"][i]),
                                       "void": void, "truncated": trunc}
                    if None not in rec["games"]:
                        rec["score"] = sum(1.0 if x["winner"] == "A" else 0.5 if x["winner"] == "tie" else 0.0
                                           for x in rec["games"]) / 2
                        done_pairs[p] = pending.pop(p)
                        completed[wk.index] += 1
                        part.write(json.dumps(done_pairs[p]) + "\n")
            # decisions: one forward per network over every ready worker's waiting rows
            rows = [np.flatnonzero((c["flags"] & NEEDS_ACTION) > 0) for _, c in got]
            acts = [np.zeros(len(r), dtype=np.int64) for r in rows]
            for side in (0, 1):
                if actors[side] is None:
                    continue
                sel = [r[c["slot"][r] % 2 == side] for (_, c), r in zip(got, rows)]
                k = sum(len(s) for s in sel)
                if not k:
                    continue
                obs = obs_concat([obs_of(c, s) for (_, c), s in zip(got, sel) if len(s)])
                mask = np.concatenate([c["mask"][s] for (_, c), s in zip(got, sel) if len(s)])
                tf = time.time()
                out_a = actors[side](obs, mask)
                fwd_secs += time.time() - tf
                forwards += 1
                fwd_rows += k
                dec[side] += k
                o = 0
                for j, ((_, c), r, s) in enumerate(zip(got, rows, sel)):
                    if len(s):
                        acts[j][np.searchsorted(r, s)] = out_a[o:o + len(s)]
                        o += len(s)
            for j, ((wk, c), r) in enumerate(zip(got, rows)):
                if len(r) == 0:  # nothing to answer: every slot of this worker is idle
                    if completed[wk.index] != quota[wk.index]:
                        raise RuntimeError(f"worker {wk.index} has no decisions left but finished "
                                           f"{completed[wk.index]} of {quota[wk.index]} pairs")
                    finished.add(wk.index)
                    continue
                wk.send(c["slot"][r], acts[j])
                waiting.add(wk.index)
            if a.progress and time.time() - last_print > a.progress:
                last_print = time.time()
                print(f"  {len(done_pairs)}/{a.pairs} pairs, {sum(dec) / (last_print - t0):,.0f} decisions/s", flush=True)
        wall = time.time() - (t_first or t0)
        node_cpu = sum(wk.cpu_seconds() for wk in workers)
    finally:
        for wk in workers:
            wk.close()
        part.close()
    py_cpu = time.process_time() - p0
    recs = [done_pairs[p] for p in sorted(done_pairs)]
    assert len(recs) == a.pairs and recs[0]["pair"] == first and recs[-1]["pair"] == end - 1, (len(recs), a.pairs)
    games = 2 * len(recs)
    voids = sum(g["void"] for r in recs for g in r["games"])
    meta.update({"status": "done", "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_seconds": wall,
                 "startup_seconds": (t_first or t0) - t0, "games": games, "void_games": voids,
                 "truncated_games": sum(g["truncated"] for r in recs for g in r["games"]),
                 "decisions": {"a": dec[0], "b": dec[1]}, "rejected_choices": rejected,
                 "forwards": forwards, "mean_forward_rows": fwd_rows / max(forwards, 1), "forward_seconds": fwd_secs,
                 "node_cpu_seconds": node_cpu, "python_cpu_seconds": py_cpu,
                 "games_per_hour": games / wall * 3600, "decisions_per_hour": sum(dec) / wall * 3600,
                 "summary": summarize(recs)})
    tmp = Path(f"{stem}.jsonl.tmp")
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    tmp.rename(f"{stem}.jsonl")
    Path(f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
    Path(f"{stem}.part.jsonl").unlink(missing_ok=True)
    cpus = meta["node"]["cpus"]
    print(f"shard {a.shard}: {meta['summary']['line']}; {games} games in {wall:.0f} s = {games / wall * 3600:,.0f} games/h "
          f"({games / wall * 3600 / cpus:,.0f} per CPU of {cpus}), {sum(dec) / wall:,.0f} decisions/s, "
          f"node CPU {1e6 * node_cpu / max(sum(dec), 1):.0f} us/decision, python CPU {1e6 * py_cpu / max(sum(dec), 1):.0f} "
          f"us/decision, forward rows {fwd_rows / max(forwards, 1):.0f}, void {voids}, rejected {rejected}", flush=True)


# --- pooling ---------------------------------------------------------------------------------------------------
def summarize(recs: list[dict], drop_void: bool = False) -> dict:
    if drop_void:
        recs = [r for r in recs if not any(g["void"] for g in r["games"])]
    s = np.array([r["score"] for r in recs], dtype=float)
    n = len(s)
    if n == 0:
        return {"pairs": 0, "line": "no pairs"}
    m = float(s.mean())
    half = 1.96 * float(s.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
    games = [g for r in recs for g in r["games"]]
    wins = sum(g["winner"] == "A" for g in games)
    ties = sum(g["winner"] == "tie" for g in games)
    naive = 1.96 * math.sqrt(max(m * (1 - m), 1e-9) / len(games))
    return {"pairs": n, "games": len(games), "a_wins": wins, "ties": ties, "win_rate": m, "ci95": half,
            "naive_ci95": naive, "line": f"A win rate {m:.1%} ± {half:.1%} (pair-mean 95% interval, {n} pairs = "
                                         f"{len(games)} games; naive ±{naive:.1%}; {ties} ties)"}


def merge(dirs: list[str], partial: bool = False, drop_void: bool = False, out: str | None = None) -> dict:
    recs, metas, files = [], [], []
    for d in map(Path, dirs):
        fs = sorted(d.glob("shard-*.jsonl")) if d.is_dir() else [d]
        if not partial:
            fs = [f for f in fs if not f.name.endswith(".part.jsonl")]
        for f in fs:
            files.append(str(f))
            recs += [json.loads(line) for line in open(f) if line.strip()]
            m = Path(str(f).replace(".part.jsonl", ".meta.json").replace(".jsonl", ".meta.json"))
            if m.exists() and not str(f).endswith(".part.jsonl"):
                metas.append(json.loads(m.read_text()))
    if not recs:
        raise SystemExit(f"no shard files in {dirs}")
    exps = sorted({r["exp"] for r in recs})
    if len(exps) > 1:
        raise SystemExit(f"shards from different experiments (A, B, format, seed): {exps}")
    ids = [r["pair"] for r in recs]
    dup = len(ids) - len(set(ids))
    if dup:
        raise SystemExit(f"{dup} duplicate pair indices across the shard files: overlapping ranges")
    seeds = {tuple(r["seed"]) for r in recs}
    assert len(seeds) == len(recs), "two pairs share a battle seed"
    recs.sort(key=lambda r: r["pair"])
    lo, hi = recs[0]["pair"], recs[-1]["pair"]
    summ = summarize(recs, drop_void)
    summ.update({"exp": exps[0], "files": files, "pair_range": [lo, hi + 1], "missing_in_range": hi + 1 - lo - len(recs),
                 "void_games": sum(g["void"] for r in recs for g in r["games"]),
                 "truncated_games": sum(g["truncated"] for r in recs for g in r["games"])})
    if metas:
        m0 = metas[0]
        wall = sum(m["wall_seconds"] for m in metas)
        cpu_h = sum(m["wall_seconds"] * m["node"]["cpus"] for m in metas) / 3600
        decs = sum(m["decisions"]["a"] + m["decisions"]["b"] for m in metas)
        games = sum(m["games"] for m in metas)
        summ.update({"a": m0["a"], "b": m0["b"], "format": m0["format"], "seed": m0["seed"], "shards": len(metas),
                     "shard_wall_seconds": wall, "games_per_cpu_hour": games / cpu_h, "decisions_per_cpu_hour": decs / cpu_h,
                     "decisions_per_game": decs / max(games, 1), "nodes": sorted({m["node"]["host"] for m in metas})})
    if out:
        Path(out).write_text(json.dumps(summ, indent=1))
    return summ


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="play one shard")
    r.add_argument("--a", required=True)
    r.add_argument("--b", required=True)
    r.add_argument("--pairs", type=int, required=True, help="duplicate pairs in this shard (2 games each)")
    r.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)),
                   help="shard index (default $SLURM_ARRAY_TASK_ID or 0): pairs [shard*pairs, (shard+1)*pairs)")
    r.add_argument("--first", type=int, help="first pair index instead of shard*pairs")
    r.add_argument("--seed", type=int, default=1, help="experiment seed: pair p's teams are a function of (seed, p)")
    r.add_argument("--out", required=True)
    r.add_argument("--format", default="gen4randombattle")
    r.add_argument("--workers", type=int, default=0, help="Node workers (default: CPUs - 1)")
    r.add_argument("--battles", type=int, default=48, help="battles per worker")
    r.add_argument("--device", default="cuda")
    r.add_argument("--max-turns", type=int, default=1000)
    r.add_argument("--v2-src", help="netv2 src tree (porygon.v2) for v2: nets; prepended to sys.path")
    r.add_argument("--stall-timeout", type=float, default=600)
    r.add_argument("--progress", type=float, default=60, help="seconds between progress lines (0: none)")
    r.add_argument("--overwrite", action="store_true")
    m = sub.add_parser("merge", help="pool shard files of one experiment")
    m.add_argument("dirs", nargs="+")
    m.add_argument("--partial", action="store_true", help="include .part.jsonl files (biased toward short pairs)")
    m.add_argument("--drop-void", action="store_true", help="drop pairs with a void game instead of scoring it 0.5")
    m.add_argument("--out", help="write the summary JSON here")
    m.add_argument("--json", action="store_true")
    a = p.parse_args()
    if a.cmd == "run":
        if a.v2_src:
            sys.path.insert(0, a.v2_src)
        run(a)
    else:
        s = merge(a.dirs, a.partial, a.drop_void, a.out)
        print(json.dumps(s, indent=1) if a.json else
              f"{s.get('a', '?')} vs {s.get('b', '?')}: {s['line']}; pairs {s['pair_range'][0]}..{s['pair_range'][1] - 1} "
              f"({s['missing_in_range']} missing), void {s['void_games']}"
              + (f"; {s['games_per_cpu_hour']:,.0f} games/CPU-h, {s['decisions_per_cpu_hour']:,.0f} decisions/CPU-h"
                 if "games_per_cpu_hour" in s else ""))


if __name__ == "__main__":
    main()
