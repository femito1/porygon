"""Trainer v2 --eval-async (`porygon.v2.train_eval`). Without a server: a fake trainer (only the attributes the eval
code uses) with a tiny CPU net and fake eval subprocesses checks the bookkeeping (snapshot of the live weights, skip
while busy, the result handled at the snapshot's steps like a synchronous eval: log line, TensorBoard, best_model.pt =
the snapshot's weights, --frozen-best, the anchor-drop guard; crashes; snapshot cleanup; termination at exit); the
synchronous path through the same code; the subprocess entry point with a fake Evaluator. With a server: one real
eval subprocess on a local Showdown server started on a free port (skipped when none can be started).
Pool health (same fake trainer): pool-net load failures through `_opp_act` (counted per kind, logged once, the guard
tripped only by the main-loop check), a rotated --frozen-best file, the realised opponent mix and its log-line
fields, and the dry-pool alarm."""

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import porygon.v2.train as train_mod
from porygon.v2 import train_eval
from porygon.v2.model import Net, load, save
from porygon.v2.spec import fields
from porygon.v2.train import PoolLoads, Trainer, parse_args

ROOT = Path(__file__).resolve().parent.parent
CPU = torch.device("cpu")
TINY = dict(d_model=32, n_layers=1, n_heads=4, ffn_mult=1, d_move=16, d_species=16, d_move_emb=16, d_item=8,
            d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)
W = "value_head.0.weight"  # the tensor compared between the live net, snapshots and best_model.pt
PORT = 8765  # never contacted by the fakes

# a fake eval subprocess: argv = out, steps, mode, res (JSON); writes the result the way train_eval does
FAKE = r"""
import json, os, signal, sys, time
out, steps, mode, res = sys.argv[1], int(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
if mode == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("fake eval", steps, mode, flush=True)
if mode == "crash":
    sys.exit("Traceback (most recent call last): fake eval crashed")
time.sleep(120 if mode in ("hang", "slow") else float(mode))
env = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH")}
with open(out + ".tmp", "w") as f:
    json.dump({"steps": steps, "res": res, "seconds": 12.25, "env": env}, f)
os.replace(out + ".tmp", out)
"""


@pytest.fixture
def tmp(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)  # the laptop disk is nearly full


def tiny_net(seed: int = 0) -> Net:
    torch.manual_seed(seed)
    return Net(4, **TINY)


class Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


def fake_trainer(tmp: Path, *argv: str) -> Trainer:
    """A Trainer without collector or GPU: only what the eval methods, `_opp_act` and `_pool_alarm` use."""
    t = object.__new__(Trainer)
    t.run_dir = tmp / "run"
    t.args = parse_args(["--format", "gen4randombattle", "--ports", str(PORT), "--eval-battles", "4",
                         "--run-dir", str(t.run_dir), *argv])
    t.pool_dir = t.run_dir / "pool"
    t.pool_dir.mkdir(parents=True)
    t.gen, t.device, t.net = 4, CPU, tiny_net()
    t.counters = {"steps": 0, "updates": 0, "games": 0, "best": -1.0}
    t.stop, t.evaluator, t.eval_job, t.writer = threading.Event(), None, None, Writer()
    t.snapshots, t.compile, t.amp, t.pool_loads = {}, False, False, PoolLoads()
    t.workers = SimpleNamespace(restarts=0)  # (a BattleWorkers: no pool_per_worker)
    t.logs = []
    t.log = t.logs.append
    return t


def with_fake_subprocess(t: Trainer) -> dict:
    """Eval points start FAKE instead of porygon.v2.train_eval; set plan["mode"] / plan["res"] before each."""
    plan = {}
    t._eval_command = lambda snap, out, steps: [sys.executable, "-c", FAKE, str(out), str(steps), plan["mode"],
                                                json.dumps(plan["res"])]
    return plan


def wait_until(cond, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, "timed out"
        time.sleep(0.02)


def wait_handled(t: Trainer, timeout: float = 30.0) -> None:
    """Poll as the training loop does until the running eval has been handled."""
    wait_until(lambda: t.poll_eval() or t.eval_job is None, timeout)


def weights(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"][W]


def test_async_eval_bookkeeping(tmp):
    t = fake_trainer(tmp, "--eval-async", "--frozen-best", "--guard-anchor-drop", "0.2")
    plan = with_fake_subprocess(t)
    ev, frozen = t.run_dir / "eval", t.pool_dir / "frozen"
    w0 = t.net.state_dict()[W].clone()

    def eval_point(steps: int, mode: str, **res):
        t.counters["steps"] = steps
        plan.update(mode=mode, res=res)
        t.start_eval()

    # eval point: a snapshot of the live weights (+ the counters) and one subprocess on it (playing for 2 s)
    eval_point(1000, "2", heuristic=0.6, max_power=0.8)
    snap1 = ev / "snap_00000001000.pt"
    assert t.eval_job is not None and t.eval_job.steps == 1000 and t.eval_job.snap == snap1
    assert t.logs[-1].startswith("eval @ 1,000 started (async, pid ")
    with torch.no_grad():
        t.net.state_dict()[W].add_(1.0)  # training goes on: the live weights leave the snapshot's
    # the next eval point while it runs: skipped (nothing written, nothing counted)
    eval_point(2000, "0", heuristic=0.0, max_power=0.0)
    assert t.logs[-1].startswith("eval @ 2,000 skipped: previous still running (eval @ 1,000, ")
    assert t.eval_job.steps == 1000 and not (ev / "snap_00000002000.pt").exists()
    t.poll_eval()
    assert t.eval_job is not None and t.counters["best"] == -1.0  # still playing: nothing happens
    assert torch.equal(weights(snap1), w0) and torch.equal(load(snap1).state_dict()[W], w0)
    assert torch.load(snap1, weights_only=False)["counters"] == {"steps": 1000, "updates": 0, "games": 0, "best": -1.0}
    # the result, handled at the snapshot's steps as a synchronous eval of the snapshot's weights
    t.counters["steps"] = 2500
    wait_handled(t)
    assert t.logs[-1] == "[eval @ 1,000] heuristic=60.0%  max_power=80.0%  (12 s) (async, finished at 2,500)"
    assert t.writer.scalars == [("eval/win_rate_heuristic", 0.6, 1000), ("eval/win_rate_max_power", 0.8, 1000),
                                ("eval/seconds", 12.25, 1000)]
    assert t.counters["best"] == 0.6 and t.guard_evals == 0
    best = torch.load(t.run_dir / "best_model.pt", weights_only=False)
    assert torch.equal(best["state_dict"][W], w0)  # not the live (moved) weights
    assert not torch.equal(t.net.state_dict()[W], w0)
    assert best["eval"] == {"heuristic": 0.6, "max_power": 0.8}
    assert best["counters"] == {"steps": 1000, "updates": 0, "games": 0, "best": 0.6}
    assert torch.equal(load(t.run_dir / "best_model.pt").state_dict()[W], w0)
    assert [p.name for p in frozen.iterdir()] == ["best-00000001000.pt"] and torch.equal(
        weights(frozen / "best-00000001000.pt"), w0)
    assert snap1.exists()  # the current best's snapshot is kept
    out = json.loads((ev / "result_00000001000.json").read_text())
    assert out["env"]["CUDA_VISIBLE_DEVICES"] == "" and out["env"]["OMP_NUM_THREADS"] == "1"
    assert out["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT / "src")
    log = (ev / "eval.log").read_text()
    assert "fake eval 1000 2" in log and "fake eval 2000" not in log
    # a worse score: the guard counts it; its snapshot is deleted; best_model.pt untouched
    eval_point(3000, "0", heuristic=0.3, max_power=0.5)
    wait_handled(t)
    assert t.guard_evals == 1 and t.counters["best"] == 0.6 and torch.equal(weights(t.run_dir / "best_model.pt"), w0)
    assert sorted(p.name for p in ev.glob("snap_*.pt")) == ["snap_00000001000.pt"]
    # a crash (exit without a result): logged with the end of eval.log, best and guard untouched, the next can start
    eval_point(4000, "crash")
    wait_handled(t)
    assert t.logs[-1].startswith("eval @ 4,000 failed: exit code 1, no result")
    assert "fake eval crashed" in t.logs[-1] and "fake eval 4000 crash" in t.logs[-1]
    assert t.guard_evals == 1 and t.counters["best"] == 0.6 and len(t.writer.scalars) == 6
    assert sorted(p.name for p in ev.glob("snap_*.pt")) == ["snap_00000001000.pt"]
    # a new best: best_model.pt and the frozen copy = this snapshot (the moved weights); the old best's files go
    eval_point(5000, "0", heuristic=0.7, max_power=0.5)
    wait_handled(t)
    assert t.counters["best"] == 0.7 and t.guard_evals == 0
    assert torch.equal(weights(t.run_dir / "best_model.pt"), w0 + 1.0)
    assert torch.load(t.run_dir / "best_model.pt", weights_only=False)["counters"]["steps"] == 5000
    assert sorted(p.name for p in ev.glob("snap_*.pt")) == ["snap_00000005000.pt"]
    assert [p.name for p in frozen.iterdir()] == ["best-00000005000.pt"]
    # twice more than --guard-anchor-drop below the best: the guard trips
    eval_point(6000, "0", heuristic=0.45, max_power=0.5)
    wait_handled(t)
    assert t.guard_evals == 1 and not t.stop.is_set()
    eval_point(7000, "0", heuristic=0.4, max_power=0.5)
    wait_handled(t)
    assert t.stop.is_set() and (t.run_dir / "GUARD_TRIPPED").exists() and "GUARD TRIPPED" in t.logs[-1]
    assert sorted(p.name for p in ev.glob("snap_*.pt")) == ["snap_00000005000.pt"]
    assert sorted(p.name for p in ev.glob("result_*.json")) == [f"result_{s:011d}.json" for s in (1000, 3000, 5000,
                                                                                               6000, 7000)]
    assert not list(ev.glob("*.tmp"))
    # stopping: no new eval
    eval_point(8000, "0", heuristic=0.5, max_power=0.5)
    assert t.eval_job is None and not (ev / "snap_00000008000.pt").exists()


def test_async_eval_stopped_at_exit(tmp):
    t = fake_trainer(tmp, "--eval-async")
    plan = with_fake_subprocess(t)
    ev = t.run_dir / "eval"
    # a subprocess still playing: SIGTERM ends it, its snapshot is deleted
    plan.update(mode="slow", res={})
    t.counters["steps"] = 1000
    t.start_eval()
    proc = t.eval_job.proc
    start = time.time()
    t.stop_eval()
    assert proc.returncode == -signal.SIGTERM and time.time() - start < 5 and t.eval_job is None
    assert not (ev / "snap_00000001000.pt").exists() and t.logs[-1] == "eval @ 1,000 stopped at exit (exit code -15)"
    t.stop_eval()  # nothing running: no-op
    # one that ignores SIGTERM: killed after the timeout
    plan.update(mode="hang")
    t.counters["steps"] = 2000
    t.start_eval()
    proc = t.eval_job.proc
    wait_until(lambda: "fake eval 2000 hang" in (ev / "eval.log").read_text())  # its SIGTERM handler is set
    start = time.time()
    t.stop_eval(timeout=0.5)
    assert proc.returncode == -signal.SIGKILL and 0.5 <= time.time() - start < 5
    assert not list(ev.glob("snap_*.pt")) and t.counters["best"] == -1.0 and not t.writer.scalars


def test_sync_eval_unchanged(tmp, monkeypatch):
    calls = []

    class FakeEvaluator:
        def __init__(self, net, battle_format, port, anchor, device, concurrency=16):
            calls.append((net, battle_format, port, anchor, device))
            self.net = net

        def run(self, n_battles):
            calls.append((n_battles, self.net.training))
            return {"anchor": 0.55, "heuristic": 0.4, "max_power": 0.9}

    monkeypatch.setattr(train_mod, "Evaluator", FakeEvaluator)
    t = fake_trainer(tmp, "--frozen-best", "--anchor", str(tmp / "anchor.pt"))
    t.counters.update(steps=1000, updates=3, games=7)
    t.net.train()
    t.evaluate()
    assert calls == [(t.net, "gen4randombattle", PORT, tmp / "anchor.pt", CPU), (4, False)] and t.net.training
    assert t.logs == ["[eval @ 1,000] anchor=55.0%  heuristic=40.0%  max_power=90.0%  (0 s)"]
    assert [x[0] for x in t.writer.scalars] == ["eval/win_rate_anchor", "eval/win_rate_heuristic",
                                                "eval/win_rate_max_power", "eval/seconds"]
    assert all(x[2] == 1000 for x in t.writer.scalars) and t.counters["best"] == 0.55  # score = anchor
    best = torch.load(t.run_dir / "best_model.pt", weights_only=False)
    assert best["counters"] == {"steps": 1000, "updates": 3, "games": 7, "best": 0.55}
    assert best["eval"] == {"anchor": 0.55, "heuristic": 0.4, "max_power": 0.9}
    assert torch.equal(best["state_dict"][W], t.net.state_dict()[W])
    assert torch.equal(weights(t.pool_dir / "frozen" / "best-00000001000.pt"), t.net.state_dict()[W])
    assert not (t.run_dir / "eval").exists() and t.eval_job is None
    t.evaluate()  # same Evaluator reused; a lower score never replaces the best
    assert len(calls) == 3 and t.counters["best"] == 0.55


def test_entry_point(tmp, monkeypatch):
    """`python -m porygon.v2.train_eval` as the trainer calls it, in-process with a fake Evaluator."""
    from poke_env.ps_client.account_configuration import CONFIGURATION_FROM_PLAYER_COUNTER as names
    seen = {}

    class FakeEvaluator:
        opponents = {"anchor": None, "heuristic": None, "max_power": None}

        def __init__(self, net, battle_format, port, anchor, device, concurrency=16):
            seen.update(net=net, format=battle_format, port=port, anchor=anchor, device=device, concurrency=concurrency)

        def run(self, n_battles):
            seen.update(n=n_battles, threads=torch.get_num_threads())
            return {"anchor": 0.5, "heuristic": 0.25, "max_power": 0.75}

    monkeypatch.setattr(train_mod, "Evaluator", FakeEvaluator)
    net = tiny_net(1)
    snap = save(net, tmp / "eval" / "snap_00000001234.pt", counters={"steps": 1234, "updates": 2, "best": -1.0})
    t = fake_trainer(tmp, "--eval-async", "--ports", "8100-8103", "--anchor", "anchor.pt")
    cmd = t._eval_command(snap, tmp / "eval" / "result_00000001234.json", 1234)
    assert cmd[:3] == [sys.executable, "-m", "porygon.v2.train_eval"]
    args = train_eval.parse_args(cmd[3:])
    assert (args.snapshot, args.format, args.port, args.battles, args.steps, args.anchor, args.concurrency) == \
        (snap, "gen4randombattle", 8100, 4, 1234, Path("anchor.pt"), None)
    threads, before = torch.get_num_threads(), dict(names)
    try:
        out = train_eval.main(cmd[3:])
        first = dict(seen)
        assert min(names[k] for k in ("NetPlayer", "SimpleHeuristicsPlayer", "MaxBasePowerPlayer")) >= 10_000
        default = train_eval.main([str(snap), "--format", "gen4randombattle", "--battles", "2"])  # no --steps, --out
    finally:
        torch.set_num_threads(threads)
        names.clear()
        names.update(before)
    assert out["steps"] == 1234 and out["seconds"] >= 0
    assert out["res"] == {"anchor": 0.5, "heuristic": 0.25, "max_power": 0.75}
    assert json.loads((tmp / "eval" / "result_00000001234.json").read_text()) == out
    assert not list((tmp / "eval").glob("*.tmp"))
    assert (first["format"], first["port"], first["anchor"], first["device"], first["concurrency"], first["n"],
            first["threads"]) == ("gen4randombattle", 8100, Path("anchor.pt"), CPU, 16, 4, 1)
    assert not first["net"].training and all(p.device == CPU for p in first["net"].parameters())
    assert torch.equal(first["net"].state_dict()[W], net.state_dict()[W])
    assert default["steps"] == 1234 and seen["n"] == 2 and seen["anchor"] is None  # steps from the snapshot


# ---------------------------------------------------------------- pool health

def opp_batch(n: int = 3) -> dict:
    """An opponent-decision batch as the collectors pass it to `_opp_act`: zero observations, actions 0 and 6 legal."""
    b = {k: np.zeros((n, *shape), dtype) for k, (shape, dtype) in fields(4).items()}
    b["action_mask"][:, [0, 6]] = 1.0
    b["drop_seed"] = np.zeros(n, np.int64)
    return b


def ep(kind: str, opp: str) -> dict:
    """A finished episode's info as the collectors report it."""
    return {"kind": kind, "opp": opp, "turns": 20, "decisions": [9] if kind == "pool" else [9, 9],
            "result": [1] if kind == "pool" else [1, -1]}


def update(t: Trainer, eps: list[dict]) -> dict:
    """The main loop's per-update pool bookkeeping."""
    st = Trainer._episode_stats(eps, 1.0)
    t._pool_alarm(eps, st)
    return st


def test_pool_snapshot_load_failures(tmp):
    t = fake_trainer(tmp, "--snapshot-cache", "1")  # alternating names: every call loads
    pool = t.pool_dir
    save(tiny_net(1), pool / "00000000100.pt")
    save(tiny_net(2), pool / "00000000200.pt")
    (pool / "00000000300.pt").write_bytes(b"not a checkpoint")
    b = opp_batch()
    good = ["00000000100.pt", "00000000200.pt"]

    def play(name):
        acts = t._opp_act(name, b)
        assert len(acts) == 3 and set(acts.tolist()) <= {0, 6}  # a failed net plays uniformly over legal actions

    play(good[0])
    play(good[0])  # cached: no load
    for _ in range(3):
        play("00000000300.pt")  # never cached: every decision retries
    assert t.logs == [] and not t.stop.is_set()  # the collector thread only records
    st = update(t, [ep("pool", "snapshot")])
    assert (st["pool/load_attempts_snapshot"], st["pool/load_failures_snapshot"]) == (4, 3)
    assert (st["pool/load_attempts_frozen"], st["pool/load_failures_frozen"]) == (0, 0)
    assert len(t.logs) == 1 and t.logs[0].startswith(f"pool load FAILED snapshot {pool / '00000000300.pt'}: ")
    assert not t.stop.is_set()  # 3 of 4 failed, but fewer than 20 attempts
    # 1 of 20 failed (5%): no trip, the failing net is not logged again
    for i in range(19):
        play(good[i % 2])
    play("00000000300.pt")
    st = update(t, [ep("pool", "snapshot")])
    assert (st["pool/load_attempts_snapshot"], st["pool/load_failures_snapshot"]) == (20, 1)
    assert len(t.logs) == 1 and not t.stop.is_set()
    # 2 of 20 (10%): the main-loop check trips the guard
    for i in range(18):
        play(good[i % 2])
    play("00000000300.pt")
    play("00000000300.pt")
    assert not t.stop.is_set()
    st = update(t, [ep("pool", "snapshot")])
    assert t.stop.is_set() and t.logs[-1].startswith("GUARD TRIPPED: pool: 2 of 20 snapshot loads failed")
    assert (t.run_dir / "GUARD_TRIPPED").read_text().startswith("pool: ")
    assert (t.counters["pool_load_attempts_snapshot"], t.counters["pool_load_failures_snapshot"]) == (44, 6)
    assert t.counters["pool_load_failures_frozen"] == 0 and not (t.run_dir / "POOL_ALARM").exists()
    line = Trainer._line(t, 10, st | {"perf/steps_per_s": 1.0, "perf/collect_steps_per_s": 1.0, "perf/lr": 1e-4,
                                      "perf/update_s": 2.0})
    assert " ptop - plf 2 mix 0.00/0.00/1.00/0.00 dst - " in line


def test_pool_frozen_load_failure_and_rotation(tmp):
    t = fake_trainer(tmp)
    frozen = t.pool_dir / "frozen"
    save(tiny_net(1), frozen / "init.pt")
    save(Net(9, **TINY), frozen / "g9.pt")  # a gen9 net in a gen4 run
    save(tiny_net(3), frozen / "best-00000000500.pt")
    b = opp_batch()
    t._opp_act("frozen/init.pt", b)
    # best-400 was rotated out by --frozen-best (best-500 is newer): its battles play best-500, not a failure
    t._opp_act("frozen/best-00000000400.pt", b)
    t._opp_act("frozen/best-00000000400.pt", b)  # best-500 is cached now: no load
    st = update(t, [ep("pool", "frozen")])
    assert (st["pool/load_attempts_frozen"], st["pool/load_failures_frozen"]) == (2, 0)
    assert set(t.snapshots) == {"frozen/init.pt", "frozen/best-00000000500.pt"} and not t.stop.is_set()
    assert t.logs == ["pool opponent frozen/best-00000000400.pt was replaced by --frozen-best mid-battle; playing "
                      "frozen/best-00000000500.pt instead"]
    # failures: a wrong-gen frozen net (twice: logged once), and a missing best with nothing newer
    t._opp_act("frozen/g9.pt", b)
    t._opp_act("frozen/g9.pt", b)
    t._opp_act("frozen/best-00000000600.pt", b)
    assert not t.stop.is_set()
    st = update(t, [ep("pool", "frozen")])
    assert (st["pool/load_attempts_frozen"], st["pool/load_failures_frozen"]) == (3, 3)
    assert t.logs[1].startswith(f"pool load FAILED frozen {frozen / 'g9.pt'}: AssertionError('gen 9 snapshot')")
    assert t.logs[2].startswith(f"pool load FAILED frozen {frozen / 'best-00000000600.pt'}: FileNotFoundError")
    assert t.stop.is_set() and t.logs[3].startswith("GUARD TRIPPED: pool: 3 frozen-net load failure(s)")
    assert len(t.logs) == 4 and (t.run_dir / "GUARD_TRIPPED").read_text().startswith("pool: ")
    assert t.counters["pool_load_failures_frozen"] == 3 and t.counters["pool_load_attempts_frozen"] == 5


def test_pool_mix(tmp):
    t = fake_trainer(tmp)
    eps = ([ep("selfplay", "self")] * 4 + [ep("pool", "frozen")] * 2 + [ep("pool", "snapshot")] * 2
           + [ep("pool", "heuristic"), ep("pool", "max_power")])
    st = update(t, eps)
    assert [st[f"mix/{k}"] for k in ("selfplay", "frozen", "snapshot", "baseline")] == [0.4, 0.2, 0.2, 0.2]
    assert st["perf/pool_episode_share"] == pytest.approx(0.6) and st["pool/win_rate_frozen"] == 1.0
    perf = {"perf/steps_per_s": 1.0, "perf/collect_steps_per_s": 1.0, "perf/lr": 1e-4, "perf/update_s": 2.0}
    assert " ptop - plf 0 mix 0.40/0.20/0.20/0.20 dst - " in Trainer._line(t, 10, st | perf)
    st = update(t, [])  # no finished game this update
    assert not any(k.startswith("mix/") for k in st) and st["perf/pool_episode_share"] == 0.0
    assert " plf 0 mix - dst " in Trainer._line(t, 10, st | perf)
    assert not t.stop.is_set() and t.logs == []


def test_pool_alarm(tmp):
    selfplay = [ep("selfplay", "self")] * 3
    # pool battles configured: 4 dry updates, a pool game resets the count, then 5 dry updates trip
    t = fake_trainer(tmp / "a", "--pool-battles", "2")
    for i in range(4):
        update(t, selfplay)
    update(t, selfplay + [ep("pool", "heuristic")])
    for i in range(4):
        update(t, selfplay)
        assert not t.stop.is_set()
    update(t, selfplay)
    assert t.stop.is_set()
    msg = "POOL ALARM: 2 pool battles configured but no pool episode finished in 5 updates in a row (worker restarts 0)"
    assert t.logs == [msg, f"GUARD TRIPPED: pool: {msg}; stopping (best model: best_model.pt)"]
    assert (t.run_dir / "POOL_ALARM").read_text() == msg + "\n"
    assert (t.run_dir / "GUARD_TRIPPED").read_text() == f"pool: {msg}\n"
    # never: no pool battles, the alarm off, or a fastenv run without pool battles per worker
    for name, argv, workers in (("b", [], None), ("c", ["--pool-battles", "2", "--pool-alarm-updates", "0"], None),
                                ("d", ["--pool-battles", "2"], SimpleNamespace(restarts=0, pool_per_worker=0))):
        t = fake_trainer(tmp / name, *argv)
        t.workers = workers or t.workers
        for i in range(10):
            update(t, selfplay)
        assert not t.stop.is_set() and t.logs == [] and not (t.run_dir / "POOL_ALARM").exists(), name


# ---------------------------------------------------------------- with a local Showdown server

def showdown_dir() -> Path | None:
    """A built Showdown checkout: $PS_DIR, this checkout's, or (in a git worktree under .claude/worktrees/) the main
    checkout's."""
    cands = [os.environ.get("PS_DIR"), ROOT / "pokemon-showdown"]
    if ROOT.parent.name == "worktrees":
        cands.append(ROOT.parents[2] / "pokemon-showdown")
    for c in cands:
        if c and (Path(c) / "dist" / "server" / "index.js").exists() and (Path(c) / "config" / "config.js").exists():
            return Path(c)
    return None


def _handshake(port: int, timeout: float = 60.0) -> Exception | None:
    """Open one websocket (as scripts/start_servers.sh does), retrying until `timeout`; the last error if none."""
    import websockets

    async def go():
        async with websockets.connect(f"ws://localhost:{port}/showdown/websocket", open_timeout=10):
            pass
    end = time.time() + timeout
    while True:
        try:
            asyncio.run(go())
            return None
        except Exception as e:
            if time.time() > end:
                return e
            time.sleep(1.0)


@contextmanager
def showdown_server(log_path: Path):
    """One Showdown server (no security) on a free port, in its own process group (stopped with it; never
    `scripts/stop_servers.sh`, which stops every server on the machine)."""
    ps = showdown_dir()
    if ps is None or shutil.which("node") is None:
        pytest.skip("no built Showdown checkout (set PS_DIR) or no node")
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(["node", "pokemon-showdown", "start", "--skip-build", "--no-security", str(port)],
                                cwd=ps, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        end = time.time() + 120
        while b"listening" not in log_path.read_bytes():
            if proc.poll() is not None or time.time() > end:
                pytest.skip(f"Showdown did not start: {log_path.read_text()[-500:]}")
            time.sleep(0.25)
        err = _handshake(port)
        if err is not None:
            pytest.skip(f"no websocket handshake with the local Showdown server on port {port}: {err!r}")
        yield port
    finally:  # the whole group: the server's worker processes too
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # anything of the group still alive
        except ProcessLookupError:
            pass
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            pass


def test_async_eval_end_to_end(tmp):
    """The real subprocess: a tiny random gen4 snapshot plays 4 games vs a tiny anchor and each bot."""
    with showdown_server(tmp / "showdown.log") as port:
        save(tiny_net(1), tmp / "anchor.pt")
        t = fake_trainer(tmp, "--eval-async", "--ports", str(port), "--eval-battles", "4", "--anchor",
                         str(tmp / "anchor.pt"))
        t.counters["steps"] = 1000
        start = time.time()
        t.start_eval()
        wait_handled(t, timeout=300)
        wall = time.time() - start
        ev = t.run_dir / "eval"
        log = (ev / "eval.log").read_text()
        assert (ev / "result_00000001000.json").exists(), log[-3000:]
        out = json.loads((ev / "result_00000001000.json").read_text())
        assert out["steps"] == 1000 and set(out["res"]) == {"anchor", "heuristic", "max_power"}
        assert all(v in (0.0, 0.25, 0.5, 0.75, 1.0) for v in out["res"].values()), out
        assert t.logs[-1].startswith("[eval @ 1,000] anchor=") and t.logs[-1].endswith("(async, finished at 1,000)")
        assert "1 torch thread(s)" in log and "Traceback" not in log
        assert t.counters["best"] == out["res"]["anchor"] and (t.run_dir / "best_model.pt").exists()
        print(f"\nasync eval: 12 games (4 vs each of anchor, heuristic, max_power) in {out['seconds']:.1f} s of play, "
              f"{wall:.1f} s wall incl. start-up; {out['res']}")
