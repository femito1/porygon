"""--actor-stream (porygon.v2.actor_buffers + porygon.v2.train). CPU: the double buffer (refresh flips, records the
version, the old copy is untouched until the next refresh), forwards through acquire() equal a plain copy's, a
threaded stress test (every forward's output is exactly the one of the version acquire() reported: no torn or stale
weights), the trainer's _act / _refresh_actor / version counts identical with and without the flag (fake fastenv), the
collector-thread wrapper, and short real trainer runs on fastenv (node + a built Showdown). CUDA (skipped without it):
the stream's priority and the collector thread's current stream, the stress test with the actor on the high-priority
stream while the learner keeps the default stream busy (plus the check that this harness does see the races a missing
event allows), _act on the actor stream (bf16, compiled damage layer) and the fastenv trainer runs on the GPU."""

import copy
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from porygon.v2.actor_buffers import ActorBuffers
from porygon.v2.model import Net, batch_to_tensors, model_keys, save
from porygon.v2.synthetic import synthetic_obs
from porygon.v2.train import Trainer, parse_args

CPU = torch.device("cpu")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SMALL = dict(d_model=32, n_layers=1, n_heads=4, ffn_mult=1, d_move=16, d_species=16, d_move_emb=16, d_item=8,
             d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)
SMALL_ARGS = ["--d-model", "32", "--n-layers", "1", "--n-heads", "4", "--ffn-mult", "1", "--d-move", "16",
              "--model-json", '{"d_species":16,"d_move_emb":16,"d_item":8,"d_ability":8,"d_type":8,"pair_hidden":8,'
                              '"value_hidden":32}']


@pytest.fixture
def tmp(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)  # the laptop disk is nearly full


@pytest.fixture
def torch_globals():
    """The trainer sets the torch thread count and the float32 matmul precision process-wide: restore them."""
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    yield
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision(precision)


def chain(d: int, device=CPU) -> nn.Module:
    """Two bias-free d × d layers. With every weight v / d (`fill`), an all-ones input gives exactly v² in every output
    (d a power of 2: every partial sum is a multiple of v² / d that fp32 holds exactly while v < 2¹² / √d), so a
    forward that read another version, two versions or a partly copied weight does not give its version's output."""
    return nn.Sequential(nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)).to(device)


@torch.no_grad()
def fill(net: nn.Module, v: int) -> None:
    for p in net.parameters():
        p.fill_(v / p.shape[1])


def levels(net: nn.Module) -> set[float]:
    """The v of `fill(net, v)` that each (constant) weight holds."""
    out = set()
    for p in net.parameters():
        p = p.detach()
        assert torch.all(p == p.flatten()[0])
        out.add(float(p.flatten()[0]) * p.shape[1])
    return out


def small_net(seed: int, gen: int = 9) -> Net:
    torch.manual_seed(seed)
    net = Net(gen, **SMALL)
    with torch.no_grad():  # the zero-initialised policy layer would give uniform logits
        net.policy_head[-1].weight.normal_(std=0.5)
        net.policy_head[-1].bias.normal_(std=0.1)
    return net


def obs_batch(n: int, seed: int, gen: int = 9) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ob = synthetic_obs(gen, n, 6, rng)
    none = ob["action_mask"].sum(1) == 0
    ob["action_mask"][none, 0] = 1.0  # every row with a legal action, as the collector's
    ob["drop_seed"] = np.where(rng.random(n) < 0.5, rng.integers(1, 2 ** 62, n), 0)  # half the rows with id dropout
    return ob


@torch.no_grad()
def perturb(net: nn.Module, seed: int) -> None:
    """The same 'update' on every net it is applied to (a fixed generator)."""
    g = torch.Generator().manual_seed(seed)
    for p in net.parameters():
        p.add_(0.05 * torch.randn(p.shape, generator=g).to(p.device))


# ---------------------------------------------------------------- the double buffer (CPU)

def test_refresh_flips_and_keeps_the_old_copy():
    d = 8
    src = chain(d)
    fill(src, 1)
    buf = ActorBuffers(src, version=1)
    assert buf.stream is None and buf.front == 0 and buf.versions == [1, 1]
    assert buf.nets[0] is not buf.nets[1] and all(n is not src and not n.training for n in buf.nets)
    assert buf.describe().startswith("actor-stream: CPU: double buffer only")
    x = torch.ones(3, d)
    with buf.acquire() as (net, v):
        assert buf.lock.locked() and net is buf.nets[0] and v == 1
        assert torch.equal(net(x), torch.full((3, d), 1.0))
    assert not buf.lock.locked()
    fill(src, 2)  # the learner moves on: no copy sees it before a refresh
    assert levels(buf.nets[0]) == levels(buf.nets[1]) == {1.0}
    buf.refresh(src, 2)
    assert buf.front == 1 and buf.versions == [1, 2]
    with buf.acquire() as (net, v):
        assert net is buf.nets[1] and v == 2 and torch.equal(net(x), torch.full((3, d), 4.0))
    assert levels(buf.nets[0]) == {1.0}  # the old copy is untouched until the next refresh
    fill(src, 3)
    buf.refresh(src, 3)
    assert buf.front == 0 and buf.versions == [3, 2]
    with buf.acquire() as (net, v):
        assert net is buf.nets[0] and v == 3 and torch.equal(net(x), torch.full((3, d), 9.0))
    assert levels(buf.nets[1]) == {2.0}
    with pytest.raises(ValueError):  # an exception inside the block releases the lock
        with buf.acquire():
            raise ValueError
    assert not buf.lock.locked()


@torch.no_grad()
def test_acquire_forward_equals_a_plain_copy():
    """A v2 net (damage layer, id dropout on half the rows) through acquire() gives exactly the forward of a plain
    eval-mode deep copy of the same weights, before and after refreshes."""
    net = small_net(0)
    buf = ActorBuffers(net, CPU)
    t = batch_to_tensors(obs_batch(24, 1), CPU, keys=model_keys(9) + ["drop_seed"])
    for step in range(3):
        ref = copy.deepcopy(net).eval()(t, aux=False)
        with buf.acquire() as (actor, v):
            out = actor(t, aux=False)
        assert v == step
        for k in ("logits", "value", "win_logit"):
            assert torch.equal(out[k], ref[k]), k
        perturb(net, 10 + step)
        buf.refresh(net, step + 1)
    with buf.acquire() as (actor, _):
        assert not torch.equal(actor(t, aux=False)["logits"], ref["logits"])  # the refresh changed what acts


# ---------------------------------------------------------------- threaded stress test (CPU and CUDA)

class _NoEvent:
    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass


class _NoEvents(dict):
    def __setitem__(self, k, v):  # acquire still records an event; refresh finds none to wait on
        pass


class NoReadyWait(ActorBuffers):
    """Broken on purpose (harness check): the actor's stream does not wait for the copy into the front."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ready = [_NoEvent(), _NoEvent()]


class NoUsedWait(ActorBuffers):
    """Broken on purpose (harness check): the copy into the back does not wait for the forwards that read it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.used = [_NoEvents(), _NoEvents()]


def stress(cls, device, d: int, rows: int, rounds: int, group: int, actor_gate: int = 0, learner_gate: int = 0,
           dummy: int = 0) -> dict:
    """Actor thread (on the buffers' stream on CUDA): forwards through acquire() on an all-ones input until the
    learner is done, in groups of `group`, each group first delayed on its stream by `actor_gate` GPU cycles (so its
    forwards are still queued when the learner copies), each output checked after its group against v² for the
    version v acquire() yielded. Learner (this thread, the default stream), `rounds` times: `dummy` heavy matmuls and
    a delay of `learner_gate` cycles, then a refresh (the copy queues behind them while the flip happens at once);
    then, on an idle default stream, two back-to-back refreshes (the second copies into the copy the actor's queued
    forwards read)."""
    cuda = device.type == "cuda"
    src = chain(d, device)
    fill(src, 1)
    buf = cls(src, device, version=1)
    done = threading.Event()
    res = {"forwards": 0, "bad": [], "seen": set(), "error": None, "stream_ok": True}

    def check(pending):
        got = torch.stack([torch.stack([y.amin(), y.amax()]) for _, y in pending]).double().cpu()
        for (v, _), (lo, hi) in zip(pending, got.tolist()):
            res["forwards"] += 1
            res["seen"].add(v)
            if abs(lo - v * v) > 1e-6 * v * v or abs(hi - v * v) > 1e-6 * v * v:
                res["bad"].append((v, lo, hi))

    def actor():
        try:
            if cuda:
                torch.cuda.set_stream(buf.stream)
            x = torch.ones(rows, d, device=device)  # (allocated and filled on the actor's stream)
            while not done.is_set():
                if actor_gate:
                    torch.cuda._sleep(actor_gate)
                pending = []
                for _ in range(group):
                    with buf.acquire() as (net, v):
                        if cuda and torch.cuda.current_stream() != buf.stream:
                            res["stream_ok"] = False
                        y = net(x)
                    pending.append((v, y))
                check(pending)
                time.sleep(0)  # (CPU: let the learner take the lock)
        except BaseException as e:
            res["error"] = e

    th = threading.Thread(target=actor, daemon=True)
    th.start()
    big = torch.randn(2048, 2048, device=device) / 45 if dummy else None
    v = 1
    try:
        for _ in range(rounds):
            for _ in range(dummy):
                big = torch.tanh(big @ big)
            if learner_gate:
                torch.cuda._sleep(learner_gate)
            v += 1
            fill(src, v)
            buf.refresh(src, v)
            if cuda:
                torch.cuda.current_stream().synchronize()
            time.sleep(0.003)
            for _ in range(2):
                v += 1
                fill(src, v)
                buf.refresh(src, v)
            if cuda:
                torch.cuda.current_stream().synchronize()
            time.sleep(0.003)
    finally:
        done.set()
        th.join(120)
    assert not th.is_alive(), "actor thread hung"
    if res["error"] is not None:
        raise res["error"]
    res["versions"] = v
    return res


def test_stress_cpu():
    res = stress(ActorBuffers, CPU, d=256, rows=32, rounds=60, group=1)
    assert res["forwards"] >= 100 and len(res["seen"]) >= 10, res
    assert not res["bad"], f"{len(res['bad'])} of {res['forwards']} forwards torn or stale: {res['bad'][:5]}"


# ---------------------------------------------------------------- the trainer (CPU, fake fastenv)

def fake_fastenv(monkeypatch) -> None:
    """Trainer(args) without Node workers (as test_v2_stacked_opp.py's test_flag_wiring)."""
    import porygon.fastenv_v2 as fv2
    monkeypatch.setattr(fv2, "build_env", lambda args, seed=None: SimpleNamespace(
        env=SimpleNamespace(close=lambda: None), pool_per_worker=0, restarts=0, close=lambda: None))
    monkeypatch.setattr(fv2, "FastCollector", lambda *a, **kw: SimpleNamespace())


def make_trainer(tmp: Path, name: str, *extra: str, device: str = "cpu") -> Trainer:
    args = parse_args(["--collector", "fastenv", "--device", device, "--seed", "0", "--ema-decay", "0",
                       "--run-dir", str(tmp / name), *SMALL_ARGS, *extra])
    return Trainer(args)


def close(tr: Trainer) -> None:
    tr.log_file.close()
    tr.writer.close()


def test_trainer_act_same_with_and_without_the_flag(tmp, monkeypatch, torch_globals):
    """Same seed, so the same initial weights: _act's actions, log-probs, values and probabilities are identical with
    and without --actor-stream (same multinomial seed), and again after each identical 'update' + _refresh_actor; the
    lag accounting (version counts, _take_versions) is the same."""
    fake_fastenv(monkeypatch)
    off, on = make_trainer(tmp, "off"), make_trainer(tmp, "on", "--actor-stream")
    try:
        assert off.actor_buffers is None and on.actor_buffers is not None
        assert isinstance(off.actor_lock, type(threading.Lock())) and off.actor is not off.net
        assert on.actor is on.actor_buffers.nets[0] and on.actor_lock is on.actor_buffers.lock
        log_off, log_on = ((tmp / n / "train.log").read_text() for n in ("off", "on"))
        assert "actor-stream" not in log_off
        assert "actor-stream: CPU: double buffer only" in log_on
        assert "warning: --actor-stream overlaps collection with the update without --decoupled" in log_on
        batch = obs_batch(32, 3)
        first = None
        for step in range(3):
            if step:
                for tr in (off, on):
                    perturb(tr.net, 100 + step)
                    tr.counters["updates"] = step
                    tr._refresh_actor()
            res = []
            for tr in (off, on):
                torch.manual_seed(7 + step)
                res.append(tr._act(batch))
            for a, b in zip(*res):
                assert np.array_equal(a, b)
            first = res[0] if first is None else first
        assert not np.array_equal(first[3], res[0][3])  # the refreshes changed the acting weights
        assert on.actor_buffers.front == 0 and on.actor_buffers.versions == [2, 1] and on.actor_version == 2
        assert off._take_versions() == on._take_versions() == {0: 32, 1: 32, 2: 32}
    finally:
        close(off)
        close(on)


def test_no_warning_with_decoupled_or_no_overlap(tmp, monkeypatch, torch_globals):
    fake_fastenv(monkeypatch)
    for i, extra in enumerate([["--decoupled"], ["--no-overlap"]]):
        close(make_trainer(tmp, f"r{i}", "--actor-stream", *extra))
        log = (tmp / f"r{i}" / "train.log").read_text()
        assert "actor-stream: CPU" in log and "warning: --actor-stream" not in log


def test_collector_wrapper_runs_the_collector(tmp, monkeypatch, torch_globals):
    """run()'s thread target with the flag: on CPU the plain collector loop, with the same arguments."""
    fake_fastenv(monkeypatch)
    tr = make_trainer(tmp, "w", "--actor-stream")
    close(tr)
    seen = []
    tr.collector = SimpleNamespace(run=lambda *a: seen.append(a))
    th = threading.Thread(target=tr._collect_on_actor_stream, args=(1, 2, 3, 4))
    th.start()
    th.join(10)
    assert seen == [(1, 2, 3, 4)]


# ---------------------------------------------------------------- CUDA

@needs_cuda
def test_cuda_stream_priority_and_thread_stream(tmp, monkeypatch, torch_globals):
    dev = torch.device("cuda")
    buf = ActorBuffers(chain(64, dev), dev)
    lo, hi = torch.cuda.Stream.priority_range()
    assert hi < lo and buf.stream.priority == hi and buf.priority_range == (lo, hi), (lo, hi, buf.stream.priority)
    assert buf.stream != torch.cuda.default_stream() and buf.stream.device.type == "cuda"
    assert f"priority {hi}" in buf.describe()
    fake_fastenv(monkeypatch)
    tr = make_trainer(tmp, "s", "--actor-stream", "--decoupled", "--no-damage", device="cuda")
    close(tr)
    seen = []
    tr.collector = SimpleNamespace(run=lambda *a: seen.append(torch.cuda.current_stream()))
    th = threading.Thread(target=tr._collect_on_actor_stream, args=(0, None, None, None))
    th.start()
    th.join(30)
    assert seen == [tr.actor_buffers.stream]  # the collector thread's current stream
    assert torch.cuda.current_stream() == torch.cuda.default_stream()  # the main thread's is unchanged
    assert f"actor-stream: collector thread on a CUDA stream of priority {hi}" in (tmp / "s" / "train.log").read_text()


# 2 × (512 × 2048 @ 2048 × 2048) per forward; versions up to 1 + 3 × 24 = 73 < 2¹² / √2048 (exact fp32 sums);
# gates: 3e7 cycles ≈ 20 ms (actor, per group of 16), 2.25e8 ≈ 150 ms (learner, before the copy) at 1.5 GHz
CUDA_STRESS = dict(d=2048, rows=512, rounds=24, group=16, actor_gate=30_000_000, learner_gate=225_000_000, dummy=4)


@needs_cuda
def test_stress_cuda(torch_globals):
    """The actor on the high-priority stream while the learner keeps the default stream busy: no forward sees torn
    or stale weights (exact fp32 sums: TF32 off)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    res = stress(ActorBuffers, torch.device("cuda"), **CUDA_STRESS)
    print(f"\nActorBuffers: {len(res['bad'])} of {res['forwards']} forwards torn or stale, "
          f"{len(res['seen'])} of {res['versions']} versions seen")
    assert res["stream_ok"] and res["forwards"] >= 4 * CUDA_STRESS["group"] and len(res["seen"]) >= 4, res
    assert not res["bad"], f"{len(res['bad'])} of {res['forwards']} forwards torn or stale: {res['bad'][:5]}"


@needs_cuda
@pytest.mark.parametrize("broken", [NoReadyWait, NoUsedWait], ids=["no-ready-wait", "no-used-wait"])
def test_stress_cuda_harness_sees_missing_events(broken, torch_globals):
    """Harness check, not a check of the patch: with either wait removed the same harness must see bad forwards on
    this GPU; if it does not, test_stress_cuda's pass shows less than it should (the race window is too small here)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    res = stress(broken, torch.device("cuda"), **CUDA_STRESS)
    print(f"\n{broken.__name__}: {len(res['bad'])} of {res['forwards']} forwards torn or stale")
    assert res["bad"], f"{broken.__name__}: no bad forward in {res['forwards']}: the harness cannot see this race here"


@needs_cuda
def test_cuda_act_on_the_actor_stream(tmp, monkeypatch, torch_globals):
    """_act from the collector thread on the actor stream (bf16 autocast; the damage layer compiled, or eager if the
    compiler is unavailable) while the main thread refreshes: finite outputs, counts per acting version, the compiled
    layer shared by both copies, and the same probabilities as the front copy's forward on the default stream."""
    fake_fastenv(monkeypatch)
    tr = make_trainer(tmp, "c", "--actor-stream", "--decoupled", device="cuda")
    close(tr)
    b = tr.actor_buffers
    assert b.nets[0].__dict__.get("_damage_compiled") is not None
    assert b.nets[1].__dict__.get("_damage_compiled") is b.nets[0].__dict__["_damage_compiled"]
    batch, n_calls = obs_batch(64, 5), 40
    results, errors = [], []
    first = threading.Event()

    def collector_run(*_):
        try:
            for _ in range(n_calls):
                results.append((torch.cuda.current_stream() == b.stream, tr._act(batch)))
                first.set()
        except BaseException as e:
            errors.append(e)
        first.set()

    tr.collector = SimpleNamespace(run=collector_run)
    th = threading.Thread(target=tr._collect_on_actor_stream, args=(0, None, None, None))
    th.start()
    assert first.wait(600)  # (the first call compiles the damage layer)
    for u in (1, 2, 3):
        perturb(tr.net, 200 + u)
        tr.counters["updates"] = u
        tr._refresh_actor()
        time.sleep(0.02)
    th.join(600)
    assert not th.is_alive() and not errors, errors
    assert len(results) == n_calls and all(ok for ok, _ in results)
    assert all(np.isfinite(r[2]).all() and np.isfinite(r[3]).all() for _, r in results)
    vc = tr._take_versions()
    assert sum(vc.values()) == 64 * n_calls and set(vc) <= {0, 1, 2, 3}, vc
    t = batch_to_tensors(batch, tr.device, keys=model_keys(9) + ["drop_seed"])
    with torch.no_grad(), b.acquire() as (net, v), torch.autocast("cuda", dtype=torch.bfloat16):  # default stream
        ref = torch.softmax(net(t, aux=False)["logits"].float(), -1).cpu().numpy()
    assert v == 3
    results.clear()
    tr.collector = SimpleNamespace(run=lambda *_: results.append(tr._act(batch)))
    th = threading.Thread(target=tr._collect_on_actor_stream, args=(0, None, None, None))
    th.start()
    th.join(120)
    assert np.allclose(results[0][3], ref, atol=2e-2)


# ---------------------------------------------------------------- the real trainer on fastenv

def _fastenv_ready() -> bool:
    """node, the fastenv JS and a built Showdown where fastenv/js/ps.js looks for one."""
    if shutil.which(os.environ.get("PORYGON_NODE") or "node") is None:
        return False
    from porygon.fastenv import JS
    if not (JS / "host.js").exists():
        return False
    cands = [os.environ.get("PORYGON_PS")] + [os.path.normpath(JS / rel) for rel in (
        "../../pokemon-showdown", "../../../../../pokemon-showdown")]
    return any(c and (Path(c) / "dist" / "sim" / "index.js").exists() for c in cands)


needs_fastenv = pytest.mark.skipif(not _fastenv_ready(), reason="needs node, fastenv/js and a built Showdown")


@needs_fastenv
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
@pytest.mark.parametrize("extra", [[], ["--actor-stream", "--decoupled"], ["--actor-stream", "--no-overlap"]],
                         ids=["flag-off", "overlap-decoupled", "no-overlap"])
def test_trainer_runs_on_fastenv(tmp, device, extra, torch_globals):
    """3 updates of the real trainer (collector thread, refreshes, PPO updates) on one fastenv worker with self-play
    and pool battles against a saved snapshot (loaded and played on the collector thread): 3 update lines, policy lag
    0 without overlap and at most 1 with it, the snapshot played, the acting copy refreshed before every update."""
    run = tmp / "run"
    save(small_net(3), run / "pool" / "00000000100.pt")
    args = parse_args(["--format", "gen9randombattle", "--collector", "fastenv", "--fastenv-workers", "1",
                       "--fastenv-battles", "8", "--n-battles", "4", "--pool-battles", "4", "--p-baseline", "0",
                       "--rollout", "120", "--batch-size", "64", "--n-epochs", "1", "--eval-battles", "0",
                       "--snapshot-every", "100000", "--checkpoint-every", "100000", "--max-updates", "3",
                       "--device", device, "--seed", "0", "--run-dir", str(run), *SMALL_ARGS, *extra])
    tr = Trainer(args)
    fired = threading.Event()
    watchdog = threading.Timer(600, lambda: (fired.set(), tr.stop.set()))
    watchdog.start()
    try:
        tr.run()
    finally:
        watchdog.cancel()
        tr.log_file.close()
    assert not fired.is_set(), "watchdog: the run hung"
    log = (run / "train.log").read_text()
    upd = [line for line in log.splitlines() if " upd " in line]
    assert len(upd) == 3, log[-3000:]
    lags = [line.split(" lag ")[1].split()[0] for line in upd]
    if "--no-overlap" in extra:
        assert all(lag == "0.00/0" for lag in lags), lags
    else:
        assert all(float(lag.split("/")[1]) <= 1 for lag in lags), lags
    assert tr.collector.opp_decisions > 0  # the pool snapshot acted
    assert (run / "final_model.pt").exists()
    if extra:
        assert tr.actor_buffers.versions[tr.actor_buffers.front] == 2  # refreshed before updates 1, 2 and 3
        assert ("actor-stream: CPU" if device == "cpu" else "actor-stream: collector thread on a CUDA stream") in log
    else:
        assert tr.actor_buffers is None and "actor-stream" not in log
