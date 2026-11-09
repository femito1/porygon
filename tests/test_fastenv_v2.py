"""FastCollector (porygon.fastenv_v2) with the v2 JS encoder feeds netv2's rollout builder. Needs netv2's
porygon.v2 package (branch ws/netv2) importable: skipped otherwise."""

import shutil

import numpy as np
import pytest

v2 = pytest.importorskip("porygon.v2.train")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


@pytest.mark.parametrize("fmt", ["gen4randombattle", "gen9randombattle"])
def test_collector_rollout(fmt):
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import DONE, REW, Packer

    rng = np.random.default_rng(0)

    def act(batch):
        mask = batch["action_mask"] > 0
        probs = mask / mask.sum(1, keepdims=True)
        acts = (rng.random(mask.shape) * mask).argmax(1)
        n = len(acts)
        return acts, np.log(probs[np.arange(n), acts]), np.zeros(n, np.float32), probs.astype(np.float32)

    packer = Packer(int(fmt[3]))
    workers = FastWorkers(FastEnv(fmt, 1, 4, seed=2, encoder="v2", shaping=(0, 0, 0)))
    try:
        col = FastCollector(workers.env, packer, act, seed=0)
        while len(col.episodes) < 4:
            col.step()
        segs, eps = col.take()
        data = v2.build_rollout(segs, packer, 0.9999, 0.754)
        rows = [t for s in segs.values() for t in s]
        assert len(data["action"]) == len(rows) > 0
        assert all(t[REW] == 0 or t[DONE] for t in rows), "sparse reward only at the end"
        assert all(sorted(e["result"]) in ([-1, 1], [0, 0]) for e in eps)
    finally:
        workers.close()


def test_collector_pool_snapshots(tmp_path):
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    (tmp_path / "snap_a.pt").write_bytes(b"")
    rng = np.random.default_rng(1)
    seen = []

    def act(batch):
        mask = batch["action_mask"] > 0
        probs = mask / mask.sum(1, keepdims=True)
        acts = (rng.random(mask.shape) * mask).argmax(1)
        n = len(acts)
        return acts, np.log(probs[np.arange(n), acts]), np.zeros(n, np.float32), probs.astype(np.float32)

    def opp(name, batch):
        seen.append((name, len(batch["action_mask"])))
        assert "aux_item" not in batch and "drop_seed" in batch
        return (rng.random(batch["action_mask"].shape) * (batch["action_mask"] > 0)).argmax(1)

    workers = FastWorkers(FastEnv("gen9randombattle", 1, 4, seed=3, encoder="v2", shaping=(0, 0, 0)))
    try:
        col = FastCollector(workers.env, Packer(9), act, opp, seed=0, pool_per_worker=2, pool_dir=tmp_path)
        while len(col.episodes) < 6:
            col.step()
        segs, eps = col.take()
        kinds = {e["kind"] for e in eps}
        assert kinds == {"pool", "selfplay"}, kinds
        assert seen and all(n == "snap_a.pt" for n, _ in seen)
        assert all(s % 2 == 0 for s in segs if (s // 2) % 4 < 2), "pool battles: only the learner side has transitions"
    finally:
        workers.close()


def test_collector_env_bots():
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    rng = np.random.default_rng(2)

    def act(batch):
        mask = batch["action_mask"] > 0
        probs = mask / mask.sum(1, keepdims=True)
        acts = (rng.random(mask.shape) * mask).argmax(1)
        n = len(acts)
        return acts, np.log(probs[np.arange(n), acts]), np.zeros(n, np.float32), probs.astype(np.float32)

    workers = FastWorkers(FastEnv("gen4randombattle", 1, 3, seed=8, encoder="v2", shaping=(0, 0, 0),
                                  bot="heuristic", bot_battles=1))
    try:
        col = FastCollector(workers.env, Packer(4), act, seed=0)
        while len(col.episodes) < 8:
            col.step()
        segs, eps = col.take()
        bot_eps = [e for e in eps if e["kind"] == "pool"]
        assert bot_eps and all(e["opp"] == "heuristic" and len(e["result"]) == 1 for e in bot_eps)
        assert 1 not in segs, "the bot's side has no transitions"
    finally:
        workers.close()


def _versioned_act(rng, state):
    """Uniform policy whose value output is the 'actor version' state['v'] (tags which weights acted)."""
    def act(batch):
        mask = batch["action_mask"] > 0
        probs = mask / mask.sum(1, keepdims=True)
        acts = (rng.random(mask.shape) * mask).argmax(1)
        n = len(acts)
        return (acts, np.log(probs[np.arange(n), acts]), np.full(n, state["v"], np.float32),
                probs.astype(np.float32))
    return act


@pytest.mark.parametrize("on_policy", [True, False])
def test_collector_on_policy_drain(on_policy):
    """on_policy: every transition of rollout k was acted, and every non-terminal one bootstrapped, by actor k
    (except the few counted as stale); without it, the previous rollout's open transitions carry over."""
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import DONE, NEXT_V, VAL, Packer

    state = {"v": 0.0}
    workers = FastWorkers(FastEnv("gen9randombattle", 2, 6, seed=5, encoder="v2", shaping=(0, 0, 0)))
    try:
        col = FastCollector(workers.env, Packer(9), _versioned_act(np.random.default_rng(3), state), seed=0,
                            on_policy=on_policy)
        for k in range(3):
            state["v"] = float(k)
            while col.n_closed < 120:
                col.step()
            if col.on_policy:
                col.drain()
            segs, _ = col.take()
            rows = [t for s in segs.values() for t in s]
            old = sum(t[VAL] != k for t in rows)
            assert old == round(col.last_stats["perf/fastenv_stale_frac"] * len(rows))
            if k == 0:
                assert old == 0
            elif on_policy:  # residual: sides waiting on the opponent's forced switch at the drain (~10-20%)
                assert old <= 24 // 3, f"rollout {k}: {old} transitions acted by an earlier actor"
                assert all(t[NEXT_V] == k for t in rows if not t[DONE]), "bootstraps come from the current actor"
                assert col.last_stats["perf/fastenv_drained_frac"] > 0
            else:
                assert old >= 6, "carry-over expected without the drain"
    finally:
        workers.close()


def test_env_bot_mix():
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    workers = FastWorkers(FastEnv("gen4randombattle", 2, 2, seed=9, encoder="v2", shaping=(0, 0, 0),
                                  bot="heuristic,max_power", bot_battles=1))
    try:
        assert [workers.env.worker_bot(i) for i in range(2)] == ["heuristic", "max_power"]
        col = FastCollector(workers.env, Packer(4), _versioned_act(np.random.default_rng(4), {"v": 0.0}), seed=0)
        while len({e["opp"] for e in col.episodes if e["kind"] == "pool"}) < 2:
            col.step()
        assert {e["opp"] for e in col.episodes if e["kind"] == "pool"} == {"heuristic", "max_power"}
    finally:
        workers.close()


@pytest.mark.parametrize("with_snapshot", [False, True])
def test_dynamic_pool_bots(tmp_path, with_snapshot):
    """netv2 pool semantics: a pool battle is a baseline bot with p_baseline or while the pool is empty, else a
    snapshot; bot battles give the learner side's transitions only and one 'pool' episode each."""
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    if with_snapshot:
        (tmp_path / "snap.pt").write_bytes(b"")
    rng = np.random.default_rng(6)
    seen = []

    def opp(name, batch):
        seen.append(name)
        return (rng.random(batch["action_mask"].shape) * (batch["action_mask"] > 0)).argmax(1)

    workers = FastWorkers(FastEnv("gen9randombattle", 1, 4, seed=4, encoder="v2", shaping=(0, 0, 0),
                                  dyn_bots="heuristic,max_power"), pool_per_worker=3)
    try:
        col = FastCollector(workers.env, Packer(9), _versioned_act(rng, {"v": 0.0}), opp, seed=1,
                            pool_per_worker=3, pool_dir=tmp_path, p_baseline=0.5)
        while len(col.episodes) < 30:
            col.step()
        segs, eps = col.take()
        opps = [e["opp"] for e in eps if e["kind"] == "pool"]
        assert {"heuristic", "max_power"} <= set(opps)
        if with_snapshot:
            assert "snapshot" in opps and seen
            share = opps.count("snapshot") / len(opps)
            assert 0.2 < share < 0.8, share
        else:
            assert "snapshot" not in opps and not seen
        assert all(len(e["result"]) == 1 and e["result"][0] in (-1, 0, 1) for e in eps if e["kind"] == "pool")
        assert [e for e in eps if e["kind"] == "selfplay"], "battle 3 is self-play"
        assert not any(s % 2 == 1 and (s // 2) < 3 for s in segs), "pool battles: learner side only"
    finally:
        workers.close()


def test_worker_restart():
    """A dead worker is replaced (restart_dead): collection goes on, closed transitions stay usable."""
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    workers = FastWorkers(FastEnv("gen9randombattle", 2, 3, seed=12, encoder="v2", shaping=(0, 0, 0)))
    workers.env.restart_dead = True
    try:
        col = FastCollector(workers.env, Packer(9), _versioned_act(np.random.default_rng(7), {"v": 0.0}), seed=0)
        for _ in range(5):
            col.step()
        workers.env.workers[1].proc.kill()
        for _ in range(200):
            col.step()
            if workers.restarts:
                break
        assert workers.restarts == 1
        n0 = len(col.episodes)
        while len(col.episodes) < n0 + 4:
            col.step()
        segs, _ = col.take()
        data = v2.build_rollout(segs, Packer(9), 0.9999, 0.754)
        assert len(data["action"]) > 0 and np.isfinite(data["adv"]).all()
    finally:
        workers.close()


def test_frozen_pool_opponents(tmp_path):
    """p_frozen: pool battles vs pool_dir/frozen/*.pt (name 'frozen/<file>', episodes labelled 'frozen'), also when
    no snapshot exists yet; the rest baselines (as PoolOpponent at ws/netv2 0ac086c)."""
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    (tmp_path / "frozen").mkdir()
    (tmp_path / "frozen" / "p1.pt").write_bytes(b"")
    rng = np.random.default_rng(8)
    seen = set()

    def opp(name, batch):
        seen.add(name)
        return (rng.random(batch["action_mask"].shape) * (batch["action_mask"] > 0)).argmax(1)

    workers = FastWorkers(FastEnv("gen9randombattle", 1, 4, seed=14, encoder="v2", shaping=(0, 0, 0),
                                  dyn_bots="heuristic,max_power"), pool_per_worker=3)
    try:
        col = FastCollector(workers.env, Packer(9), _versioned_act(rng, {"v": 0.0}), opp, seed=2,
                            pool_per_worker=3, pool_dir=tmp_path, p_baseline=0.3, p_frozen=0.5)
        while len(col.episodes) < 30:
            col.step()
        opps = [e["opp"] for e in col.episodes if e["kind"] == "pool"]
        assert seen == {"frozen/p1.pt"}
        assert "frozen" in opps and ({"heuristic", "max_power"} & set(opps)) and "snapshot" not in opps
        # no snapshots: frozen whenever r < 0.5 or r >= 0.8 -> 70% frozen
        share = opps.count("frozen") / len(opps)
        assert 0.4 < share < 0.95, share
    finally:
        workers.close()
