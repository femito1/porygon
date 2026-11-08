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
