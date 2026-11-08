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
