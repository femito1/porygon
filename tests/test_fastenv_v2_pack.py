"""FastCollector's run-grouped packing (`_pack_runs`, 2026-11-09 speed item) is byte-identical to packing the rows one
by one, for any order of (batch, row) pairs (runs from the same batch, interleaved batches, single rows), and the
per-rollout collector timing counters (perf/col_*) are reported and reset by `take`. Needs node + a built Showdown."""

import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def test_pack_runs_matches_per_row():
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector
    from porygon.v2.workers import Packer

    env = FastEnv("gen9randombattle", 2, 6, seed=3, encoder="v2", shaping=(0, 0, 0))
    try:
        rng = np.random.default_rng(0)
        col = FastCollector(env, Packer(9), lambda batch: None, seed=0)  # never acts: only its first batches are used
        bs = list(col.ready.values())
        rows = [(b, int(i)) for b in bs for i in np.flatnonzero(b.needs_action)]
        assert len(bs) >= 2 and len(rows) >= 6
        for trial in range(20):
            k = int(rng.integers(1, len(rows) + 1))
            pick = [rows[int(j)] for j in rng.integers(0, len(rows), k)]  # repeats, interleaving, any order
            if trial == 0:
                pick = sorted(pick, key=lambda t: (id(t[0]), t[1]))  # long runs from one batch
            ref = np.concatenate([col._pack(b, np.array([i])) for b, i in pick])
            got = col._pack_runs(iter(pick))
            assert got.dtype == ref.dtype and got.shape == ref.shape
            assert np.array_equal(got, ref)
        stats = col.rollout_stats()
        assert all(not k.startswith("perf/col_") for k in stats)  # nothing acted yet
    finally:
        env.close()


def test_collector_timing_counters():
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector
    from porygon.v2.workers import Packer

    rng = np.random.default_rng(0)

    def act(batch):
        mask = batch["action_mask"] > 0
        probs = mask / mask.sum(1, keepdims=True)
        acts = (rng.random(mask.shape) * mask).argmax(1)
        n = len(acts)
        return acts, np.log(probs[np.arange(n), acts]), np.zeros(n, np.float32), probs.astype(np.float32)

    env = FastEnv("gen9randombattle", 1, 4, seed=2, encoder="v2", shaping=(0, 0, 0))
    try:
        col = FastCollector(env, Packer(9), act, seed=0)
        for _ in range(30):
            col.step()
        st = col.rollout_stats()
        assert st["perf/col_acts"] > 0 and st["perf/col_fwd_rows"] > 0
        assert st["perf/col_act_s"] >= st["perf/col_fwd_s"] >= 0 and st["perf/col_poll_s"] >= 0
        col.take()
        assert all(not k.startswith("perf/col_") for k in col.rollout_stats())
    finally:
        env.close()
