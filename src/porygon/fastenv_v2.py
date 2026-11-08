"""porygon.fastenv as a drop-in for netv2's rollout collection (`porygon.v2.workers.Collector`, branch ws/netv2):
same transitions (packed spec-v2 blobs + action, log-prob, value, id-dropout seed, summed rewards, opponent-action
soft target, bootstrap value, done), same `step` / `take` / `run` / counters, so `porygon.v2.train` can switch with
a few lines:

    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    env = FastEnv(args.format, n_workers, per_worker, seed=args.seed, encoder="v2", shaping=tuple(args.shaping))
    self.workers = FastWorkers(env)                      # instead of BattleWorkers(specs)
    self.collector = FastCollector(env, self.packer, self._act, id_dropout=..., seed=args.seed)

Needs the v2 JS encoder (`fastenv/js/encoder_v2.js`: obs/<field> and aux/<field> columns named as
`porygon.v2.spec.fields` / `aux_fields`). Self-play, plus snapshot pool battles (`pool_per_worker`, `pool_dir`,
opp_fn as in Collector); baseline-bot opponents (SimpleHeuristics / MaxBasePower are poke-env Python) are not
available. Transitions close as in Collector: at the side's next decision (bootstrap = its
value) or at the end of the battle (done, bootstrap 0); open transitions carry over between rollouts.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from collections import defaultdict
from typing import Callable

import numpy as np

from porygon.fastenv import Batch, FastEnv
from porygon.v2.spec import N_OPP_CLASSES, fields
from porygon.v2.workers import ACT, DONE, NEXT_V, OBS, OPP_C, OPP_T, REW, Packer  # noqa: F401  (layout)


class FastWorkers:
    """Stands in for BattleWorkers where the trainer only reads counters and closes it."""

    def __init__(self, env: FastEnv):
        self.env, self.restarts = env, 0

    def close(self) -> None:
        self.env.close()


class FastCollector:
    def __init__(self, env: FastEnv, packer: Packer, act_fn: Callable, opp_fn: Callable | None = None, *,
                 id_dropout: bool = True, seed: int | None = None, pool_per_worker: int = 0,
                 pool_dir=None, recent: int = 4, **_ignored):
        """pool_per_worker: in every worker, battles 0..k-1 are pool battles: side two is a frozen snapshot from
        pool_dir (`*.pt`, picked per battle like PoolOpponent: half the time one of the `recent` newest, else
        uniform), played through opp_fn(name, batch); no snapshot yet -> that battle is self-play."""
        self.env, self.packer, self.act_fn, self.opp_fn = env, packer, act_fn, opp_fn
        self.pool_per_worker, self.pool_dir, self.recent = pool_per_worker, pool_dir, recent
        self.snapshot: dict[int, str | None] = {}  # battle id -> snapshot name of the current pool battle
        self.opp_packer = Packer(packer.gen, aux=False)
        self.id_dropout = id_dropout
        self.rng = np.random.default_rng(seed)
        self.model_keys = list(fields(packer.gen)) + ["aux_opp_class"]
        self.ready: dict[int, Batch] = dict(env.poll_all())
        self.in_flight: set[int] = set()
        self.open: dict[int, list] = {}
        self.segments: dict[int, list] = defaultdict(list)
        self.n_closed = 0
        self.episodes: list[dict] = []
        self.forward_batches = 0
        self.opp_decisions = 0
        self.decisions: dict[int, int] = defaultdict(int)  # per slot, current battle
        b0 = next(iter(self.ready.values()))
        self.col = {k: (f"obs/{k}" if f"obs/{k}" in b0.cols else f"aux/{k}") for k in packer.layout}
        missing = [k for k, c in self.col.items() if c not in b0.cols]
        assert not missing, f"fastenv encoder lacks spec fields {missing}"
        for b in self.ready.values():
            self._receive(b)  # first decisions: picks the pool battles' opponents

    def _is_pool(self, battle: int) -> bool:
        return battle % self.env.battles_per_worker < self.pool_per_worker

    def _pick_snapshot(self) -> str | None:
        if self.pool_dir is None or self.opp_fn is None:
            return None
        from pathlib import Path
        snaps = sorted(Path(self.pool_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not snaps:
            return None
        pick = snaps[-self.recent:] if self.rng.random() < 0.5 else snaps
        return pick[self.rng.integers(len(pick))].name

    def _opponent(self, s: int) -> str | None:
        """Snapshot playing slot s (side two of a pool battle), else None (the learner plays it)."""
        return self.snapshot.get(s // 2) if s % 2 == 1 else None

    # -- packing: one bytes blob per row in Packer layout
    def _pack(self, b: Batch, rows: np.ndarray) -> np.ndarray:
        raw = np.empty((len(rows), self.packer.size), np.uint8)
        for k, (off, nb, shape, dtype) in self.packer.layout.items():
            a = np.ascontiguousarray(b.cols[self.col[k]][rows], dtype=dtype)
            raw[:, off:off + nb] = a.reshape(len(rows), -1).view(np.uint8)
        return raw

    def _close(self, key: int, tr: list, next_value: float, done: bool) -> None:
        tr[NEXT_V], tr[DONE] = next_value, done
        self.segments[key].append(tr)
        self.n_closed += 1

    def _receive(self, b: Batch) -> None:
        pending_result: dict[int, list] = {}
        for i in range(len(b)):
            s = int(b.slot[i])
            tr = self.open.get(s)
            if tr is not None:
                tr[REW] += float(b.reward[i])
            if b.flags[i] & 2:  # terminal row
                tr = self.open.pop(s, None)
                if tr is not None:
                    self._close(s, tr, 0.0, True)
                r = pending_result.setdefault(s // 2, [None, None, 0])
                r[s % 2] = int(b.win[i])
                r[2] = int(b.turn[i])
                if None not in r[:2]:
                    base = (s // 2) * 2
                    snap = self.snapshot.pop(s // 2, None)
                    dec = [self.decisions.pop(base, 0), self.decisions.pop(base + 1, 0)]
                    if snap is None:
                        self.episodes.append({"kind": "selfplay", "opp": "self", "turns": r[2], "decisions": dec,
                                              "result": r[:2]})
                    else:
                        self.episodes.append({"kind": "pool", "opp": "snapshot", "snapshot": snap, "turns": r[2],
                                              "decisions": dec[:1], "result": r[:1]})
            elif b.flags[i] & 4 and s % 2 == 0 and self._is_pool(s // 2):  # a new pool battle: pick its opponent
                self.snapshot[s // 2] = self._pick_snapshot()

    def _act(self) -> None:
        ws = list(self.ready)
        batches = [self.ready.pop(w) for w in ws]
        all_rows = [np.flatnonzero(b.needs_action) for b in batches]
        acts_out = [np.zeros(len(r), np.int64) for r in all_rows]
        # snapshot opponents: one forward per snapshot name, no transitions
        by_name: dict[str, list] = defaultdict(list)
        rows = []
        for bi, (b, r) in enumerate(zip(batches, all_rows)):
            mine = []
            for k, i in enumerate(r):
                name = self._opponent(int(b.slot[i]))
                if name is None:
                    mine.append(k)
                else:
                    by_name[name].append((bi, k))
            rows.append(np.asarray(mine, np.int64))
        for name, items in by_name.items():
            raw = np.concatenate([self._pack(batches[bi], all_rows[bi][[k]]) for bi, k in items])
            ob = {k: raw[:, off:off + nb].copy().view(dtype).reshape(len(items), *shape)
                  for k, (off, nb, shape, dtype) in self.packer.layout.items() if k in self.opp_packer.layout}
            ob["drop_seed"] = np.zeros(len(items), np.int64)
            acts = self.opp_fn(name, ob)
            self.opp_decisions += len(items)
            for (bi, k), a in zip(items, acts):
                acts_out[bi][k] = int(a)
        n = sum(map(len, rows))
        if n:
            rows_k = rows
            rows = [all_rows[bi][rk] for bi, rk in enumerate(rows_k)]
            raws = [self._pack(b, r) for b, r in zip(batches, rows)]
            raw = np.concatenate(raws)
            batch = {k: raw[:, off:off + nb].copy().view(dtype).reshape(n, *shape)
                     for k, (off, nb, shape, dtype) in self.packer.layout.items() if k in self.model_keys}
            seeds = (self.rng.integers(1, 2 ** 62, n, dtype=np.int64) if self.id_dropout
                     else np.zeros(n, np.int64))
            batch["drop_seed"] = seeds
            actions, logps, values, probs = self.act_fn(batch)
            self.forward_batches += 1
            slots = np.concatenate([b.slot[r] for b, r in zip(batches, rows)])
            pos = {}
            for i, s in enumerate(map(int, slots)):
                prev = self.open.get(s)
                if prev is not None:
                    self._close(s, prev, float(values[i]), False)
                self.open[s] = [raw[i].tobytes(), int(actions[i]), float(logps[i]), float(values[i]), int(seeds[i]),
                                0.0, None, -1, 0.0, False]
                self.decisions[s] += 1
                pos[s] = i
            cls = batch["aux_opp_class"]
            for s, i in pos.items():  # both sides decided in this forward: opponent-action target from its policy
                j = pos.get(s ^ 1)
                if j is None:
                    continue
                c, p = cls[i], probs[j]
                ok = c >= 0
                if not ok.any():
                    continue
                target = np.zeros(N_OPP_CLASSES, np.float32)
                np.add.at(target, c[ok], p[ok])
                mass = target.sum()
                tr = self.open[s]
                tr[OPP_C] = int(c[int(actions[j])])
                if mass >= 0.5:
                    tr[OPP_T] = target / mass
            k = 0
            for bi, rk in enumerate(rows_k):
                acts_out[bi][rk] = np.asarray(actions[k:k + len(rk)])
                k += len(rk)
        for w, b, a in zip(ws, batches, acts_out):
            self.env.send_to(w, b, a)
            self.in_flight.add(w)

    # -- Collector interface
    def step(self, timeout: float = 1.0) -> None:
        if self.ready:
            self._act()
        for w, b in self.env.poll(timeout):
            self.in_flight.discard(w)
            self._receive(b)
            self.ready[w] = b

    def take(self) -> tuple[dict, list[dict]]:
        segs, eps = self.segments, self.episodes
        self.segments, self.episodes, self.n_closed = defaultdict(list), [], 0
        return segs, eps

    def run(self, rollout: int, requests: threading.Semaphore, out: queue.Queue, stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                if not requests.acquire(timeout=1.0):
                    continue
                start = time.time()
                while self.n_closed < rollout and not stop.is_set():
                    self.step()
                if stop.is_set():
                    break
                segs, eps = self.take()
                out.put((segs, eps, time.time() - start))
        except BaseException as e:
            traceback.print_exc()
            out.put(e)
