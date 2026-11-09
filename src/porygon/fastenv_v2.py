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
opp_fn as in Collector), plus baseline bots inside the env (`FastEnv(bot="heuristic", bot_battles=k)`: poke-env's
SimpleHeuristicsPlayer / MaxBasePowerPlayer ported to JS, episodes reported as kind "pool", opp = the bot). Transitions close as in Collector: at the side's next decision (bootstrap = its
value) or at the end of the battle (done, bootstrap 0); open transitions carry over between rollouts.

On-policy collection (`on_policy=True`, what `porygon.v2.train --collector fastenv --no-overlap` uses): at the end
of every rollout the collector waits for the workers' in-flight batches and closes every open learner transition
whose side has its next decision waiting, bootstrapping with the *current* actor's value of that decision (one
extra forward, no action sent). The next rollout then starts from fresh decisions acted by the refreshed actor, so
every transition's behaviour log-prob comes from the weights the update starts from (Collector's carry-over leaves
~2 open transitions per battle acted by the previous weights). The few open transitions whose side has no decision
waiting at that moment (the other side is force-switching) still carry over: `rollout_stats()` counts them
(`perf/fastenv_stale_frac`). With overlap (the trainer's default) use `--decoupled`: the behaviour log-probs stored
here are exact for the weights that acted.

`add_args(parser)` / `build(args, ...)` give `porygon.v2.train` its `--collector fastenv` switch
(patch: `patches/netv2-fastenv-collector.patch`).
"""

from __future__ import annotations

import queue
import sys
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

    def __init__(self, env: FastEnv, pool_per_worker: int = 0):
        self.env, self.pool_per_worker = env, pool_per_worker

    @property
    def restarts(self) -> int:  # dead fastenv workers replaced so far (the trainer logs perf/worker_restarts)
        return self.env.restarts

    def close(self) -> None:
        self.env.close()


# ---------------------------------------------------------------- porygon.v2.train --collector fastenv

def add_args(add: Callable) -> None:
    """The trainer's fastenv flags (`add` = parser.add_argument)."""
    add("--collector", choices=("showdown", "fastenv"), default="showdown",
        help="rollouts from Showdown servers + poke-env workers, or from fastenv (in-process Showdown, porygon.fastenv)")
    add("--fastenv-workers", type=int, default=0, help="fastenv Node workers (0: usable CPUs - 2)")
    add("--fastenv-battles", type=int, default=48, help="battles per fastenv worker (>= 32 keeps workers busy)")
    add("--fastenv-pool-share", type=float, default=None,
        help="share of each worker's battles played vs the pool (default: pool_battles / (n_battles + pool_battles));"
             " of those, a p_baseline share vs env-side baseline bots (heuristic / max_power by worker), the rest vs"
             " pool snapshots (self-play until the first snapshot exists)")


def build_env(args, seed: int | None = None) -> FastWorkers:
    """FastWorkers(FastEnv) for porygon.v2.train args (format, shaping, n_battles / pool_battles and the --fastenv-*
    flags). Workers default to the usable CPUs - 2 (the trainer's main and collector threads). Pool battles: the
    first round(B * share) battles of every worker; each new one draws its opponent as PoolOpponent (frozen net /
    env-side heuristic or max_power bot / snapshot: FastCollector(p_frozen=..., p_baseline=...))."""
    import os
    n = args.fastenv_workers or max(1, len(os.sched_getaffinity(0)) - 2)
    total = args.n_battles + args.pool_battles
    share = args.fastenv_pool_share if args.fastenv_pool_share is not None else (
        args.pool_battles / total if total else 0.0)
    pool = int(round(args.fastenv_battles * share))
    seed = seed if seed is not None else int.from_bytes(os.urandom(3), "little")  # restarts must not replay teams
    env = FastEnv(args.format, n, args.fastenv_battles, seed=seed, encoder="v2", shaping=tuple(args.shaping),
                  dyn_bots="heuristic,max_power" if pool else "")
    env.restart_dead = True  # a crashed worker is replaced (its battles start over) instead of ending training
    print(f"fastenv: {n} workers x {args.fastenv_battles} battles ({pool} pool battles per worker: frozen net with p "
          f"{getattr(args, 'p_frozen', 0.0)}, baseline bot with p {args.p_baseline}, else a snapshot), seed {seed}",
          flush=True)
    return FastWorkers(env, pool_per_worker=pool)


class FastCollector:
    def __init__(self, env: FastEnv, packer: Packer, act_fn: Callable, opp_fn: Callable | None = None, *,
                 id_dropout: bool = True, seed: int | None = None, pool_per_worker: int = 0,
                 pool_dir=None, recent: int = 4, on_policy: bool = False, p_baseline: float = 0.0,
                 p_frozen: float = 0.0, **_ignored):
        """pool_per_worker: in every worker, battles bot_battles..bot_battles+k-1 are pool battles: side two is a
        frozen snapshot from pool_dir (`*.pt`, picked per battle like PoolOpponent: half the time one of the
        `recent` newest, else uniform), played through opp_fn(name, batch). Each new pool battle draws its opponent
        exactly as netv2's PoolOpponent.new_battle (ws/netv2 0ac086c): a frozen net (uniform over
        pool_dir/frozen/*.pt, name 'frozen/<file>') with p_frozen, a baseline bot (uniform over the env's dyn_bots,
        FastEnv(dyn_bots='heuristic,max_power')) with p_baseline, else a snapshot; no snapshot yet -> a frozen net
        if any, else a baseline. Without dyn_bots a baseline draw is self-play."""
        self.env, self.packer, self.act_fn, self.opp_fn = env, packer, act_fn, opp_fn
        self.pool_per_worker, self.pool_dir, self.recent = pool_per_worker, pool_dir, recent
        self.snapshot: dict[int, str | None] = {}  # battle id -> snapshot name of the current pool battle
        self.dynbot: dict[int, int] = {}  # battle id -> index into env.dyn_bots of the current pool battle's bot
        self.p_baseline, self.p_frozen = p_baseline, p_frozen
        self.label: dict[int, str] = {}  # battle id -> "snapshot" | "frozen" for snapshot-played pool battles
        self._snaps, self._frozen, self._snaps_t = [], [], -1e9  # cached pool listings (re-listed every 10 s)
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
        self.on_policy = on_policy
        self.rollout_index = 0  # rollouts taken so far; transitions remember the rollout they were acted in
        self.born: dict[int, int] = {}  # slot -> rollout index of its open transition
        self.stale = 0  # closed transitions acted in an earlier rollout (carry-over)
        self.drained = 0  # transitions closed by the end-of-rollout bootstrap
        self.drain_s = 0.0
        b0 = next(iter(self.ready.values()))
        self.col = {k: (f"obs/{k}" if f"obs/{k}" in b0.cols else f"aux/{k}") for k in packer.layout}
        missing = [k for k, c in self.col.items() if c not in b0.cols]
        assert not missing, f"fastenv encoder lacks spec fields {missing}"
        for b in self.ready.values():
            self._receive(b)  # first decisions: picks the pool battles' opponents

    def _is_bot(self, battle: int) -> bool:  # side two played inside the env (FastEnv(bot=..., bot_battles=k))
        return battle % self.env.battles_per_worker < self.env.bot_battles

    def _is_pool(self, battle: int) -> bool:  # snapshot battles come after the env-side bot battles
        k = battle % self.env.battles_per_worker - self.env.bot_battles
        return 0 <= k < self.pool_per_worker

    def _list_snapshots(self) -> list:
        if self.pool_dir is None or self.opp_fn is None:
            return []
        if time.time() - self._snaps_t > 10.0:
            from pathlib import Path
            try:
                self._snaps = sorted(Path(self.pool_dir).glob("*.pt"), key=lambda p: p.stat().st_mtime)
                self._frozen = sorted((Path(self.pool_dir) / "frozen").glob("*.pt")) if self.p_frozen > 0 else []
            except OSError:  # a snapshot being replaced while listed: keep the previous listing
                pass
            self._snaps_t = time.time()
        return self._snaps

    def _pick_snapshot(self) -> str | None:
        snaps = self._list_snapshots()
        if not snaps:
            return None
        pick = snaps[-self.recent:] if self.rng.random() < 0.5 else snaps
        return pick[self.rng.integers(len(pick))].name

    def _new_pool_battle(self, battle: int) -> None:
        """Opponent of a new pool battle: PoolOpponent.new_battle's draw (frozen / baseline / snapshot)."""
        snaps = self._list_snapshots()
        frozen = self._frozen
        if (snaps and not snaps[-1].exists()) or (frozen and not all(f.exists() for f in frozen)):
            self._snaps_t = -1e9  # a listed file was replaced (e.g. --frozen-best rotates best-*.pt): list again
            snaps, frozen = self._list_snapshots(), self._frozen
        r = self.rng.random()
        if frozen and (r < self.p_frozen or (not snaps and r >= self.p_frozen + self.p_baseline)):
            self.snapshot[battle] = "frozen/" + frozen[self.rng.integers(len(frozen))].name
            self.label[battle] = "frozen"
        elif snaps and r >= self.p_frozen + self.p_baseline:
            self.snapshot[battle] = self._pick_snapshot()
            self.label[battle] = "snapshot"
        elif self.env.dyn_bots:
            self.dynbot[battle] = int(self.rng.integers(len(self.env.dyn_bots)))
        # else: a baseline draw without env bots -> this battle is self-play

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
        if self.born.pop(key, self.rollout_index) != self.rollout_index:
            self.stale += 1

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
                if (self._is_bot(s // 2) or (s // 2) in self.dynbot) and s % 2 == 0:  # bot battle: learner side only
                    k = self.dynbot.pop(s // 2, None)
                    self.episodes.append({"kind": "pool", "turns": r[2],
                                          "opp": self.env.worker_bot(s // self.env.span) if k is None
                                          else self.env.dyn_bots[k],
                                          "decisions": [self.decisions.pop(s, 0)], "result": [r[0]]})
                    self.decisions.pop(s + 1, None)
                    pending_result.pop(s // 2)
                elif None not in r[:2]:
                    base = (s // 2) * 2
                    snap = self.snapshot.pop(s // 2, None)
                    label = self.label.pop(s // 2, "snapshot")
                    dec = [self.decisions.pop(base, 0), self.decisions.pop(base + 1, 0)]
                    if snap is None:
                        self.episodes.append({"kind": "selfplay", "opp": "self", "turns": r[2], "decisions": dec,
                                              "result": r[:2]})
                    else:
                        self.episodes.append({"kind": "pool", "opp": label, "snapshot": snap, "turns": r[2],
                                              "decisions": dec[:1], "result": r[:1]})
            elif b.flags[i] & 4 and s % 2 == 0 and self._is_pool(s // 2):  # a new pool battle: pick its opponent
                self._new_pool_battle(s // 2)

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
                s = int(b.slot[i])
                if s % 2 == 1 and (s // 2) in self.dynbot:  # hand side two to the env-side bot for this battle
                    acts_out[bi][k] = FastEnv.bot_action(self.dynbot[s // 2])
                    continue
                name = self._opponent(s)
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
                self.born[s] = self.rollout_index
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
            try:
                self.env.send_to(w, b, a)
            except OSError as e:  # the worker died since its batch arrived
                if not self.env.restart_dead:
                    raise
                self.env.dead.append((w, repr(e)))
                continue
            self.in_flight.add(w)

    def _poll(self, timeout: float) -> None:
        for w, b in self.env.poll(timeout):
            self.in_flight.discard(w)
            self._receive(b)
            self.ready[w] = b
        for w, why in self.env.take_dead():
            self._restart(w, why)

    def _restart(self, w: int, why: str) -> None:
        """Worker w died: drop what was open for its slots (closed transitions stay), start a fresh process."""
        lo, hi = w * self.env.span, (w + 1) * self.env.span
        for s in [s for s in self.open if lo <= s < hi]:
            self.open.pop(s)
            self.born.pop(s, None)
        for s in range(lo, hi):
            self.decisions.pop(s, None)
        for bt in range(lo // 2, hi // 2):
            self.snapshot.pop(bt, None)
            self.dynbot.pop(bt, None)
            self.label.pop(bt, None)
        self.in_flight.discard(w)
        self.ready.pop(w, None)
        b = self.env.restart_worker(w)
        print(f"fastenv: worker {w} died ({why}); restarted it ({self.env.restarts} restarts so far)",
              file=sys.stderr, flush=True)
        self._receive(b)
        self.ready[w] = b

    # -- Collector interface
    def step(self, timeout: float = 1.0) -> None:
        if self.ready:
            self._act()
        self._poll(timeout)

    def drain(self) -> None:
        """End of an on-policy rollout: receive every in-flight batch, then close each open learner transition whose
        side has its next decision waiting, with the current actor's value of that decision as the bootstrap (those
        decisions stay `ready` and are acted afresh by the next rollout's actor)."""
        t0 = time.time()
        while self.in_flight:
            self._poll(1.0)
        picks = []  # (batch, row) of learner decisions that close an open transition
        for b in self.ready.values():
            for i in np.flatnonzero(b.needs_action):
                s = int(b.slot[i])
                if s in self.open and self._opponent(s) is None:
                    picks.append((b, i, s))
        if picks:
            raw = np.concatenate([self._pack(b, np.array([i])) for b, i, _ in picks])
            n = len(picks)
            batch = {k: raw[:, off:off + nb].copy().view(dtype).reshape(n, *shape)
                     for k, (off, nb, shape, dtype) in self.packer.layout.items() if k in self.model_keys}
            batch["drop_seed"] = (self.rng.integers(1, 2 ** 62, n, dtype=np.int64) if self.id_dropout
                                  else np.zeros(n, np.int64))
            _, _, values, _ = self.act_fn(batch)
            for (_, _, s), v in zip(picks, values):
                self._close(s, self.open.pop(s), float(v), False)
            self.drained += n
        self.drain_s += time.time() - t0

    def rollout_stats(self) -> dict[str, float]:
        """Per-rollout collector diagnostics (reset by `take`): share of transitions acted in an earlier rollout,
        share closed by the end-of-rollout bootstrap, seconds spent draining."""
        n = max(self.n_closed, 1)
        return {"perf/fastenv_stale_frac": self.stale / n, "perf/fastenv_drained_frac": self.drained / n,
                "perf/fastenv_drain_s": self.drain_s}

    def take(self) -> tuple[dict, list[dict]]:
        self.last_stats = self.rollout_stats()
        segs, eps = self.segments, self.episodes
        self.segments, self.episodes, self.n_closed = defaultdict(list), [], 0
        self.stale = self.drained = 0
        self.drain_s = 0.0
        self.rollout_index += 1
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
                if self.on_policy:
                    self.drain()
                segs, eps = self.take()
                out.put((segs, eps, time.time() - start))
        except BaseException as e:
            traceback.print_exc()
            out.put(e)
