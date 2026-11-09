"""Battle workers and rollout collection for the v2 trainer (`porygon.v2.train`).

One process per battle (as v1 `porygon.async_env`), started from a forkserver (clean restarts at any time). A
self-play worker runs a poke-env `SinglesEnv` with both sides played by the learner; a pool worker plays the learner
(side 0) against a frozen v2 snapshot or a baseline bot (`PoolOpponent`). Every message to the trainer carries, per
learner side, the packed observation (`Packer`: spec v2 fields + aux targets in one bytes blob) or None when that
side has no decision this step (waiting while the other side force-switches), plus the step's rewards and whether
the battle ended (the observations are then the next battle's).

Decision-point timing (checked on the local server, 2026-11-08, ~2300 decisions over 35 gen4 / gen9 battles): when
`PokeEnv.step` returns, `env.battle1` and `env.battle2` reflect the same decision point - same turn, the opponent's
active species and HP fraction in one equal the active's in the other in every check (only Illusion / forme names
differ), `agent{1,2}_to_move` is exactly `not battle.wait`, and every legal action of the other side maps to an
opponent-action class (`aux_targets`). Requests arrive after the turn's protocol lines and the server sends nothing
more until both sides chose, so neither battle object changes while the worker encodes.

`Collector` (trainer side) batches whatever observations are waiting into one forward pass, answers the workers and
turns decisions into transitions. A side's transition is closed when its next decision is evaluated (its value is the
transition's bootstrap `next_value`) or the battle ends (`done`, next_value 0); rewards received in between (and the
final ±1) are added to it. Open transitions carry over between rollouts, so a rollout never waits for a battle and
every closed transition has an exact bootstrap value: GAE per (worker, side) segment is then exact truncated GAE.
For a self-play step where both sides decide at once, each side's opponent-action soft target is the other side's
policy distribution mapped to its frame (`aux_opp_class`).
"""

import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Callable

import numpy as np
from gymnasium.spaces import Box
from poke_env import ServerConfiguration
from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import BattleOrder, DefaultBattleOrder, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from porygon.v2.encoder import aux_targets, encode
from porygon.v2.spec import N_OPP_CLASSES, aux_fields, fields

_DUMMY = np.zeros(1, np.float32)


def local_server(port: int) -> ServerConfiguration:  # = porygon.env.local_server (that module imports SB3)
    return ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                               "https://play.pokemonshowdown.com/action.php?")


def gen_of(battle_format: str) -> int:
    return int(battle_format[3:battle_format.index("random")]) if battle_format.startswith("gen") else 9


# ---------------------------------------------------------------- env

class V2Env(SinglesEnv):
    """SinglesEnv whose observations the worker builds itself (encode + aux_targets need both battles). Reward:
    sparse ±1 at the end (0 for a tie) unless shaping weights (fainted_value, hp_value, status_value) are given."""

    def __init__(self, *, shaping: dict[str, float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.shaping = {k: v for k, v in (shaping or {}).items() if v}
        self.observation_spaces = {a: Box(0, 1, (1,), np.float32) for a in self.possible_agents}

    def calc_reward(self, battle: AbstractBattle) -> float:
        if self.shaping:
            return self.reward_computing_helper(battle, victory_value=1.0, **self.shaping)
        if not battle.finished:
            return 0.0
        return 1.0 if battle.won else -1.0 if battle.lost else 0.0

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return _DUMMY


# ---------------------------------------------------------------- packed observations

class Packer:
    """spec.fields(gen) (+ spec.aux_fields(gen)) <-> one bytes blob per observation; batches unpack in bulk."""

    def __init__(self, gen: int, aux: bool = True):
        self.gen = gen
        self.spec = {**fields(gen), **(aux_fields(gen) if aux else {})}
        self.layout = {}
        off = 0
        for k, (shape, dtype) in self.spec.items():
            nb = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self.layout[k] = (off, nb, shape, np.dtype(dtype))
            off += nb
        self.size = off

    def pack(self, obs: dict, aux: dict | None = None) -> bytes:
        parts = []
        for k, (_, _, shape, dtype) in self.layout.items():
            a = obs[k] if k in obs else aux[k]
            assert a.shape == shape and a.dtype == dtype, (k, a.shape, a.dtype)
            parts.append(a.tobytes())
        return b"".join(parts)

    def unpack(self, blobs: list[bytes], keys=None) -> dict[str, np.ndarray]:
        raw = np.frombuffer(b"".join(blobs), np.uint8).reshape(len(blobs), self.size)
        out = {}
        for k in (keys or self.layout):
            off, nb, shape, dtype = self.layout[k]
            out[k] = raw[:, off:off + nb].copy().view(dtype).reshape(len(blobs), *shape)
        return out


def empty_like_blob(packer: Packer) -> bytes:
    """An all-zero observation blob (tests, placeholders)."""
    return bytes(packer.size)


# ---------------------------------------------------------------- opponent pool

class PoolOpponent:
    """Each battle: a frozen opponent with p_frozen (uniform over `pool_dir/frozen/*.pt`: fixed strong nets kept for
    the whole run against self-play cycling, e.g. the supervised init and the best checkpoint), a baseline bot with
    p_baseline (SimpleHeuristicsPlayer / MaxBasePowerPlayer), else a v2 snapshot from pool_dir (`*.pt`, by mtime):
    half the time one of the `recent` newest, otherwise uniform (no snapshot yet: a frozen net if any, else a
    baseline). Snapshots are not run here: `snapshot` names the file (relative to pool_dir) and the worker asks the
    trainer, which batches every pool battle's snapshot decisions on its device (the damage layer alone costs ~0.2 s
    per call on one CPU thread)."""

    def __init__(self, pool_dir: Path | None, battle_format: str, *, p_baseline: float = 0.2, recent: int = 4,
                 p_frozen: float = 0.0, rng=None):
        self.pool_dir = Path(pool_dir) if pool_dir else None
        self.format, self.p_baseline, self.recent, self.p_frozen = battle_format, p_baseline, recent, p_frozen
        common = dict(battle_format=battle_format, start_listening=False)
        self.baselines = {"heuristic": SimpleHeuristicsPlayer(**common), "max_power": MaxBasePowerPlayer(**common)}
        self.rng = rng or np.random.default_rng()
        self.label, self.snapshot = "heuristic", None

    def new_battle(self) -> None:
        snaps, frozen = [], []
        if self.pool_dir is not None and self.pool_dir.is_dir():
            snaps = sorted(self.pool_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
            if self.p_frozen > 0:
                frozen = sorted((self.pool_dir / "frozen").glob("*.pt"))
        r = self.rng.random()
        if frozen and (r < self.p_frozen or (not snaps and r >= self.p_frozen + self.p_baseline)):
            self.label, self.snapshot = "frozen", "frozen/" + frozen[self.rng.integers(len(frozen))].name
        elif snaps and r >= self.p_frozen + self.p_baseline:
            pick = snaps[-self.recent:] if self.rng.random() < 0.5 else snaps
            self.label, self.snapshot = "snapshot", pick[self.rng.integers(len(pick))].name
        else:
            self.label, self.snapshot = ("heuristic", "max_power")[self.rng.integers(2)], None

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        return self.baselines[self.label].choose_move(battle)


# ---------------------------------------------------------------- worker process

@dataclass
class WorkerSpec:
    kind: str  # "selfplay" | "pool"
    port: int
    battle_format: str
    shaping: dict = field(default_factory=dict)
    pool_dir: str | None = None
    p_baseline: float = 0.2
    recent: int = 4
    p_frozen: float = 0.0


def battle_worker(conn: Connection, spec: WorkerSpec) -> None:
    """Protocol: sends (obs per learner side | None, rewards per side, done, info, opp_request); receives
    (actions per side, opponent action) or None = stop. opp_request = (snapshot file name, packed observation of
    the opponent) when a pool snapshot opponent must decide, else None."""
    try:
        _battle_loop(conn, spec)
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        os._exit(0)  # never wait on poke-env's threads / an unfinished battle


def _battle_loop(conn: Connection, spec: WorkerSpec) -> None:
    gen = gen_of(spec.battle_format)
    packer, opp_packer = Packer(gen), Packer(gen, aux=False)
    env = V2Env(battle_format=spec.battle_format, server_configuration=local_server(spec.port), shaping=spec.shaping,
                log_level=40, open_timeout=None, challenge_timeout=None, strict=False)
    pool = spec.kind == "pool"
    opponent = PoolOpponent(spec.pool_dir, spec.battle_format, p_baseline=spec.p_baseline,
                            recent=spec.recent, p_frozen=spec.p_frozen) if pool else None
    n_sides = 1 if pool else 2
    a1, a2 = env.possible_agents
    decisions = [0, 0]

    def ob(b, other):
        o = encode(b)
        return packer.pack(o, aux_targets(b, other, o))

    def observe():
        b1, b2 = env.battle1, env.battle2
        out = [ob(b1, b2) if env.agent1_to_move else None]
        if not pool:
            out.append(ob(b2, b1) if env.agent2_to_move else None)
        for s, o in enumerate(out):
            decisions[s] += o is not None
        return out

    def opp_request():
        if pool and opponent.snapshot is not None and env.agent2_to_move:
            return opponent.snapshot, opp_packer.pack(encode(env.battle2))
        return None

    def reset():
        decisions[:] = [0, 0]
        env.reset()
        if pool:
            opponent.new_battle()
        return observe()

    def result(b):
        return 1 if b.won else -1 if b.lost else 0

    conn.send((reset(), [0.0] * n_sides, False, None, opp_request()))
    while (reply := conn.recv()) is not None:
        actions, opp_action = reply
        rewards = [0.0] * n_sides
        while True:
            if pool:
                opp = 0
                if env.agent2_to_move:
                    if opponent.snapshot is not None:
                        opp = opp_action
                    else:
                        opp = env.order_to_action(opponent.choose_move(env.battle2), env.battle2, strict=False)
                act = {a1: np.int64(actions[0]), a2: np.int64(opp)}
            else:
                act = {a1: np.int64(actions[0]), a2: np.int64(actions[1])}
            _, rew, term, trunc, _ = env.step(act)
            rewards[0] += float(rew[a1])
            if not pool:
                rewards[1] += float(rew[a2])
            done = any(term.values()) or any(trunc.values())
            if done or env.agent1_to_move or (not pool and env.agent2_to_move):
                break
            if pool and opponent.snapshot is not None and env.agent2_to_move:
                break  # learner waiting, the snapshot opponent decides: ask the trainer
            actions = [0] * n_sides  # learner waiting while a baseline opponent force-switches: step on
        info = None
        if done:
            info = {"kind": spec.kind, "opp": opponent.label if pool else "self", "turns": env.battle1.turn,
                    "decisions": decisions[:n_sides],
                    "result": [result(env.battle1)] + ([] if pool else [result(env.battle2)])}
            obs = reset()
        else:
            obs = observe()
        conn.send((obs, rewards, done, info, opp_request()))


class BattleWorkers:
    """Worker processes; any of them can be restarted (forkserver: safe after CUDA / threads exist)."""

    def __init__(self, specs: list[WorkerSpec]):
        self.specs = specs
        self.ctx = mp.get_context("forkserver")
        # No preload of this module: importing poke_env starts its POKE_LOOP thread, and a child forked from the
        # forkserver would inherit the loop object but not the thread, so every poke-env Player built in a worker
        # (PoolOpponent's SimpleHeuristicsPlayer / MaxBasePowerPlayer: create_in_poke_loop) blocks forever. That hung
        # every pool worker of R1 (2026-11-08/09; fastenv scripts/fastenv/debug_pool_worker.py reproduces it). Each
        # worker now imports poke_env itself (a few seconds per start). SinglesEnv runs its own loop + thread.
        self.conns: list[Connection | None] = [None] * len(specs)
        self.procs: list = [None] * len(specs)
        self.restarts = 0
        for i in range(len(specs)):
            self.start(i)

    def start(self, i: int) -> None:
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=battle_worker, args=(child, self.specs[i]), daemon=True)
        proc.start()
        child.close()
        self.conns[i], self.procs[i] = parent, proc

    def restart(self, i: int) -> None:
        self.restarts += 1
        self.kill(i)
        self.start(i)

    def kill(self, i: int) -> None:
        try:
            self.conns[i].close()
        except OSError:
            pass
        p = self.procs[i]
        if p is not None and p.is_alive():
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(None)
            except (OSError, ValueError):
                pass
        deadline = time.time() + 10
        for p in self.procs:
            p.join(max(0.1, deadline - time.time()))
        for i in range(len(self.procs)):
            self.kill(i)


# ---------------------------------------------------------------- collector (trainer side)

# transition fields
OBS, ACT, LOGP, VAL, SEED, REW, OPP_T, OPP_C, NEXT_V, DONE = range(10)


class Collector:
    """Drives the workers: forward every waiting observation in one batch (`act_fn`), answer, record transitions.

    act_fn(batch: dict[str, np.ndarray]) -> (actions [M], log_probs [M], values [M], probs [M, A]); batch holds the
    model fields plus `drop_seed` (int64, 0 = no id dropout). opp_fn(snapshot name, batch) -> actions [M] plays the
    pool's snapshot opponents."""

    def __init__(self, workers: BattleWorkers, packer: Packer, act_fn: Callable, opp_fn: Callable | None = None, *,
                 id_dropout: bool = True, stall_seconds: float = 900.0, seed: int | None = None):
        self.workers, self.packer, self.act_fn, self.opp_fn = workers, packer, act_fn, opp_fn
        self.opp_packer = Packer(packer.gen, aux=False)
        self.opp_decisions = 0
        self.id_dropout, self.stall_seconds = id_dropout, stall_seconds
        self.model_keys = list(fields(packer.gen)) + ["aux_opp_class"]
        self.rng = np.random.default_rng(seed)
        self.index = {}
        self.ready: dict[int, tuple] = {}  # worker -> (obs per side, opponent request) awaiting actions
        self.in_flight: dict[int, float] = {i: time.time() for i in range(len(workers.specs))}  # worker -> since
        self.open: dict[tuple[int, int], list] = {}
        self.segments: dict[tuple[int, int], list] = defaultdict(list)
        self.n_closed = 0
        self.episodes: list[dict] = []
        self.forward_batches = 0

    # -- one iteration
    def step(self, timeout: float = 1.0) -> None:
        if self.ready:
            self._act()
        conns = {self.workers.conns[w]: w for w in self.in_flight}
        for c in wait(list(conns), timeout=timeout):
            self._receive(conns[c])
        now = time.time()
        for w, since in list(self.in_flight.items()):
            if now - since > self.stall_seconds:
                print(f"worker {w} silent for {now - since:.0f} s: restarting", file=sys.stderr)
                self._restart(w)

    def _restart(self, w: int) -> None:
        for side in (0, 1):
            self.open.pop((w, side), None)
        self.ready.pop(w, None)
        self.workers.restart(w)
        self.in_flight[w] = time.time()

    def _receive(self, w: int) -> None:
        try:
            obs, rewards, done, info, opp_req = self.workers.conns[w].recv()
        except (EOFError, OSError, ConnectionResetError) as e:
            print(f"worker {w} died ({e!r}): restarting", file=sys.stderr)
            self._restart(w)
            return
        del self.in_flight[w]
        for side, r in enumerate(rewards):
            tr = self.open.get((w, side))
            if tr is not None:
                tr[REW] += r
        if done:
            for side in range(len(rewards)):
                tr = self.open.pop((w, side), None)
                if tr is not None:
                    self._close((w, side), tr, 0.0, True)
            if info is not None:
                self.episodes.append(info)
        self.ready[w] = (obs, opp_req)

    def _close(self, key, tr: list, next_value: float, done: bool) -> None:
        tr[NEXT_V], tr[DONE] = next_value, done
        self.segments[key].append(tr)
        self.n_closed += 1

    def _act(self) -> None:
        rows = [(w, side, blob) for w, (obs, _) in self.ready.items() for side, blob in enumerate(obs)
                if blob is not None]
        if rows:
            batch = self.packer.unpack([r[2] for r in rows], self.model_keys)
            seeds = (self.rng.integers(1, 2 ** 62, len(rows), dtype=np.int64) if self.id_dropout
                     else np.zeros(len(rows), np.int64))
            batch["drop_seed"] = seeds
            actions, logps, values, probs = self.act_fn(batch)
            self.forward_batches += 1
            pos = {}
            for i, (w, side, blob) in enumerate(rows):
                key = (w, side)
                prev = self.open.get(key)
                if prev is not None:
                    self._close(key, prev, float(values[i]), False)
                self.open[key] = [blob, int(actions[i]), float(logps[i]), float(values[i]), int(seeds[i]), 0.0,
                                  None, -1, 0.0, False]
                pos[key] = i
            cls = batch["aux_opp_class"]
            for (w, side), i in pos.items():  # self-play: both sides decided in this forward
                j = pos.get((w, 1 - side))
                if j is None:
                    continue
                c, p = cls[i], probs[j]
                ok = c >= 0
                if not ok.any():
                    continue
                target = np.zeros(N_OPP_CLASSES, np.float32)
                np.add.at(target, c[ok], p[ok])
                mass = target.sum()
                tr = self.open[(w, side)]
                tr[OPP_C] = int(c[int(actions[j])])
                if mass >= 0.5:
                    tr[OPP_T] = target / mass
        opp_actions = self._opponents()
        for w, (obs, _) in list(self.ready.items()):
            acts = [int(self.open[(w, s)][ACT]) if blob is not None and (w, s) in self.open else 0
                    for s, blob in enumerate(obs)]
            try:
                self.workers.conns[w].send((acts, opp_actions.get(w, 0)))
                self.in_flight[w] = time.time()
            except (OSError, BrokenPipeError) as e:
                print(f"worker {w} unreachable ({e!r}): restarting", file=sys.stderr)
                self._restart(w)
        self.ready = {}

    def _opponents(self) -> dict[int, int]:
        """Snapshot opponents' actions, one batched forward per snapshot."""
        by_name: dict[str, list] = defaultdict(list)
        for w, (_, req) in self.ready.items():
            if req is not None:
                by_name[req[0]].append((w, req[1]))
        out = {}
        for name, reqs in by_name.items():
            batch = self.opp_packer.unpack([b for _, b in reqs])
            batch["drop_seed"] = np.zeros(len(reqs), np.int64)
            acts = self.opp_fn(name, batch)
            self.opp_decisions += len(reqs)
            out.update({w: int(a) for (w, _), a in zip(reqs, acts)})
        return out

    def take(self) -> tuple[dict, list[dict]]:
        segs, eps = self.segments, self.episodes
        self.segments, self.episodes, self.n_closed = defaultdict(list), [], 0
        return segs, eps

    # -- background thread
    def run(self, rollout: int, requests: threading.Semaphore, out: queue.Queue, stop: threading.Event) -> None:
        """Collect rollouts of >= `rollout` closed transitions, one per acquired request token."""
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
        except BaseException as e:  # surface in the training thread
            traceback.print_exc()
            out.put(e)
