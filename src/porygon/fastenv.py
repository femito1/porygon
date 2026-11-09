"""Batched in-process Showdown self-play: W Node workers (fastenv/js/host.js), each running B battles with
Showdown's own simulator, a first-person tracker and the observation encoder in JS. No server, no websocket, no
poke-env. Contract: .orchestra/CONTRACTS.md, "Environment API".

    env = FastEnv("gen9randombattle", n_workers=8, battles_per_worker=64, seed=1)
    batch = env.reset()                         # rows of every worker
    while True:
        acts = policy(batch.obs[batch.needs_action], batch.mask[batch.needs_action])
        batch = env.step(acts)                  # one action per needs_action row, in row order

A row is one side of one battle at one moment: a decision (needs_action: obs, mask) or a terminal row (done:
the battle ended for this side; obs/mask zero). Every row carries the reward since that slot's previous row
(difference of poke-env's reward_computing_helper state value, victory included), so a trajectory per slot is
rows in order, new battle after each done. `step` is lockstep over workers; the asynchronous `poll`/`send` pair
lets a trainer act on whichever workers are ready (AsyncPPO style).
"""

from __future__ import annotations

import json
import os
import select
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

JS = Path(__file__).resolve().parents[2] / "fastenv" / "js"
MAGIC, MAGIC_HELLO = 0x50594746, 0x4c454846
DTYPES = {"f32": "<f4", "i32": "<i4", "u32": "<u4", "i16": "<i2", "u16": "<u2", "u8": "u1", "i8": "i1"}
SHAPING = (0.05, 0.05, 0.0)  # porygon.env.SHAPING: (fainted, hp, status)
NEEDS_ACTION, DONE, FIRST, REJECTED = 1, 2, 4, 8


class Batch:
    """Rows of one or more workers. Columns (numpy arrays, first axis = row):
    slot int64 (global: worker * 2B + battle * 2 + side), reward f32 (since this slot's previous row), flags u8
    (NEEDS_ACTION | DONE | FIRST | REJECTED), win i8 (terminal rows: +1 / -1 / 0 tie or void), turn u16, mask bool
    [A], opp_prev i16 [4] (the opponent's choice at this slot's previous decision: its action index in its own
    frame, move id, switched-in species id, tera 0/1; -1 / 0 when none), opp_team i16 [6, 8] (opponent's true team
    in token order: species, item, ability, 4 moves, tera type), opp_revealed u8 [6]; then the encoder's
    observation: `obs` f32 [OBS_DIM] (v1) or `obs/<field>` (multi-field encoders, see `.obs`), and `aux/<field>`."""

    def __init__(self, cols: dict[str, np.ndarray]):
        self.cols = cols

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self.__dict__["cols"][name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def obs(self):
        """v1: the f32 [n, OBS_DIM] array; multi-field encoders: {field: array}."""
        if "obs" in self.cols:
            return self.cols["obs"]
        return {k[4:]: v for k, v in self.cols.items() if k.startswith("obs/")}

    @property
    def aux(self) -> dict[str, np.ndarray]:
        return {k[4:]: v for k, v in self.cols.items() if k.startswith("aux/")}

    @property
    def needs_action(self) -> np.ndarray:
        return (self.flags & NEEDS_ACTION) > 0

    @property
    def done(self) -> np.ndarray:
        return (self.flags & DONE) > 0

    def __len__(self) -> int:
        return len(self.cols["slot"])

    def take(self, rows) -> "Batch":
        return Batch({k: v[rows] for k, v in self.cols.items()})

    @staticmethod
    def concat(batches: list["Batch"]) -> "Batch":
        return Batch({k: np.concatenate([b.cols[k] for b in batches]) for k in batches[0].cols})


def obs_rows(obs, rows):
    """Select rows of an observation (array or dict of arrays)."""
    return {k: v[rows] for k, v in obs.items()} if isinstance(obs, dict) else obs[rows]


def obs_concat(parts: list):
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.concatenate(parts)


def node_binary() -> str:
    return os.environ.get("PORYGON_NODE") or shutil.which("node") or "node"


class Worker:
    def __init__(self, index: int, fmt: str, battles: int, seed: int, encoder: str, shaping, node: str,
                 max_turns: int, duplicate: bool, bot: str, bot_battles: int, dyn_bots: str = ""):
        self.index, self.battles = index, battles
        to_child_r, self.to_child = os.pipe()
        self.from_child, from_child_w = os.pipe()
        cmd = [node, "--max-old-space-size=2048", str(JS / "host.js"), "--format", fmt, "--battles", str(battles),
               "--seed", str(seed), "--encoder", encoder, "--fd-in", str(to_child_r), "--fd-out", str(from_child_w),
               "--shaping", ",".join(map(str, shaping[:3])), "--max-turns", str(max_turns),
               "--duplicate", str(int(duplicate))] + (["--bot", bot, "--bot-battles", str(bot_battles)] if bot else []) \
            + (["--bots", dyn_bots] if dyn_bots else [])
        self.proc = subprocess.Popen(cmd, pass_fds=(to_child_r, from_child_w), stdin=subprocess.DEVNULL)
        os.close(to_child_r)
        os.close(from_child_w)
        self.buf = bytearray(1 << 20)
        self.columns = None  # from the hello message
        self.waiting = True  # a batch is (or will be) on its way

    def _read_into(self, view: memoryview) -> None:
        got = 0
        while got < len(view):
            k = os.readv(self.from_child, [view[got:]])
            if k == 0:
                raise RuntimeError(f"fastenv worker {self.index} exited (code {self.proc.poll()})")
            got += k

    def _hello(self) -> None:
        head = bytearray(8)
        self._read_into(memoryview(head))
        magic, n = struct.unpack("<2I", head)
        assert magic == MAGIC_HELLO, f"fastenv worker {self.index}: bad hello {magic:#x}"
        body = bytearray(n)
        self._read_into(memoryview(body))
        self.info = json.loads(body)
        self.columns = [(name, np.dtype(DTYPES[dt]), tuple(shape)) for name, dt, shape in self.info["columns"]]

    def read(self) -> Batch:
        if self.columns is None:
            self._hello()
        head = bytearray(16)
        self._read_into(memoryview(head))
        magic, n, ncol, a = struct.unpack("<4I", head)
        assert magic == MAGIC and ncol == len(self.columns), f"fastenv worker {self.index}: bad batch header"
        sizes = [n * int(np.prod(shape)) * dt.itemsize for _, dt, shape in self.columns]
        total = sum(sizes)
        if len(self.buf) < total:
            self.buf = bytearray(2 * total)
        mv = memoryview(self.buf)[:total]
        self._read_into(mv)
        cols, o = {}, 0
        for (name, dt, shape), size in zip(self.columns, sizes):
            cols[name] = np.frombuffer(mv[o:o + size], dtype=dt).reshape(n, *shape).copy()
            o += size
        cols["slot"] = cols["slot"].reshape(n).astype(np.int64) + self.index * 2 * self.battles
        for k in ("reward", "turn", "flags", "win"):
            cols[k] = cols[k].reshape(n)
        cols["mask"] = cols["mask"].astype(bool)
        self.waiting = False
        return Batch(cols)

    def send(self, local_slots: np.ndarray, actions: np.ndarray) -> None:
        msg = np.empty((len(local_slots), 2), dtype="<i4")
        msg[:, 0] = local_slots
        msg[:, 1] = actions
        data = struct.pack("<2I", 1, len(local_slots)) + msg.tobytes()
        view = memoryview(data)
        while view:
            view = view[os.write(self.to_child, view):]
        self.waiting = True

    def kill(self) -> None:
        """Dead or misbehaving worker: kill it and release its pipes."""
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass
        for fd in (self.from_child, self.to_child):
            try:
                os.close(fd)
            except OSError:
                pass

    def close(self) -> None:
        os.close(self.from_child)  # a host blocked writing a batch gets EPIPE and exits
        try:
            os.write(self.to_child, struct.pack("<2I", 2, 0))
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        for fd in (self.to_child,):
            try:
                os.close(fd)
            except OSError:
                pass


class FastEnv:
    """W node workers x B battles. Global slots: worker w owns [w * 2B, (w + 1) * 2B)."""

    def __init__(self, battle_format: str = "gen9randombattle", n_workers: int = 4, battles_per_worker: int = 64,
                 seed: int = 0, encoder: str | None = None, shaping=SHAPING, node: str | None = None,
                 max_turns: int = 1000, duplicate: bool = False, bot: str = "", bot_battles: int = 0,
                 dyn_bots: str = ""):
        """bot: an env-side baseline ('heuristic' / 'max_power' = poke-env SimpleHeuristicsPlayer /
        MaxBasePowerPlayer, 'random') playing side two of the first bot_battles battles of every worker (those
        sides emit no rows); a comma list ('heuristic,max_power') assigns the bots to workers round-robin
        (worker i plays bots[i % len(bots)]).
        dyn_bots: bots ('heuristic,max_power') the caller may hand side two of any battle to, one battle at a time,
        by answering that side's row with action -2-k (`bot_action(k)`); the bot then plays the side until the
        battle ends and the side emits no more rows (not even its terminal row). FastCollector uses it for netv2's
        pool mix."""
        encoder = encoder or os.environ.get("PORYGON_FASTENV_ENCODER", "v1")
        self.bots = [b for b in bot.split(",") if b] if bot else []
        self.dyn_bots = [b for b in dyn_bots.split(",") if b]
        self.bot, self.bot_battles = bot, (bot_battles if self.bots else 0)
        self.format, self.n_workers, self.battles_per_worker = battle_format, n_workers, battles_per_worker
        node = node or node_binary()
        self.seed, self.restarts = seed, 0
        self.restart_dead = False  # True: a dead worker is reported in `dead` (and restartable) instead of raising
        self.dead: list[tuple[int, str]] = []
        self._new_worker = lambda i, s: Worker(i, battle_format, battles_per_worker, s, encoder, shaping, node,
                                               max_turns, duplicate, self.worker_bot(i), self.bot_battles,
                                               ",".join(self.dyn_bots))
        self.workers = [self._new_worker(i, seed * 1000 + i) for i in range(n_workers)]
        self.span = 2 * battles_per_worker
        self._pending: dict[int, Batch] = {}  # worker -> batch whose decisions are not answered yet

    @staticmethod
    def bot_action(k: int) -> int:
        """The action that hands a side-two row's battle to dyn_bots[k]."""
        return -2 - k

    def worker_bot(self, worker: int) -> str:
        """The env-side bot of that worker's bot battles ('' if none)."""
        return self.bots[worker % len(self.bots)] if self.bots else ""

    # --- asynchronous interface ---
    def poll(self, timeout: float | None = None) -> list[tuple[int, Batch]]:
        """Batches from every worker that has one ready (blocks until at least one, or timeout)."""
        fds = {w.from_child: w for w in self.workers if w.waiting}
        if not fds:
            return []
        ready, _, _ = select.select(list(fds), [], [], timeout)
        out = []
        for fd in ready:
            w = fds[fd]
            try:
                b = w.read()
            except (RuntimeError, OSError, AssertionError) as e:
                if not self.restart_dead:
                    raise
                w.waiting = False
                self.dead.append((w.index, repr(e)))
                continue
            self._pending[w.index] = b
            out.append((w.index, b))
        return out

    def take_dead(self) -> list[tuple[int, str]]:
        """Workers found dead since the last call (restart_dead mode): (index, reason)."""
        d, self.dead = self.dead, []
        return d

    def restart_worker(self, i: int) -> Batch:
        """Replace worker i by a fresh process (new seed; all its battles start over) and return its first batch.
        The caller drops whatever it held for that worker's slots."""
        self.workers[i].kill()
        self.restarts += 1
        self._pending.pop(i, None)
        self.workers[i] = self._new_worker(i, self.seed * 1000 + i + 7919 * self.restarts)
        b = self.workers[i].read()
        self._pending[i] = b
        return b

    def send(self, worker: int, actions: np.ndarray) -> None:
        """Actions for every needs_action row of that worker's last batch, in row order."""
        self.send_to(worker, self._pending[worker], actions)

    def send_to(self, worker: int, batch: Batch, actions: np.ndarray) -> None:
        """Same, given that batch (the worker's last one) explicitly."""
        self._pending.pop(worker, None)
        rows = batch.needs_action
        assert len(actions) == int(rows.sum()), (len(actions), int(rows.sum()))
        self.workers[worker].send(batch.slot[rows] - worker * self.span, np.asarray(actions))

    def poll_all(self) -> list[tuple[int, Batch]]:
        """Waits for every worker that owes a batch; returns them in worker order."""
        got: dict[int, Batch] = {}
        while any(w.waiting for w in self.workers):
            got.update(self.poll())
        return sorted(got.items())

    # --- lockstep interface ---
    def reset(self) -> Batch:
        return Batch.concat([b for _, b in self.poll_all()])

    def step(self, actions: np.ndarray) -> Batch:
        k = 0
        for i in range(self.n_workers):
            m = int(self._pending[i].needs_action.sum())
            self.send(i, actions[k:k + m])
            k += m
        assert k == len(actions)
        return self.reset()

    def close(self) -> None:
        for w in self.workers:
            w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def random_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform over legal actions, vectorised."""
    keys = rng.random(mask.shape) * mask
    return keys.argmax(1)
