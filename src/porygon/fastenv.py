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

import os
import select
import shutil
import struct
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

JS = Path(__file__).resolve().parents[2] / "fastenv" / "js"
MAGIC = 0x50594746
SHAPING = (0.05, 0.05, 0.0)  # porygon.env.SHAPING: (fainted, hp, status)
NEEDS_ACTION, DONE, FIRST, REJECTED = 1, 2, 4, 8


@dataclass
class Batch:
    slot: np.ndarray          # int64 [n]: global slot = worker * 2B + battle * 2 + side
    obs: np.ndarray           # float32 [n, obs_dim]
    mask: np.ndarray          # bool [n, n_actions]
    reward: np.ndarray        # float32 [n]: since this slot's previous row
    flags: np.ndarray         # uint8 [n]: NEEDS_ACTION | DONE | FIRST | REJECTED
    win: np.ndarray           # int8 [n]: terminal rows: +1 win, -1 loss, 0 tie
    turn: np.ndarray          # uint16 [n]
    opp_prev: np.ndarray      # int16 [n, 4]: opponent's choice at this slot's previous decision (see aux.js)
    opp_team: np.ndarray      # int16 [n, 6, 8]: opponent's true team in token order (species, item, ability, 4 moves, tera)
    opp_revealed: np.ndarray  # uint8 [n, 6]: 1 where that opponent token is a revealed mon

    @property
    def needs_action(self) -> np.ndarray:
        return (self.flags & NEEDS_ACTION) > 0

    @property
    def done(self) -> np.ndarray:
        return (self.flags & DONE) > 0

    def __len__(self) -> int:
        return len(self.slot)

    @staticmethod
    def concat(batches: list[Batch]) -> Batch:
        return Batch(*[np.concatenate([getattr(b, f.name) for b in batches]) for f in fields(Batch)])


def node_binary() -> str:
    return os.environ.get("PORYGON_NODE") or shutil.which("node") or "node"


class Worker:
    def __init__(self, index: int, fmt: str, battles: int, seed: int, encoder: str, shaping, node: str,
                 max_turns: int):
        self.index, self.battles = index, battles
        to_child_r, self.to_child = os.pipe()
        self.from_child, from_child_w = os.pipe()
        cmd = [node, "--max-old-space-size=2048", str(JS / "host.js"), "--format", fmt, "--battles", str(battles),
               "--seed", str(seed), "--encoder", encoder, "--fd-in", str(to_child_r), "--fd-out", str(from_child_w),
               "--shaping", ",".join(map(str, shaping[:3])), "--max-turns", str(max_turns)]
        self.proc = subprocess.Popen(cmd, pass_fds=(to_child_r, from_child_w), stdin=subprocess.DEVNULL)
        os.close(to_child_r)
        os.close(from_child_w)
        self.buf = bytearray(1 << 20)
        self.waiting = True  # a batch is (or will be) on its way

    def _read_into(self, view: memoryview) -> None:
        got = 0
        while got < len(view):
            k = os.readv(self.from_child, [view[got:]])
            if k == 0:
                raise RuntimeError(f"fastenv worker {self.index} exited (code {self.proc.poll()})")
            got += k

    def read(self) -> Batch:
        head = bytearray(16)
        self._read_into(memoryview(head))
        magic, n, d, a = struct.unpack("<4I", head)
        assert magic == MAGIC, f"fastenv worker {self.index}: bad batch header {magic:#x}"
        sizes = [4 * n * d, 4 * n, 4 * n, 2 * n * 48, 2 * n * 4, 2 * n, n * a, n, n, n * 6]
        total = sum(sizes)
        if len(self.buf) < total:
            self.buf = bytearray(2 * total)
        mv = memoryview(self.buf)[:total]
        self._read_into(mv)
        out, o = [], 0
        for size, dt in zip(sizes, ["<f4", "<f4", "<u4", "<i2", "<i2", "<u2", "u1", "u1", "i1", "u1"]):
            out.append(np.frombuffer(mv[o:o + size], dtype=dt).copy())
            o += size
        obs, reward, slot, opp_team, opp_prev, turn, mask, flags, win, revealed = out
        self.waiting = False
        return Batch(
            slot=slot.astype(np.int64) + self.index * 2 * self.battles, obs=obs.reshape(n, d),
            mask=mask.reshape(n, a).astype(bool), reward=reward, flags=flags, win=win, turn=turn,
            opp_prev=opp_prev.reshape(n, 4), opp_team=opp_team.reshape(n, 6, 8), opp_revealed=revealed.reshape(n, 6))

    def send(self, local_slots: np.ndarray, actions: np.ndarray) -> None:
        msg = np.empty((len(local_slots), 2), dtype="<i4")
        msg[:, 0] = local_slots
        msg[:, 1] = actions
        data = struct.pack("<2I", 1, len(local_slots)) + msg.tobytes()
        view = memoryview(data)
        while view:
            view = view[os.write(self.to_child, view):]
        self.waiting = True

    def close(self) -> None:
        try:
            os.write(self.to_child, struct.pack("<2I", 2, 0))
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        for fd in (self.to_child, self.from_child):
            try:
                os.close(fd)
            except OSError:
                pass


class FastEnv:
    """W node workers x B battles. Global slots: worker w owns [w * 2B, (w + 1) * 2B)."""

    def __init__(self, battle_format: str = "gen9randombattle", n_workers: int = 4, battles_per_worker: int = 64,
                 seed: int = 0, encoder: str | None = None, shaping=SHAPING, node: str | None = None,
                 max_turns: int = 1000):
        encoder = encoder or os.environ.get("PORYGON_FASTENV_ENCODER", "v1")
        self.format, self.n_workers, self.battles_per_worker = battle_format, n_workers, battles_per_worker
        node = node or node_binary()
        self.workers = [Worker(i, battle_format, battles_per_worker, seed * 1000 + i, encoder, shaping, node,
                               max_turns) for i in range(n_workers)]
        self.span = 2 * battles_per_worker
        self._pending: dict[int, Batch] = {}  # worker -> batch whose decisions are not answered yet

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
            b = w.read()
            self._pending[w.index] = b
            out.append((w.index, b))
        return out

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
