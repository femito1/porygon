"""One-turn lookahead whose leaves are scored by a v2 network (gen9), gated on the same raw network (Bet 1, 11-15).

At every decision with >= 2 legal actions:
  1. the raw v2 net on the live battle (`porygon.netsearch.net_forward`): T-1 probabilities p and P(win); a_net is
     drawn from softmax(logits / T) (T 0.5 = the benchmarked raw player), exactly as NetSearchPlayer draws it;
  2. K worlds = determinizations of the opponent's hidden information (`porygon.search.convert`, our search's sampler);
  3. per world, every (our legal action, their engine option) pair is simulated for one turn with poke-engine's
     `generate_instructions`; chance branches below `min_prob` (and below the most likely one) are dropped and the
     rest renormalised. A branch is scored 1 / 0 / 0.5 when the battle is over, else by the net's P(win) of the leaf
     observation (`porygon.v3leaf.LeafEncoder`: the root's real v2 observation patched with the engine state,
     turns=1). A branch after which only the opponent has to switch (a faint, their U-turn) continues with their
     switch-in that is worst for us (min over switch-ins of the expected score of `none` vs that switch); one where we
     have to switch is scored directly (the net makes forced switches in training);
  4. the K payoff matrices M_k[a, j] (our expected P(win)) form a zero-sum Bayesian game: nature draws world k, the
     opponent knows it and plays y_k, we play one mixed strategy x. Regret matching+ with linear averaging
     (`solve_bayes`) gives x and the y_k; Q(a) = mean_k M_k[a] . y_k is our win probability with action a against the
     opponent's equilibrium play (the game value for actions in x's support, less outside it);
  5. gate (`porygon.netsearch`'s margin rule): a_L = argmax x (first on ties) is the lookahead's own move; play it
     instead of a_net iff a_L != a_net and D = Q(a_L) - Q(a_net) > margin. margin +inf = the raw net, -inf = the pure
     lookahead. Q(a_net) undefined (not an engine option in every world) => a_net; any failure => a_net, counted as a
     fallback.

The value: G9q's value head (netv2 PPO) is trained with BCE(sigmoid(win_logit), (lambda-return + 1) / 2) on sparse
+-1 rewards (its trainer argv has no --shaping; the default is 0 0 0), so sigmoid(win_logit) is a win probability
and no potential term is added (porygon.lookahead's V + Phi rule is for the shaped v1 values).

Worlds expand in worker processes (`LookaheadPool`: `python -m porygon.lookahead2 --worker`, one world per request,
length-prefixed pickles over pipes); the leaves of all worlds are scored in one batched forward (a copy of the net on
`leaf_device`); the game is solved in the player's process. No time budget: the same position and worlds give the same
decision on any machine, only slower under contention.

Needs PORYGON_GEN=9 (gen9 engine build first on PYTHONPATH) and netv2's porygon.v2 importable (NETV2_SRC, appended to
porygon.__path__ as scripts/fastenv/netsearch_h2h.py does; the workers do the same from the inherited environment).
"""

import asyncio
import logging
import math
import os
import pickle
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("porygon.lookahead2")

DEFAULT_MARGIN = 0.03
SOLVE_ITERS = 1000
ERROR_SHARE_MAX = 0.1  # a world whose share of failed (engine panic) cells is above this is dropped


# ---------------------------------------------------------------- pure functions (no engine, no network)

def _rm(r: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    """Regret matching: positive regrets normalised over the last axis; uniform (over `allowed`) when none."""
    pos = np.maximum(r, 0.0) if allowed is None else np.where(allowed, np.maximum(r, 0.0), 0.0)
    tot = pos.sum(-1, keepdims=True)
    unif = np.ones_like(pos) if allowed is None else allowed.astype(float)
    unif = unif / unif.sum(-1, keepdims=True)
    return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), unif)


@dataclass
class Solution:
    x: np.ndarray  # [n] our average strategy
    ys: list  # per world [m_k] the opponent's average strategy
    q: np.ndarray  # [n] Q(a) = sum_k w_k M_k[a] . y_k
    value: float  # x . q
    expl: float  # exploitability of (x, y): our best deviation gain + the opponents' (>= 0, 0 at equilibrium)


def solve_bayes(mats: list[np.ndarray], weights: list[float] | None = None, iters: int = SOLVE_ITERS) -> Solution:
    """The zero-sum Bayesian game max_x sum_k w_k min_{y_k} x' M_k y_k (rows: our actions, the same in every world;
    columns: the opponent's options in world k). Alternating regret matching+ with linearly weighted averages of both
    sides' strategies; padded columns are masked."""
    k, n = len(mats), mats[0].shape[0]
    m = max(a.shape[1] for a in mats)
    A = np.zeros((k, n, m))
    real = np.zeros((k, m), bool)
    for i, a in enumerate(mats):
        assert a.shape[0] == n, "every world needs the same rows"
        A[i, :, :a.shape[1]] = a
        real[i, :a.shape[1]] = True
    w = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, float) / float(np.sum(weights))
    rx, ry = np.zeros(n), np.zeros((k, m))
    sx, sy = np.zeros(n), np.zeros((k, m))
    for t in range(1, iters + 1):  # as porygon.lookahead.solve, vectorised over the worlds
        y = _rm(ry, real)
        ux = np.einsum("k,knm,km->n", w, A, y)  # our score per action against y
        x = _rm(rx)
        rx = np.maximum(rx + ux - x @ ux, 0.0)
        x = _rm(rx)
        uy = -np.einsum("n,knm->km", x, A)  # the opponent minimises our score
        ry = np.where(real, np.maximum(ry + uy - (y * uy).sum(1, keepdims=True), 0.0), 0.0)
        sx += t * x
        sy += t * _rm(ry, real)
    x = sx / sx.sum()
    yk = sy / sy.sum(1, keepdims=True)
    q = np.einsum("k,knm,km->n", w, A, yk)
    value = float(x @ q)
    xa = np.einsum("n,knm->km", x, A)
    their_best = float(sum(w[i] * xa[i, real[i]].min() for i in range(k)))
    expl = float(q.max() - value) + (value - their_best)
    return Solution(x, [yk[i, :mats[i].shape[1]] for i in range(k)], q, value, expl)


@dataclass
class Gate:
    action: int  # the move to play
    net: int  # the network's move
    search: int  # the lookahead's own move (argmax x)
    gap: float  # Q(search) - Q(net); NaN if Q(net) is undefined
    override: bool  # played the lookahead's move instead of the network's


def gate(a_net: int, rows: list[int], x: np.ndarray, q: np.ndarray, margin: float) -> Gate:
    """The margin rule on the lookahead's statistics (aligned with `rows`, our action indices). margin -inf always
    plays argmax x (the pure lookahead); a finite margin keeps a_net when it has no row."""
    a_l = int(rows[int(np.argmax(x))])
    gap = float(q[rows.index(a_l)] - q[rows.index(a_net)]) if a_net in rows else math.nan
    override = a_l != a_net and (margin == -math.inf or (math.isfinite(gap) and gap > margin))
    return Gate(a_l if override else a_net, a_net, a_l, gap, bool(override))


def ref_value(ref: tuple, v: np.ndarray, offset: int = 0) -> float:
    """Expected score of a branch reference from a world expansion: ("L", leaf index) the net's value, ("T", value) a
    finished battle, ("S", [[(p, ref)...] per switch-in]) the opponent's forced switch: the switch-in worst for us."""
    kind = ref[0]
    if kind == "L":
        return float(v[offset + ref[1]])
    if kind == "T":
        return float(ref[1])
    if kind == "S":
        return min(sum(p * ref_value(r, v, offset) for p, r in sub) for sub in ref[1])
    raise ValueError(f"bad reference {ref!r}")


def world_matrix(cells: list[list[list | None]], v: np.ndarray, offset: int = 0) -> tuple[np.ndarray, int]:
    """(payoff matrix [rows, cols] of one world, failed cells). A failed cell (None: the engine panicked on the pair)
    takes its row's mean over the other cells; a column that failed for every row is dropped; a row that failed
    everywhere fails the world (ValueError)."""
    n, m = len(cells), len(cells[0]) if cells else 0
    M = np.full((n, m), np.nan)
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            if cell is not None:
                M[i, j] = sum(p * ref_value(r, v, offset) for p, r in cell)
    bad = int(np.isnan(M).sum())
    if bad:
        M = M[:, ~np.isnan(M).all(0)]
        if M.shape[1] == 0 or np.isnan(M).all(1).any():
            raise ValueError("a row or every column of the world failed")
        M = np.where(np.isnan(M), np.nanmean(M, 1, keepdims=True), M)
    return M, bad


# ---------------------------------------------------------------- one world (runs in a worker process)

def engine_choice(choice: str) -> str:
    """An MCTS option ('switch xatu', 'No Move', 'flamethrower-tera') as generate_instructions takes it."""
    return "none" if choice == "No Move" else choice.removeprefix("switch ")


def expand_world(root: Any, conv: Any, legal: np.ndarray, min_prob: float = 0.0) -> dict:
    """Every (our legal action, their engine option) pair of one world, simulated for one turn. Returns rows (our
    action indices, ascending), cols (their engine options), cells[i][j] = [(p, ref)] or None (engine error), the
    stacked leaf observations, and counts. root: v3leaf.LeafRoot; conv: search.Converted."""
    import poke_engine as pe
    from porygon import v3leaf as L
    t0 = time.perf_counter()
    enc = L.LeafEncoder(root, conv)
    state = enc.state
    r = pe.monte_carlo_tree_search(state, iterations=1)
    rows: dict[int, str] = {}
    for o in r.side_one:
        c = o.move_choice
        if c == "No Move":
            continue
        a = conv.action(c)
        if a is not None and 0 <= a < len(legal) and legal[a] and a not in rows:
            rows[a] = c
    cols = [o.move_choice for o in r.side_two]
    order = sorted(rows)
    obs: list[dict] = []
    count = {"branches": 0, "terminal": 0, "switch_expansions": 0, "errors": 0}

    def outcomes(s, ours: str, theirs: str) -> list[tuple[float, Any]]:
        branches = pe.generate_instructions(s, ours, theirs)
        if not branches:
            raise ValueError(f"no instructions for {ours!r} vs {theirs!r}")
        top = max(b.percentage for b in branches)
        kept = [b for b in branches if b.percentage >= min(min_prob * 100, top)]
        total = sum(b.percentage for b in kept)
        count["branches"] += len(kept)
        return [(b.percentage / total, s.apply_instructions(b)) for b in kept]

    def node(s, expand: bool = True) -> tuple:
        done = L.terminal(s)
        if done is not None:
            count["terminal"] += 1
            return ("T", done)
        if expand and L.needs_switch(s.side_two) and not L.needs_switch(s.side_one):
            count["switch_expansions"] += 1
            a2 = int(s.side_two.active_index)
            subs = [[(p, node(c, False)) for p, c in outcomes(s, "none", m.id)]
                    for i, m in enumerate(s.side_two.pokemon) if i != a2 and m.hp > 0]
            if subs:
                return ("S", subs)
        obs.append(enc.encode(s, turns=1))
        return ("L", len(obs) - 1)

    cells: list[list[list | None]] = []
    for a in order:
        row = []
        for c2 in cols:
            try:
                row.append([(p, node(s)) for p, s in outcomes(state, engine_choice(rows[a]), engine_choice(c2))])
            except BaseException as e:  # a Rust panic is a pyo3 PanicException (a BaseException)
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                count["errors"] += 1
                row.append(None)
        cells.append(row)
    stacked = {k: np.stack([o[k] for o in obs]) for k in obs[0]} if obs else None
    return {"rows": order, "cols": cols, "cells": cells, "obs": stacked, "n_leaves": len(obs),
            "seconds": time.perf_counter() - t0, **count}


def _setup_paths() -> None:
    """netv2's porygon.v2 on porygon.__path__ (from NETV2_SRC), as netsearch_h2h.py sets it up for the harness."""
    import porygon
    nv = Path(os.environ.get("NETV2_SRC", "/vol/bitbucket/fr1226/porygon-ws-netv2/src")) / "porygon"
    if str(nv) not in porygon.__path__:
        porygon.__path__.append(str(nv))


def worker_main() -> None:
    """Serve expand_world requests: 8-byte little-endian length + pickle, both ways."""
    out = sys.stdout.buffer
    sys.stdout = sys.stderr  # nothing but replies on the reply pipe
    _setup_paths()
    import torch
    torch.set_num_threads(1)
    inp = sys.stdin.buffer
    while True:
        hdr = inp.read(8)
        if len(hdr) < 8:
            break
        (n,) = struct.unpack("<Q", hdr)
        payload = pickle.loads(inp.read(n))
        try:
            reply = ("ok", expand_world(**payload))
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            reply = ("error", f"{type(e).__name__}: {e}")
        data = pickle.dumps(reply, protocol=5)
        out.write(struct.pack("<Q", len(data)))
        out.write(data)
        out.flush()


class LookaheadPool:
    """`n` expand_world worker subprocesses on the running event loop; `expand` awaits a free one. A request that
    takes longer than `timeout` seconds kills its worker (replaced) and raises."""

    def __init__(self, n: int, python: str | None = None, timeout: float = 300.0):
        self.n, self.python, self.timeout = n, python, timeout
        self.idle: asyncio.Queue | None = None

    async def _spawn(self) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            self.python or sys.executable, "-m", "porygon.lookahead2", "--worker",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=2**26)

    async def expand(self, payload: dict) -> dict:
        if self.idle is None:
            self.idle = asyncio.Queue()
            for _ in range(self.n):
                self.idle.put_nowait(await self._spawn())
        proc, body = await self.idle.get(), None
        try:
            data = pickle.dumps(payload, protocol=5)
            proc.stdin.write(struct.pack("<Q", len(data)) + data)
            await proc.stdin.drain()
            hdr = await asyncio.wait_for(proc.stdout.readexactly(8), self.timeout)
            body = await asyncio.wait_for(proc.stdout.readexactly(struct.unpack("<Q", hdr)[0]), self.timeout)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError) as e:
            raise RuntimeError(f"lookahead worker died or timed out: {type(e).__name__}") from e
        finally:
            if body is None:
                if proc.returncode is None:
                    proc.kill()
                proc = await self._spawn()
            self.idle.put_nowait(proc)
        status, result = pickle.loads(body)
        if status != "ok":
            raise RuntimeError(f"lookahead worker: {result}")
        return result

    async def close(self) -> None:
        while self.idle is not None and not self.idle.empty():
            proc = self.idle.get_nowait()
            if proc.returncode is None:
                proc.kill()
                await proc.wait()


# ---------------------------------------------------------------- leaf scoring and the decision

_GPU_LOCK = threading.Lock()  # one batched leaf forward at a time per process


def leaf_values(net: Any, stacked: list[dict], batch: int = 2048) -> tuple[np.ndarray, float]:
    """(P(win) (float64) of every leaf of the worlds' stacked observations, in order, batched; seconds of forward
    inside the lock, i.e. without waiting for other decisions' forwards)."""
    import torch
    from porygon.v2.model import batch_to_tensors
    keys = list(stacked[0])
    cat = {k: np.concatenate([s[k] for s in stacked]) for k in keys}
    n = len(cat[keys[0]])
    device = next(net.parameters()).device
    out = []
    with _GPU_LOCK, torch.inference_mode():
        t0 = time.perf_counter()
        for i in range(0, n, batch):
            x = batch_to_tensors({k: v[i:i + batch] for k, v in cat.items()}, device)
            out.append(torch.sigmoid(net(x, aux=False)["win_logit"].double()).cpu().numpy())
        t_fwd = time.perf_counter() - t0
    return (np.concatenate(out) if out else np.zeros(0)), t_fwd


@dataclass
class Result:
    rows: list  # our action indices (common to the worlds used)
    x: np.ndarray
    q: np.ndarray
    value: float
    expl: float
    worlds: int  # worlds used
    failed_worlds: int
    leaves: int
    terminal: int
    errors: int
    cols: float  # mean opponent options per world
    t_expand: float  # wall seconds of the world expansions (parallel)
    t_score: float  # leaf scoring, incl. waiting for other decisions' forwards
    t_fwd: float  # the forward itself
    t_solve: float
    cpu_expand: float  # sum of the worlds' expansion seconds (worker CPU)


def combine(worlds: list[dict], values: np.ndarray, iters: int = SOLVE_ITERS) -> tuple[list, Solution, int]:
    """Payoff matrices of the expanded worlds (leaf values in `values`, the worlds' leaves in order) on the rows
    common to them -> (rows, Bayesian-game solution, worlds dropped because their matrix failed)."""
    mats, offset, dropped = [], 0, 0
    for w in worlds:
        try:
            mats.append((w["rows"], world_matrix(w["cells"], values, offset)[0]))
        except ValueError:
            dropped += 1
        offset += w["n_leaves"]
    if not mats:
        raise ValueError("every world's matrix failed")
    common = [a for a in mats[0][0] if all(a in r for r, _ in mats[1:])]
    if not common:
        raise ValueError("no action is an engine option in every world")
    sub = [M[[r.index(a) for a in common]] for r, M in mats]
    return common, solve_bayes(sub, iters=iters), dropped


async def lookahead(pool: LookaheadPool, leaf_net: Any, battle: Any, first_seen: int | None, rng: Any, mask: np.ndarray,
                    *, worlds: int, min_prob: float = 0.0, iters: int = SOLVE_ITERS, batch: int = 2048) -> Result:
    """The lookahead's statistics for one decision (module docstring, steps 2-4)."""
    from porygon import search as S
    from porygon import v3leaf as L
    hist = S.History(battle)
    convs = [S.convert(battle, rng, first_seen, hist) for _ in range(worlds)]
    root = L.LeafRoot(battle, first_seen)
    legal = mask > 0
    t0 = time.perf_counter()
    res = await asyncio.gather(*(pool.expand({"root": root, "conv": c, "legal": legal, "min_prob": min_prob})
                                 for c in convs), return_exceptions=True)
    t1 = time.perf_counter()
    ok = []
    for r in res:
        if isinstance(r, BaseException):
            log.debug("world failed: %s", r)
            continue
        n_cells = sum(len(row) for row in r["cells"])
        if not r["rows"] or not r["cols"] or (n_cells and r["errors"] / n_cells > ERROR_SHARE_MAX):
            continue
        ok.append(r)
    if not ok:
        errs = [str(r) for r in res if isinstance(r, BaseException)][:1]
        raise RuntimeError(f"every world failed {errs}")
    stacked = [r["obs"] for r in ok if r["obs"] is not None]
    values, t_fwd = await asyncio.to_thread(leaf_values, leaf_net, stacked, batch) if stacked else (np.zeros(0), 0.0)
    t2 = time.perf_counter()
    rows, sol, dropped = combine(ok, values, iters)
    t3 = time.perf_counter()
    return Result(rows, sol.x, sol.q, sol.value, sol.expl, len(ok) - dropped, len(res) - len(ok) + dropped,
                  int(sum(r["n_leaves"] for r in ok)), int(sum(r["terminal"] for r in ok)),
                  int(sum(r["errors"] for r in ok)), float(np.mean([len(r["cols"]) for r in ok])),
                  t1 - t0, t2 - t1, t_fwd, t3 - t2, float(sum(r["seconds"] for r in ok)))


# ---------------------------------------------------------------- player

def _player_classes():
    """LookaheadPlayer / LoggedLookaheadPlayer, built on first use (porygon.search needs PORYGON_GEN at import)."""
    from poke_env.battle import AbstractBattle, Battle
    from poke_env.environment import SinglesEnv
    from poke_env.player import BattleOrder, DefaultBattleOrder

    from porygon import netsearch as NS
    from porygon import search as S

    class LookaheadPlayer(S.SearchPlayer):
        """The lookahead's margin rule as a poke-env player (module docstring). net: the v2 Net for the root (the raw
        player's distribution; keep it on the CPU so a_net is drawn exactly as the raw NetPlayer draws it);
        leaf_net: the same checkpoint on the leaf device (default: net). margin +inf = the raw net."""

        def __init__(self, net: Any, *args: Any, leaf_net: Any = None, margin: float = DEFAULT_MARGIN,
                     temperature: float = 0.5, deterministic: bool = False, worlds: int = 4, workers: int = 11,
                     min_prob: float = 0.0, iters: int = SOLVE_ITERS, batch: int = 2048, seed: int | None = None,
                     **kwargs: Any):
            super().__init__(None, *args, prior_weight=0.0, samples=worlds, workers=workers, eval_config="stock",
                             seed=seed, **kwargs)
            assert int(net.gen) == S.GEN_NUM == 9, f"lookahead2 is gen9 (net gen{net.gen}, PORYGON_GEN={S.GEN_NUM})"
            self.net = net.eval()
            self.leaf_net = (leaf_net or net).eval()
            self.margin, self.temperature, self.deterministic = float(margin), float(temperature), deterministic
            self.worlds, self.min_prob, self.iters, self.batch = int(worlds), float(min_prob), int(iters), int(batch)
            self.la_pool = LookaheadPool(workers)
            self.np_rng = np.random.default_rng(seed)
            self.stats.update(overrides=0, gaps_undefined=0, leaves=0, failed_worlds=0, la_seconds=0.0)

        def net_move(self, probs: np.ndarray) -> int:
            p = NS.sharpen(probs, self.temperature)
            if self.deterministic:
                return int(np.argmax(p))
            return int(self.np_rng.choice(len(p), p=p / p.sum()))

        async def choose_move(self, battle: AbstractBattle) -> BattleOrder:
            if battle.wait:
                return DefaultBattleOrder()
            assert isinstance(battle, Battle)
            probs, mask, win = NS.net_forward(self.net, battle)
            if not (mask > 0).any():
                return self.choose_random_move(battle)
            a_net = action = self.net_move(probs)
            self.stats["decisions"] += 1
            if mask.sum() > 1 and self.margin != math.inf:
                t0 = time.perf_counter()
                try:
                    res = await lookahead(self.la_pool, self.leaf_net, battle, self._weather_first_seen(battle),
                                          self.rng, mask, worlds=self.worlds, min_prob=self.min_prob,
                                          iters=self.iters, batch=self.batch)
                    g = gate(a_net, res.rows, res.x, res.q, self.margin)
                    action = self.last_action[battle.battle_tag] = g.action
                    self.stats["searched"] += 1
                    self.stats["agree"] += g.search == a_net
                    self.stats["overrides"] += g.override
                    self.stats["gaps_undefined"] += not math.isfinite(g.gap)
                    self.stats["leaves"] += res.leaves
                    self.stats["failed_worlds"] += res.failed_worlds
                    self.stats["la_seconds"] += time.perf_counter() - t0
                    self.on_gate(battle, probs, mask, win, res, g, time.perf_counter() - t0)
                except Exception as e:
                    self.stats["fallbacks"] += 1
                    log.warning("lookahead failed in %s turn %d, playing the network's move: %s", battle.battle_tag,
                                battle.turn, e)
            return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)

        def on_gate(self, battle: Battle, probs: np.ndarray, mask: np.ndarray, win: float, res: Result, g: Gate,
                    seconds: float) -> None:
            """Called after every searched decision (logging subclasses)."""

    class LoggedLookaheadPlayer(LookaheadPlayer):
        """Appends one JSON line per searched decision to `log_path` (the gate log of the slot harness)."""

        def __init__(self, net: Any, *args: Any, log_path: str, **kwargs: Any):
            super().__init__(net, *args, **kwargs)
            self.log_path = log_path

        def on_gate(self, battle, probs, mask, win, res, g, seconds) -> None:
            import json
            legal = np.flatnonzero(mask > 0)
            idx = {a: i for i, a in enumerate(res.rows)}
            rec = {"battle": battle.battle_tag, "turn": battle.turn, "legal": legal.tolist(),
                   "probs": [round(float(probs[a]), 5) for a in legal], "win": round(win, 5), "net": g.net,
                   "search": g.search, "gap": None if not math.isfinite(g.gap) else round(g.gap, 5),
                   "override": g.override, "action": g.action,
                   "q": [round(float(res.q[idx[a]]), 5) if a in idx else None for a in legal],
                   "share": [round(float(res.x[idx[a]]), 5) if a in idx else None for a in legal],
                   "root": round(res.value, 5), "expl": round(res.expl, 5), "worlds": res.worlds,
                   "failed_worlds": res.failed_worlds, "leaves": res.leaves, "terminal": res.terminal,
                   "errors": res.errors, "cols": round(res.cols, 2), "t": round(seconds, 3),
                   "t_expand": round(res.t_expand, 3), "t_score": round(res.t_score, 3),
                   "t_fwd": round(res.t_fwd, 3), "t_solve": round(res.t_solve, 3),
                   "cpu_expand": round(res.cpu_expand, 3)}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    return LookaheadPlayer, LoggedLookaheadPlayer


def __getattr__(name: str):  # porygon.lookahead2.LookaheadPlayer, built lazily
    if name in ("LookaheadPlayer", "LoggedLookaheadPlayer"):
        a, b = _player_classes()
        globals().update(LookaheadPlayer=a, LoggedLookaheadPlayer=b)
        return globals()[name]
    raise AttributeError(name)


SPEC_FLOATS = {"margin", "temperature", "min_prob"}
SPEC_INTS = {"worlds", "workers", "seed", "iters", "batch"}


def parse_spec(spec: str) -> tuple[str, dict[str, Any], dict[str, str]]:
    """`lookahead:NET[,margin=M][,temperature=T][,deterministic=1][,worlds=K][,workers=N][,min_prob=P][,iters=I]
    [,batch=B][,seed=S][,device=D][,leaf_device=D][,log=PATH]` -> (NET, player kwargs, {device, leaf_device, log})."""
    base, _, rest = spec.partition(",")
    kind, _, path = base.partition(":")
    assert kind == "lookahead" and path, f"not a lookahead spec: {spec}"
    opts = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    kw: dict[str, Any] = {}
    other = {k: opts.pop(k) for k in ("device", "leaf_device", "log") if k in opts}
    for k, v in opts.items():
        if k in SPEC_FLOATS:
            kw[k] = float(v)
        elif k in SPEC_INTS:
            kw[k] = int(v)
        elif k == "deterministic":
            kw[k] = v not in ("0", "false", "False")
        else:
            raise ValueError(f"unknown lookahead option {k!r} in {spec}")
    return path, kw, other


def load_spec(spec: str, net_loader: Any = None, **player_kwargs: Any):
    """A LookaheadPlayer from a `lookahead:` spec (parse_spec). The root net loads on `device` (default cpu), the leaf
    net (same file) on `leaf_device` (default: cuda if available, else the root's)."""
    path, kw, other = parse_spec(spec)
    if net_loader is None:
        from porygon.v2.model import load as net_loader
    device = other.get("device", "cpu")
    leaf_device = other.get("leaf_device")
    if leaf_device is None:
        import torch
        leaf_device = "cuda" if torch.cuda.is_available() else device
    net = net_loader(path, device=device)
    leaf = net if leaf_device == device else net_loader(path, device=leaf_device)
    a, b = _player_classes()
    if "log" in other:
        return b(net, leaf_net=leaf, log_path=other["log"], **kw, **player_kwargs)
    return a(net, leaf_net=leaf, **kw, **player_kwargs)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker_main()
    else:
        sys.exit("usage: python -m porygon.lookahead2 --worker")
