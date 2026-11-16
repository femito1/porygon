"""Best-first selective deepening on top of lookahead2 ("deepen"; design docs/netsearch/SCALE-SEARCH-DESIGN.md).

lookahead2 (porygon.lookahead2) expands every (our action, their option) pair of K sampled worlds for one turn, with
every chance branch, scores the leaves with the v2 net and solves the K matrices as one Bayesian zero-sum game. deepen
starts from exactly that tree (wave 0; budget 0 IS lookahead2: same worlds, cells, leaf values and solve) and then
spends an extra leaf budget, in waves, on the leaves that matter most to the decision:

  1. solve (inner nodes bottom-up as zero-sum matrix games, then the root's Bayesian game);
  2. reach of every leaf = product along its path of world weight 1/K, a row weight (root: (x + pi + [a = a_net]) / 3;
     inner nodes: (x + pi) / 2, pi = the net's policy at that state), a column weight ((y + q) / 2, q = the net's
     opponent-action head at that state; the root uses the root observation's head) and the chance probability; a
     forced opponent switch passes its reach to the switch-in worst for us;
  3. priority = reach * 4 v (1 - v) (v the leaf's P(win): settled leaves are not refined);
  4. expand the highest-priority leaves (until the wave's leaf estimate is met) for one more turn: our options kept by
     the net's policy (top until `us_mass` of probability, at most `us_cap`), theirs by the opponent-action head (top
     until `opp_mass`, at most `opp_cap`, at least `opp_min`); every pair simulated with generate_instructions,
     branches below `deep_min_prob` dropped and renormalised; new leaves scored by the net in one batched forward
     (their policy and opponent-action logits come from the same forward and are the priors of a later expansion);
  5. identical states (same world, same depth, same engine state string) share one leaf (a DAG; the forward is
     skipped for a repeated state).
The loop stops when the extra leaves reach `budget` (or nothing is left to expand); the final solve uses the
lookahead's 1000 regret-matching iterations at the root. The decision is lookahead2's margin gate unchanged
(porygon.lookahead2.gate): play argmax x instead of the net's T-0.5 draw iff Q gap > margin.

Pure parts (no engine, no network): keep_by_mass, our_prior, opp_prior, solve_games, Tree. Engine parts run in worker
processes (`python -m porygon.deepen --worker`, the LookaheadPool protocol): op "root" (lookahead2's expand_world
plus per-leaf metadata) and op "expand" (one more turn below given states). Needs PORYGON_GEN=9 and netv2's
porygon.v2 importable, as porygon.lookahead2.
"""

import asyncio
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from porygon import lookahead2 as LA

log = logging.getLogger("porygon.deepen")

SOLVE_ITERS = LA.SOLVE_ITERS


@dataclass(frozen=True)
class Config:
    budget: int = 0  # extra leaves beyond lookahead2's tree (0 = lookahead2)
    wave: int = 1024  # target new leaves per wave (one batched forward)
    us_mass: float = 0.9
    us_cap: int = 3
    opp_mass: float = 0.9
    opp_cap: int = 4
    opp_min: int = 2
    min_prob: float = 0.0  # root (lookahead2's)
    deep_min_prob: float = 0.03  # below the root
    max_turns: int = 4  # deepest leaf: turns since the root
    mid_iters: int = 200  # regret-matching iterations of the solves between waves (selection only)
    node_iters: int = 300  # inner nodes' final solve
    branch_guess: float = 2.5  # leaves per expanded cell (selection's estimate) before any is measured


# ---------------------------------------------------------------- priors and option selection (pure)

def keep_by_mass(p: np.ndarray | list, mass: float, cap: int, floor: int = 1) -> list[int]:
    """Indices of the most probable options (ties: lower index first) until their mass reaches `mass`, at most `cap`,
    at least `floor` (when there are that many), returned ascending."""
    order = sorted(range(len(p)), key=lambda i: (-float(p[i]), i))
    out, tot = [], 0.0
    for i in order:
        if len(out) >= cap or (tot >= mass - 1e-12 and len(out) >= floor):
            break
        out.append(i)
        tot += float(p[i])
    return sorted(out)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    e = np.exp(z - z.max())
    return e / e.sum()


def our_prior(logits: np.ndarray, actions: list[int]) -> np.ndarray:
    """The net's policy over our engine options at a state (actions: poke-env indices, -1 = none / struggle)."""
    if len(actions) <= 1 or any(a < 0 for a in actions):
        return np.full(len(actions), 1.0 / max(len(actions), 1))
    return _softmax(np.asarray(logits, float)[actions])


def opp_prior(logits: np.ndarray, classes: list[int]) -> np.ndarray:
    """The net's opponent-action head over the opponent's engine options (classes: spec.N_OPP_CLASSES, -1 = not
    placeable). Options of one class share its probability (switches to unseen mons); an unplaceable or masked option
    gets the mean of the placed ones; uniform when none is placed."""
    m = len(classes)
    z = np.asarray(logits, float)
    legal = z > -1e30
    if m <= 1 or not legal.any():
        return np.full(m, 1.0 / max(m, 1))
    p = np.zeros_like(z)
    p[legal] = _softmax(z[legal])
    count: dict[int, int] = {}
    for c in classes:
        if c >= 0 and legal[c]:
            count[c] = count.get(c, 0) + 1
    s = [p[c] / count[c] if c >= 0 and c in count else math.nan for c in classes]
    placed = [v for v in s if not math.isnan(v)]
    if not placed or sum(placed) <= 0:
        return np.full(m, 1.0 / m)
    fill = float(np.mean(placed))
    s = np.array([fill if math.isnan(v) else v for v in s])
    return s / s.sum()


# ---------------------------------------------------------------- batched zero-sum matrix games (pure)

def solve_games(mats: list[np.ndarray], iters: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Many small zero-sum matrix games max_x min_y x'My at once (padded, masked): the same alternating regret
    matching+ with linear averaging as lookahead2.solve_bayes (one world each). Returns (x per game, y per game,
    values x'My)."""
    n = len(mats)
    if n == 0:
        return [], [], np.zeros(0)
    R = max(a.shape[0] for a in mats)
    Cn = max(a.shape[1] for a in mats)
    A = np.zeros((n, R, Cn))
    rm = np.zeros((n, R), bool)
    cm = np.zeros((n, Cn), bool)
    for i, a in enumerate(mats):
        A[i, :a.shape[0], :a.shape[1]] = a
        rm[i, :a.shape[0]] = True
        cm[i, :a.shape[1]] = True
    rx, ry = np.zeros((n, R)), np.zeros((n, Cn))
    sx, sy = np.zeros((n, R)), np.zeros((n, Cn))
    for t in range(1, iters + 1):
        y = LA._rm(ry, cm)
        ux = np.einsum("nrc,nc->nr", A, y)
        x = LA._rm(rx, rm)
        rx = np.where(rm, np.maximum(rx + ux - (x * ux).sum(1, keepdims=True), 0.0), 0.0)
        x = LA._rm(rx, rm)
        uy = -np.einsum("nr,nrc->nc", x, A)
        ry = np.where(cm, np.maximum(ry + uy - (y * uy).sum(1, keepdims=True), 0.0), 0.0)
        sx += t * x
        sy += t * LA._rm(ry, cm)
    x = sx / sx.sum(1, keepdims=True)
    y = sy / sy.sum(1, keepdims=True)
    val = np.einsum("nr,nrc,nc->n", x, A, y)
    return ([x[i, :a.shape[0]] for i, a in enumerate(mats)], [y[i, :a.shape[1]] for i, a in enumerate(mats)], val)


# ---------------------------------------------------------------- the tree (pure; leaf values come from outside)

def cell_matrix(cells: list[list[list | None]], vals: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """(matrix, kept column indices) of a cell grid: lookahead2.world_matrix's rule (a failed cell takes its row's
    mean, a column failed everywhere is dropped, a row failed everywhere raises ValueError), keeping the mapping."""
    n, m = len(cells), len(cells[0]) if cells else 0
    M = np.full((n, m), np.nan)
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            if cell is not None:
                M[i, j] = sum(p * LA.ref_value(r, vals) for p, r in cell)
    keep = np.arange(m)
    if np.isnan(M).any():
        ok = ~np.isnan(M).all(0)
        M, keep = M[:, ok], keep[ok]
        if M.shape[1] == 0 or np.isnan(M).all(1).any():
            raise ValueError("a row or every column failed")
        M = np.where(np.isnan(M), np.nanmean(M, 1, keepdims=True), M)
    return M, keep


def remap(ref: tuple, ids: list[int]) -> tuple:
    """A worker's branch reference with local leaf indices -> the tree's item ids."""
    if ref[0] == "L":
        return ("L", ids[ref[1]])
    if ref[0] == "S":
        return ("S", [[(p, remap(r, ids)) for p, r in sub] for sub in ref[1]])
    return ref


@dataclass
class Node:
    item: int  # the item this node expands
    turns: int  # turns since the root of the node's state (its children are at turns + 1)
    rows: list[int]  # indices into the item's our-options
    cols: list[int]  # indices into the item's their-options
    cells: list  # [rows][cols] -> [(p, ref)] or None
    pi: np.ndarray  # the net's policy over the kept rows (renormalised)
    q: np.ndarray  # the opponent head over the kept cols (renormalised)
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    keep: np.ndarray | None = None  # columns kept by cell_matrix
    failed: bool = False


@dataclass
class World:
    rows: list[int]  # our poke-env action indices (ascending), as lookahead2
    cols: list[str]
    cells: list
    q0: np.ndarray  # the root's opponent head over cols


class Tree:
    """Items are leaves (the net's value) or expanded nodes (a matrix game over their children); refs are
    lookahead2's ("L", item) / ("T", value) / ("S", subs)."""

    def __init__(self, n_worlds: int):
        self.worlds: list[World] = []
        self.val: list[float] = []  # current value per item
        self.leaf_v: list[float] = []  # the net's value per item
        self.world: list[int] = []
        self.turns: list[int] = []
        self.state: list[str] = []
        self.ours: list[tuple] = []  # (choices, actions)
        self.theirs: list[tuple] = []  # (choices, classes)
        self.pi: list[np.ndarray] = []
        self.qo: list[np.ndarray] = []
        self.node_of: list[int] = []  # node index, -1 = leaf, -2 = expansion failed (stays a leaf)
        self.key: dict[tuple, int] = {}
        self.nodes: list[Node] = []
        self.n_worlds = n_worlds

    def add_leaf(self, world: int, turns: int, meta: tuple, v: float, pol: np.ndarray, opp: np.ndarray) -> int:
        state, oc, oa, tc, tcl = meta
        g = len(self.val)
        self.val.append(float(v))
        self.leaf_v.append(float(v))
        self.world.append(world)
        self.turns.append(turns)
        self.state.append(state)
        self.ours.append((oc, oa))
        self.theirs.append((tc, tcl))
        self.pi.append(our_prior(pol, oa))
        self.qo.append(opp_prior(opp, tcl))
        self.node_of.append(-1)
        self.key.setdefault((world, turns, state), g)
        return g

    def lookup(self, world: int, turns: int, state: str) -> int | None:
        return self.key.get((world, turns, state))

    # -- values
    def evaluate(self, iters: int) -> None:
        """Inner nodes bottom-up (deepest turns first), each level's games solved in one batch."""
        by: dict[int, list[Node]] = {}
        for nd in self.nodes:
            if not nd.failed:
                by.setdefault(nd.turns, []).append(nd)
        for d in sorted(by, reverse=True):
            todo, mats = [], []
            for nd in by[d]:
                try:
                    M, keep = cell_matrix(nd.cells, self.val)
                except ValueError:
                    nd.failed = True
                    self.node_of[nd.item] = -2
                    self.val[nd.item] = self.leaf_v[nd.item]
                    continue
                nd.keep = keep
                todo.append(nd)
                mats.append(M)
            xs, ys, vs = solve_games(mats, iters)
            for nd, x, y, v in zip(todo, xs, ys, vs):
                nd.x, nd.y = x, y
                self.val[nd.item] = float(v)

    def root_solve(self, iters: int):
        """lookahead2.combine on the current values -> (common rows, Solution, used world indices, per used world
        (row positions, kept columns))."""
        mats, used, info = [], [], []
        for k, w in enumerate(self.worlds):
            try:
                M, keep = cell_matrix(w.cells, self.val)
            except ValueError:
                continue
            mats.append((w.rows, M))
            used.append(k)
            info.append(keep)
        if not mats:
            raise ValueError("every world's matrix failed")
        common = [a for a in mats[0][0] if all(a in r for r, _ in mats[1:])]
        if not common:
            raise ValueError("no action is an engine option in every world")
        pos = [[r.index(a) for a in common] for r, _ in mats]
        sub = [M[p] for p, (_, M) in zip(pos, mats)]
        sol = LA.solve_bayes(sub, iters=iters)
        return common, sol, used, list(zip(pos, info))

    # -- reach and selection
    def _push(self, ref: tuple, m: float, r: np.ndarray) -> None:
        kind = ref[0]
        if kind == "L":
            r[ref[1]] += m
        elif kind == "S":
            vals = [sum(p * LA.ref_value(x, self.val) for p, x in sub) for sub in ref[1]]
            for p, x in ref[1][int(np.argmin(vals))]:
                self._push(x, m * p, r)

    def reach(self, common: list[int], sol, used: list[int], info: list, root_pi: np.ndarray,
              a_net: int | None) -> np.ndarray:
        r = np.zeros(len(self.val))
        rho = np.asarray(sol.x, float) + root_pi
        if a_net is not None and a_net in common:
            rho = rho.copy()
            rho[common.index(a_net)] += 1.0
            rho /= 3.0
        else:
            rho /= 2.0
        wk = 1.0 / len(used)
        for u, k in enumerate(used):
            w = self.worlds[k]
            pos, keep = info[u]
            q0 = w.q0[keep]
            q0 = q0 / q0.sum() if q0.sum() > 0 else np.full(len(keep), 1.0 / len(keep))
            cw = 0.5 * (np.asarray(sol.ys[u], float) + q0)
            for i, ri in enumerate(pos):
                for jj, j in enumerate(keep):
                    cell = w.cells[ri][j]
                    base = wk * rho[i] * cw[jj]
                    if cell is None or base <= 0:
                        continue
                    for p, ref in cell:
                        self._push(ref, base * p, r)
        for nd in sorted((n for n in self.nodes if not n.failed), key=lambda n: n.turns):
            m = r[nd.item]
            if m <= 0 or nd.x is None:
                continue
            rw = 0.5 * (nd.x + nd.pi)
            qk = nd.q[nd.keep]
            qk = qk / qk.sum() if qk.sum() > 0 else np.full(len(nd.keep), 1.0 / len(nd.keep))
            cw = 0.5 * (nd.y + qk)
            for i in range(len(nd.rows)):
                for jj, j in enumerate(nd.keep):
                    cell = nd.cells[i][j]
                    if cell is None:
                        continue
                    for p, ref in cell:
                        self._push(ref, m * rw[i] * cw[jj] * p, r)
        return r

    def kept(self, g: int, cfg: Config) -> tuple[list[int], list[int]]:
        rows = keep_by_mass(self.pi[g], cfg.us_mass, cfg.us_cap, 1)
        cols = keep_by_mass(self.qo[g], cfg.opp_mass, cfg.opp_cap, cfg.opp_min)
        return rows, cols

    def select(self, r: np.ndarray, target: float, cfg: Config, branches: float) -> list[tuple[int, list, list]]:
        """Leaves to expand this wave: highest reach * 4v(1-v) first (ties: lower id), until the estimated new
        leaves (kept rows x kept cols x branches per cell) reach `target`."""
        v = np.asarray(self.val)
        pr = r * 4.0 * v * (1.0 - v)
        cand = [g for g in np.flatnonzero(pr > 1e-12)
                if self.node_of[g] == -1 and self.turns[g] < cfg.max_turns and self.ours[g][0] and self.theirs[g][0]]
        cand.sort(key=lambda g: (-pr[g], g))
        out, est = [], 0.0
        for g in cand:
            if est >= target:
                break
            rows, cols = self.kept(int(g), cfg)
            out.append((int(g), rows, cols))
            est += len(rows) * len(cols) * branches
        return out

    def add_node(self, g: int, rows: list[int], cols: list[int], cells: list) -> None:
        pi = self.pi[g][rows]
        q = self.qo[g][cols]
        nd = Node(g, self.turns[g], rows, cols, cells, pi / pi.sum() if pi.sum() > 0 else np.full(len(rows), 1 / len(rows)),
                  q / q.sum() if q.sum() > 0 else np.full(len(cols), 1 / len(cols)))
        self.node_of[g] = len(self.nodes)
        self.nodes.append(nd)

    def depth(self) -> int:
        return max((self.turns[nd.item] + 1 for nd in self.nodes if not nd.failed), default=1)


# ---------------------------------------------------------------- engine side (worker processes)

def opp_classes(enc: Any, state: Any, obs: dict, choices: list[str]) -> list[int]:
    """Opponent-action classes (porygon.v2.encoder's: 0-5 switch to observation slot, 6 an unseen mon, 7 + move slot,
    7 + KM + move slot with tera) of the opponent's engine options at a state, in the leaf observation's frame
    (v3leaf.LeafEncoder's slot assignment); -1 = not placeable (a sampled move outside the candidate slots, none)."""
    from porygon.v2 import encoder as E
    from porygon.v2.spec import KM
    root = enc.root
    opp = state.side_two.pokemon
    a2 = int(state.side_two.active_index)
    slots: list[int | None] = list(range(6, 6 + root.n_opp))
    free = 6 + root.n_opp
    for k in range(root.n_opp, 6):
        m = opp[k]
        if k == a2 or m.hp < m.maxhp or m.status.lower() != "none":
            slots.append(free)
            free += 1
        else:
            slots.append(None)
    t_a = slots[a2] if a2 < len(slots) else None
    row = obs["moves"][t_a] if t_a is not None else None
    out = []
    for c in choices:
        if c.startswith("switch "):
            name = c[len("switch "):]
            i = next((i for i, m in enumerate(opp) if m.id == name and i != a2), None)
            t = slots[i] if i is not None and i < len(slots) else None
            out.append(6 if t is None else t - 6)
        elif c in ("none", "No Move", "struggle") or row is None:
            out.append(-1)
        else:
            tera = c.endswith("-tera")
            mid = c[:-len("-tera")] if tera else c
            hits = np.flatnonzero(row == E.vid("moves", mid))
            out.append(7 + int(hits[0]) + (KM if tera else 0) if len(hits) else -1)
    return out


def leaf_meta(enc: Any, state: Any, obs: dict) -> tuple:
    """(state string, our choices, our action indices, their choices, their classes) of a leaf state, in
    v3leaf.side_options' format."""
    from porygon import v3leaf as L
    from porygon.v2.spec import FIELD_NUM
    root = enc.root
    tera1 = root.gen >= 9 and not root.used_tera
    tera2 = root.gen >= 9 and root.obs["field_num"][FIELD_NUM.index("opp_used_tera")] == 0
    oc, oa = L.side_options(state.side_one, state.side_two, tera1)
    tc, _ = L.side_options(state.side_two, state.side_one, tera2)
    return state.to_string(), oc, oa, tc, opp_classes(enc, state, obs, tc)


def expand_cells(enc: Any, state: Any, ours: list[str], theirs: list[str], turns: int, min_prob: float,
                 count: dict, obs: list, metas: list) -> list:
    """cells[i][j] for our engine choices `ours` x their `theirs` from `state`, simulated one turn (lookahead2's
    expand_world rule: chance branches below min_prob dropped and renormalised, a finished battle is a "T" ref, an
    opponent-only forced switch takes their switch-in worst for us, any other state a leaf encoded with `turns`).
    New leaves are appended to obs / metas; a failed pair is None."""
    import poke_engine as pe
    from porygon import v3leaf as L

    def outcomes(s, a: str, b: str):
        branches = pe.generate_instructions(s, a, b)
        if not branches:
            raise ValueError(f"no instructions for {a!r} vs {b!r}")
        top = max(br.percentage for br in branches)
        kept = [br for br in branches if br.percentage >= min(min_prob * 100, top)]
        total = sum(br.percentage for br in kept)
        count["branches"] += len(kept)
        return [(br.percentage / total, s.apply_instructions(br)) for br in kept]

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
        o = enc.encode(s, turns=turns)
        obs.append(o)
        metas.append(leaf_meta(enc, s, o))
        return ("L", len(obs) - 1)

    cells = []
    for a in ours:
        row = []
        for b in theirs:
            try:
                row.append([(p, node(s)) for p, s in outcomes(state, a, b)])
            except BaseException as e:  # a Rust panic is a pyo3 PanicException (a BaseException)
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                count["errors"] += 1
                row.append(None)
        cells.append(row)
    return cells


def _stack(obs: list) -> dict | None:
    return {k: np.stack([o[k] for o in obs]) for k in obs[0]} if obs else None


def op_root(root: Any, conv: Any, legal: np.ndarray, min_prob: float = 0.0) -> dict:
    """lookahead2.expand_world (same rows, cols, cells, leaf observations) plus leaf metadata and the root columns'
    opponent-action classes."""
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
    metas: list[tuple] = []
    count = {"branches": 0, "terminal": 0, "switch_expansions": 0, "errors": 0}
    cells = expand_cells(enc, state, [LA.engine_choice(rows[a]) for a in order], [LA.engine_choice(c) for c in cols],
                         1, min_prob, count, obs, metas)
    col_classes = opp_classes(enc, state, root.obs, ["none" if c == "No Move" else c for c in cols])
    return {"rows": order, "cols": cols, "cells": cells, "obs": _stack(obs), "n_leaves": len(obs), "metas": metas,
            "col_classes": col_classes, "seconds": time.perf_counter() - t0, **count}


def op_expand(root: Any, conv: Any, items: list[tuple], min_prob: float) -> dict:
    """One more turn below each (item id, state string, turns, our choices, their choices): cells per item (None =
    the item failed), the new leaves' observations and metadata (refs index the request's leaves)."""
    import poke_engine as pe
    from porygon import v3leaf as L
    t0 = time.perf_counter()
    enc = L.LeafEncoder(root, conv)
    obs: list[dict] = []
    metas: list[tuple] = []
    count = {"branches": 0, "terminal": 0, "switch_expansions": 0, "errors": 0, "cells": 0}
    out, leaf_turns = [], []
    for g, s, turns, ours, theirs in items:
        n0 = len(obs)
        try:
            st = pe.State.from_string(s)
            cells = expand_cells(enc, st, [L.engine_choice(c) for c in ours], [L.engine_choice(c) for c in theirs],
                                 turns + 1, min_prob, count, obs, metas)
            count["cells"] += len(ours) * len(theirs)
            leaf_turns += [turns + 1] * (len(obs) - n0)
            out.append((g, cells))
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            del obs[n0:], metas[n0:]
            count["errors"] += 1
            out.append((g, None))
    return {"items": out, "obs": _stack(obs), "metas": metas, "leaf_turns": leaf_turns, "n_leaves": len(obs),
            "seconds": time.perf_counter() - t0, **count}


def worker_main() -> None:
    """lookahead2's worker loop, dispatching on payload["op"]."""
    import pickle
    import struct
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    LA._setup_paths()
    import torch
    torch.set_num_threads(1)
    inp = sys.stdin.buffer
    ops = {"root": op_root, "expand": op_expand}
    while True:
        hdr = inp.read(8)
        if len(hdr) < 8:
            break
        (n,) = struct.unpack("<Q", hdr)
        payload = pickle.loads(inp.read(n))
        try:
            reply = ("ok", ops[payload.pop("op")](**payload))
        except BaseException as e:
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            reply = ("error", f"{type(e).__name__}: {e}")
        data = pickle.dumps(reply, protocol=5)
        out.write(struct.pack("<Q", len(data)))
        out.write(data)
        out.flush()


class DeepenPool(LA.LookaheadPool):
    """LookaheadPool whose workers run porygon.deepen."""

    async def _spawn(self):
        return await asyncio.create_subprocess_exec(
            self.python or sys.executable, "-m", "porygon.deepen", "--worker",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=2**28)


# ---------------------------------------------------------------- leaf forward and the decision

def leaf_forward(net: Any, stacked: list[dict], batch: int = 2048, half: bool = False):
    """(P(win) float64, policy logits, opponent-action logits) of the stacked leaves, in order; seconds of forward.
    P(win) is lookahead2.leaf_values' (sigmoid of the win logit in float64); `half`: fp16 autocast (I1 parity)."""
    import contextlib

    import torch
    from porygon.v2.model import batch_to_tensors
    keys = list(stacked[0])
    cat = {k: np.concatenate([s[k] for s in stacked]) for k in keys}
    n = len(cat[keys[0]])
    device = next(net.parameters()).device
    vs, ps, qs = [], [], []
    amp = (torch.autocast(device.type, dtype=torch.float16) if half and device.type == "cuda"
           else contextlib.nullcontext())
    with LA._GPU_LOCK, torch.inference_mode(), amp:
        t0 = time.perf_counter()
        for i in range(0, n, batch):
            x = batch_to_tensors({k: v[i:i + batch] for k, v in cat.items()}, device)
            out = net(x, aux=True)
            vs.append(torch.sigmoid(out["win_logit"].double()).cpu().numpy())
            ps.append(out["logits"].float().cpu().numpy())
            qs.append(out["aux"]["opp_action"].float().cpu().numpy())
        t = time.perf_counter() - t0
    return np.concatenate(vs), np.concatenate(ps), np.concatenate(qs), t


@dataclass
class Result:  # lookahead2.Result's fields (the gate log reads them) plus the deepening's
    rows: list
    x: np.ndarray
    q: np.ndarray
    value: float
    expl: float
    worlds: int
    failed_worlds: int
    leaves: int  # forwarded leaves, all waves
    terminal: int
    errors: int
    cols: float
    t_expand: float
    t_score: float
    t_fwd: float
    t_solve: float
    cpu_expand: float
    base_leaves: int = 0  # wave 0 (= lookahead2's)
    extra_leaves: int = 0
    expansions: int = 0
    failed_expansions: int = 0
    waves: int = 0
    depth: int = 1  # deepest leaf, turns since the root
    dedup: int = 0  # new leaves that repeated a state (no forward)
    cells: int = 0  # expanded cells below the root
    branches: int = 0  # their chance branches
    extra: dict = field(default_factory=dict)


def _chunks(xs: list, n: int) -> list[list]:
    if not xs:
        return []
    size = max(1, math.ceil(len(xs) / max(n, 1)))
    return [xs[i:i + size] for i in range(0, len(xs), size)]


async def deepen(pool: LA.LookaheadPool, leaf_net: Any, battle: Any, first_seen: int | None, rng: Any,
                 mask: np.ndarray, probs: np.ndarray, a_net: int | None, *, worlds: int, cfg: Config,
                 iters: int = SOLVE_ITERS, batch: int = 2048, half: bool = False) -> Result:
    """The deepened lookahead's statistics for one decision (module docstring). probs: the net's T-1 policy at the
    root (row weights); a_net: the move the gate compares against (row weight; None = omit)."""
    from porygon import search as S
    from porygon import v3leaf as L
    hist = S.History(battle)
    convs = [S.convert(battle, rng, first_seen, hist) for _ in range(worlds)]  # lookahead2's draw order
    root = L.LeafRoot(battle, first_seen)
    legal = mask > 0
    t0 = time.perf_counter()
    res = await asyncio.gather(*(pool.expand({"op": "root", "root": root, "conv": c, "legal": legal,
                                              "min_prob": cfg.min_prob}) for c in convs), return_exceptions=True)
    t_expand = time.perf_counter() - t0
    ok, ok_conv = [], []
    for r, c in zip(res, convs):  # lookahead2.lookahead's filter
        if isinstance(r, BaseException):
            log.debug("world failed: %s", r)
            continue
        n_cells = sum(len(row) for row in r["cells"])
        if not r["rows"] or not r["cols"] or (n_cells and r["errors"] / n_cells > LA.ERROR_SHARE_MAX):
            continue
        ok.append(r)
        ok_conv.append(c)
    if not ok:
        errs = [str(r) for r in res if isinstance(r, BaseException)][:1]
        raise RuntimeError(f"every world failed {errs}")
    t1 = time.perf_counter()
    stacked = [r["obs"] for r in ok if r["obs"] is not None]
    t_fwd = 0.0
    if stacked:
        v, pol, opp, t = await asyncio.to_thread(leaf_forward, leaf_net, stacked, batch, half)
        t_fwd += t
    else:
        v, pol, opp = np.zeros(0), np.zeros((0, 1)), np.zeros((0, 1))
    _, _, q_root, t = await asyncio.to_thread(leaf_forward, leaf_net, [{k: x[None] for k, x in root.obs.items()}],
                                               batch, half)
    t_fwd += t
    t_score = time.perf_counter() - t1
    tree = Tree(len(ok))
    off = 0
    for k, r in enumerate(ok):
        ids = [tree.add_leaf(k, 1, r["metas"][i], v[off + i], pol[off + i], opp[off + i]) for i in range(r["n_leaves"])]
        off += r["n_leaves"]
        tree.worlds.append(World(r["rows"], r["cols"], [[None if c is None else [(p, remap(x, ids)) for p, x in c]
                                                          for c in row] for row in r["cells"]],
                                 opp_prior(q_root[0], r["col_classes"])))
    base = len(tree.val)
    stats = {"terminal": int(sum(r["terminal"] for r in ok)), "errors": int(sum(r["errors"] for r in ok)),
             "cpu": float(sum(r["seconds"] for r in ok)), "expansions": 0, "failed": 0, "waves": 0, "dedup": 0,
             "cells": 0, "branches": 0, "cell_leaves": 0}
    t_solve = 0.0
    extra = 0
    branches = cfg.branch_guess
    while extra < cfg.budget:
        ts = time.perf_counter()
        tree.evaluate(cfg.mid_iters)
        common, sol, used, info = tree.root_solve(cfg.mid_iters)
        pi = probs[common] / probs[common].sum() if probs[common].sum() > 0 else np.full(len(common), 1 / len(common))
        r = tree.reach(common, sol, used, info, pi, a_net)
        picks = tree.select(r, min(cfg.wave, cfg.budget - extra), cfg, branches)
        t_solve += time.perf_counter() - ts
        if not picks:
            break
        by_world: dict[int, list] = {}
        for g, rows, cols in picks:
            by_world.setdefault(tree.world[g], []).append(
                (g, tree.state[g], tree.turns[g], [tree.ours[g][0][i] for i in rows], [tree.theirs[g][0][j] for j in cols]))
        reqs = []
        n_req = max(1, pool.n)
        total = sum(len(x) for x in by_world.values())
        for k, items in by_world.items():
            share = max(1, round(n_req * len(items) / total))
            for ch in _chunks(items, share):
                reqs.append((k, ch))
        te = time.perf_counter()
        out = await asyncio.gather(*(pool.expand({"op": "expand", "root": root, "conv": ok_conv[k], "items": ch,
                                                  "min_prob": cfg.deep_min_prob}) for k, ch in reqs),
                                   return_exceptions=True)
        t_expand += time.perf_counter() - te
        kept = {g: (rows, cols) for g, rows, cols in picks}
        new_rows: list[tuple] = []  # (world, turns, meta) of each leaf to forward, in order
        stack: list[dict] = []
        wave_key: dict[tuple, int] = {}
        replies = []
        for (k, ch), o in zip(reqs, out):
            if isinstance(o, BaseException):
                log.debug("expansion request failed: %s", o)
                for g, *_ in ch:
                    tree.node_of[g] = -2
                    stats["failed"] += 1
                continue
            for key_ in ("terminal", "errors", "cells", "branches"):
                stats[key_] += o[key_]
            stats["cell_leaves"] += len(o["metas"])
            stats["cpu"] += o["seconds"]
            ids, take = [], []
            for li, meta in enumerate(o["metas"]):
                key = (k, o["leaf_turns"][li], meta[0])
                g = tree.lookup(*key)
                if g is not None:
                    ids.append(("old", g))
                elif key in wave_key:
                    ids.append(("new", wave_key[key]))
                else:
                    wave_key[key] = len(new_rows)
                    ids.append(("new", len(new_rows)))
                    new_rows.append((k, o["leaf_turns"][li], meta))
                    take.append(li)
                    continue
                stats["dedup"] += 1
            if take:
                stack.append({kk: arr[take] for kk, arr in o["obs"].items()})
            replies.append((o, ids))
        tf = time.perf_counter()
        new_ids: list[int] = []
        if new_rows:
            vv, pp, qq, t = await asyncio.to_thread(leaf_forward, leaf_net, stack, batch, half)
            t_fwd += t
            new_ids = [tree.add_leaf(k, turns, meta, vv[i], pp[i], qq[i]) for i, (k, turns, meta) in enumerate(new_rows)]
        t_score += time.perf_counter() - tf
        extra += len(new_rows)
        for o, ids in replies:
            gmap = [j if kind == "old" else new_ids[j] for kind, j in ids]
            for g, cells in o["items"]:
                if cells is None:
                    tree.node_of[g] = -2
                    stats["failed"] += 1
                    continue
                rows, cols = kept[g]
                tree.add_node(g, rows, cols, [[None if c is None else [(p, remap(x, gmap)) for p, x in c] for c in row]
                                              for row in cells])
                stats["expansions"] += 1
        stats["waves"] += 1
        if stats["cells"]:
            branches = max(stats["cell_leaves"] / stats["cells"], 0.5)  # leaves per expanded cell so far
    ts = time.perf_counter()
    tree.evaluate(cfg.node_iters)
    common, sol, used, _ = tree.root_solve(iters)
    t_solve += time.perf_counter() - ts
    return Result(common, sol.x, sol.q, sol.value, sol.expl, len(used), worlds - len(used), len(tree.val),
                  stats["terminal"], stats["errors"], float(np.mean([len(r["cols"]) for r in ok])), t_expand, t_score,
                  t_fwd, t_solve, stats["cpu"], base, extra, stats["expansions"], stats["failed"], stats["waves"],
                  tree.depth(), stats["dedup"], stats["cells"], stats["branches"])



# ---------------------------------------------------------------- player

def _player_classes():
    """DeepenPlayer / LoggedDeepenPlayer on top of lookahead2's LookaheadPlayer (built on first use)."""
    import json

    from poke_env.battle import AbstractBattle, Battle
    from poke_env.environment import SinglesEnv
    from poke_env.player import BattleOrder, DefaultBattleOrder

    from porygon import netsearch as NS

    class DeepenPlayer(LA.LookaheadPlayer):
        """lookahead2's player with the deepened lookahead (budget 0 = LookaheadPlayer's decisions). `half`: fp16
        leaf forwards."""

        def __init__(self, net: Any, *args: Any, cfg: Config = Config(), half: bool = False, **kwargs: Any):
            super().__init__(net, *args, **kwargs)
            self.cfg, self.half = cfg, bool(half)
            self.la_pool = DeepenPool(self.la_pool.n)
            self.stats.update(extra_leaves=0, expansions=0, max_depth=0)

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
                    res = await deepen(self.la_pool, self.leaf_net, battle, self._weather_first_seen(battle), self.rng,
                                       mask, probs, a_net, worlds=self.worlds, cfg=self.cfg, iters=self.iters,
                                       batch=self.batch, half=self.half)
                    g = LA.gate(a_net, res.rows, res.x, res.q, self.margin)
                    action = self.last_action[battle.battle_tag] = g.action
                    self.stats["searched"] += 1
                    self.stats["agree"] += g.search == a_net
                    self.stats["overrides"] += g.override
                    self.stats["gaps_undefined"] += not math.isfinite(g.gap)
                    self.stats["leaves"] += res.leaves
                    self.stats["extra_leaves"] += res.extra_leaves
                    self.stats["expansions"] += res.expansions
                    self.stats["max_depth"] = max(self.stats["max_depth"], res.depth)
                    self.stats["failed_worlds"] += res.failed_worlds
                    self.stats["la_seconds"] += time.perf_counter() - t0
                    self.on_gate(battle, probs, mask, win, res, g, time.perf_counter() - t0)
                except Exception as e:
                    self.stats["fallbacks"] += 1
                    log.warning("deepen failed in %s turn %d, playing the network's move: %s", battle.battle_tag,
                                battle.turn, e)
            return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)

    class LoggedDeepenPlayer(DeepenPlayer):
        """One JSON line per searched decision (lookahead2's gate-log fields plus the deepening's)."""

        def __init__(self, net: Any, *args: Any, log_path: str, **kwargs: Any):
            super().__init__(net, *args, **kwargs)
            self.log_path = log_path

        def on_gate(self, battle, probs, mask, win, res, g, seconds) -> None:
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
                   "cpu_expand": round(res.cpu_expand, 3), "base_leaves": res.base_leaves,
                   "extra_leaves": res.extra_leaves, "expansions": res.expansions,
                   "failed_expansions": res.failed_expansions, "waves": res.waves, "depth": res.depth,
                   "dedup": res.dedup, "budget": self.cfg.budget}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    return DeepenPlayer, LoggedDeepenPlayer


def __getattr__(name: str):
    if name in ("DeepenPlayer", "LoggedDeepenPlayer"):
        a, b = _player_classes()
        globals().update(DeepenPlayer=a, LoggedDeepenPlayer=b)
        return globals()[name]
    raise AttributeError(name)


CFG_INTS = {"budget", "wave", "us_cap", "opp_cap", "opp_min", "max_turns", "mid_iters", "node_iters"}
CFG_FLOATS = {"us_mass", "opp_mass", "deep_min_prob", "branch_guess"}


def parse_spec(spec: str) -> tuple[str, dict[str, Any], dict[str, str], Config, bool]:
    """`deepen:NET[,budget=B][,wave=W][,us_mass=..][,us_cap=..][,opp_mass=..][,opp_cap=..][,opp_min=..]
    [,deep_min_prob=..][,max_turns=..][,mid_iters=..][,node_iters=..][,half=1]` plus every lookahead2 option
    (margin, temperature, worlds, workers, min_prob, iters, batch, seed, device, leaf_device, log) ->
    (NET, player kwargs, {device, leaf_device, log}, Config, half). min_prob is the root's (lookahead2's)."""
    base, _, rest = spec.partition(",")
    kind, _, path = base.partition(":")
    assert kind == "deepen" and path, f"not a deepen spec: {spec}"
    la_opts, cfg_kw, half = [], {}, False
    for kv in (x for x in rest.split(",") if x):
        k, _, v = kv.partition("=")
        if k in CFG_INTS:
            cfg_kw[k] = int(v)
        elif k in CFG_FLOATS:
            cfg_kw[k] = float(v)
        elif k == "half":
            half = v not in ("0", "false", "False")
        else:
            la_opts.append(kv)
    p, kw, other = LA.parse_spec(",".join([f"lookahead:{path}"] + la_opts))
    if "min_prob" in kw:
        cfg_kw["min_prob"] = kw["min_prob"]
    return p, kw, other, Config(**cfg_kw), half


def load_spec(spec: str, net_loader: Any = None, **player_kwargs: Any):
    """A DeepenPlayer from a `deepen:` spec (lookahead2.load_spec's devices)."""
    path, kw, other, cfg, half = parse_spec(spec)
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
        return b(net, leaf_net=leaf, log_path=other["log"], cfg=cfg, half=half, **kw, **player_kwargs)
    return a(net, leaf_net=leaf, cfg=cfg, half=half, **kw, **player_kwargs)


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker_main()
    else:
        sys.exit("usage: python -m porygon.deepen --worker")
