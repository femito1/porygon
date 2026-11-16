"""porygon.lookahead2's pure parts: the Bayesian matrix-game solver, leaf scoring of a world's branch tree, the
combination of worlds and the margin gate (no engine, no network, no server)."""

import math

import numpy as np
import pytest

from porygon import lookahead2 as la


def test_solve_single_world_mixed():
    pennies = np.array([[1.0, 0.0], [0.0, 1.0]])  # our win probability; they minimise
    s = la.solve_bayes([pennies])
    assert np.allclose(s.x, 0.5, atol=0.02) and np.allclose(s.ys[0], 0.5, atol=0.02)
    assert np.allclose(s.q, 0.5, atol=0.02) and abs(s.value - 0.5) < 0.02 and s.expl < 0.02
    rps = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
    s = la.solve_bayes([rps])
    assert np.allclose(s.x, 1 / 3, atol=0.02) and np.allclose(s.ys[0], 1 / 3, atol=0.02) and s.expl < 0.02


def test_solve_dominance_and_q_outside_support():
    m = np.array([[0.8, 0.6], [0.3, 0.2]])  # row 0 dominates; column 1 is then best for them
    s = la.solve_bayes([m])
    assert s.x[0] > 0.98 and s.ys[0][1] > 0.98
    assert abs(s.q[0] - 0.6) < 0.01 and abs(s.q[1] - 0.2) < 0.01  # Q of the dominated row: vs their equilibrium


def test_solve_bayesian_one_strategy_for_all_worlds():
    # world 1: A dominates; world 2: B dominates (but by less): one x for both worlds -> A
    m1 = np.array([[0.9, 0.9], [0.1, 0.1]])
    m2 = np.array([[0.2, 0.2, 0.2], [0.6, 0.6, 0.6]])  # padded to 3 columns
    s = la.solve_bayes([m1, m2])
    assert s.x[0] > 0.98
    assert abs(s.q[0] - 0.55) < 1e-6 and abs(s.q[1] - 0.35) < 1e-6
    assert len(s.ys[0]) == 2 and len(s.ys[1]) == 3
    # weights move the answer
    s = la.solve_bayes([m1, m2], weights=[1, 3])
    assert s.x[1] > 0.98


def test_solve_padding_never_picked():
    # world 2 has only one real column, worth 0.9 to us: padded columns (scored 0 before masking) must stay unused
    m1 = np.array([[0.5, 0.4], [0.5, 0.6]])
    m2 = np.array([[0.9], [0.9]])
    s = la.solve_bayes([m1, m2])
    assert len(s.ys[1]) == 1 and abs(s.ys[1][0] - 1) < 1e-9
    assert all(q > 0.6 for q in s.q)


def test_ref_value_and_world_matrix():
    v = np.array([0.2, 0.7, 0.4, 0.9])
    assert la.ref_value(("L", 1), v) == pytest.approx(0.7)
    assert la.ref_value(("L", 0), v, offset=3) == pytest.approx(0.9)
    assert la.ref_value(("T", 1.0), v) == 1.0
    # the opponent's forced switch: the switch-in worst for us
    sw = ("S", [[(1.0, ("L", 1))], [(0.5, ("L", 2)), (0.5, ("T", 0.0))]])
    assert la.ref_value(sw, v) == pytest.approx(0.2)
    cells = [[[(0.25, ("L", 0)), (0.75, ("T", 1.0))], [(1.0, sw)]],
             [[(1.0, ("L", 3))], None]]  # a failed cell takes its row's mean
    M, bad = la.world_matrix(cells, v)
    assert bad == 1 and M.shape == (2, 2)
    assert M[0, 0] == pytest.approx(0.8) and M[0, 1] == pytest.approx(0.2) and M[1, 1] == pytest.approx(0.9)
    # a column that failed everywhere is dropped; a row that failed everywhere fails the world
    M, bad = la.world_matrix([[[(1.0, ("L", 0))], None], [[(1.0, ("L", 1))], None]], v)
    assert bad == 2 and M.shape == (2, 1)
    with pytest.raises(ValueError):
        la.world_matrix([[None, None], [[(1.0, ("L", 1))], [(1.0, ("L", 2))]]], v)


def test_combine_common_rows_and_offsets():
    w1 = {"rows": [6, 7, 8], "cols": ["a", "b"], "n_leaves": 2,
          "cells": [[[(1.0, ("L", 0))], [(1.0, ("L", 1))]], [[(1.0, ("T", 0.0))], [(1.0, ("T", 0.0))]],
                    [[(1.0, ("T", 1.0))], [(1.0, ("T", 1.0))]]]}
    w2 = {"rows": [6, 8], "cols": ["c"], "n_leaves": 1, "cells": [[[(1.0, ("L", 0))]], [[(1.0, ("T", 0.5))]]]}
    v = np.array([0.3, 0.5, 0.6])  # w1's two leaves, then w2's one
    rows, sol, dropped = la.combine([w1, w2], v)
    assert rows == [6, 8] and dropped == 0
    # row 8 = 1.0 in w1, 0.5 in w2 -> 0.75 and x plays it from the first iteration; the opponent is then
    # indifferent in w1 and its average stays uniform: row 6 = mean(0.3, 0.5) in w1, 0.6 in w2 -> 0.5
    assert sol.q[1] == pytest.approx(0.75, abs=1e-3) and sol.q[0] == pytest.approx(0.5, abs=1e-3)
    assert sol.x[1] > 0.98


def test_gate():
    rows, x, q = [1, 6, 7], np.array([0.1, 0.7, 0.2]), np.array([0.40, 0.55, 0.52])
    g = la.gate(7, rows, x, q, 0.02)  # D = 0.03 > 0.02
    assert g.override and g.action == 6 and g.search == 6 and g.gap == pytest.approx(0.03)
    g = la.gate(7, rows, x, q, 0.05)
    assert not g.override and g.action == 7
    g = la.gate(6, rows, x, q, 0.0)  # the net already plays the lookahead's move
    assert not g.override and g.action == 6 and g.gap == 0.0
    g = la.gate(9, rows, x, q, 0.0)  # no row for the net's move: Q undefined -> the net's move
    assert not g.override and g.action == 9 and math.isnan(g.gap)
    assert la.gate(9, rows, x, q, -math.inf).action == 6  # pure lookahead
    assert la.gate(1, rows, x, q, math.inf).action == 1  # raw net


def test_parse_spec():
    path, kw, other = la.parse_spec("lookahead:nets/a.pt,margin=0.02,temperature=0.5,worlds=6,workers=11,"
                                    "min_prob=0.01,leaf_device=cuda,log=/tmp/g.jsonl,deterministic=0")
    assert path == "nets/a.pt"
    assert kw == {"margin": 0.02, "temperature": 0.5, "worlds": 6, "workers": 11, "min_prob": 0.01,
                  "deterministic": False}
    assert other == {"leaf_device": "cuda", "log": "/tmp/g.jsonl"}
    assert la.parse_spec("lookahead:x.pt,margin=-inf")[1]["margin"] == -math.inf
    with pytest.raises(ValueError):
        la.parse_spec("lookahead:x.pt,ms=1000")
    with pytest.raises(AssertionError):
        la.parse_spec("netsearch:x.pt")
