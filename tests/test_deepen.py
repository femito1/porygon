"""porygon.deepen's pure parts: option selection by prior mass, the priors, the batched matrix-game solver, the tree
(values bottom-up, the root's Bayesian game = lookahead2's at budget 0, reach and selection) and the spec (no engine,
no network, no server)."""

import math

import numpy as np
import pytest

from porygon import deepen as D
from porygon import lookahead2 as LA

META = lambda s="s", oc=("a", "b"), oa=(6, 7), tc=("x", "y"), tcl=(7, 8): (s, list(oc), list(oa), list(tc), list(tcl))  # noqa: E731
FLAT = np.zeros(26)
OPP = np.zeros(D_N := 55)


def test_keep_by_mass():
    assert D.keep_by_mass([0.5, 0.3, 0.2], 0.9, 3) == [0, 1, 2]
    assert D.keep_by_mass([0.5, 0.3, 0.2], 0.8, 3) == [0, 1]
    assert D.keep_by_mass([0.2, 0.5, 0.3], 0.5, 3) == [1]
    assert D.keep_by_mass([0.2, 0.5, 0.3], 0.5, 3, floor=2) == [1, 2]
    assert D.keep_by_mass([0.1] * 10, 0.9, 4) == [0, 1, 2, 3]  # cap; ties by index
    assert D.keep_by_mass([1.0], 0.9, 3, floor=2) == [0]


def test_priors():
    lg = np.full(26, -5.0)
    lg[6], lg[7] = 1.0, 0.0
    p = D.our_prior(lg, [6, 7])
    assert p == pytest.approx([math.e / (math.e + 1), 1 / (math.e + 1)])
    assert D.our_prior(lg, [-1]) == pytest.approx([1.0])
    z = np.full(55, np.finfo(np.float32).min, dtype=np.float64)
    z[6], z[7], z[8] = 0.0, math.log(2.0), 0.0  # unseen, move slot 0, move slot 1
    q = D.opp_prior(z, [7, 6, 6, 8])  # two unseen switch-ins share class 6
    assert q == pytest.approx(np.array([2, 0.5, 0.5, 1]) / 4)
    q = D.opp_prior(z, [7, -1, 3, 8])  # unplaceable and masked options get the mean of the placed ones
    assert q == pytest.approx(np.array([0.5, 0.375, 0.375, 0.25]) / 1.5)
    assert D.opp_prior(z, [-1, -1]) == pytest.approx([0.5, 0.5])


def test_solve_games_matches_solve_bayes_and_padding():
    rng = np.random.default_rng(0)
    mats = [rng.random((3, 4)), rng.random((2, 2)), np.array([[0.7]]), np.array([[1.0, 0.0], [0.0, 1.0]])]
    xs, ys, vs = D.solve_games(mats, 300)
    for M, x, y, v in zip(mats, xs, ys, vs):
        s = LA.solve_bayes([M], iters=300)
        assert x == pytest.approx(s.x, abs=1e-12) and y == pytest.approx(s.ys[0], abs=1e-12)
        assert v == pytest.approx(float(x @ M @ y))
    assert vs[2] == pytest.approx(0.7) and vs[3] == pytest.approx(0.5, abs=0.02)


def _tree_one_world():
    """One world, rows 6 / 7, cols a / b; cell (7, b) has two chance branches, (6, a) a forced opponent switch."""
    t = D.Tree(1)
    vals = [0.6, 0.4, 0.5, 0.3, 0.9, 0.2]
    ids = [t.add_leaf(0, 1, META(f"s{i}"), v, FLAT, OPP) for i, v in enumerate(vals)]
    cells = [[[(1.0, ("S", [[(1.0, ("L", ids[0]))], [(1.0, ("L", ids[5]))]]))], [(1.0, ("L", ids[1]))]],
             [[(1.0, ("L", ids[2]))], [(0.5, ("L", ids[3])), (0.5, ("L", ids[4]))]]]
    t.worlds.append(D.World([6, 7], ["a", "b"], cells, np.array([0.5, 0.5])))
    return t, ids, cells


def test_root_solve_is_lookahead2_combine():
    t, ids, cells = _tree_one_world()
    common, sol, used, info = t.root_solve(1000)
    w = {"rows": [6, 7], "cols": ["a", "b"], "cells": cells, "n_leaves": 0}
    rows, ref, dropped = LA.combine([w], np.array(t.val))
    assert common == rows and used == [0] and not dropped
    assert sol.x == pytest.approx(ref.x, abs=0) and sol.q == pytest.approx(ref.q, abs=0)
    # two worlds, one with a failed cell and an extra row
    t2 = D.Tree(2)
    a = [t2.add_leaf(0, 1, META(f"a{i}"), v, FLAT, OPP) for i, v in enumerate([0.3, 0.6, 0.5])]
    b = [t2.add_leaf(1, 1, META(f"b{i}"), v, FLAT, OPP) for i, v in enumerate([0.8, 0.1, 0.4, 0.7])]
    c1 = [[[(1.0, ("L", a[0]))], [(1.0, ("L", a[1]))]], [[(1.0, ("L", a[2]))], None]]
    c2 = [[[(1.0, ("L", b[0]))]], [[(1.0, ("L", b[1]))]], [[(1.0, ("L", b[2])), ]]]
    t2.worlds += [D.World([6, 8], ["x", "y"], c1, np.ones(2) / 2), D.World([6, 7, 8], ["z"], c2, np.ones(1))]
    common, sol, used, info = t2.root_solve(1000)
    ws = [{"rows": [6, 8], "cols": ["x", "y"], "cells": c1, "n_leaves": 0},
          {"rows": [6, 7, 8], "cols": ["z"], "cells": c2, "n_leaves": 0}]
    rows, ref, _ = LA.combine(ws, np.array(t2.val))
    assert common == rows == [6, 8] and sol.q == pytest.approx(ref.q, abs=0)
    assert info[1][0] == [0, 2]


def test_expansion_changes_value_and_failed_node_falls_back():
    t, ids, cells = _tree_one_world()
    g = ids[1]  # cell (6, b), value 0.4
    kids = [t.add_leaf(0, 2, META(f"k{i}"), v, FLAT, OPP) for i, v in enumerate([0.9, 0.7, 0.8, 0.6])]
    t.add_node(g, [0, 1], [0, 1], [[[(1.0, ("L", kids[0]))], [(1.0, ("L", kids[1]))]],
                                    [[(1.0, ("L", kids[2]))], [(1.0, ("L", kids[3]))]]])
    t.evaluate(1000)
    assert t.val[g] == pytest.approx(0.7, abs=0.01)  # row 0 dominates (0.9/0.7 vs 0.8/0.6); col 1 for them
    assert t.leaf_v[g] == pytest.approx(0.4) and t.depth() == 2
    M, keep = D.cell_matrix(t.worlds[0].cells, t.val)
    assert M[0, 1] == pytest.approx(t.val[g]) and M[0, 0] == pytest.approx(0.2)  # the root sees the deeper value
    # a node whose every row failed keeps the leaf value and never expands again
    h = ids[2]
    t.add_node(h, [0], [0, 1], [[None, None]])
    t.evaluate(100)
    assert t.node_of[h] == -2 and t.val[h] == pytest.approx(0.5)


def test_reach_is_a_distribution_and_flows_down():
    t, ids, _ = _tree_one_world()
    common, sol, used, info = t.root_solve(1000)
    r = t.reach(common, sol, used, info, np.array([0.5, 0.5]), a_net=6)
    assert r.sum() == pytest.approx(1.0)  # no terminal: every path ends in a leaf
    assert r[ids[5]] > 0 and r[ids[0]] == 0  # the forced switch passes everything to the switch-in worst for us
    # row weight with a_net: (x + pi + 1[a_net]) / 3
    rho6 = (sol.x[0] + 0.5 + 1) / 3
    ys = 0.5 * (sol.ys[0] + 0.5)
    assert r[ids[1]] == pytest.approx(rho6 * ys[1])
    # an expanded leaf passes its reach on
    g = ids[2]
    kids = [t.add_leaf(0, 2, META(f"k{i}"), 0.5, FLAT, OPP) for i in range(2)]
    t.add_node(g, [0], [0, 1], [[[(1.0, ("L", kids[0]))], [(0.5, ("L", kids[1])), (0.5, ("T", 1.0))]]])
    t.evaluate(500)
    common, sol, used, info = t.root_solve(1000)
    r = t.reach(common, sol, used, info, np.array([0.5, 0.5]), a_net=None)
    lost = r[g] * 0.5 * (t.nodes[-1].y[1] + 0.5) * 0.5  # the terminal branch
    assert r[kids[0]] + r[kids[1]] == pytest.approx(r[g] - lost)


def test_select_priority_target_and_limits():
    t, ids, _ = _tree_one_world()
    common, sol, used, info = t.root_solve(1000)
    r = t.reach(common, sol, used, info, np.array([0.5, 0.5]), a_net=6)
    cfg = D.Config(budget=100, us_cap=2, opp_cap=2, max_turns=4)
    v = np.array(t.val)
    pr = r * 4 * v * (1 - v)
    picks = t.select(r, target=1.0, cfg=cfg, branches=2.0)
    assert len(picks) == 1 and picks[0][0] == int(np.argmax(pr))
    allp = t.select(r, target=1e9, cfg=cfg, branches=2.0)
    assert [g for g, *_ in allp] == sorted((g for g in range(len(v)) if pr[g] > 0), key=lambda g: (-pr[g], g))
    assert ids[0] not in [g for g, *_ in allp]  # reach 0
    t.turns[allp[0][0]] = 4  # at max_turns: not expandable
    assert allp[0][0] not in [g for g, *_ in t.select(r, 1e9, cfg, 2.0)]
    t.val[ids[1]] = 1.0  # settled
    assert ids[1] not in [g for g, *_ in t.select(r, 1e9, cfg, 2.0)]


def test_dedup_key():
    t = D.Tree(1)
    a = t.add_leaf(0, 1, META("same"), 0.5, FLAT, OPP)
    b = t.add_leaf(0, 1, META("same"), 0.5, FLAT, OPP)
    assert t.lookup(0, 1, "same") == a != b
    assert t.lookup(0, 2, "same") is None and t.lookup(1, 1, "same") is None


def test_remap():
    ref = ("S", [[(0.5, ("L", 0)), (0.5, ("T", 1.0))], [(1.0, ("L", 1))]])
    assert D.remap(ref, [7, 9]) == ("S", [[(0.5, ("L", 7)), (0.5, ("T", 1.0))], [(1.0, ("L", 9))]])


def test_parse_spec():
    path, kw, other, cfg, half = D.parse_spec("deepen:nets/a.pt,budget=2048,margin=0.005,temperature=0.5,worlds=8,"
                                              "workers=11,leaf_device=cuda,opp_cap=5,deep_min_prob=0.05,half=1,"
                                              "min_prob=0.0,log=/tmp/g.jsonl")
    assert path == "nets/a.pt" and half
    assert kw == {"margin": 0.005, "temperature": 0.5, "worlds": 8, "workers": 11, "min_prob": 0.0}
    assert other == {"leaf_device": "cuda", "log": "/tmp/g.jsonl"}
    assert cfg == D.Config(budget=2048, opp_cap=5, deep_min_prob=0.05, min_prob=0.0)
    assert D.parse_spec("deepen:x.pt")[3] == D.Config()
    with pytest.raises(AssertionError):
        D.parse_spec("lookahead:x.pt")
