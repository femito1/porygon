"""porygon.netsearch's pure parts (root statistics, the margin gate, sharpening) on synthetic engine results; the
engine / network paths are checked on the cluster (scripts/fastenv/netsearch_study.py check)."""
import math

import numpy as np
import pytest

from porygon import netsearch as NS

A = 10  # gen4 action space: switches 0-5, moves 6-9


class Conv:
    """Stand-in for porygon.search.Converted: engine choice -> action index."""

    def __init__(self, moves=("tackle", "growl"), switches=("a", "b")):
        self.names = {f"switch {s}": i for i, s in enumerate(switches)} | {m: 6 + i for i, m in enumerate(moves)}

    def action(self, choice):
        return self.names.get(choice)


def opts(**kv):
    """[(choice, visits, total score)] from choice=(visits, mean score)."""
    return [(c.replace("_", " "), v, v * q) for c, (v, q) in kv.items()]


def test_root_stats_q_share_root():
    convs = [Conv(), Conv()]
    res = [opts(tackle=(300, 0.6), growl=(100, 0.4)), opts(tackle=(100, 0.5), growl=(300, 0.7), switch_a=(0, 0.0))]
    st = NS.root_stats(convs, res, A)
    assert st.ok.all() and not st.errors
    np.testing.assert_allclose(st.q[[6, 7]], [0.55, 0.55])
    assert math.isnan(st.q[0])  # never visited
    np.testing.assert_allclose(st.share[[6, 7]], [0.5, 0.5])
    np.testing.assert_allclose(st.root, np.mean([(180 + 40) / 400, (50 + 210) / 400]))
    assert st.visits == 400


def test_root_stats_drops_failed_and_unmappable():
    convs = [Conv(), Conv(), Conv()]
    res = [opts(tackle=(10, 0.9)), RuntimeError("panic"), opts(thunderbolt=(5, 0.1))]
    st = NS.root_stats(convs, res, A)
    assert st.ok.tolist() == [True, False, False] and len(st.errors) == 2
    assert st.q[6] == pytest.approx(0.9) and st.share[6] == pytest.approx(1.0)


def test_no_move_is_ignored():
    st = NS.root_stats([Conv()], [[("No Move", 50, 25.0)] + opts(tackle=(10, 0.5))], A)
    assert st.share[6] == pytest.approx(1.0) and st.visits == 10


def test_sharpen_matches_temperature_softmax():
    logits = np.array([1.0, -0.5, 0.3, 2.0])
    p1 = np.exp(logits) / np.exp(logits).sum()
    z = logits / 0.5
    want = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    np.testing.assert_allclose(NS.sharpen(p1, 0.5), want, rtol=1e-12)
    p = np.array([0.0, 0.25, 0.75, 0.0])
    np.testing.assert_allclose(NS.sharpen(p, 0.5), [0, 0.1, 0.9, 0], rtol=1e-12)
    assert NS.sharpen(p, 1.0) is p


def _stats():
    # tackle most visited (pick) with Q .60; growl Q .55; switch a Q .40
    return NS.root_stats([Conv(), Conv()], [opts(tackle=(500, 0.6), growl=(300, 0.55), switch_a=(200, 0.4))] * 2, A)


def test_gate_margin_and_endpoints():
    st, mask = _stats(), np.zeros(A)
    mask[[0, 6, 7]] = 1
    assert NS.search_pick(st, mask) == 6
    g = NS.gate(7, st, mask, 0.03)  # gap .05 > .03: override
    assert g.override and g.action == 6 and g.gap == pytest.approx(0.05)
    g = NS.gate(7, st, mask, 0.06)  # gap .05 <= .06: keep the network's move
    assert not g.override and g.action == 7
    assert NS.gate(6, st, mask, -math.inf).action == 6 and not NS.gate(6, st, mask, -math.inf).override
    for a in (0, 6, 7):
        assert NS.gate(a, st, mask, math.inf).action == a  # +inf = raw network
        assert NS.gate(a, st, mask, -math.inf).action == 6  # -inf = pure search


def test_gate_undefined_gap_keeps_network_unless_minus_inf():
    st, mask = _stats(), np.zeros(A)
    mask[[1, 6, 7]] = 1  # switch b is legal but no engine option: Q undefined
    g = NS.gate(1, st, mask, 0.0)
    assert math.isnan(g.gap) and not g.override and g.action == 1
    assert NS.gate(1, st, mask, -math.inf).action == 6


def test_search_pick_prior_weight_and_illegal():
    st, mask = _stats(), np.zeros(A)
    mask[[0, 7]] = 1  # the most visited (tackle) is illegal here
    assert NS.search_pick(st, mask) == 7
    probs = np.zeros(A)
    probs[0] = 1.0
    assert NS.search_pick(st, mask, probs, prior_weight=0.5) == 0  # share .2 + .5 > .3


def test_gap_t():
    convs = [Conv()] * 3
    res = [opts(tackle=(100, q1), growl=(100, q2)) for q1, q2 in ((0.6, 0.5), (0.7, 0.5), (0.5, 0.5))]
    m, se, n = NS.gap_t(NS.root_stats(convs, res, A), 6, 7)
    assert n == 3 and m == pytest.approx(0.1) and se == pytest.approx(0.1 / math.sqrt(3))


def test_load_spec_parses_options():
    class Net:
        gen = NS.S.GEN_NUM

        def eval(self):
            return self

    seen = {}

    def loader(path, device):
        seen.update(path=path, device=device)
        return Net()

    p = NS.load_spec("netsearch:/x/net.pt,margin=-inf,temperature=0.5,ms=200,samples=2,c_puct=1.5,device=cpu",
                     net_loader=loader, battle_format=NS.S.inference.FORMAT, start_listening=False)
    assert seen == {"path": "/x/net.pt", "device": "cpu"}
    assert p.margin == -math.inf and p.temperature == 0.5 and p.ms == 200 and p.samples == 2 and p.c_puct == 1.5
    with pytest.raises(ValueError):
        NS.load_spec("netsearch:/x/net.pt,bogus=1", net_loader=loader, start_listening=False)
