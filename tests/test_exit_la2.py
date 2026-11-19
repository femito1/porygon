"""ExIt-LA2 pilot targets (scripts/fastenv/exit_la2.py): pi_LA over the T 0.5 draw with lookahead2's gate, the sqrt
transform, KL, the verdict rule, the reader's S-line prefix parse. Pure numpy, no engine or network."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "exit_la2", Path(__file__).resolve().parents[1] / "scripts" / "fastenv" / "exit_la2.py")
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)


def _probs():
    p = np.zeros(26)
    p[[6, 7, 8, 1]] = [0.5, 0.3, 0.15, 0.05]
    return p


def test_pi_la_moves_only_overridden_mass():
    p = _probs()
    rows, x = [1, 6, 7, 8], np.array([0.0, 0.0, 1.0, 0.0])  # a_L = 7
    q = np.array([0.40, 0.50, 0.52, 0.516])  # gaps to 7: from 1 0.12, from 6 0.02, from 8 0.004
    pi, p05 = X.pi_la(p, rows, x, q, 0.005)
    assert abs(pi.sum() - 1) < 1e-12 and abs(p05.sum() - 1) < 1e-12
    want = p05.copy()
    want[7] += want[1] + want[6]  # 8: gap 0.004 <= 0.005, kept
    want[1] = want[6] = 0
    assert np.allclose(pi, want)


def test_pi_la_margin_inf_is_the_net_and_undefined_q_keeps_the_draw():
    p = _probs()
    rows, x, q = [6, 7, 8], np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.9, 0.1])  # action 1 has no row
    pi, p05 = X.pi_la(p, rows, x, q, math.inf)
    assert np.allclose(pi, p05)
    pi, p05 = X.pi_la(p, rows, x, q, 0.0)
    assert np.isclose(pi[1], p05[1]) and np.isclose(pi[7], p05[6] + p05[7] + p05[8])


def test_sqrt_target_inverts_sharpening():
    p = _probs()
    p1 = p / p.sum()
    p05 = X.sharpen(p1, 0.5)
    assert np.allclose(X.sqrt_target(p05), p1, atol=1e-7)  # C's target = the start net's own T-1 policy
    pi = np.zeros(26)
    pi[[6, 7]] = [0.2, 0.8]
    t = X.sqrt_target(pi).astype(np.float64)
    assert np.allclose(X.sharpen(t, 0.5), pi)  # a net playing t at T 1 plays pi at T 0.5
    assert X.sqrt_target(np.stack([pi, p05])).shape == (2, 26)


def test_kl_rows():
    a = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[0.5, 0.5, 0.0], [0.5, 0.25, 0.25]])
    assert np.allclose(X.kl_rows(a, b), [0.0, math.log(2)])


def test_verdict_rule():
    r = lambda w, h: {"win_rate": w, "ci95": h}  # noqa: E731
    assert X.verdict_line(r(0.52, 0.007), r(0.515, 0.007)) == "GO"
    assert X.verdict_line(r(0.52, 0.007), r(0.505, 0.007)) == "MIXED"
    assert X.verdict_line(r(0.49, 0.007), r(0.50, 0.007)) == "HARMFUL"
    assert X.verdict_line(r(0.50, 0.007), r(0.51, 0.007)) == "FLAT"
    assert X.verdict_line(r(0.505, 0.007), r(0.51, 0.007)) == "NOT SHOWN"


def test_s_line_prefix_parse_and_pick():
    line = json.dumps({"t": "S", "game": "7:3:1", "slot": 5, "side": 1, "turn": 4, "idx": 6, "e": 33,
                       "state": {"x": [1, {"state": 2}]}}, separators=(",", ":"))
    cut = line.find(',"state":')
    assert json.loads(line[:cut] + "}") == {"t": "S", "game": "7:3:1", "slot": 5, "side": 1, "turn": 4, "idx": 6,
                                             "e": 33}
    assert X.pick(1, "7:3:1", 0, 9) == X.pick(1, "7:3:1", 0, 9)
    assert 0 <= X.pick(1, "g", 1, 3) < 3
    assert X.game_hash("7:3:1") < 2 ** 63
