"""The tunable leaf evaluation (tunable-eval engine patch) and engine-only self-play (porygon.selfplay)."""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import poke_engine as pe
import pytest

from porygon.engine_worker import search
from porygon.selfplay import EvalConfig, initial_state, mirrored, play, winner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pairs():
    out = subprocess.run(["node", str(ROOT / "scripts/gen_teams.js"), "8"], capture_output=True, text=True,
                         check=True).stdout
    return [json.loads(line) for line in out.splitlines()]


def states(pairs, n=8, per=25, seed=0):
    rng = random.Random(seed)
    for teams in pairs[:n]:
        s = initial_state(*teams)
        for _ in range(per):
            if winner(s):
                break
            yield s
            pe.set_eval(None)
            a1 = rng.choice(pe.monte_carlo_tree_search(s, iterations=500).side_one).move_choice
            a2 = rng.choice(pe.monte_carlo_tree_search(mirrored(s), iterations=500).side_one).move_choice
            fix = lambda c: "none" if c == "No Move" else c.removeprefix("switch ")  # noqa: E731
            b = pe.generate_instructions(s, fix(a1), fix(a2))
            s = s.apply_instructions(rng.choices(b, weights=[x.percentage for x in b])[0])


def test_stock_weights_reproduce_stock_evaluation(pairs):
    w = [100.0] + [0.0] * (len(pe.eval_feature_names()) - 1)
    for s in states(pairs):
        pe.set_eval(None)
        stock = pe.evaluate(s)
        pe.set_eval(w)
        assert abs(pe.evaluate(s) - stock) < 1e-3
        f, fm = np.array(pe.eval_features(s)), np.array(pe.eval_features(mirrored(s)))
        assert np.abs(f + fm).max() < 1e-4  # side one minus side two
    pe.set_eval(None)


def test_eval_config_changes_the_search(pairs):
    s = initial_state(*pairs[0]).to_string()
    flat = {"weights": [0.0] * len(pe.eval_feature_names()), "scale": 1.0, "root_weight": 1.0}
    res = search(s, 50, eval_config=flat)  # every leaf scores 0.5 except won / lost battles
    assert all(abs(total / max(n, 1) - 0.5) < 0.05 for _, n, total in res)
    search(s, 10)  # and the next request without a config is back to the stock evaluation
    pe.set_eval(None)


def test_selfplay_game_and_luck(pairs):
    stock = EvalConfig.load("stock")
    g = play(pairs[1], stock, stock, 300, random.Random(0), record=True, model=stock)
    assert g["winner"] in (1, 2) and len(g["features"]) == len(g["luck"]) == g["decisions"]
    assert all(-1 <= x <= 1 for x in g["luck"])
