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
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]


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



def test_search_player_defaults_to_the_tuned_evaluation():
    from porygon.search import DEFAULT_EVAL, check_tunable_engine
    cfg = json.loads(Path(DEFAULT_EVAL).read_text())
    assert len(cfg["weights"]) == len(pe.eval_feature_names())
    check_tunable_engine(None)  # this venv's engine has the patch


@pytest.mark.skipif(not (ROOT / ".venv-stock/bin/python").exists(), reason="no stock-engine venv")
def test_stock_engine_is_refused_for_a_tuned_evaluation():
    from porygon.search import check_tunable_engine
    with pytest.raises(RuntimeError, match="set_eval"):
        check_tunable_engine(str(ROOT / ".venv-stock/bin/python"))


def test_decision_log_oracle_record(pairs, tmp_path):
    """The true-team position logged for offline network studies (porygon.decisions, --log-oracle)."""
    from types import SimpleNamespace

    from porygon.decisions import LoggedSearchPlayer
    from tests.test_search import battle_from

    jirachi = {"speciesId": "jirachi", "level": 80, "moves": ["ironhead", "uturn", "bodyslam", "firepunch"],
               "ability": "Serene Grace", "item": "Choice Scarf", "evs": dict.fromkeys(STATS, 85),
               "ivs": dict.fromkeys(STATS, 31)}
    theirs = [jirachi] + [s for s in pairs[0][1] if s["speciesId"] != "jirachi"][:5]
    (tmp_path / "battle-gen4randombattle-1.json").write_text(
        json.dumps({"p1": "me", "p2": "you", "p1team": pairs[0][0], "p2team": theirs}))
    battle = battle_from("|turn|1")
    fake = SimpleNamespace(oracle_dir=str(tmp_path), username="me", weather_seen={})
    rec = LoggedSearchPlayer.oracle_record(fake, battle)
    assert "error" not in rec, rec
    side = pe.State.from_string(rec["state"]).side_two
    assert side.pokemon[0].item == "CHOICESCARF" and side.pokemon[0].moves[0].id == "IRONHEAD"
    assert [p.id.lower() for p in side.pokemon[1:]] == [s["speciesId"] for s in theirs[1:]]  # the true bench
