"""Value lookahead: the matrix-game solver, and leaf encoding / potential against poke-env on real battles
(local server on port 8000)."""

import asyncio
import random

import numpy as np
import pytest

pytest.importorskip("poke_engine")
from poke_env import AccountConfiguration  # noqa: E402
from poke_env.player import SimpleHeuristicsPlayer  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from porygon import lookahead as la  # noqa: E402
from porygon.encoder import FIELD_DIM, MON_DIM, embed_battle  # noqa: E402
from porygon.env import PolicyPlayer, local_server  # noqa: E402
from porygon.expert import state_value  # noqa: E402
from porygon.search import History, SearchPlayer, convert  # noqa: E402
from tests.test_search import MODEL  # noqa: E402


def test_solve():
    rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    x, y = la.solve(rps)
    assert np.allclose(x, 1 / 3, atol=0.02) and np.allclose(y, 1 / 3, atol=0.02)
    x, y = la.solve(np.array([[1.0, 0.5], [0.0, 0.2]]))  # row 0 dominates; column 1 is then best for them
    assert x[0] > 0.98 and y[1] > 0.98


# Leaf features that legitimately differ from embed_battle at the root: the leaf is one turn later.
LATER = {"turn", "weather_age", "first_turn"}


def feature(i: int) -> str:
    if i < FIELD_DIM:
        if i == la.TAIL:
            return "turn"
        return "weather_age" if la.NW <= i < 2 * la.NW else f"field[{i}]"
    o = (i - FIELD_DIM) % MON_DIM
    return {la.FIRST_TURN: "first_turn", la.HP: "hp"}.get(o, f"mon[{o}]")


class Checker(PolicyPlayer):
    """Encodes every root as a leaf and compares it with embed_battle."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked, self.mismatches = 0, []

    def choose_move(self, battle):
        if not battle.wait and battle.active_pokemon is not None:
            conv = convert(battle, random.Random(self.checked), None, History(battle))
            leaves = la.Leaves(la.Root(battle), conv)
            got, want = leaves.encode(leaves.state), embed_battle(battle)
            for i in np.nonzero(np.abs(got - want) > 1e-3)[0]:
                name = feature(i)
                if name not in LATER and not (name == "hp" and abs(got[i] - want[i]) < 0.02):
                    self.mismatches.append((battle.turn, name, float(want[i]), float(got[i])))
            phi, done = la.potential(leaves.state)
            assert not done and abs(phi - state_value(battle)) < 0.01, (phi, state_value(battle))
            self.checked += 1
        return super().choose_move(battle)


@pytest.mark.skipif(not MODEL.exists(), reason="needs a trained model")
def test_leaves_match_embed_battle_and_lookahead_plays(n_battles=3):
    policy = PPO.load(MODEL, device="cpu").policy.eval()
    tag = random.randint(0, 10**6)
    common = dict(battle_format="gen4randombattle", server_configuration=local_server(8000),
                  max_concurrent_battles=n_battles, log_level=40)
    checker = Checker(policy, account_configuration=AccountConfiguration(f"ltest{tag}", None), **common)
    opponent = SimpleHeuristicsPlayer(account_configuration=AccountConfiguration(f"lheur{tag}", None), **common)
    asyncio.run(checker.battle_against(opponent, n_battles=n_battles))
    assert checker.checked > 10 * n_battles
    # rare root-only differences: convert trusts the request over poke-env's stale must_recharge, and
    # misses the PP of some opponent hidden powers
    assert len({(t, n) for t, n, *_ in checker.mismatches}) <= max(1, checker.checked // 100), checker.mismatches[:10]

    player = SearchPlayer(policy, ms=50, samples=2, workers=4, value_weight=0.5, value_min_prob=0.01,
                          account_configuration=AccountConfiguration(f"lsrch{tag}", None), **common)
    asyncio.run(player.battle_against(opponent, n_battles=n_battles))
    assert player.n_finished_battles == n_battles
    assert player.stats["searched"] > 10 * n_battles
    assert player.stats["fallbacks"] == 0 and player.stats["value_failed"] == 0
    assert player.lookahead.failed <= player.stats["searched"] // 20
