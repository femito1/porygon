"""porygon.fastenv invariants on a few in-process battles per gen (needs node + a built pokemon-showdown;
PORYGON_PS / PORYGON_NODE override). Observation parity with poke-env is tests/test_fastenv_parity.py."""

import shutil

import numpy as np
import pytest

from porygon.fastenv import FastEnv, random_actions

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


@pytest.mark.parametrize("fmt,n_actions", [("gen4randombattle", 10), ("gen9randombattle", 26)])
def test_rows_rewards_and_aux(fmt, n_actions):
    rng = np.random.default_rng(0)
    returns: dict[int, float] = {}
    finished = []  # (slot, return, win)
    with FastEnv(fmt, n_workers=2, battles_per_worker=3, seed=5) as env:
        b = env.reset()
        assert b.mask.shape[1] == n_actions and b.obs.shape[1] == 4079
        while len(finished) < 24:
            na = b.needs_action
            assert (b.mask[na].sum(1) > 0).all(), "decision row without a legal action"
            assert not b.mask[~na].any() and not b.obs[~na].any(), "terminal rows carry no observation"
            assert (b.flags & 3 != 3).all(), "a row is either a decision or terminal"
            for i in range(len(b)):
                s = int(b.slot[i])
                returns[s] = returns.get(s, 0.0) + float(b.reward[i])
                if b.done[i]:
                    finished.append((s, returns.pop(s), int(b.win[i])))
            # aux: opponent team in token order, every slot a real mon, revealed tokens first
            assert (b.opp_team[:, :, 0] > 1).all()
            rev = b.opp_revealed.astype(bool)
            assert (np.diff(rev.astype(int), axis=1) <= 0).all()
            b = env.step(random_actions(b.mask[na], rng))
    for s, ret, win in finished:
        # shaped return = final state value: win/loss +-1 plus the faint/HP balance (within +-0.6)
        assert abs(ret - win) <= 0.6 + 1e-4, (ret, win)


def test_duplicate_pairs_replay_teams():
    rng = np.random.default_rng(1)
    with FastEnv("gen4randombattle", n_workers=1, battles_per_worker=2, seed=9, duplicate=True) as env:
        b = env.reset()
        games = {0: 0, 1: 0}
        teams: dict[tuple[int, int], np.ndarray] = {}
        for _ in range(4000):
            for i in range(len(b)):  # rows in order: a battle's terminal rows precede the next battle's
                s = int(b.slot[i])
                if b.done[i] and s % 2 == 0:
                    games[s // 2] += 1
                elif b.flags[i] & 4:  # first decision of a battle: the opponent's true team
                    teams.setdefault((s // 2, games[s // 2]), {})[s % 2] = np.sort(b.opp_team[i, :, 0])
            if min(games.values()) >= 2:
                break
            b = env.step(random_actions(b.mask[b.needs_action], rng))
        for pos in (0, 1):
            g0, g1 = teams[(pos, 0)], teams[(pos, 1)]
            # game 1 swaps the teams: p1's opponent in game 1 is p2's opponent (= p1's team) of game 0
            assert (g1[0] == g0[1]).all() and (g1[1] == g0[0]).all()


@pytest.mark.parametrize("fmt", ["gen4randombattle", "gen9randombattle"])
def test_v2_rows_consistent(fmt):
    """Multi-row batches: every row's v2 action_mask equals the env mask (catches per-row stride bugs that
    single-decision parity harnesses cannot see)."""
    rng = np.random.default_rng(3)
    with FastEnv(fmt, n_workers=1, battles_per_worker=6, seed=4, encoder="v2") as env:
        b = env.reset()
        for _ in range(60):
            na = b.needs_action
            assert (b.obs["action_mask"][na] > 0).tolist() == b.mask[na].tolist()
            cls = b.aux["aux_opp_class"][na]
            assert cls.shape[1] == b.mask.shape[1] and (cls >= -1).all()
            b = env.step(random_actions(b.mask[na], rng))


def test_env_side_bot_battles():
    rng = np.random.default_rng(5)
    with FastEnv("gen9randombattle", n_workers=1, battles_per_worker=4, seed=6, bot="max_power", bot_battles=2) as env:
        b = env.reset()
        sides = set()
        for _ in range(200):
            sides |= {(int(s) // 2, int(s) % 2) for s in b.slot}
            b = env.step(random_actions(b.mask[b.needs_action], rng))
        assert {(0, 1), (1, 1)}.isdisjoint(sides) and {(2, 1), (3, 1), (0, 0)} <= sides
