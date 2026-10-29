"""Battle -> poke-engine conversion on synthetic logs, and a few real battles on a local server (port 8000)."""

import asyncio
import logging
import random
from pathlib import Path

import pytest

pe = pytest.importorskip("poke_engine")
from poke_env import AccountConfiguration  # noqa: E402
from poke_env.battle import Battle  # noqa: E402
from poke_env.player import SimpleHeuristicsPlayer  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from porygon.env import local_server  # noqa: E402
from porygon.search import (  # noqa: E402
    DEFAULT_SET, History, SearchPlayer, calc_stats, convert, make_mon, sample_species, species_info)

MODEL = Path(__file__).resolve().parents[1] / "runs/20261019-185359/best_model.zip"
START = """|player|p1|me|
|player|p2|you|
|teamsize|p1|6
|teamsize|p2|6
|switch|p1a: Blissey|Blissey, L84, F|100/100
|switch|p2a: Jirachi|Jirachi, L80|100/100"""


def battle_from(log: str) -> Battle:
    battle = Battle("battle-gen4randombattle-1", "me", logging.getLogger("test"), gen=4)
    battle.player_role = "p1"
    for line in (START + "\n" + log).splitlines():
        battle.parse_message(line.split("|"))
    return battle


def state_of(battle: Battle, seed: int = 0) -> pe.State:
    return pe.State.from_string(convert(battle, random.Random(seed)).state_str)


def test_wish_future_sight_rest():
    log = """|turn|1
|move|p2a: Jirachi|Wish|p2a: Jirachi
|move|p1a: Blissey|Future Sight|p2a: Jirachi
|-start|p1a: Blissey|Future Sight
|turn|2"""
    st = state_of(battle_from(log))
    assert st.side_two.wish[0] == 1 and st.side_one.wish == (0, 0)
    assert st.side_one.future_sight == (2, "0")
    assert st.side_one.last_used_move == "move:0" and st.side_two.last_used_move == "move:0"
    log += """
|move|p1a: Blissey|Rest|p1a: Blissey
|-status|p1a: Blissey|slp|[from] move: Rest
|turn|3
|cant|p1a: Blissey|slp
|turn|4"""
    st = state_of(battle_from(log))
    assert st.side_two.wish == (0, 0) and st.side_one.future_sight[0] == 0
    assert st.side_one.pokemon[0].status == "SLEEP" and st.side_one.pokemon[0].rest_turns == 2


def test_opponent_choice_lock_and_toxic():
    log = """|turn|1
|move|p2a: Jirachi|Iron Head|p1a: Blissey
|move|p1a: Blissey|Toxic|p2a: Jirachi
|-status|p2a: Jirachi|tox
|turn|2"""
    battle = battle_from(log)
    battle.opponent_active_pokemon.item = "choicescarf"
    side = state_of(battle).side_two
    moves = side.pokemon[0].moves
    assert moves[0].id == "IRONHEAD" and not moves[0].disabled and all(m.disabled for m in moves[1:])
    assert side.last_used_move == "move:0" and side.side_conditions.toxic_count == 1
    # two different moves in one stint -> never guess a choice item
    battle = battle_from(log + "\n|move|p2a: Jirachi|U-turn|p1a: Blissey\n|turn|3")
    assert "p2: Jirachi" in History(battle).evidence.unlocked
    assert not any(state_of(battle, s).side_two.pokemon[0].item.startswith("CHOICE") for s in range(30))


def test_switch_in_resets_last_move_and_protect_counter():
    st = state_of(battle_from("|turn|1"))
    assert st.side_one.last_used_move == "switch:0" and st.side_two.last_used_move == "switch:0"
    st = state_of(battle_from("|turn|1\n|move|p2a: Jirachi|Protect|p2a: Jirachi\n|-singleturn|p2a: Jirachi|Protect\n"
                              "|turn|2"))
    assert st.side_two.side_conditions.protect == 1 and st.side_one.side_conditions.protect == 0


def test_timed_volatiles_and_substitute_health():
    log = """|turn|1
|move|p2a: Jirachi|Taunt|p1a: Blissey
|-start|p1a: Blissey|move: Taunt
|move|p1a: Blissey|Confuse Ray|p2a: Jirachi
|-start|p2a: Jirachi|confusion
|turn|2
|-activate|p2a: Jirachi|confusion
|move|p2a: Jirachi|Substitute|p2a: Jirachi
|-start|p2a: Jirachi|Substitute
|-damage|p2a: Jirachi|75/100
|move|p1a: Blissey|Ice Beam|p2a: Jirachi
|-activate|p2a: Jirachi|Substitute|[damage]
|turn|3"""
    st = state_of(battle_from(log))
    assert st.side_one.volatile_statuses == {"TAUNT"} and st.side_one.volatile_status_durations.taunt == 2
    assert st.side_two.volatile_statuses == {"CONFUSION", "SUBSTITUTE"}  # no stray NONE (patched round trip)
    assert st.side_two.volatile_status_durations.confusion == 1
    full = st.side_two.pokemon[0].maxhp // 4
    assert 1 <= st.side_two.substitute_health < full  # Ice Beam took some of it
    st = state_of(battle_from(log + "\n|-end|p2a: Jirachi|Substitute\n|-end|p2a: Jirachi|confusion\n|turn|4"))
    assert st.side_two.substitute_health == 0 and st.side_two.volatile_statuses == set()


def test_engine_is_patched_for_gen4_durations():
    """scripts/build_engine.sh: a taunt that has lasted 2 turns may end now (1/3), so the turn branches."""
    st = state_of(battle_from("|turn|1\n|move|p2a: Jirachi|Taunt|p1a: Blissey\n|-start|p1a: Blissey|move: Taunt"
                              "\n|move|p1a: Blissey|Seismic Toss|p2a: Jirachi\n|turn|2\n|turn|3"))
    theirs = st.side_two.pokemon[0].moves[0].id.lower()
    branches = pe.generate_instructions(st, "seismictoss", theirs)
    assert any("RemoveVolatileStatus SideOne: TAUNT" in repr(b.instruction_list) for b in branches), \
        "unpatched poke-engine: run scripts/build_engine.sh"


def test_root_priors_steer_the_search():
    """scripts/build_engine.sh's root-priors patch: PUCT sends visits to the prior's choice, none to a zero prior."""
    def mon(species: str, moves: list[str]) -> pe.Pokemon:
        st = calc_stats(species, 80, DEFAULT_SET.evs, DEFAULT_SET.ivs)
        return make_mon(species, 80, ("normal", "typeless"), st["hp"], st["hp"], st, "none", "none",
                        [(m, 16, False) for m in moves], "none", 0, 0, 50.0)
    side = lambda moves: pe.Side(pokemon=[mon("snorlax", moves), mon("chansey", ["splash"])]  # noqa: E731
                                 + [pe.Pokemon.create_fainted()] * 4)
    state = pe.State(side_one=side(["bodyslam", "splash"]), side_two=side(["bodyslam"]))
    visits = lambda r: {o.move_choice: o.visits for o in r.side_one}  # noqa: E731
    plain = visits(pe.monte_carlo_tree_search(state, iterations=20000))
    steered = visits(pe.monte_carlo_tree_search_priors(state, {"splash": 1.0}, 8.0, iterations=20000))
    assert steered["splash"] > 3 * plain["splash"] and steered["switch chansey"] <= 1


def test_sample_species_follows_team_rules():
    rng = random.Random(0)
    for _ in range(200):
        team = ["tyranitar", "blissey"] + sample_species(["tyranitar", "blissey"], 4, rng)
        assert len(team) == 6 and len({species_info(s)[0] for s in team}) == 6
        assert "shedinja" not in team and "chansey" not in team
        types = [t for s in team for t in species_info(s)[1]]
        assert max(types.count(t) for t in set(types)) <= 2


@pytest.mark.skipif(not MODEL.exists(), reason="needs a trained model")
@pytest.mark.parametrize("c_puct", [0.0, 4.0])
def test_search_battles(c_puct, n_battles=3):
    policy = PPO.load(MODEL, device="cpu").policy.eval()
    tag = random.randint(0, 10**6)
    common = dict(battle_format="gen4randombattle", server_configuration=local_server(8000),
                  max_concurrent_battles=n_battles, log_level=40)
    player = SearchPlayer(policy, ms=50, samples=2, workers=4, c_puct=c_puct,
                          account_configuration=AccountConfiguration(f"stest{tag}", None), **common)
    opponent = SimpleHeuristicsPlayer(account_configuration=AccountConfiguration(f"htest{tag}", None), **common)
    asyncio.run(player.battle_against(opponent, n_battles=n_battles))
    assert player.n_finished_battles == n_battles
    assert player.stats["searched"] > 10 * n_battles
    assert player.stats["fallbacks"] == 0
