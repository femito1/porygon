"""gen9 search against a live local server: tera choices map to actions 22-25 and both sides' tera state converts.

Runs only as  PORYGON_GEN=9 PYTHONPATH=<gen9 poke-engine>:src pytest tests/test_search_gen9.py  with a Showdown
server on port 8000 (scripts/start_servers.sh 1 8000); skipped otherwise.
"""

import asyncio
import collections
import random
import socket

import pytest

from porygon import inference


def _server_up(port: int = 8000) -> bool:
    try:
        socket.create_connection(("localhost", port), timeout=1).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(inference.GEN_NUM != 9 or not _server_up(), reason="needs PORYGON_GEN=9 and a server")


def test_gen9_search_and_tera_conversion():
    import poke_engine as pe
    from poke_env import AccountConfiguration

    from porygon import search as S
    from porygon.env import local_server
    from porygon.evaluate import BASELINES

    c: collections.Counter = collections.Counter()

    class Check(S.SearchPlayer):
        def on_decision(self, battle, probs, mask, shares):
            c["tera chosen"] += self.last_action[battle.battle_tag] >= S.TERA_OFFSET
            st = pe.State.from_string(S.convert(battle, random.Random(0), None, S.History(battle)).state_str)
            our_tera = S.request_tera(battle)
            for ours, side, team in ((True, st.side_one, battle.team), (False, st.side_two, battle.opponent_team)):
                for (key, mon), em in zip(team.items(), side.pokemon):
                    if mon.is_terastallized:
                        c["tera"] += 1
                        c["tera ok"] += (em.terastallized and em.tera_type.lower() == mon.tera_type.name.lower()
                                         and tuple(t.lower() for t in em.types) == S.dex_types(mon.species))
                    elif ours:
                        c["own"] += 1
                        c["own ok"] += em.tera_type.lower() == our_tera.get(key)

    common = dict(battle_format="gen9randombattle", server_configuration=local_server(8000),
                  max_concurrent_battles=3, log_level=40)
    tag = random.randint(0, 10**6)
    player = Check(None, prior_weight=0, eval_config="stock", ms=100, samples=1, workers=3,
                   account_configuration=AccountConfiguration(f"g9t{tag}", None), **common)
    opponent = BASELINES["heuristic"](account_configuration=AccountConfiguration(f"g9h{tag}", None), **common)
    asyncio.run(player.battle_against(opponent, n_battles=6))
    assert player.n_finished_battles == 6
    assert player.stats["fallbacks"] == 0 and player.stats["failed_samples"] == 0
    assert c["own"] > 0 and c["own ok"] == c["own"]
    assert c["tera ok"] == c["tera"]
    assert c["tera chosen"] > 0
