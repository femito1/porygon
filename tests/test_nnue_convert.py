"""porygon.nnue.convert_full (NNUE gate 1, step 2): dumped fastenv states -> pe.State (side one = the side to move).
gen4 on the laptop (its engine build); gen9 runs on the cluster (scripts/fastenv/nnue_parity.py)."""

import os
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
ROOT = Path(__file__).resolve().parents[1]
GEN4 = os.environ.get("PORYGON_GEN", "4") == "4"


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    from tests.test_nnue_dump import dump_games
    path = tmp_path_factory.mktemp("nnue") / "g4.jsonl"
    _, lines = dump_games(path, fmt="gen4randombattle", games=2, seed=11)
    return path, lines


@pytest.mark.skipif(not GEN4, reason="laptop engine build is gen4")
def test_convert_full_states(dump):
    from porygon.nnue.convert_full import convert_full, state_dict
    _, lines = dump
    states = [x for x in lines if x["t"] == "S"]
    assert len(states) > 50
    for x in states:
        st = x["state"]
        d = state_dict(convert_full(st))
        me, opp = st["sides"][x["side"]], st["sides"][1 - x["side"]]
        for side, src in (("side_one", me), ("side_two", opp)):
            mons = d[side]["pokemon"]
            assert [m["id"] for m in mons[:len(src["mons"])]] == [m["species"] for m in src["mons"]]
            assert [m["hp"] for m in mons[:6]] == [0 if m["fainted"] else m["hp"] for m in src["mons"]]
            assert int(d[side]["active_index"]) == max(src["active"], 0)
        assert d["side_one"]["force_switch"] == bool(me["request"]["forceSwitch"])
        assert d["side_two"]["force_switch"] is False
