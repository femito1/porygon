"""fastenv host.js --dump-states (NNUE gate 1, step 1): one state line per decision row, one game line per battle;
`e` points at the side's request (or error) message in its client stream. Needs node + a built pokemon-showdown."""

import json
import shutil
from collections import Counter

import numpy as np
import pytest

from porygon.fastenv import FastEnv, random_actions

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def dump_games(path, fmt="gen9randombattle", games=4, seed=7, streams=True, battles=2):
    """Plays `games` random-action battles on one worker, dumping to `path`; returns (decision rows, lines)."""
    rng = np.random.default_rng(seed)
    rows = Counter()
    with FastEnv(fmt, n_workers=1, battles_per_worker=battles, seed=seed, encoder="stub",
                 dump_states=str(path), dump_streams=streams) as env:
        b, done = env.reset(), 0
        while done < games:
            rows.update(int(s) for s in b.slot[b.needs_action])
            done += int(b.done.sum()) // 2
            b = env.step(random_actions(b.mask[b.needs_action], rng))
        rows.update(int(s) for s in b.slot[b.needs_action])  # the last batch's rows are dumped too
    lines = [json.loads(x) for x in open(path)]
    return rows, lines


def test_dump_states_lines(tmp_path):
    rows, lines = dump_games(tmp_path / "d.jsonl")
    states = [x for x in lines if x["t"] == "S"]
    games = {x["game"]: x for x in lines if x["t"] == "G"}
    assert len(games) >= 4
    assert Counter(x["slot"] for x in states) == rows, "one state line per decision row"
    for game, g in games.items():
        assert g["inputlog"][0].startswith(">start ") and sum(x.startswith(">player ") for x in g["inputlog"]) == 2
        for side in (0, 1):
            mine = [x for x in states if x["game"] == game and x["side"] == side]
            assert [x["idx"] for x in mine] == list(range(len(mine)))
            stream = g["streams"][side]
            for x in mine:
                msg = stream[x["e"]]
                assert msg.startswith("|request|") or msg.startswith("|error|"), msg[:40]
                st = x["state"]
                assert st["side"] == side and st["turn"] == x["turn"]
                me = st["sides"][side]
                assert me["request"] is not None and not me["request"]["wait"]
                assert len(me["mons"]) == 6 and (me["active"] >= 0 or me["request"]["forceSwitch"])
                for s in st["sides"]:
                    if s["active"] >= 0:
                        a = s["mons"][s["active"]]
                        assert a["active"] or a["fainted"]  # a fainted active stays in its slot until replaced
                    for m in s["mons"]:
                        assert 0 <= m["hp"] <= m["maxhp"] and len(m["moves"]) <= 4 and m["level"] > 0
    json.dumps(states[0])  # plain JSON (no cycles)


def test_dump_without_streams(tmp_path):
    _, lines = dump_games(tmp_path / "d.jsonl", games=2, streams=False)
    assert all("streams" not in x for x in lines if x["t"] == "G")
