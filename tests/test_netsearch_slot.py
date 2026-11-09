"""The slot runner's scoring, looks and Stage 2 selection (scripts/fastenv/netsearch_slot.py) on synthetic run
directories, and NetSearchPlayer's own engine pool. The runner's control flow is checked by a dry run
(NS_DRY=1 scripts/fastenv/netsearch_slot.sh scripts/fastenv/plans/dry.env OUT), the harness by the smoke test."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("netsearch_slot", HERE.parent / "scripts/fastenv/netsearch_slot.py")
SL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SL)


def block(run: Path, arm: int, j: int, outcomes: list[tuple[float, float]], name: str = "pax") -> None:
    """arm-K/block-J with one pair per outcome (s0, s1) in {0, 0.5, 1}: a server log and the client's results."""
    d = run / f"arm-{arm}" / f"block-{j}"
    d.mkdir(parents=True)
    with open(d / "battles.jsonl", "w") as bf, open(d / "results.jsonl", "w") as rf:
        for i, pair in enumerate(outcomes):
            for g, s in enumerate(pair):
                winner = name if s == 1 else ("pbx" if s == 0 else None)
                bf.write(json.dumps({"pair": i, "game": g, "winner": winner, "p1": name, "p2": "pbx"}) + "\n")
                rf.write(json.dumps({"a": name, "status": {1: "win", 0: "loss", 0.5: "tie"}[s]}) + "\n")
    (d / "summary.json").write_text(json.dumps({"seconds": 10, "search_stats_a": {"decisions": 100, "fallbacks": 1},
                                                "search_stats_b": {"decisions": 90, "fallbacks": 0}}))


def test_scoring_matches_netv2():
    nv = Path(__file__).resolve().parents[2] / "ws-netv2" / "src"
    if not (nv / "porygon" / "v2" / "evaluate.py").exists():
        pytest.skip("netv2's porygon.v2 not on this machine")
    sys.dont_write_bytecode = True  # read-only: no __pycache__ in netv2's tree
    import porygon
    if str(nv / "porygon") not in porygon.__path__:
        porygon.__path__.append(str(nv / "porygon"))
    from porygon.v2 import evaluate as E
    rows = [{"pair": p, "game": g, "winner": w, "p1": "paA", "p2": "pbB"}
            for p, g, w in [(0, 0, "paA"), (0, 1, "pbB"), (1, 0, "paA"), (1, 1, "paA"), (2, 0, None), (2, 1, "pbB"),
                            (3, 0, "paA")]]
    ours = {"paa"}
    m, g, inc = SL.pair_means(rows, ours)
    m2, g2, inc2 = E.pair_means(rows, ours, "k")
    assert sorted(m) == sorted(m2) and g == len(g2) and inc == inc2 == 1
    assert SL.interval(m) == pytest.approx(E.interval(m2))


def test_pairs_ties_and_incomplete():
    rows = [{"pair": 0, "game": 0, "winner": "pa1", "p1": "pa1", "p2": "pb1"},
            {"pair": 0, "game": 1, "winner": None, "p1": "pb1", "p2": "pa1"},
            {"pair": 1, "game": 0, "winner": "pb1", "p1": "pa1", "p2": "pb1"},
            {"pair": 5, "game": 0, "winner": "x", "p1": "q", "p2": "r"}]  # another matchup: ignored
    m, g, inc = SL.pair_means(rows, {"pa1"})
    assert m == [0.75] and g == 3 and inc == 1


def test_look_futility_done_and_nested(tmp_path):
    bad = [(0, 0)] * 40 + [(1, 0)] * 10  # 10% -> upper bound far below 50%
    block(tmp_path, 1, 1, bad)
    row, rc = SL.look(tmp_path, 1, blocks_max=4, futility=0.5)
    assert row["decision"] == "futility" and rc == SL.FUTILITY_EXIT and row["pairs"] == 50
    row, rc = SL.look(tmp_path, 1, blocks_max=4, futility=None)  # fixed n: never stops
    assert row["decision"] == "continue" and rc == 0
    good = [(1, 0)] * 30 + [(1, 1)] * 20
    block(tmp_path, 2, 1, good)
    assert SL.look(tmp_path, 2, 2, 0.5)[0]["decision"] == "continue"
    first = SL.arm_stats(tmp_path, 2, upto=1)
    block(tmp_path, 2, 2, bad)
    assert SL.arm_stats(tmp_path, 2, upto=1)["mean"] == first["mean"]  # look 1's data stay fixed
    row, rc = SL.look(tmp_path, 2, 2, 0.5)
    assert row["decision"] == "done" and rc == 0 and row["pairs"] == 100  # the last block ends the arm
    s = SL.arm_stats(tmp_path, 2)
    assert s["mean"] == pytest.approx((0.7 + 0.1) / 2) and s["stats_a"]["fallbacks"] == 2
    lines = (tmp_path / "looks.jsonl").read_text().splitlines()
    assert len(lines) == 4


def test_blocks_in_numeric_order(tmp_path):
    for j in (1, 2, 10):
        block(tmp_path, 1, j, [(1, 1)] if j == 10 else [(0, 0)])
    assert [b.name for b in SL.blocks_of(tmp_path, 1)] == ["block-1", "block-2", "block-10"]
    assert SL.arm_stats(tmp_path, 1, upto=2)["mean"] == 0


def test_select_rule():
    arm = lambda k, margin, mean, stopped=False: {"arm": k, "margin": margin, "mean": mean, "pairs": 200,
                                                  "stopped": stopped}
    s = SL.select([arm(1, 0.02, 0.530), arm(2, 0.03, 0.522), arm(3, 0.04, 0.515)], go=0.515, tie=0.01)
    assert s["selected"] == 2 and s["go"]  # 2 is within 1 point of the best and has the larger margin
    s = SL.select([arm(1, 0.02, 0.530), arm(2, 0.03, 0.519), arm(3, 0.04, 0.60, stopped=True)], 0.515, 0.01)
    assert s["selected"] == 1 and s["go"]  # a stopped arm is out; 0.519 is not within 1 point of 0.530
    s = SL.select([arm(1, 0.02, 0.51), arm(2, 0.03, 0.505)], 0.515, 0.01)
    assert s["selected"] == 2 and not s["go"]
    assert SL.select([arm(1, 0.02, 0.3, stopped=True)], 0.515, 0.01)["selected"] is None


def test_spec_margin():
    assert SL.spec_margin("netsearch:n.pt,margin=0.023,ms=1000") == 0.023
    assert SL.spec_margin("netsearch:n.pt,margin=inf") == math.inf
    assert SL.spec_margin("netsearch:n.pt,margin=-inf") == -math.inf
    assert SL.spec_margin("runs/net.pt,temperature=0.5") == math.inf


def test_report(tmp_path):
    (tmp_path / "plan.env").write_text("NAME=t\nTAG=x\nOPP='search:none'\nFUTILITY=0.5\nARMS=('a' 'b')\n")
    (tmp_path / "arms.txt").write_text("netsearch:n.pt,margin=0.02\nnetsearch:n.pt,margin=0.03\n")
    block(tmp_path, 1, 1, [(1, 0)] * 10)
    block(tmp_path, 2, 1, [(1, 1)] * 6 + [(0, 0)] * 4)
    r = SL.report(tmp_path, go=0.515, tie=0.01, void_fallbacks=0.02)
    assert [a["margin"] for a in r["arms"]] == [0.02, 0.03]
    assert r["arms"][0]["fallback_rate_a"] == pytest.approx(0.01) and not r["arms"][0]["void"]
    assert r["selection"]["selected"] == 2 and r["selection"]["go"]
    assert "GO" in SL.format_report(r)


def test_own_engine_pool():
    """A NetSearchPlayer's engine processes are its own, not porygon.search's shared pool for its worker count."""
    from porygon import netsearch as NS
    from porygon import search as S

    class FakeNet:
        gen = S.GEN_NUM

        def eval(self):
            return self

    fmt = f"gen{S.GEN_NUM}randombattle"
    p = NS.NetSearchPlayer(FakeNet(), workers=3, battle_format=fmt, start_listening=False)
    q = NS.NetSearchPlayer(FakeNet(), workers=3, own_pool=False, battle_format=fmt, start_listening=False)
    shared = S.engine_pool(3)
    assert p.pool is not shared and p.pool.n == 3 and q.pool is shared
    spec = NS.load_spec("netsearch:x.pt,margin=0.02,workers=3,own_pool=0", net_loader=lambda path, device: FakeNet(),
                        battle_format=fmt, start_listening=False)
    assert spec.pool is shared and spec.margin == 0.02
