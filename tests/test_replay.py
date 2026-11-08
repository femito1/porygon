"""porygon.replay: first-person views from logged message streams and from re-simulated input logs.

Fixtures (tests/fixtures/):
  live_streams.json.gz   one gen9 and one gen4 battle played live on a local server by two random poke-env players
                         that logged every websocket message and, at each decision, the live mask, action and a
                         SHA-1 of the live observation (score_reconstruction.py record-live)
  resim_battles.jsonl    two gen9 and one gen4 battle made with Showdown's simulator at the commit in their
                         `>version` line (random choices, some tera): input log + spectator log (tera, U-turn /
                         Flip Turn, faints with forced switches, sleep `cant`)
The re-simulation test needs node and a Showdown build at that commit (the repo's pokemon-showdown/ or
PORYGON_SHOWDOWN_DIR); it is skipped otherwise.
"""

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from porygon import replay
from porygon.encoder import OBS_DIM, embed_battle

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"


def test_live_parity():
    """Replaying a live player's logged stream reproduces its live decisions exactly (obs, mask, action)."""
    games = json.load(gzip.open(FIX / "live_streams.json.gz", "rt"))
    for fmt, game in games.items():
        gen = int(fmt[3])
        for player, data in game["players"].items():
            rep = replay.replay_stream(data["stream"], player, gen, encoder=embed_battle, battle_tag=game["tag"])
            assert rep.finished
            pairs = list(replay.align_logged(rep.decisions, data["decisions"]))
            assert len(pairs) == len(data["decisions"]) > 5
            for ours, live in pairs:
                assert ours.mask.tolist() == live["mask"]
                assert ours.mask[live["action"]] == 1
                x = np.asarray(ours.obs, dtype=np.float32).tobytes()
                assert hashlib.sha1(x).hexdigest() == live["obs_sha1"], (fmt, player, ours.turn)


def test_parse_choice():
    assert replay.parse_choice("move 2 terastallize") == ("move", "2", "terastallize")
    assert replay.parse_choice("move knockoff") == ("move", "knockoff", "")
    assert replay.parse_choice("switch 3") == ("switch", "3", "")
    assert replay.parse_choice("move conversion2") == ("move", "conversion2", "")
    assert replay.parse_choice("move 1 2") == ("move", "1", "")
    assert replay.parse_choice("default") == ("default", "", "")


def showdown_dir(version: str) -> Path | None:
    cands = [os.environ.get("PORYGON_SHOWDOWN_DIR"), ROOT / "pokemon-showdown"]
    try:  # a git worktree: the main checkout's pokemon-showdown
        common = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], capture_output=True,
                                text=True).stdout.strip()
        cands.append(Path(common).resolve().parent / "pokemon-showdown")
    except OSError:
        pass
    for c in cands:
        if not c or not (Path(c) / "dist" / "sim" / "battle-stream.js").exists():
            continue
        head = subprocess.run(["git", "-C", str(c), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        if head.strip() == version:
            return Path(c)
    return None


def test_resim_roundtrip():
    """Re-simulating the fixture input logs reproduces the stored logs; every chosen action is legal."""
    recs = [json.loads(line) for line in open(FIX / "resim_battles.jsonl")]
    version = recs[0]["inputlog"].split("\n")[0].split()[1]
    ps = showdown_dir(version)
    if shutil.which("node") is None or ps is None:
        pytest.skip(f"needs node and a Showdown build at {version[:8]}")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / version).symlink_to(ps)
        out = subprocess.run(["node", str(ROOT / "scripts" / "resim_inputlog.js"), "--builds", tmp,
                              str(FIX / "resim_battles.jsonl")], capture_output=True, text=True, check=True).stdout
    results = [json.loads(line) for line in out.splitlines()]
    assert [r["id"] for r in results] == [r["id"] for r in recs]
    seen = {"tera": 0, "forced": 0, "turn": 0}
    for r in results:
        assert r["ok"] and r["exact"], r.get("diff")
        assert len(r["teams"]["p1"]) == len(r["teams"]["p2"]) == 6
        gen = int(r["id"][3])
        sides = replay.resim_sides(r, gen, embed_battle)
        for side, sr in sides.items():
            assert sr.finished and sr.won == (r["winner"] == side)
            for d in sr.decisions:
                seen[d.kind] += 1
                if d.choice is None:  # the battle ended before this request was answered
                    continue
                assert d.action >= 0 and d.legal, (r["id"], side, d.turn, d.choice, d.mask)
                seen["tera"] += d.action >= replay.TERA_OFFSET
                assert d.obs.shape == (OBS_DIM,)
    assert seen["tera"] > 0 and seen["forced"] > 0 and seen["turn"] > 100
