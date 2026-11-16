"""NNUE gate 1, step 4: porygon.nnue.features (State -> student inputs), porygon.nnue.student (torch model = numpy
forward), the feature-parity script and the report's pure parts. gen4 on the laptop (its engine build); the gate's
gen9 runs are on the cluster (scripts/fastenv/nnue_gate1.sh)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tests.test_nnue_convert import GEN4, ROOT, dump, pytestmark  # noqa: F401  (module fixture, node skip)

sys.path.insert(0, str(ROOT / "scripts" / "fastenv"))


@pytest.fixture(scope="module")
def states(dump):
    import poke_engine as pe
    from porygon.nnue.convert_full import convert_full
    _, lines = dump
    return [pe.State.from_string(convert_full(x["state"]).to_string()) for x in lines if x["t"] == "S"]


@pytest.mark.skipif(not GEN4, reason="laptop engine build is gen4")
def test_features_shapes_and_invariants(states):
    from porygon.nnue import features as F
    names = F.column_names()
    assert len(names) == len(set(names))
    for st in states[:200]:
        f = F.features(st)
        assert f["ids"].shape == (12, 8) and f["mon"].shape == (12, F.F_MON)
        assert f["side"].shape == (2, len(F.side_cols())) and f["field"].shape == (F.F_FIELD,)
        assert len(F.flat(f)) == len(names)
        assert np.array_equal(F.flat(f), F.flat(F.features(st)))  # deterministic
        for s in (0, 6):  # the active mon comes first; a fainted mon is all zeros but its fainted bit
            assert f["mon"][s, F.MON_COLS.index("active")] == 1 or f["mon"][s, F.MON_COLS.index("fainted")] == 1
        dead = f["mon"][:, F.MON_COLS.index("fainted")] == 1
        assert not f["ids"][dead].any()
        assert f["field"][:len(F.WEATHERS)].sum() == 1 and f["field"][len(F.WEATHERS):-1].sum() == 1
    assert "vol_lockedmove" not in F.side_cols()  # untracked by poke-env (main's parity condition)
    assert not any(c.startswith("weather_turns") or c.startswith("terrain_turns") for c in F.FIELD_COLS)


@pytest.mark.skipif(not GEN4, reason="laptop engine build is gen4")
def test_base_species_formes():
    from porygon.nnue.features import base_species
    assert base_species("ROTOMWASH") == base_species("ROTOM") == "ROTOM"  # gen4 dex (gen9: MORPEKOHANGRY, MINIORMETEOR)
    assert base_species("DEOXYSATTACK") == "DEOXYS"


@pytest.mark.skipif(not GEN4, reason="laptop engine build is gen4")
def test_student_numpy_matches_torch(states, tmp_path):
    torch = pytest.importorskip("torch")
    from porygon.nnue import features as F
    from porygon.nnue.student import NumpyStudent, Student, n_params
    spec = F.spec()
    net = Student(spec)
    assert sum(p.numel() for p in net.parameters()) == n_params(spec)
    assert 50_000 <= n_params(spec) <= 200_000  # PREREG
    fs = [F.features(st) for st in states[:64]]
    B = lambda k: np.stack([f[k] for f in fs])  # noqa: E731
    with torch.no_grad():
        lt = net(torch.as_tensor(B("ids")).long(), torch.as_tensor(B("mon")), torch.as_tensor(B("side")),
                 torch.as_tensor(B("field"))).numpy()
    NumpyStudent.save(str(tmp_path / "s.npz"), net.numpy_weights(), spec)
    ns = NumpyStudent.load(str(tmp_path / "s.npz"))
    assert np.abs(ns.logits(B("ids"), B("mon"), B("side"), B("field")) - lt).max() < 1e-4
    assert abs(ns.p_win(fs[5]) - 1 / (1 + np.exp(-lt[5]))) < 1e-5


def test_report_auc_and_levels():
    from lookahead2_study import _auc
    from nnue_report import auc, criteria, level
    r = np.random.default_rng(3)
    s, y = np.round(r.normal(size=2000), 1), (r.random(2000) < 0.4).astype(float)
    assert abs(auc(s, y) - _auc(s, y)) < 1e-12
    m = lambda st, tt, et: {"student": {"auc": st[0], "ll_cal": st[1]}, "teacher": {"auc": tt[0], "ll_cal": tt[1]},  # noqa: E731
                            "eval": {"auc": et[0], "ll_cal": et[1]}}
    good = criteria(m((0.745, 0.585), (0.75, 0.58), (0.72, 0.61)))
    weak = criteria(m((0.73, 0.59), (0.75, 0.57), (0.71, 0.61)))   # c2 met, c1 not
    bad = criteria(m((0.725, 0.60), (0.75, 0.57), (0.72, 0.61)))
    assert good["c1"] and good["c2"] and weak["c2"] and not weak["c1"] and not bad["c2"]
    assert level(good, good) == "PASS" and level(good, weak) == "PARTIAL" and level(good, bad) == "FAIL"


@pytest.mark.skipif(not GEN4, reason="laptop engine build is gen4")
def test_feature_parity_script(tmp_path):
    """Re-play mode on 4 gen4 games, then the same-games check against its own dumps."""
    env = dict(os.environ, PORYGON_GEN="4", PYTHONPATH=str(ROOT / "src"))
    cmd = [sys.executable, str(ROOT / "scripts/fastenv/nnue_feature_parity.py"), "--format", "gen4randombattle",
           "--play", "4", "--workers", "2", "--seed", "5", "--min-games", "1", "--min-decisions", "20", "--no-verdict"]
    r = subprocess.run(cmd + ["--dump-dir", str(tmp_path / "a"), "--out", str(tmp_path / "A.txt")], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run(cmd + ["--dump-dir", str(tmp_path / "b"), "--out", str(tmp_path / "B.txt"), "--same-as",
                              str(tmp_path / "a" / "dump-*.jsonl")], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    text = (tmp_path / "B.txt").read_text()
    assert "same games as" in text and ": yes (" in text
    d = json.loads((tmp_path / "B.json").read_text())
    assert d["decisions"] >= 20 and not any(k.endswith(".hp") for k in d["column_mismatch"])  # seen HP = convert's
    lines = [json.loads(x) for x in open(tmp_path / "a" / "dump-0.jsonl")]
    seen = [m.get("seen_hp", "absent") for x in lines if x["t"] == "S" for m in x["state"]["sides"][1 - x["side"]]["mons"]]
    assert "absent" not in seen and any(v is None for v in seen) and any(isinstance(v, (int, float)) for v in seen)
