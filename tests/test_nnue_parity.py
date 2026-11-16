"""scripts/fastenv/nnue_parity.py (NNUE gate 1, step 3) on 2 gen4 fixture games: convert_full vs
search.convert(truth=, oracle_all=True) on the replayed decisions. The gate's parity (>= 50 gen9 games) runs on the
cluster (nnue/gate1/PARITY.txt)."""

import json
import os
import subprocess
import sys

from tests.test_nnue_convert import ROOT, dump, pytestmark  # noqa: F401  (module fixture, node skip)


def test_parity_script(dump):
    """scripts/fastenv/nnue_parity.py on the fixture games: runs, compares, writes its report."""
    path, _ = dump
    out = path.parent / "PARITY.txt"
    env = dict(os.environ, PORYGON_GEN="4", PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/fastenv/nnue_parity.py"), "--dump", str(path), "--out",
                        str(out), "--format", "gen4randombattle", "--min-games", "1", "--min-decisions", "20",
                        "--no-verdict"], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    text = out.read_text()
    assert "decisions compared" in text and "side_one.pokemon.id" in text
    rates = json.loads((path.parent / "PARITY.json").read_text())
    assert rates["decisions"] >= 20
    assert rates["rates"]["side_one.pokemon.id"] == 0.0  # species are exact
