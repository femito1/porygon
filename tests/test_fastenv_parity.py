"""Fast parity check of the JS tracker + encoder_v1 (fastenv/js) against poke-env + porygon.encoder.embed_battle on a
few in-process battles per generation (scripts/fastenv/parity_v1.py; the full check runs >= 300 battles per gen)."""
import importlib.util
import shutil
import sys
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _parity():
    spec = importlib.util.spec_from_file_location("parity_v1", ROOT / "scripts/fastenv/parity_v1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _have_showdown() -> bool:
    if shutil.which("node") is None:
        return False
    r = subprocess.run(["node", "-e", f"require({str(ROOT / 'fastenv/js/ps.js')!r})"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _have_showdown(), reason="node / built pokemon-showdown not available")
@pytest.mark.parametrize("fmt", ["gen4randombattle", "gen9randombattle"])
def test_js_encoder_matches_poke_env(fmt):
    parity = _parity()
    proc = subprocess.run(["node", str(ROOT / "scripts/fastenv/record_battles.js"), "4", fmt, "-", "7"],
                          capture_output=True, text=True, check=True)
    st = parity.Stats()
    parity.check_lines(proc.stdout.splitlines(), st, limit=10)
    assert st.battles == 4 and st.decisions > 100
    assert parity.report(st, fmt), st.examples


NETV2 = ROOT.parent / "ws-netv2" / "src" / "porygon" / "v2" / "encoder.py"


@pytest.mark.skipif(not _have_showdown() or not NETV2.exists(), reason="needs node, Showdown and the ws-netv2 worktree")
@pytest.mark.parametrize("fmt", ["gen4randombattle", "gen9randombattle"])
def test_js_encoder_v2_matches_reference(fmt):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/fastenv/parity_v2.py"), "--record", "3", fmt,
                        "--seed", "7"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert " 0 with any mismatch" in r.stdout
