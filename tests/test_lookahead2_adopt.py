"""LA2-1350 (docs/netsearch/LA2-1350-PREREG.md): the gate, smoke checks, reading and VERDICT of
scripts/fastenv/lookahead2_adopt.py on fake inputs, and the chain scripts/fastenv/lookahead2_adopt.sbatch end to end in
its laptop mock (LA2A_MOCK): the gate's cases CONFIRMED / NOT CONFIRMED / FAILED / missing and the reading at lower
95% bounds 52.9 / 50.1 / 48.0 / 46.9%. No games, no Slurm."""
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scripts/fastenv/lookahead2_adopt.sbatch"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AD = _load("lookahead2_adopt", "scripts/fastenv/lookahead2_adopt.py")
SL = _load("netsearch_slot", "scripts/fastenv/netsearch_slot.py")

# RESULTS.txt as netv2's scripts/netv2_conf1350/conf1350.sbatch writes it (its echo lines and reading() wording)
HEAD = ("F1350: G9q-cd@1350 1000-game confirmation vs Foul Play gen9 (runs/g9q-cd1350/PREREG.txt 342e63f8), job 1238699; "
        "net 01355036714.pt md5 6cf40c0c; measurement, reading per the PREREG")
L2 = "vs Foul Play gen9 [fp9 harness, seed 8135222 (new)]: "
RUNLINE = ("/vol/bitbucket/fr1226/porygon-ws-netv2/runs/foulplay-net-20261119-140000-1238699: {p}% ± 2.9% (500 duplicate "
           "pairs, 1000 games; won both 140, split 290, lost both 70; per-game {p}% ± 3.1%)")
CTX = "context only (not a paired test): G9q-cd@805 vs Foul Play gen9 57.8% ± 2.8% [55.0, 60.6] (runs/conf800/RESULTS.txt)"
RESULTS = {
    "CONFIRMED": [HEAD, L2 + RUNLINE.format(p="56.1") + " => [53.2, 59.0] => CONFIRMED (lower 95% bound > 50%)", CTX],
    "NOT CONFIRMED": [HEAD, L2 + RUNLINE.format(p="51.0") + " => [48.1, 53.9] => NOT CONFIRMED (lower 95% bound <= 50%)",
                      CTX],
    "FAILED": [HEAD, L2 + "FAILED", CTX, "failed: /vol/bitbucket/fr1226/porygon-ws-netv2/runs/conf1350/FAILED-ii"],
    "missing": None,
}
LOWS = {0.529: AD.BETTER, 0.501: AD.BETTER, 0.480: AD.NON_INFERIOR, 0.469: AD.NOT_ADOPTED}


def write_results(path: Path, case: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS[case] is not None:
        path.write_text("\n".join(RESULTS[case]) + "\n")


def fake_result(run: Path, lo: float, pairs: int = 500, void: bool = False, harness_fail: bool = False) -> None:
    """A result.json with the fields netsearch_slot.py report writes, at a given lower bound."""
    run.mkdir(parents=True, exist_ok=True)
    hw = 0.029
    arm = {"arm": 1, "blocks": 5, "pairs": pairs, "games": 2 * pairs, "incomplete_pairs": 0, "mean": lo + hw, "hw": hw,
           "lo": lo, "hi": lo + 2 * hw, "won_both": 130, "lost_both": 110, "split": pairs - 240, "seconds": 5.5 * 3600,
           "stats_a": {"decisions": 30000, "searched": 25000, "fallbacks": 3, "leaves": 9e7, "failed_worlds": 0},
           "stats_b": {"decisions": 30000, "searched": 25000, "fallbacks": 2, "leaves": 9e7, "failed_worlds": 0},
           "gate": {"decisions": 25000, "overrides": 5000}, "gate_opp": {"decisions": 25000, "overrides": 4900},
           "per_block": [{"block": f"block-{j}", "pairs": pairs // 5, "seconds": 3960} for j in range(1, 6)],
           "spec": "lookahead:NET1355,margin=0.005,temperature=0.5,worlds=8,workers=11,leaf_device=cuda",
           "margin": 0.005, "stopped": False, "seq_decision": None, "fallback_rate_a": 1e-4, "fallback_rate_b": 7e-5,
           "override_rate": 0.2, "override_rate_opp": 0.196, "void": void}
    r = {"run": str(run), "name": "la2x1350", "tag": "8135444", "opponent": "lookahead:NET805,margin=0.005,...",
         "arms": [arm]}
    (run / "result.json").write_text(json.dumps(r))
    if harness_fail:
        (run / "HARNESS_FAIL").touch()


# ---------------------------------------------------------------- the helper

@pytest.mark.parametrize("case", list(RESULTS))
def test_gate_cases(tmp_path, case):
    p = tmp_path / "RESULTS.txt"
    write_results(p, case)
    ok, why = AD.gate(p, "6cf40c0c")
    assert ok == (case == "CONFIRMED"), why
    want = {"CONFIRMED": "F1350 CONFIRMED:", "NOT CONFIRMED": "F1350 NOT CONFIRMED:", "FAILED": "F1350 FAILED:",
            "missing": "missing"}[case]
    assert want in why


def test_gate_guards(tmp_path):
    p = tmp_path / "RESULTS.txt"
    p.write_text(RESULTS["CONFIRMED"][0] + "\n")  # no line 2
    assert AD.gate(p, "6cf40c0c") == (False, f"{p} has no line 2")
    p.write_text("\n".join(RESULTS["CONFIRMED"]).replace("md5 6cf40c0c", "md5 d0c5b6cf") + "\n")  # another net
    ok, why = AD.gate(p, "6cf40c0c")
    assert not ok and "does not name the net md5 6cf40c0c" in why
    p.write_text("\n".join([HEAD, CTX, RESULTS["CONFIRMED"][1]]) + "\n")  # CONFIRMED on line 3 only
    assert not AD.gate(p, "6cf40c0c")[0]


def test_reading():
    for lo, want in LOWS.items():
        assert AD.reading(lo) == want
    assert AD.reading(0.50) == AD.NON_INFERIOR and AD.reading(0.47) == AD.NOT_ADOPTED  # strict >
    assert AD.reading(math.nan) == AD.NOT_ADOPTED


@pytest.mark.parametrize("lo", list(LOWS))
def test_verdict_line1(tmp_path, lo):
    fake_result(tmp_path / "run", lo)
    text = AD.verdict(tmp_path / "run", "0.0075", "G9q-cd@1355.04M", "G9q-cd@805.03M")
    line1, line2 = text.splitlines()[:2]
    assert line1.startswith(LOWS[lo] + ": lower 95% ")
    assert (LOWS[lo] == AD.NOT_ADOPTED) == ("stays lookahead2 on G9q-cd@805.03M" in line2)
    assert f"lower 95% {lo:.3%}" in line1 and "δ* 0.0075" in line1 and "500 pairs" in line1


@pytest.mark.parametrize("kw", [{"pairs": 449}, {"void": True}, {"harness_fail": True}])
def test_verdict_void(tmp_path, kw):
    fake_result(tmp_path / "run", 0.529, **kw)
    assert AD.verdict(tmp_path / "run", "0.005", "a", "b").startswith("VOID: ")


def test_verdict_on_a_real_report(tmp_path):
    """The fields verdict() reads are the ones netsearch_slot.py report writes (synthetic blocks)."""
    run = tmp_path / "run"
    (run / "plan.env").parent.mkdir(parents=True)
    (run / "plan.env").write_text("NAME=la2x1350\nTAG=8135444\nOPP='lookahead:B,margin=0.005'\n")
    (run / "arms.txt").write_text("lookahead:A,margin=0.005\n")
    for j in range(1, 6):
        d = run / "arm-1" / f"block-{j}"
        d.mkdir(parents=True)
        with open(d / "battles.jsonl", "w") as bf, open(d / "results.jsonl", "w") as rf:
            for i in range(100):
                for g in range(2):
                    won = (i + g + j) % 5 < 3  # 60% wins
                    bf.write(json.dumps({"pair": i, "game": g, "winner": "pax" if won else "pbx", "p1": "pax",
                                         "p2": "pbx"}) + "\n")
                    rf.write(json.dumps({"a": "pax", "status": "win" if won else "loss"}) + "\n")
        (d / "summary.json").write_text(json.dumps({"seconds": 3600, "search_stats_a": {"decisions": 100, "fallbacks": 0},
                                                    "search_stats_b": {"decisions": 100, "fallbacks": 0}}))
        (d / "gate.jsonl").write_text(json.dumps({"override": True, "gap": 0.01}) + "\n")
        (d / "gate-opp.jsonl").write_text(json.dumps({"override": False, "gap": 0.0}) + "\n")
    r = SL.report(run, 0.515, 0.01, 0.02)
    (run / "result.json").write_text(json.dumps(r, default=str))
    text = AD.verdict(run, "0.005", "A", "B")
    assert text.splitlines()[0].startswith(AD.reading(r["arms"][0]["lo"]) + ": ") and r["arms"][0]["pairs"] == 500


def test_smoke_checks(tmp_path):
    run = tmp_path / "run"
    fake_result(run, 0.5, pairs=2)
    r = json.loads((run / "result.json").read_text())
    r["arms"][0]["games"] = 4
    (run / "result.json").write_text(json.dumps(r))
    ok, line = AD.smoke(run, 4)
    assert ok and line.startswith("PASS")
    r["arms"][0]["stats_b"]["leaves"] = 0  # the benchmark bot never searched
    (run / "result.json").write_text(json.dumps(r))
    ok, line = AD.smoke(run, 4)
    assert not ok and "arm 1 opponent searched/leaves > 0 NO" in line


# ---------------------------------------------------------------- the chain, mocked end to end

def mock_dir(tmp_path: Path, case: str, lo: float | None = None) -> Path:
    m = tmp_path / "mock"
    s = m / "fastenv/runs/la2-1350"
    (s / "code/scripts/fastenv").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/fastenv/lookahead2_adopt.py", s / "code/scripts/fastenv/")
    (s / "code/COMMIT").write_text("mock\n")
    (s / "PREREG.txt").write_text("mock prereg\n")
    (s / "PREREG.txt").chmod(0o444)
    md5 = subprocess.run(["md5sum"], input=b"mock prereg\n", capture_output=True).stdout.decode()[:32]
    (s / "PREREG.md5").write_text(md5 + "\n")
    write_results(m / "netv2/runs/conf1350/RESULTS.txt", case)
    (m / "DELTA_IN").write_text("0.0075\n")
    if lo is not None:
        fake_result(m / "run", lo)
    return m


def run_chain(m: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(LA2A_MOCK=str(m), LA2A_MOCK_PY=sys.executable)
    return subprocess.run(["bash", str(CHAIN)], env=env, capture_output=True, text=True, timeout=120)


needs_tools = pytest.mark.skipif(not (shutil.which("flock") and shutil.which("md5sum")), reason="flock / md5sum")


@needs_tools
@pytest.mark.parametrize("case", ["NOT CONFIRMED", "FAILED", "missing"])
def test_chain_withdrawn(tmp_path, case):
    m = mock_dir(tmp_path, case)
    p = run_chain(m)
    s = m / "fastenv/runs/la2-1350"
    assert p.returncode == 0, p.stdout + p.stderr
    w = (s / "WITHDRAWN").read_text()
    assert w.startswith("WITHDRAWN: ") and "nothing played" in w
    assert (s / "RESULT").read_text().startswith("WITHDRAWN: ")
    assert not (s / "DELTA").exists() and not (s / "VERDICT.txt").exists() and not (s / "FAILED").exists()
    out = (m / "fastenv/LEDGER-OUTBOX.md").read_text().splitlines()
    assert len(out) == 1 and " | fastenv | RELEASE | mock | LA2-1350 WITHDRAWN (" in out[0]
    assert out[0].endswith("slot FREE") and out[0].count(" | ") == 4


@needs_tools
@pytest.mark.parametrize("lo", list(LOWS))
def test_chain_confirmed_reading(tmp_path, lo):
    m = mock_dir(tmp_path, "CONFIRMED", lo)
    p = run_chain(m)
    s = m / "fastenv/runs/la2-1350"
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (s / "WITHDRAWN").exists() and not (s / "FAILED").exists()
    assert (s / "DELTA").read_text() == "0.0075\n"
    v1 = (s / "VERDICT.txt").read_text().splitlines()[0]
    assert v1.startswith(LOWS[lo] + ": ") and not os.access(s / "VERDICT.txt", os.W_OK)
    assert (s / "RESULT").read_text().strip() == v1
    out = (m / "fastenv/LEDGER-OUTBOX.md").read_text().splitlines()
    assert len(out) == 1 and f" | fastenv | RELEASE | mock | LA2-1350 done: {LOWS[lo]}: " in out[0]
    assert out[0].count(" | ") == 4 and "chain done" in (s / "STATUS").read_text()


@needs_tools
def test_chain_void_prereg_and_refusal(tmp_path):
    m = mock_dir(tmp_path, "CONFIRMED", 0.529)
    (m / "run/HARNESS_FAIL").touch()
    p = run_chain(m)
    s = m / "fastenv/runs/la2-1350"
    assert p.returncode == 1 and (s / "VERDICT.txt").read_text().startswith("VOID: ")
    assert "stage verdict: VOID" in (s / "FAILED").read_text()
    assert "LA2-1350 FAILED at stage verdict (VOID" in (m / "fastenv/LEDGER-OUTBOX.md").read_text()
    p = run_chain(m)  # the stage dir now holds a chain: refused, still one RELEASE line
    assert p.returncode == 1 and "refused" in (m / "fastenv/LEDGER-OUTBOX.md").read_text().splitlines()[-1]
    m2 = mock_dir(tmp_path / "b", "CONFIRMED", 0.529)
    (m2 / "fastenv/runs/la2-1350/PREREG.md5").write_text("0" * 32 + "\n")
    p = run_chain(m2)
    assert p.returncode == 1 and "stage prereg: PREREG md5" in (m2 / "fastenv/runs/la2-1350/FAILED").read_text()
    assert not (m2 / "fastenv/runs/la2-1350/DELTA").exists()
