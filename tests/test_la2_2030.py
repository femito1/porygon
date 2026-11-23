"""LA2-2030 (docs/netsearch/LA2-2030-PREREG.md): the F2030 gate, the stage-1 reading, the FP200 measurement text and
its Foul Play iteration check (scripts/fastenv/lookahead2_adopt.py), foulplay_instances.sh's fixed-iteration launch,
and the chain scripts/fastenv/la2_2030.sbatch end to end in its laptop mock (LA22_MOCK): WITHDRAWN cases, the reading
at lower 95% bounds 52.9 / 50.1 / 48.0 / 46.9%, FP200 only after ADOPTED, the one automatic replay of an
infrastructure VOID, an iteration-check failure. No games, no Slurm."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scripts/fastenv/la2_2030.sbatch"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AD = _load("lookahead2_adopt", "scripts/fastenv/lookahead2_adopt.py")

# RESULTS.txt as netv2's scripts/netv2_conf2030/conf2030.sbatch writes it (the real 11-23 file's lines 1-2)
HEAD = ("F2030: G9q-cd@2030 1000-game confirmation vs Foul Play gen9 at a FIXED 200000 iterations per state "
        "(runs/g9q-cd2030/PREREG.txt 7a68bf3e), job 1552777; net 02035041391.pt md5 55940b3f; measurement, reading per "
        "the PREREG")
L2 = ("vs Foul Play gen9, fixed 200000 iterations per 1000 ms state [wsp fp9_iters 82a207a6, seed 8203222 (new)]: "
      "/vol/bitbucket/fr1226/porygon-ws-wsp/fpcal/runs/foulplay-net-20261123-130543-1552777-200000: {p}% ± 2.9% (500 "
      "duplicate pairs, 1000 games; won both 175, split 258, lost both 67; per-game {p}% ± 3.0%)")
L3 = "infrastructure voids replayed: 0"
RESULTS = {
    "CONFIRMED": [HEAD, L2.format(p="60.8") + " => [57.9, 63.7] => CONFIRMED (lower 95% bound > 50%)", L3],
    "NOT CONFIRMED": [HEAD, L2.format(p="51.0") + " => [48.1, 53.9] => NOT CONFIRMED (lower 95% bound <= 50%)", L3],
    "FAILED": [HEAD, "vs Foul Play gen9, fixed 200000 iterations per 1000 ms state [wsp fp9_iters 82a207a6, seed "
               "8203222 (new)]: FAILED (two infrastructure voids)", L3],
    "missing": None,
}
LOWS = {0.529: AD.BETTER, 0.501: AD.BETTER, 0.480: AD.NON_INFERIOR, 0.469: AD.NOT_ADOPTED}
ITERS_LINE = "2026-11-23T14:10:40Z all iteration counts OK (1000 single-option 39, N 1300, N/2 252, decisions 366)\n"


def write_results(path: Path, case: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS[case] is not None:
        path.write_text("\n".join(RESULTS[case]) + "\n")


def fake_result(run: Path, lo: float, pairs: int = 500, void: bool = False, harness_fail: bool = False,
                fp: bool = False, iters: str | None = ITERS_LINE) -> None:
    """A result.json with the fields netsearch_slot.py report writes; fp: a Foul Play opponent (no opponent stats)
    with 5 blocks, each with ITERS_OK (iters) or none (iters None)."""
    run.mkdir(parents=True, exist_ok=True)
    hw = 0.029
    nb = 5 if pairs >= 5 else 1
    arm = {"arm": 1, "blocks": nb, "pairs": pairs, "games": 2 * pairs, "incomplete_pairs": 0, "mean": lo + hw, "hw": hw,
           "lo": lo, "hi": lo + 2 * hw, "won_both": 150, "lost_both": 80, "split": pairs - 230, "seconds": 3.3 * 3600,
           "stats_a": {"decisions": 30000, "searched": 25000, "fallbacks": 3, "leaves": 9e7, "failed_worlds": 0},
           "stats_b": {} if fp else {"decisions": 30000, "searched": 25000, "fallbacks": 2, "leaves": 9e7,
                                     "failed_worlds": 0},
           "gate": {"decisions": 25000, "overrides": 4500},
           "gate_opp": {"decisions": 0, "overrides": 0} if fp else {"decisions": 25000, "overrides": 4400},
           "per_block": [{"block": f"block-{j}", "pairs": pairs // nb, "seconds": 2400} for j in range(1, nb + 1)],
           "spec": "lookahead:NET2035,margin=0.0075,temperature=0.5,worlds=8,workers=11,leaf_device=cuda",
           "margin": 0.0075, "stopped": False, "seq_decision": None, "fallback_rate_a": 1e-4,
           "fallback_rate_b": None if fp else 7e-5, "override_rate": 0.18, "override_rate_opp": None if fp else 0.172,
           "void": void}
    r = {"run": str(run), "name": "la2fp200" if fp else "la2x2030", "tag": "8233792" if fp else "8233571",
         "opponent": "foulplay:ms=1000,parallelism=1,iters=200000" if fp else "lookahead:NET1355,margin=0.0075,...",
         "arms": [arm]}
    (run / "result.json").write_text(json.dumps(r))
    if fp:
        for j in range(1, nb + 1):
            b = run / "arm-1" / f"block-{j}"
            b.mkdir(parents=True, exist_ok=True)
            (b / "results.jsonl").write_text('{"status": "win"}\n' * 2 * (pairs // nb))
            if iters is not None:
                (b / "ITERS_OK").write_text(iters)
    if harness_fail:
        (run / "HARNESS_FAIL").touch()


def void_run(run: Path) -> None:
    """An FP200 attempt that never started: HARNESS_FAIL, 0 games, handshake timeouts (as netsearch_slot.sh logs it)."""
    (run / "arm-1/block-1/fp").mkdir(parents=True)
    (run / "HARNESS_FAIL").touch()
    (run / "run.log").write_text(
        "2026-11-24T02:00:00Z Foul Play instance fp1a1b1x0 did not start: websockets.exceptions.InvalidMessage "
        "TimeoutError: timed out during handshake restarting (exit 1)\n")


# ---------------------------------------------------------------- the helper

@pytest.mark.parametrize("case", list(RESULTS))
def test_gate_f2030(tmp_path, case):
    p = tmp_path / "RESULTS.txt"
    write_results(p, case)
    ok, why = AD.gate(p, "55940b3f", "F2030")
    assert ok == (case == "CONFIRMED"), why
    want = {"CONFIRMED": "F2030 CONFIRMED:", "NOT CONFIRMED": "F2030 NOT CONFIRMED:", "FAILED": "F2030 FAILED:",
            "missing": "F2030 wrote no RESULTS.txt"}[case]
    assert want in why


def test_gate_f2030_other_net(tmp_path):
    p = tmp_path / "RESULTS.txt"
    p.write_text("\n".join(RESULTS["CONFIRMED"]).replace("md5 55940b3f", "md5 6cf40c0c") + "\n")
    ok, why = AD.gate(p, "55940b3f", "F2030")
    assert not ok and "does not name the net md5 55940b3f" in why


def test_iters_counts():
    c = AD.iters_counts(ITERS_LINE)
    assert c == {"1000 single-option": 39, "N": 1300, "N/2": 252, "decisions": 366}
    assert AD.iters_counts("all iteration counts OK ()") == {}


def test_fp200_text(tmp_path):
    run = tmp_path / "run"
    fake_result(run, 0.571, fp=True)
    text = AD.fp200(run, "0.0075", "G9q-cd@2035.04M", 200000, "8233792")
    l1 = text.splitlines()[0]
    assert l1.startswith("LA2-FP200: lookahead2(G9q-cd@2035.04M, δ* 0.0075, K 8, T 0.5) vs Foul Play gen9 at a FIXED "
                         "200000 iterations per state: 60.00% ± 2.90% [57.10%, 62.90%] (500 duplicate pairs, 1000 games)")
    assert "no decision rule" in l1
    assert "iteration check (wsp fp_iters_check.py, every block): OK (1000 single-option 195, N 6500, N/2 1260, " \
           "decisions 1830)" in text


@pytest.mark.parametrize("bad", ["no_iters", "mismatch_block", "mismatch_run", "pairs", "harness", "fallbacks", "no_n"])
def test_fp200_void(tmp_path, bad):
    run = tmp_path / "run"
    fake_result(run, 0.571, fp=True, pairs=449 if bad == "pairs" else 500, void=bad == "fallbacks",
                harness_fail=bad == "harness", iters=None if bad == "no_iters" else
                ("all iteration counts OK (decisions 3)\n" if bad == "no_n" else ITERS_LINE))
    if bad == "mismatch_block":
        (run / "arm-1/block-3/ITERS_MISMATCH").write_text("expected 200000 ...")
    if bad == "mismatch_run":
        (run / "ITERS_MISMATCH").write_text("arm 1 block 2: expected 200000 ...")
    assert AD.fp200(run, "0.0075", "x", 200000, "8233792").startswith("VOID: ")


def test_smoke_fp(tmp_path):
    run = tmp_path / "run"
    fake_result(run, 0.5, pairs=2, fp=True)
    ok, line = AD.smoke(run, 4, fp=True)
    assert ok and line.startswith("PASS") and "Foul Play iteration check ok" in line and "opponent" not in line
    ok, line = AD.smoke(run, 4)  # the h2h smoke's checks would demand opponent search stats
    assert not ok
    (run / "arm-1/block-1/ITERS_OK").unlink()
    ok, line = AD.smoke(run, 4, fp=True)
    assert not ok and "Foul Play iteration check NO" in line


# ---------------------------------------------------------------- foulplay_instances.sh: the fixed-iteration launch

def test_foulplay_instances_wrapper(tmp_path):
    """With FPI_WRAP / FPI_ITERS, run.py starts as `python FPI_WRAP --iters-per-s N run.py ...` (wsp's launch)."""
    fp, venv, logs = tmp_path / "fp", tmp_path / "venv", tmp_path / "logs"
    (venv / "bin").mkdir(parents=True)
    fp.mkdir()
    logs.mkdir()
    py = venv / "bin/python"
    py.write_text('#!/bin/sh\necho "ARGV $*"\necho "Waiting for a challenge"\nsleep 30\n')
    py.chmod(0o755)
    env = dict(os.environ, FPI_WRAP="/x/fp_fixed_iters.py", FPI_ITERS="200000")
    p = subprocess.Popen(["setsid", "bash", str(ROOT / "scripts/fastenv/foulplay_instances.sh"), str(fp), str(venv), "",
                          "8000", "gen9randombattle", "1000", "1", str(logs), "fpx0"], env=env)
    try:
        for _ in range(100):
            if (logs / "fpx0.log").exists() and "Waiting" in (logs / "fpx0.log").read_text():
                break
            time.sleep(0.1)
        text = (logs / "fpx0.log").read_text()
    finally:
        os.killpg(p.pid, 9)
        p.wait()
    assert "ARGV /x/fp_fixed_iters.py --iters-per-s 200000 run.py --websocket-uri ws://localhost:8000/showdown/websocket" \
           in text and "--search-time-ms 1000 --search-parallelism 1" in text


# ---------------------------------------------------------------- the chain, mocked end to end

def md5(b: bytes) -> str:
    return subprocess.run(["md5sum"], input=b, capture_output=True).stdout.decode()[:32]


def mock_dir(tmp_path: Path, case: str, lo: float | None = None) -> Path:
    m = tmp_path / "mock"
    s = m / "fastenv/runs/la2-2030"
    (s / "code/scripts/fastenv").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/fastenv/lookahead2_adopt.py", s / "code/scripts/fastenv/")
    (s / "code/COMMIT").write_text("mock\n")
    (s / "PREREG.txt").write_text("mock prereg\n")
    (s / "PREREG.txt").chmod(0o444)
    (s / "PREREG.md5").write_text(md5(b"mock prereg\n") + "\n")
    write_results(m / "netv2/runs/conf2030/RESULTS.txt", case)
    (m / "wsp").mkdir()
    for f in ("fp_fixed_iters.py", "fp_iters_check.py", "fp9_iters.sbatch"):
        (m / "wsp" / f).write_text(f"stand-in {f}\n")
    (m / "DELTA_IN").write_text("0.0075\n")
    if lo is not None:
        fake_result(m / "run", lo)
    return m


def run_chain(m: Path, wrong_md5: bool = False) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(LA22_MOCK=str(m), LA22_MOCK_PY=sys.executable,
               LA22_MOCK_FWRAP_MD5=md5(b"stand-in fp_fixed_iters.py\n")[:8],
               LA22_MOCK_FCHK_MD5=md5(b"stand-in fp_iters_check.py\n")[:8],
               LA22_MOCK_FPI_MD5="00000000" if wrong_md5 else md5(b"stand-in fp9_iters.sbatch\n")[:8])
    return subprocess.run(["bash", str(CHAIN)], env=env, capture_output=True, text=True, timeout=120)


def outbox(m: Path) -> list[str]:
    return (m / "fastenv/LEDGER-OUTBOX.md").read_text().splitlines()


needs_tools = pytest.mark.skipif(not (shutil.which("flock") and shutil.which("md5sum") and shutil.which("setsid")),
                                 reason="flock / md5sum / setsid")


@needs_tools
@pytest.mark.parametrize("case", ["NOT CONFIRMED", "FAILED", "missing"])
def test_chain_withdrawn(tmp_path, case):
    m = mock_dir(tmp_path, case)
    p = run_chain(m)
    s = m / "fastenv/runs/la2-2030"
    assert p.returncode == 0, p.stdout + p.stderr
    w = (s / "WITHDRAWN").read_text()
    assert w.startswith("WITHDRAWN: ") and "F2030" in w.splitlines()[0] and "nothing played" in w
    assert not (s / "DELTA").exists() and not (s / "VERDICT.txt").exists() and not (s / "FAILED").exists()
    out = outbox(m)
    assert len(out) == 1 and " | fastenv | RELEASE | mock | LA2-2030 WITHDRAWN (" in out[0]
    assert out[0].endswith("slot FREE") and out[0].count(" | ") == 4


@needs_tools
@pytest.mark.parametrize("lo", list(LOWS))
def test_chain_reading_and_fp200(tmp_path, lo):
    m = mock_dir(tmp_path, "CONFIRMED", lo)
    fake_result(m / "fprun-1", 0.571, fp=True)
    p = run_chain(m)
    s = m / "fastenv/runs/la2-2030"
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (s / "WITHDRAWN").exists() and not (s / "FAILED").exists()
    v1 = (s / "VERDICT.txt").read_text().splitlines()[0]
    assert v1.startswith(LOWS[lo] + ": ") and not os.access(s / "VERDICT.txt", os.W_OK)
    assert "lookahead2(G9q-cd@2035.04M, δ* 0.0075) vs lookahead2(G9q-cd@1355.04M, δ 0.0075)" in v1
    res = (s / "RESULT").read_text().splitlines()
    out = outbox(m)
    assert len(out) == 1 and f" | fastenv | RELEASE | mock | LA2-2030 done: {LOWS[lo]}: " in out[0]
    assert out[0].count(" | ") == 4 and out[0].endswith("slot FREE")
    if LOWS[lo] == AD.NOT_ADOPTED:
        assert not (s / "FP200.txt").exists() and not (s / "fp-run").exists()
        assert res == [v1, "FP200 not run (stage 1 NOT ADOPTED)"] and "FP200 not run" in out[0]
    else:
        f1 = (s / "FP200.txt").read_text().splitlines()[0]
        assert f1.startswith("LA2-FP200: lookahead2(G9q-cd@2035.04M, δ* 0.0075, K 8, T 0.5) vs Foul Play gen9 at a "
                             "FIXED 200000 iterations per state: 60.00% ± 2.90% [57.10%, 62.90%]")
        assert not os.access(s / "FP200.txt", os.W_OK) and res == [v1, f1]
        assert "FP200: lookahead2(G9q-cd@2035.04M" in out[0] and "voids replayed: 0" in (s / "FP200.txt").read_text()


@needs_tools
def test_chain_fp200_void_replayed_once(tmp_path):
    m = mock_dir(tmp_path, "CONFIRMED", 0.529)
    void_run(m / "fprun-1")
    fake_result(m / "fprun-2", 0.571, fp=True)
    p = run_chain(m)
    s = m / "fastenv/runs/la2-2030"
    assert p.returncode == 0, p.stdout + p.stderr
    assert (s / "VOID-1").exists() and (s / "fp-run-void1/HARNESS_FAIL").exists() and not (s / "VOID-2").exists()
    text = (s / "FP200.txt").read_text()
    assert text.startswith("LA2-FP200: ") and "infrastructure voids replayed: 1 | " in text
    assert "replaying once, same seed 8233792" in (s / "STATUS").read_text()


@needs_tools
def test_chain_fp200_two_voids(tmp_path):
    m = mock_dir(tmp_path, "CONFIRMED", 0.529)
    void_run(m / "fprun-1")
    void_run(m / "fprun-2")
    p = run_chain(m)
    s = m / "fastenv/runs/la2-2030"
    assert p.returncode == 1 and not (s / "FP200.txt").exists()
    assert "stage fp200: two infrastructure voids" in (s / "FAILED").read_text()
    out = outbox(m)
    assert len(out) == 1 and "LA2-2030 FAILED at stage fp200" in out[0] and "stage 1 stands: ADOPTED (BETTER)" in out[0]


@needs_tools
def test_chain_fp200_not_a_void(tmp_path):
    """A harness failure that is not a 0-game handshake start (here: an iteration mismatch) is not replayed."""
    m = mock_dir(tmp_path, "CONFIRMED", 0.529)
    fake_result(m / "fprun-1", 0.571, fp=True, harness_fail=True)
    (m / "fprun-1/ITERS_MISMATCH").write_text("arm 1 block 2: expected 200000 or 100000, saw 51000\n")
    fake_result(m / "fprun-2", 0.571, fp=True)
    p = run_chain(m)
    s = m / "fastenv/runs/la2-2030"
    assert p.returncode == 1 and not (s / "VOID-1").exists()
    assert (s / "FP200.txt").read_text().startswith("VOID: ") and "stage fp200: VOID" in (s / "FAILED").read_text()
    assert len(outbox(m)) == 1


@needs_tools
def test_chain_refusals(tmp_path):
    m = mock_dir(tmp_path, "CONFIRMED", 0.529)
    p = run_chain(m, wrong_md5=True)
    s = m / "fastenv/runs/la2-2030"
    assert p.returncode == 1 and "stage checks: wsp fixed-iteration launcher md5" in (s / "FAILED").read_text()
    assert not (s / "DELTA").exists()
    p = run_chain(m)  # the stage dir now holds a chain: refused
    assert p.returncode == 1 and "refused" in outbox(m)[-1]
    m2 = mock_dir(tmp_path / "b", "CONFIRMED", 0.529)
    (m2 / "fastenv/runs/la2-2030/PREREG.md5").write_text("0" * 32 + "\n")
    p = run_chain(m2)
    assert p.returncode == 1 and "stage prereg: PREREG md5" in (m2 / "fastenv/runs/la2-2030/FAILED").read_text()
