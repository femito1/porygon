"""LA2-gen4 stage 1 (docs/netsearch/LA2-GEN4-PREREG.md): scripts/fastenv/la2_gen4.py's pure parts (the split into
parity battles and study positions, the parity / δ* readings, the leaf-identity comparison, the smoke and the
GO / STOP verdict) and the slot chain scripts/fastenv/la2_gen4.sbatch end to end in its laptop mock (LA2G4_MOCK).
No games, no engine, no Slurm."""
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / "scripts/fastenv/la2_gen4.sbatch"
spec = importlib.util.spec_from_file_location("la2_gen4", ROOT / "scripts/fastenv/la2_gen4.py")
LG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LG)


def rec(game: str, side: int, idx: int = 40, all_dec: bool = True) -> dict:
    r = {"id": f"{game}-p{side + 1}", "battle": game.replace(":", "-"), "format": "gen4randombattle", "side": side,
         "player": f"p{side + 1}", "stream": ["|init|battle"], "game": game, "z": 1,
         "decisions": [{"idx": idx + side, "n_legal": 5, "turn": 12, "didx": 6, "legal": [0, 2, 6, 7, 8],
                        "probs": [0.1, 0.1, 0.5, 0.2, 0.1]}]}
    if all_dec:
        r["all_decisions"] = [{"idx": 3 * k, "n_legal": 3, "turn": k, "didx": k} for k in range(20)]
    return r


def write_parts(d: Path, recs: list[dict], per: int = 7) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(recs), per):
        with gzip.open(d / f"rec-a0-w0-{i // per:04d}.jsonl.gz", "wt") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs[i:i + per]))


def run_split(tmp_path: Path, recs: list[dict], parity: int, study: int, seed: int = 4241124):
    write_parts(tmp_path / "gen", recs)
    args = [sys.executable, str(ROOT / "scripts/fastenv/la2_gen4.py"), "split", "--gen", str(tmp_path / "gen"), "--out",
            str(tmp_path / "set"), "--seed", str(seed), "--parity", str(parity), "--study", str(study)]
    return subprocess.run(args, capture_output=True, text=True)


# ---------------------------------------------------------------- split

def test_choose_one_record_per_game_in_hash_order():
    games = [f"7:{b}:{k}" for b in range(5) for k in range(4)]
    recs = [rec(g, s) for g in games for s in (0, 1)]
    recs.append(rec("9:0:0", 1))  # only side two has a record: it is taken whatever the drawn side
    out = LG.choose(recs, 4241124)
    assert [r["game"] for r in out] == sorted(games + ["9:0:0"], key=lambda g: LG.game_key(4241124, g))
    assert all(r["side"] == LG.side_of(4241124, r["game"]) for r in out if r["game"] != "9:0:0")
    assert next(r for r in out if r["game"] == "9:0:0")["side"] == 1
    sides = [r["side"] for r in out]
    assert 0 < sum(sides) < len(sides)  # both sides drawn
    assert LG.choose(list(reversed(recs)), 4241124) == out  # independent of the input order
    assert [r["game"] for r in LG.choose(recs, 1)] != [r["game"] for r in out]  # the seed orders the games


def test_split_files_and_md5s(tmp_path):
    games = [f"5:{b}:{k}" for b in range(10) for k in range(5)]
    recs = [rec(g, s) for g in games for s in (0, 1)]
    p = run_split(tmp_path, recs, 12, 30)
    assert p.returncode == 0 and p.stdout.startswith("OK: 12 parity battles, 30 study positions"), p.stdout + p.stderr
    s = tmp_path / "set"
    info = json.loads((s / "split.json").read_text())
    assert info["parity_battles"] == 12 and info["study_positions"] == 30 and info["games_with_record"] == 50
    with gzip.open(s / "parity-records.jsonl.gz", "rt") as fh:
        par = [json.loads(x) for x in fh]
    with gzip.open(s / "study-records.jsonl.gz", "rt") as fh:
        stu = [json.loads(x) for x in fh]
    rows = [json.loads(x) for x in (s / "study-rows.jsonl").open()]
    assert len(par) == 12 and len(stu) == 30 and len(rows) == 30
    assert not {r["game"] for r in par} & {r["game"] for r in stu}  # disjoint battles
    assert len({r["game"] for r in stu}) == 30  # one position per game
    assert [(r["rec"], r["side"], r["idx"]) for r in rows] == [(x["id"], x["side"], x["decisions"][0]["idx"]) for x in stu]
    for name, m in info["md5"].items():
        assert hashlib.md5((s / name).read_bytes()).hexdigest() == m
    first = info["md5"]
    shutil.rmtree(s)
    assert run_split(tmp_path, recs, 12, 30).returncode == 0  # deterministic files (gzip mtime 0)
    assert json.loads((s / "split.json").read_text())["md5"] == first


def test_split_too_few_and_no_all_decisions(tmp_path):
    recs = [rec(f"5:0:{k}", s) for k in range(10) for s in (0, 1)]
    p = run_split(tmp_path / "a", recs, 5, 6)
    assert p.returncode == 1 and "FAIL: 10 games with a record, need 11" in p.stdout
    recs = [rec(f"5:0:{k}", s, all_dec=False) for k in range(10) for s in (0, 1)]
    p = run_split(tmp_path / "b", recs, 3, 4)
    assert p.returncode == 1 and "without all_decisions" in p.stdout


# ---------------------------------------------------------------- readings

def p1json(a_agree=1.0, a_dp=2e-4, b_dp=0.02, void=False):
    return {"pairs": 880, "failed_records": 1, "dets": 2, "format": "gen4randombattle", "records": 300,
            "records_with_pairs": 299, "void": void,
            "A": {"agree": a_agree, "elements": 60000, "mean_abs_dp": a_dp, "q50": 1e-4, "q90": 4e-4, "q99": 1e-3,
                  "bad": {}},
            "B": {"agree": 0.987, "elements": 60000, "mean_abs_dp": b_dp, "q50": 0.001, "q90": 0.05, "q99": 0.2,
                  "mean_abs_dp_revealed": 0.04, "n_revealed": 500, "mean_abs_dp_not_revealed": 0.003,
                  "bad": {"move_num.pp_frac": 600, "hp": 100}}}


@pytest.mark.parametrize("kw,word", [({}, "PASS"), ({"a_agree": 0.9899}, "FAIL"), ({"a_dp": 0.0101}, "FAIL"),
                                     ({"b_dp": 0.0301}, "FAIL"), ({"b_dp": 0.03}, "PASS"), ({"void": True}, "VOID")])
def test_parity_line(kw, word):
    w, line = LG.parity_line(p1json(**kw))
    assert w == word and line.startswith(f"{word}: gen4randombattle leaf parity (ns-10 method; gen9 standard")
    assert "300 battles (299 with pairs, 1 failed), 880 pairs x 2 determinizations" in line
    assert "B disagreements: move_num.pp_frac 600, hp 100" in line


def bench_json(delta=0.0075, usable=1400):
    curves = [{"delta": d, "override": 0.2 - d, "precision_R": 0.8 + d * 10, "judgeR_gain_vs_raw": 0.005}
              for d in (0.0, 0.0025, 0.005, 0.0075, 0.01)]
    return {"usable": usable, "curves": curves,
            "margin_rule": {"judge": "R", "delta": delta, "override": 0.1925 if delta is not None else None,
                            "judgeR_gain_vs_raw": 0.00501}}


def test_delta_line():
    d, line = LG.delta_line(bench_json(), 1403)
    assert d == 0.0075 and line == ("0.0075 override 0.1925 precision_R 0.8750 judgeR_gain_vs_raw +0.00501 usable 1400 "
                                    "of 1403")
    assert LG.delta_line(bench_json(usable=1374), 1403)[0] is None  # 97.9% usable < 98%
    assert LG.delta_line(bench_json(usable=1375), 1403)[0] == 0.0075
    d, line = LG.delta_line(bench_json(delta=None), 1403)
    assert d is None and line.startswith("NONE")


def test_probs_parity():
    r = rec("5:0:0", 0)
    row = {"rec": r["id"], "idx": r["decisions"][0]["idx"], "mask": [1, 0, 1, 0, 0, 0, 1, 1, 1, 0],
           "probs": [0.1, 0, 0.1000004, 0, 0, 0, 0.5, 0.2, 0.0999996, 0]}
    out = LG.probs_parity([row], {r["id"]: r})
    assert out["compared_actions"] == 5 and out["max_abs_dp"] == pytest.approx(4e-7) and out["legal_mismatch"] == 0
    row2 = dict(row, mask=[1, 1, 1, 0, 0, 0, 1, 1, 1, 0])
    assert LG.probs_parity([row2], {r["id"]: r})["legal_mismatch"] == 1


def test_world_and_obs_diffs():
    obs = {"hp": np.zeros((3, 12, 2), np.float32), "species": np.ones((3, 12), np.int16)}
    w = {"rows": [6, 7], "cols": ["a", "b"], "cells": [[[(1.0, ("L", 0))]]], "n_leaves": 3, "branches": 4,
         "terminal": 0, "switch_expansions": 0, "errors": 0, "obs": obs}
    w2 = {**w, "obs": {k: v.copy() for k, v in obs.items()}}
    assert LG.world_diffs(w, w2) == []
    w2["obs"]["hp"][1, 2, 0] = 1e-7
    assert LG.world_diffs(w, w2) == ["obs.hp"]
    w3 = {**w, "obs": {**obs, "species": obs["species"].astype(np.int32)}}  # same values, another dtype
    assert LG.world_diffs(w, w3) == ["obs.species"]
    neg = {"hp": -np.zeros((3, 12, 2), np.float32), "species": obs["species"]}  # -0.0 is not byte-identical
    assert LG.obs_diffs(obs, neg) == ["hp"]
    assert LG.world_diffs(w, {**w, "cols": ["a", "c"]}) == ["cols"]
    assert LG.world_diffs(w, {**w, "obs": None}) == ["obs None"]


# ---------------------------------------------------------------- smoke and verdict

def fake_result(run: Path, lo: float, pairs: int = 1000, void: bool = False, harness_fail: bool = False,
                games: int | None = None, fallbacks: int = 3, failed_worlds: int = 0) -> None:
    run.mkdir(parents=True, exist_ok=True)
    hw = 0.0215
    nb = 10 if pairs >= 10 else 1
    arm = {"arm": 1, "blocks": nb, "pairs": pairs, "games": games if games is not None else 2 * pairs,
           "incomplete_pairs": 0, "mean": lo + hw, "hw": hw, "lo": lo, "hi": lo + 2 * hw, "won_both": 300,
           "lost_both": 250, "split": pairs - 550, "seconds": 6.1 * 3600,
           "stats_a": {"decisions": 60000, "searched": 52000, "fallbacks": fallbacks, "leaves": 6e7,
                       "failed_worlds": failed_worlds, "overrides": 8300, "la_seconds": 40000.0},
           "stats_b": {}, "gate": {"decisions": 52000, "overrides": 8300}, "gate_opp": {"decisions": 0, "overrides": 0},
           "per_block": [{"block": f"block-{j}", "pairs": pairs // nb, "seconds": 2200} for j in range(1, nb + 1)],
           "spec": "lookahead:NET4,margin=0.0075,temperature=0.5,worlds=8,workers=11,leaf_device=cuda",
           "fallback_rate_a": fallbacks / 60000, "fallback_rate_b": None, "override_rate": 8300 / 52000,
           "override_rate_opp": None, "void": void}
    (run / "result.json").write_text(json.dumps({"run": str(run), "opponent": "NET4,temperature=0.5", "arms": [arm]}))
    (run / "arm-1/block-1").mkdir(parents=True, exist_ok=True)
    (run / "arm-1/block-1/evaluate.log").write_text("ok\n")
    if harness_fail:
        (run / "HARNESS_FAIL").touch()


@pytest.mark.parametrize("lo,word", [(0.5001, "GO"), (0.5, "STOP"), (0.4999, "STOP"), (0.53, "GO"), (0.46, "STOP")])
def test_verdict_rule(tmp_path, lo, word):
    run = tmp_path / "run"
    fake_result(run, lo)
    lines = LG.verdict(run, "0.0075", "R1b-F-cd@640M", "4241125").splitlines()
    assert lines[0] == word
    assert lines[1].startswith(f"pair-mean {lo + 0.0215:.2%} [{lo:.2%}, {lo + 0.043:.2%}] = ")
    assert "1000 duplicate pairs, 2000 games): lookahead2(R1b-F-cd@640M, δ* 0.0075, K 8, T 0.5) vs raw R1b-F-cd@640M" \
           in lines[1]
    assert lines[2] == "override rate 16.0% of 52000 searched decisions"
    assert ("stays the incumbent search bot" in lines[3]) == (word == "STOP")
    assert "rule (main, verbatim): GO iff the pair-mean lower 95% > 50.0%" in lines[-1]


@pytest.mark.parametrize("bad", ["pairs", "harness", "fallbacks"])
def test_verdict_void(tmp_path, bad):
    run = tmp_path / "run"
    fake_result(run, 0.55, pairs=899 if bad == "pairs" else 1000, harness_fail=bad == "harness",
                void=bad == "fallbacks")
    assert LG.verdict(run, "0.0075", "x", "4241125").startswith("VOID: ")


def test_smoke(tmp_path):
    run = tmp_path / "run"
    fake_result(run, 0.5, pairs=2)
    ok, line = LG.smoke(run, 4)
    assert ok and line.startswith("PASS") and "opponent" not in line
    fake_result(run, 0.5, pairs=2, games=3)
    assert not LG.smoke(run, 4)[0]
    fake_result(run, 0.5, pairs=2, failed_worlds=1)
    assert not LG.smoke(run, 4)[0]
    fake_result(run, 0.5, pairs=2)
    (run / "arm-1/block-1/evaluate.log").write_text("Traceback (most recent call last):\n")
    ok, line = LG.smoke(run, 4)
    assert not ok and "no Traceback NO" in line


# ---------------------------------------------------------------- the slot chain, mocked end to end

def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def mock_dir(tmp_path: Path, lo: float, delta: str = "0.0075", line_delta: str | None = None,
             prep_ok: bool = True, prereg_rw: bool = False) -> Path:
    m = tmp_path / "mock"
    s = m / "fastenv/council/la2-gen4"
    (s / "code/scripts/fastenv").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/fastenv/la2_gen4.py", s / "code/scripts/fastenv/")
    (s / "code/COMMIT").write_text("mock\n")
    (m / "fastenv/nets").mkdir(parents=True)
    net = b"stand-in gen4 net\n"
    (m / "fastenv/nets/g4r1bf-cd-640.04M.pt").write_bytes(net)
    (s / "prep/set").mkdir(parents=True)
    rows = b'{"rec": "x-p1", "side": 0, "idx": 40}\n'
    (s / "prep/set/study-rows.jsonl").write_bytes(rows)
    (s / "DELTA").write_text(delta + "\n")
    (s / "DELTA").chmod(0o444)
    sha = hashlib.sha256(net).hexdigest()
    pre = (f"# mock PREREG\nmachine line: `LA2G4 DELTA*={line_delta or delta} STUDY_ROWS_MD5={md5(rows)} "
           f"NET_SHA256={sha}`\n").encode()
    (s / "PREREG.txt").write_bytes(pre)
    if not prereg_rw:
        (s / "PREREG.txt").chmod(0o444)
    (s / "PREREG.md5").write_text(md5(pre) + "  PREREG.txt\n")
    prep = ("DELTA* = 0.0075 ...\n  gen4 PASS: gen4randombattle leaf parity (ns-10 method ...)\n"
            "  gen9 P1 rows old vs new: IDENTICAL: a vs b\n  gen9 identity: PASS: gen9 v3leaf new vs old\n")
    (s / "PREP-RESULT.txt").write_text(prep if prep_ok else prep.replace("IDENTICAL", "DIFFERENT"))
    fake_result(m / "run", lo)
    return m


def run_chain(m: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(LA2G4_MOCK=str(m), LA2G4_MOCK_PY=sys.executable,
               LA2G4_MOCK_NET_MD5=md5((m / "fastenv/nets/g4r1bf-cd-640.04M.pt").read_bytes()))
    return subprocess.run(["bash", str(CHAIN)], env=env, capture_output=True, text=True, timeout=120)


def outbox(m: Path) -> list[str]:
    p = m / "fastenv/LEDGER-OUTBOX.md"
    return p.read_text().splitlines() if p.exists() else []


@pytest.mark.parametrize("lo,word", [(0.512, "GO"), (0.5, "STOP"), (0.47, "STOP")])
def test_chain_mock_reading(tmp_path, lo, word):
    m = mock_dir(tmp_path, lo)
    p = run_chain(m)
    s = m / "fastenv/council/la2-gen4"
    assert p.returncode == 0, p.stdout + p.stderr
    v = (s / "VERDICT.txt").read_text().splitlines()
    assert v[0] == word and not os.access(s / "VERDICT.txt", os.W_OK)
    assert (s / "RESULT").read_text().strip() == word
    out = outbox(m)
    assert len(out) == 1 and f"| fastenv | RELEASE | mock | LA2-gen4 done: {word}: pair-mean" in out[0]
    assert out[0].endswith("slot FREE") and "override rate" in out[0]
    assert not (s / "FAILED").exists()
    st = (s / "STATUS").read_text()
    assert "prereg OK" in st and "checks OK: δ* 0.0075 = DELTA = the PREREG's" in st and f"VERDICT: {word}" in st
    p2 = run_chain(m)  # a stage dir that holds a chain is refused
    assert p2.returncode == 1 and "refused" in outbox(m)[-1]


@pytest.mark.parametrize("case", ["delta_mismatch", "prep_not_pass", "prereg_writable", "void"])
def test_chain_mock_failures(tmp_path, case):
    m = mock_dir(tmp_path, 0.3 if case == "void" else 0.52, line_delta="0.01" if case == "delta_mismatch" else None,
                 prep_ok=case != "prep_not_pass", prereg_rw=case == "prereg_writable")
    if case == "void":
        fake_result(m / "run", 0.52, pairs=850)
    p = run_chain(m)
    s = m / "fastenv/council/la2-gen4"
    assert p.returncode == 1
    assert (s / "FAILED").exists()
    out = outbox(m)
    assert len(out) == 1 and "| RELEASE | mock | LA2-gen4 FAILED at stage" in out[0] and out[0].endswith("slot FREE")
    want = {"delta_mismatch": "stage checks", "prep_not_pass": "stage checks", "prereg_writable": "stage prereg",
            "void": "stage verdict"}[case]
    assert want in out[0]
    assert (s / "VERDICT.txt").exists() == (case == "void")
