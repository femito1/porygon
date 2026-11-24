"""Council test P2 (scripts/fastenv/p2_exit_soft.py): the completed-Q advantage, the soft target pi_imp ∝ p·exp(βA)
(support kept, expected Q̂ not lower: the PORTFOLIO's formal check), the exact KL, the β bisection, the calibration
split, the verdict rule, and the shards subcommand on synthetic label parts. Pure numpy, no engine or network."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "p2_exit_soft", Path(__file__).resolve().parents[1] / "scripts" / "fastenv" / "p2_exit_soft.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

A = 26


def _rand(rng, n):
    """n rows: p a softmax over 2..9 random legal actions (some with tiny mass), A a completed advantage."""
    p, adv = np.zeros((n, A)), np.zeros((n, A))
    for i in range(n):
        legal = rng.choice(A, size=rng.integers(2, 10), replace=False)
        lg = rng.normal(0, 3, len(legal))
        p[i, legal] = np.exp(lg - lg.max()) / np.exp(lg - lg.max()).sum()
        adv[i, legal] = rng.normal(-0.05, 0.08, len(legal))
    return p, adv


def test_completed_adv_rows_and_missing_actions():
    adv = P.completed_adv(A, [1, 6, 7, 22], [1, 6, 7], [0.40, 0.50, 0.52], 0.51)
    assert adv[1] == pytest.approx(-0.11) and adv[6] == pytest.approx(-0.01) and adv[7] == pytest.approx(0.01)
    assert adv[22] == 0.0  # legal without a row: Q completed with V
    assert np.count_nonzero(adv) == 3  # illegal actions: 0
    with pytest.raises(ValueError):
        P.completed_adv(A, [1, 6], [1, 9], [0.4, 0.5], 0.5)


def test_pi_imp_beta_zero_is_p_and_support_is_kept():
    rng = np.random.default_rng(0)
    p, adv = _rand(rng, 200)
    assert np.allclose(P.pi_imp(p, adv, 0.0), p, atol=1e-15)
    for beta in (1.0, 20.0, 200.0):
        t = P.pi_imp(p, adv, beta)
        assert np.allclose(t.sum(-1), 1.0)
        assert np.array_equal(t > 0, p > 0)  # zero-avoiding: supp(pi_imp) = supp(p)
    one = P.pi_imp(p[0], adv[0], 5.0)
    assert np.allclose(one, P.pi_imp(p[:1], adv[:1], 5.0)[0])


def test_expected_qhat_not_lower_and_v_cancels():
    rng = np.random.default_rng(1)
    p, adv = _rand(rng, 500)
    for beta in (0.5, 10.0, 100.0):
        t = P.pi_imp(p, adv, beta)
        assert ((t * adv).sum(-1) - (p * adv).sum(-1) >= -1e-12).all()  # E_pi_imp[Q̂] >= E_p[Q̂] (Chebyshev)
        shifted = np.where(p > 0, adv + 0.3, 0.0)  # another V: a constant shift on the support
        assert np.allclose(P.pi_imp(p, shifted, beta), t, atol=1e-12)


def test_kl_exact_and_increasing():
    rng = np.random.default_rng(2)
    p, adv = _rand(rng, 300)
    for beta in (0.0, 3.0, 40.0):
        t = P.pi_imp(p, adv, beta)
        direct = np.where(t > 0, t * (np.log(np.where(t > 0, t, 1)) - np.log(np.where(p > 0, p, 1))), 0).sum(-1)
        assert np.allclose(P.kl_imp(p, adv, beta), direct, atol=1e-10)
    means = [P.kl_imp(p, adv, b).mean() for b in np.linspace(0, 60, 25)]
    assert means[0] == pytest.approx(0.0, abs=1e-12) and all(np.diff(means) > 0)


def test_bisect_beta_hits_the_target():
    rng = np.random.default_rng(3)
    p, adv = _rand(rng, 400)
    b = P.bisect_beta(p, adv, 0.05)
    assert b > 0 and P.kl_imp(p, adv, b).mean() == pytest.approx(0.05, abs=1e-9)
    with pytest.raises(ValueError):
        P.bisect_beta(p, np.zeros_like(adv), 0.05)  # no advantage anywhere: unreachable


def test_calibration_split_fixed_and_about_a_tenth():
    ids = [f"7191101{g:03d}:{k}:0-p{s}" for g in range(40) for k in range(25) for s in (1, 2)]
    sel = [P.in_cal(i) for i in ids]
    assert sel == [P.in_cal(i) for i in ids]
    assert 0.08 < np.mean(sel) < 0.12


def _ev(wr, ci, pairs=4000, seed=1):
    return {"win_rate": wr, "ci95": ci, "pairs": pairs, "seed": seed}


def test_verdict_rule():
    assert P.verdict_p2(_ev(0.52, 0.01), _ev(0.515, 0.01)) == "GO"  # lowers 51.0 and 50.5
    assert P.verdict_p2(_ev(0.5149, 0.01), _ev(0.53, 0.01)) == "NO-GO"  # lower 50.49 vs C
    assert P.verdict_p2(_ev(0.5151, 0.01), _ev(0.5101, 0.01)) == "GO"  # 50.51 and 50.01
    assert P.verdict_p2(_ev(0.53, 0.01), _ev(0.5099, 0.01)) == "NO-GO"  # 49.99 vs S
    assert P.verdict_p2(_ev(0.48, 0.01), _ev(0.48, 0.01)) == "NO-GO"


def _part(d: Path, name: str, rng, n: int):
    """A synthetic label part (npz + meta) in exit_la2.py label's format, and its arm-E shard (exit_la2 shards)."""
    p, adv = _rand(rng, n)
    mask = (p > 0).astype(np.float32)
    meta, pi = [], np.zeros((n, A))
    for i in range(n):
        legal = np.flatnonzero(mask[i]).tolist()
        rows = legal[:-1] if i % 3 == 0 else legal  # every third row: the last legal action has no Q
        q = 0.5 + rng.normal(0, 0.05, len(rows))
        x = np.zeros(len(rows))
        x[int(np.argmax(q))] = 1.0
        piv, p05 = P.X.pi_la(p[i], rows, x, q, 0.005)
        pi[i] = piv
        meta.append({"id": f"{name}:{i}", "n_legal": len(legal), "a_l": rows[int(np.argmax(x))], "rows": rows,
                     "q": [round(float(v), 6) for v in q], "x": x.tolist(), "value": round(float(x @ q), 6),
                     "moved": round(float(np.abs(piv - p05).sum() / 2), 6)})
    game = np.arange(n, dtype=np.int64) + 1000 * len(list(d.glob("*.npz")))
    z = rng.choice([-1.0, 1.0], n).astype(np.float32)
    obs = {"obs/action_mask": mask, "obs/other": rng.normal(size=(n, 3)).astype(np.float32)}
    np.savez_compressed(d / f"{name}.npz", **obs, probs=p.astype(np.float32), pi=pi.astype(np.float32), z=z, game=game)
    (d / f"{name}.meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
    return obs, z, game


def test_calibrate_and_shards_on_synthetic_parts(tmp_path):
    rng = np.random.default_rng(4)
    lab, esh, out = tmp_path / "labels", tmp_path / "E", tmp_path / "Ep"
    lab.mkdir()
    esh.mkdir()
    for k in range(3):
        _part(lab, f"label-x-{k:04d}", rng, 300)
    for i, part in enumerate(P.X.label_parts(lab)):  # ns-27's E shards, as exit_la2.py shards writes them
        z = np.load(part)
        np.savez(esh / f"shard-{i:05d}.npz", **{k: z[k] for k in z.files if k.startswith("obs/")},
                 target=P.X.sqrt_target(z["pi"]), z=z["z"], game=z["game"])
    cal = tmp_path / "cal.json"
    assert P.cmd_calibrate(SimpleNamespace(labels=str(lab), json=str(cal), kl=0.05, expect=None)) == 0
    c = json.loads(cal.read_text())
    assert c["data_check"] == "OK" and c["rows"] == 900 and c["rows_missing_q"] == 300 and c["cal_rows"] > 40
    assert c["mean_kl_cal_at_beta_recorded"] == pytest.approx(0.05, rel=1e-4) and c["min_qhat_gain_cal"] >= -1e-12
    beta = c["beta_recorded"]
    assert P.cmd_calibrate(SimpleNamespace(labels=str(lab), json=str(cal), kl=0.05, expect=beta)) == 0
    assert P.cmd_calibrate(SimpleNamespace(labels=str(lab), json=str(cal), kl=0.05, expect=beta * 1.01)) == 1
    assert P.cmd_shards(SimpleNamespace(labels=str(lab), e_shards=str(esh), beta=beta, out=str(out), parts=0)) == 0
    assert json.loads((out / "SHARDS.json").read_text())["rows"] == 900
    for i in range(3):
        e, n = np.load(esh / f"shard-{i:05d}.npz"), np.load(out / f"shard-{i:05d}.npz")
        assert sorted(e.files) == sorted(n.files)
        for k in e.files:
            if k != "target":
                assert np.array_equal(e[k], n[k])  # obs, z, game, order: ns-27's E shards exactly
        assert np.allclose(n["target"].sum(-1), 1, atol=1e-6) and np.array_equal(n["target"] > 0, e["obs/action_mask"] > 0)
    np.savez(esh / "shard-00001.npz", **{k: np.load(esh / "shard-00002.npz")[k] for k in np.load(esh / "shard-00002.npz").files})
    with pytest.raises(SystemExit):  # a shard that does not match its label part
        P.cmd_shards(SimpleNamespace(labels=str(lab), e_shards=str(esh), beta=beta, out=str(out), parts=0))


def test_misaligned_meta_is_caught(tmp_path):
    rng = np.random.default_rng(5)
    lab = tmp_path / "labels"
    lab.mkdir()
    _part(lab, "label-y-0000", rng, 200)
    mp = lab / "label-y-0000.meta.jsonl"
    lines = mp.read_text().splitlines()
    mp.write_text("\n".join(lines[1:] + lines[:1]) + "\n")  # meta rotated by one row
    arr, meta = P.read_part(lab / "label-y-0000.npz")
    with pytest.raises(ValueError):
        adv, st = P.part_adv(arr, meta)
        if P.data_ok(st):
            raise ValueError(P.data_ok(st))


def test_verdict_text(tmp_path):
    t1, t2 = tmp_path / "T1.json", tmp_path / "T2.json"
    t1.write_text(json.dumps(_ev(0.499, 0.0098, 3961, 11240221)))
    t2.write_text(json.dumps(_ev(0.503, 0.0099, 3958, 11240222)))
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"beta_recorded": 12.5, "mean_kl_cal_at_beta_recorded": 0.05, "cal_rows": 11000,
                               "mean_kl_hard_target_tE_vs_p_cal": 0.07, "rows": 119967, "rows_missing_q": 10}))
    out = tmp_path / "V.txt"
    P.cmd_verdict(SimpleNamespace(t1=str(t1), t2=str(t2), calibration=str(cal), check=str(tmp_path / "none"),
                                  out=str(out)))
    lines = out.read_text().splitlines()
    assert lines[0] == "NO-GO" and lines[1].startswith("E′ vs C") and "[48.92, 50.88]" in lines[1]
    assert lines[2].startswith("E′ vs S") and math.isclose(P.lower(json.loads(t2.read_text())), 49.31)
