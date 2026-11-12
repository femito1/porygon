"""Trainer v2: GAE over carried-over segments, packing, lr schedule, a PPO update on real observations, and a tiny
end-to-end run (+ resume) on the local Showdown server (localhost:8000; skipped when it is not running)."""

import math
import pickle
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from porygon.v2.encoder import aux_targets, encode
from porygon.v2.model import Net
from porygon.v2.spec import N_OPP_CLASSES
from porygon.v2.train import build_rollout, lr_at, lr_schedule_desc, parse_args, power_decay, ppo_update, wsd_lr
from porygon.v2.workers import ACT, DONE, LOGP, NEXT_V, OBS, OPP_C, OPP_T, REW, SEED, VAL, Packer, empty_like_blob

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SNAPS = Path("/tmp/netv2/snaps_gen9randombattle.pkl")
TINY_ARGS = ["--d-model", "32", "--n-layers", "1", "--n-heads", "4", "--ffn-mult", "2", "--d-move", "16",
             "--model-json", '{"d_species":16,"d_move_emb":16,"d_item":8,"d_ability":8,"d_type":8,'
                             '"pair_hidden":8,"value_hidden":32}']
GAMMA, LAM = 0.9, 0.8


def transition(blob, v, r, next_v, done, a=1):
    t = [None] * 10
    t[OBS], t[ACT], t[LOGP], t[VAL], t[SEED], t[REW] = blob, a, -0.5, v, 0, r
    t[OPP_T], t[OPP_C], t[NEXT_V], t[DONE] = None, -1, next_v, done
    return t


def reference_gae(seg):
    adv, g = [], 0.0
    for i in reversed(range(len(seg))):
        t = seg[i]
        nonterminal = 0.0 if t[DONE] else 1.0
        delta = t[REW] + GAMMA * t[NEXT_V] * nonterminal - t[VAL]
        g = delta + GAMMA * LAM * nonterminal * (g if i < len(seg) - 1 else 0.0)
        adv.append(g)
    return adv[::-1]


def test_gae_over_segments():
    packer = Packer(9)
    blob = empty_like_blob(packer)
    # segment values: next_v of each transition = value of the next decision of that side (or a later rollout's)
    segs = {
        (0, 0): [transition(blob, 0.1, 0.0, 0.2, False), transition(blob, 0.2, 0.0, 0.3, False),
                 transition(blob, 0.3, 0.0, 0.5, False)],  # cut mid-battle: bootstraps from 0.5
        (0, 1): [transition(blob, -0.1, 0.0, -0.2, False), transition(blob, -0.2, -1.0, 0.0, True),
                 transition(blob, 0.0, 0.0, 0.4, False)],  # episode ends inside the segment
        (1, 0): [transition(blob, 0.4, 1.0, 0.0, True)],
    }
    data = build_rollout(segs, packer, GAMMA, LAM)
    expected = [x for k in segs for x in reference_gae(segs[k])]
    assert data["adv"] == pytest.approx(expected, abs=1e-6)
    values = np.array([t[VAL] for k in segs for t in segs[k]])
    assert data["ret"] == pytest.approx(np.array(expected) + values, abs=1e-6)
    assert data["species"].shape == (7, 12) and data["action"].tolist() == [1] * 7


def test_packer_roundtrip():
    if not SNAPS.exists():
        pytest.skip("no snapshots")
    snaps = pickle.load(open(SNAPS, "rb"))[:5]
    packer = Packer(9)
    obs = [encode(a) for a, _ in snaps]
    aux = [aux_targets(a, b, o) for (a, b), o in zip(snaps, obs)]
    out = packer.unpack([packer.pack(o, x) for o, x in zip(obs, aux)])
    for k in obs[0]:
        assert np.array_equal(out[k], np.stack([o[k] for o in obs])), k
    for k in aux[0]:
        assert np.array_equal(out[k], np.stack([x[k] for x in aux])), k
    assert out["species"].flags.writeable
    one = Packer(9, aux=False).unpack([Packer(9, aux=False).pack(obs[0])])
    assert np.array_equal(one["moves"][0], obs[0]["moves"])


def test_wsd_schedule():
    lr = lambda s, cd=None: wsd_lr(s, 1e-4, 1000, cd, 10_000, 0.1)  # noqa: E731
    assert lr(0) == pytest.approx(1e-7) and lr(499) == pytest.approx(5e-5) and lr(999) == pytest.approx(1e-4)
    assert lr(5000) == lr(9999) == pytest.approx(1e-4)  # no cooldown unless asked
    assert lr(5000, 5000) == pytest.approx(1e-4) and lr(10_000, 5000) == pytest.approx(1e-5)
    assert lr(7500, 5000) == pytest.approx(1e-4 * (0.1 + 0.9 * 0.5))


def _wsd_lr_before(step, lr, warmup, cooldown_start, total, final_frac):
    """wsd_lr as it was before --lr-schedule (ws/netv2 d7a5b27), verbatim: the reference for the default path."""
    f = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
    if cooldown_start is not None and step > cooldown_start:
        x = min(1.0, (step - cooldown_start) / max(1, total - cooldown_start))
        f *= final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * x))
    return lr * f


G9_ARGS = ["--lr", "7e-5", "--total-steps", "1000000000"]  # what the netv2 jobs pass (TOTAL = 1e9)
POWER = ["--lr-schedule", "power", "--lr-horizon", "300000000"]  # G9r


def _grid():
    rng = np.random.default_rng(0)
    pts = [0, 1, 999, 1000, 1001, 499_999, 999_999, 1_000_000, 1_000_001, 25_000_000, 45_004_575, 50_004_575,
           299_999_999, 300_000_000, 300_000_001, 999_999_999, 1_000_000_000, 2_000_000_000]
    return pts + [int(v) for v in rng.integers(0, 1_200_000_000, 2000)]


def test_lr_schedule_default_is_bit_identical_to_wsd():
    """The default (--lr-schedule wsd) is the old wsd_lr bit for bit on a grid, for the G9p / G9q / R1b-F args and
    with a cosine cooldown (the G9p-cd / G9q-cd branches); the old positional call too."""
    cases = [G9_ARGS, G9_ARGS + ["--lr-schedule", "wsd"], ["--lr", "7e-5"],
             G9_ARGS[:2] + ["--cooldown-start", "45004575", "--total-steps", "50004575", "--final-lr-frac", "0.1"],
             ["--lr", "3.5e-5", "--warmup-steps", "0", "--cooldown-start", "100", "--total-steps", "10000000"]]
    for argv in cases:
        a = parse_args(argv)
        assert a.lr_schedule == "wsd" and a.lr_horizon is None
        for s in _grid():
            ref = _wsd_lr_before(s, a.lr, a.warmup_steps, a.cooldown_start, a.total_steps, a.final_lr_frac)
            assert lr_at(a, s) == ref, (argv, s)  # == on floats: bit-identical
            assert wsd_lr(s, a.lr, a.warmup_steps, a.cooldown_start, a.total_steps, a.final_lr_frac) == ref
    assert lr_schedule_desc(parse_args(G9_ARGS)).startswith("lr schedule: wsd,")


def test_lr_schedule_power_values():
    a = parse_args(G9_ARGS + POWER)
    assert a.lr_schedule == "power" and a.lr_horizon == 300_000_000 and a.total_steps == 1_000_000_000
    assert lr_schedule_desc(a).startswith("lr schedule: power H=300000000,")
    wsd = parse_args(G9_ARGS)
    ratio = lambda s: lr_at(a, s) / lr_at(wsd, s)  # noqa: E731
    x = lambda s: (s - 1e6) / 299e6  # noqa: E731
    assert ratio(0) == 1.0 and lr_at(a, 0) == pytest.approx(7e-11)  # the warm-up is unchanged
    assert ratio(500_000) == 1.0 and ratio(1_000_000) == 1.0 and lr_at(a, 1_000_000) == pytest.approx(7e-5)
    assert ratio(25_000_000) == pytest.approx((8 * x(25e6) + 1) ** -1.5)
    assert lr_at(a, 25_000_000) == pytest.approx(3.33e-5, rel=2e-3)
    assert lr_at(a, 50_000_000) == pytest.approx(1.99e-5, rel=2e-3)
    assert lr_at(a, 100_000_000) == pytest.approx(1.0e-5, rel=1e-2)
    for s in (300_000_000, 300_000_001, 500_000_000, 1_000_000_000, 5_000_000_000):  # flat at the floor after H
        assert ratio(s) == pytest.approx(1 / 27) and lr_at(a, s) == pytest.approx(7e-5 / 27)
    grid = sorted(set(_grid()))
    vals = [lr_at(a, s) for s in grid if s >= 1_000_000]
    assert all(u >= v for u, v in zip(vals, vals[1:]))  # monotone non-increasing after the warm-up
    assert power_decay(0, 1_000_000, 300_000_000) == 1.0 and power_decay(10**12, 1_000_000, 300_000_000) == 1 / 27
    # the guard fallback's `--lr 7e-5 -> --lr 3.5e-5` substitution halves the whole curve
    half = parse_args(["--lr", "3.5e-5"] + G9_ARGS[2:] + POWER)
    for s in grid:
        assert lr_at(half, s) == pytest.approx(lr_at(a, s) / 2, rel=1e-12)
    with pytest.raises(SystemExit):  # power needs a horizon past the warm-up
        parse_args(G9_ARGS + ["--lr-schedule", "power"])
    with pytest.raises(SystemExit):
        parse_args(G9_ARGS + ["--lr-schedule", "power", "--lr-horizon", "1000000"])


def test_lr_schedule_power_composes_with_cooldown():
    """G9r-cd: G9r's ckpt at ~45M + 5M cosine to x0.35 ends near the G9p-cd / G9q-cd floor 7e-6."""
    a = parse_args(["--lr", "7e-5"] + POWER + ["--cooldown-start", "45000000", "--total-steps", "50000000",
                                               "--final-lr-frac", "0.35"])
    plain = parse_args(G9_ARGS + POWER)
    assert "cosine cooldown 45000000 -> 50000000 to x0.35" in lr_schedule_desc(a)
    for s in (0, 1_000_000, 30_000_000, 45_000_000):  # before the cooldown: the power curve itself
        assert lr_at(a, s) == lr_at(plain, s)
    for s in (45_000_001, 46_000_000, 47_500_000, 49_999_999, 50_000_000, 60_000_000):
        x = min(1.0, (s - 45_000_000) / 5_000_000)
        cos = 0.35 + 0.65 * 0.5 * (1 + math.cos(math.pi * x))
        assert lr_at(a, s) == pytest.approx(lr_at(plain, s) * cos, rel=1e-12)
    assert lr_at(a, 45_000_000) == pytest.approx(2.18e-5, rel=5e-3)
    assert lr_at(a, 50_000_000) == pytest.approx(7.0e-6, rel=1e-2)  # 1.99e-5 x 0.35


def test_lr_schedule_resume_continuity():
    """The schedule is a pure function of the absolute lineage step counter and the args (the trainer computes
    lr_at(args, counters["steps"]) after every rollout; a resume restores counters["steps"] from the checkpoint and
    its args come from the command line): the lr at step s is the same fresh or after a resume at s, also when the
    checkpoint saved under other flags."""
    a = parse_args(G9_ARGS + POWER)
    for s in (1_000_000, 25_038_951, 45_004_575, 299_999_999, 300_000_000, 700_000_000):
        resumed = parse_args(G9_ARGS + POWER + ["--resume", "x.pt"])
        assert lr_at(resumed, s) == lr_at(a, s)
        nxt = s + 41_523  # one A16 rollout later: continues the curve, no restart of x
        assert lr_at(resumed, nxt) == lr_at(a, nxt) <= lr_at(a, s)
    # a G9q checkpoint (wsd) resumed under power flags follows the command line, as every flag does today
    assert lr_at(parse_args(G9_ARGS + POWER + ["--resume", "g9q.pt"]), 25_000_000) == lr_at(a, 25_000_000)
    assert lr_at(parse_args(G9_ARGS + ["--resume", "g9r.pt"]), 25_000_000) == pytest.approx(7e-5)


def test_ppo_update_on_real_observations():
    if not SNAPS.exists():
        pytest.skip("no snapshots")
    snaps = pickle.load(open(SNAPS, "rb"))[:96]
    packer = Packer(9)
    rng = np.random.default_rng(0)
    seg = []
    for i, (a, b) in enumerate(snaps):
        o = encode(a)
        legal = np.flatnonzero(o["action_mask"])
        act = int(rng.choice(legal)) if len(legal) else 0
        tr = transition(packer.pack(o, aux_targets(a, b, o)), 0.0, 0.0, 0.0, i % 20 == 19, act)
        tr[LOGP] = float(-np.log(max(len(legal), 1)))  # uniform start policy
        tr[REW] = float(rng.choice([-1, 1])) if tr[DONE] else 0.0
        tr[SEED] = int(rng.integers(1, 2 ** 31))
        if i % 3 == 0:
            tgt = np.zeros(N_OPP_CLASSES, np.float32)
            tgt[rng.integers(N_OPP_CLASSES)] = 1.0
            tr[OPP_T], tr[OPP_C] = tgt, int(np.argmax(tgt))
        seg.append(tr)
    data = build_rollout({(0, 0): seg}, packer, 0.9999, 0.754)
    args = parse_args(["--batch-size", "32", "--n-epochs", "2"])
    torch.manual_seed(0)
    use_damage = True
    try:
        import porygon.v2.damage  # noqa: F401
    except ImportError:
        use_damage = False
    net = Net(9, use_damage=use_damage, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, d_move=16, d_species=16,
              d_move_emb=16, d_item=8, d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)
    opt = torch.optim.AdamW(net.param_groups(0.01), lr=1e-3)
    st = ppo_update(net, opt, data, args, torch.device("cpu"), False, True)
    for k in ("loss/pg", "loss/vf", "loss/ent", "loss/za", "loss/aux", "loss/kl", "aux/item_loss",
              "aux/opp_action_loss", "aux/unseen_loss", "aux/moves_loss", "diag/trunk_grad_ratio",
              "diag/grad_cos_policy", "diag/grad_cos_value", "diag/explained_variance"):
        assert k in st and np.isfinite(st[k]), k
    assert 0 < st["loss/kl"] < 1 and st["loss/clip"] <= 1


def test_ema_and_adv_clip():
    from porygon.v2.train import EMA, ppo_losses
    torch.manual_seed(0)
    net = Net(4, use_damage=False, use_tables=False, d_model=16, n_layers=1, n_heads=2, ffn_mult=1, d_move=8,
              d_species=8, d_move_emb=8, d_item=4, d_ability=4, d_type=4, pair_hidden=4, value_hidden=8)
    ema = EMA(net, 0.5)
    p0 = net.value_head[0].weight.detach().clone()
    with torch.no_grad():
        net.value_head[0].weight.add_(1.0)
    ema.update(net)  # effective decay min(0.5, 2/11)
    d = 2 / 11
    assert torch.allclose(ema.net.value_head[0].weight, d * p0 + (1 - d) * (p0 + 1), atol=1e-6)
    st = ema.state_dict()
    ema2 = EMA(net, 0.5)
    ema2.load_state_dict(st)
    assert ema2.n == 1 and torch.equal(ema2.net.value_head[0].weight, ema.net.value_head[0].weight)
    # advantage clipping: one huge advantage no longer dominates the surrogate
    B, A = 8, 10
    out = {"logits": torch.zeros(B, A, requires_grad=True), "win_logit": torch.zeros(B)}
    mb = {"action_mask": torch.ones(B, A), "action": torch.zeros(B, dtype=torch.long), "logp": torch.full((B,), -2.3026),
          "adv": torch.tensor([100.0] + [0.0] * 7), "ret": torch.zeros(B)}
    a0, a1 = parse_args([]), parse_args(["--adv-clip", "1.0"])
    g0 = torch.autograd.grad(ppo_losses(out, mb, a0)["pg"], out["logits"])[0]
    g1 = torch.autograd.grad(ppo_losses(out, mb, a1)["pg"], out["logits"])[0]
    assert g1.abs().max() < g0.abs().max()


def _server_up(port=8000) -> bool:
    try:
        socket.create_connection(("localhost", port), timeout=2).close()
        return True
    except OSError:
        return False


def _train(run_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [PY, "-m", "porygon.v2.train", "--format", "gen9randombattle", "--ports", "8000", "--n-battles", "2",
           "--pool-battles", "1", "--p-baseline", "0.5", "--rollout", "200", "--batch-size", "64", "--n-epochs", "2",
           "--eval-battles", "0", "--warmup-steps", "1000", "--snapshot-every", "200", "--checkpoint-every", "400",
           "--grad-ratio-every", "1", "--run-dir", str(run_dir), *TINY_ARGS, *extra]
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    return subprocess.run(cmd, cwd=run_dir.parent, env=env, capture_output=True, text=True, timeout=900)


@pytest.mark.skipif(not _server_up(), reason="no Showdown server on localhost:8000")
def test_end_to_end_and_resume(tmp_path):
    run = tmp_path / "run"
    try:
        r = _train(run, "--max-updates", "2")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        log = (run / "train.log").read_text()
        assert log.count(" upd ") == 2, log
        assert (run / "final_model.pt").exists() and list((run / "checkpoints").glob("ckpt_*.pt"))
        assert len(list((run / "pool").glob("*.pt"))) == 2
        ck = torch.load(run / "final_model.pt", weights_only=False)
        steps = ck["counters"]["steps"]
        assert steps >= 400 and ck["counters"]["updates"] == 2 and "optimizer" in ck and ck["ema"]["n"] == 2
        from porygon.v2.model import load
        ema = load(run / "ema_model.pt")
        assert ema.cfg == load(run / "final_model.pt").cfg
        r = _train(run, "--max-updates", "1", "--resume", str(run / "final_model.pt"))
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        ck2 = torch.load(run / "final_model.pt", weights_only=False)
        assert ck2["counters"]["updates"] == 3 and ck2["counters"]["steps"] >= steps + 200
        assert ck2["cfg"] == ck["cfg"] and ck2["ema"]["n"] == 3
        last = (run / "train.log").read_text().strip().splitlines()
        assert any(f"steps={steps:,}" in line for line in last), "resume did not restore the step counter"
        assert "upd 3 " in "\n".join(last)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)  # laptop disk is nearly full: never keep test runs


def test_lr_schedule_trainer_log_and_real_resume(tmp_path, monkeypatch):
    """The real Trainer (CPU, fastenv stubbed as in test_v2_actor_buffers.py): the startup line names the schedule;
    a checkpoint at step s saves the flags and the counters; a Trainer resumed from it continues at step s with the
    command line's flags and sets the same lr at s and at the next rollout as one that never stopped."""
    from types import SimpleNamespace

    import porygon.fastenv_v2 as fv2
    from porygon.v2.train import Trainer
    monkeypatch.setattr(fv2, "build_env", lambda args, seed=None: SimpleNamespace(
        env=SimpleNamespace(close=lambda: None), pool_per_worker=0, restarts=0, close=lambda: None))
    monkeypatch.setattr(fv2, "FastCollector", lambda *a, **kw: SimpleNamespace())
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    base = ["--collector", "fastenv", "--device", "cpu", "--seed", "0", "--ema-decay", "0", *TINY_ARGS,
            *G9_ARGS, *POWER]
    made = []
    try:
        a = Trainer(parse_args(base + ["--run-dir", str(tmp_path / "a")]))
        made.append(a)
        log_a = (tmp_path / "a" / "train.log").read_text()
        assert "lr schedule: power H=300000000, lr 7e-05, warm-up 1000000" in log_a
        s = 45_004_575
        a.counters.update(steps=s, updates=1084)
        a.checkpoint(tmp_path / "a" / "final_model.pt")
        ck = torch.load(tmp_path / "a" / "final_model.pt", weights_only=False)
        assert ck["args"]["lr_schedule"] == "power" and ck["args"]["lr_horizon"] == 300_000_000
        b = Trainer(parse_args(base + ["--run-dir", str(tmp_path / "b"), "--resume",
                                       str(tmp_path / "a" / "final_model.pt")]))
        made.append(b)
        assert b.counters["steps"] == s and "steps=45,004,575, lr schedule: power H=300000000" in \
            (tmp_path / "b" / "train.log").read_text()
        for step in (s, s + 41_523):
            assert lr_at(b.args, step) == lr_at(a.args, step)
        # flags from the command line win over the checkpoint's saved args (resume semantics unchanged)
        c = Trainer(parse_args(base[:-4] + ["--run-dir", str(tmp_path / "c"), "--resume",
                                            str(tmp_path / "a" / "final_model.pt")]))
        made.append(c)
        assert c.args.lr_schedule == "wsd" and lr_at(c.args, s) == 7e-5
    finally:
        for t in made:
            t.log_file.close()
            t.writer.close()
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)
        shutil.rmtree(tmp_path, ignore_errors=True)
