"""Trainer v2: GAE over carried-over segments, packing, lr schedule, a PPO update on real observations, and a tiny
end-to-end run (+ resume) on the local Showdown server (localhost:8000; skipped when it is not running)."""

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
from porygon.v2.train import build_rollout, parse_args, ppo_update, wsd_lr
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
        assert steps >= 400 and ck["counters"]["updates"] == 2 and "optimizer" in ck
        r = _train(run, "--max-updates", "1", "--resume", str(run / "final_model.pt"))
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        ck2 = torch.load(run / "final_model.pt", weights_only=False)
        assert ck2["counters"]["updates"] == 3 and ck2["counters"]["steps"] >= steps + 200
        assert ck2["cfg"] == ck["cfg"]
        last = (run / "train.log").read_text().strip().splitlines()
        assert any(f"steps={steps:,}" in line for line in last), "resume did not restore the step counter"
        assert "upd 3 " in "\n".join(last)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)  # laptop disk is nearly full: never keep test runs
