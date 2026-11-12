"""scripts/netv2_lambda_noise.py (G9t's λ pre-check): the trainer's own advantages at its λ, Monte-Carlo returns at
λ = 1, zero bias with an exact critic, and the gradient diagnostics on the same fixed chunks for every λ."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from porygon.v2.model import Net
from porygon.v2.train import Meter, build_rollout, gradient_diagnostics, parse_args, rollout_damage
from porygon.v2.workers import ACT, DONE, LOGP, NEXT_V, OBS, OPP_C, OPP_T, REW, SEED, VAL, Packer, empty_like_blob

spec = importlib.util.spec_from_file_location(
    "lam", Path(__file__).resolve().parent.parent / "scripts" / "netv2_lambda_noise.py")
lam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lam)
GAMMA = 0.9999


def tr(blob, v, r, nv, done, a=1, seed=0):
    t = [None] * 10
    t[OBS], t[ACT], t[LOGP], t[VAL], t[SEED], t[REW] = blob, a, -math.log(26), v, seed, r  # ~ the init net
    t[OPP_T], t[OPP_C], t[NEXT_V], t[DONE] = None, -1, nv, done
    return t


def rand_blob(packer, rng):
    """A random observation: float features ~ N(0, 1), ids in 1..3, every action legal."""
    o = {k: v[0] for k, v in packer.unpack([empty_like_blob(packer)]).items()}
    for k, v in o.items():
        if k == "action_mask":
            v[:] = 1
        elif v.dtype.kind == "f":
            v[:] = rng.normal(size=v.shape)
        elif v.dtype.kind in "iu":
            v[:] = rng.integers(1, 4, size=v.shape)  # small ids (present entities): every vocabulary has them
    return packer.pack(o)


def random_segs(rng, blob, n_segs=12):
    """Segments as the collector builds them: episodes ending inside, segments cut mid-game (bootstrap next_v)."""
    segs = {}
    for s in range(n_segs):
        seg = []
        for _ in range(rng.integers(1, 3)):  # 1-2 episodes, the last one maybe cut
            T = int(rng.integers(1, 9))
            vs = rng.uniform(-1, 1, T + 1).astype(np.float32)
            cut = rng.random() < 0.4
            for k in range(T):
                end = k == T - 1 and not cut
                seg.append(tr(blob(rng) if callable(blob) else blob, float(vs[k]), float(rng.choice([-1, 1])) if end else 0.0,
                              0.0 if end else float(vs[k + 1]), end, a=int(rng.integers(0, 10)),
                              seed=int(rng.integers(1, 2 ** 31))))  # (varied actions: blank observations)
            if cut:
                break
        segs[(s // 2, s % 2)] = seg
    return segs


def test_ref_lambda_is_the_trainers_own_advantages():
    rng = np.random.default_rng(0)
    packer = Packer(9)
    segs = random_segs(rng, empty_like_blob(packer))
    for lmb in (0.754, 0.5, 0.6, 1.0):
        data = build_rollout(segs, packer, GAMMA, lmb)
        adv, ret = lam.gae(*lam.seg_arrays(segs), GAMMA, lmb)
        assert np.array_equal(adv, data["adv"]) and np.array_equal(ret, data["ret"])  # bit-identical


def test_lambda_one_gives_monte_carlo_returns():
    rng = np.random.default_rng(1)
    segs = random_segs(rng, None)
    reward, value, next_v, done, last = lam.seg_arrays(segs)
    _, ret = lam.gae(reward, value, next_v, done, last, GAMMA, 1.0)
    term = lam.terminal_rows(done, last)
    i = 0
    for seg in segs.values():  # direct discounted sums: to the episode's end, else to the cut + γ^k V(cut)
        for k in range(len(seg)):
            g, disc = 0.0, 1.0
            for t in seg[k:]:
                g += disc * t[REW]
                if t[DONE]:
                    break
                disc *= GAMMA
            else:
                g += disc * seg[-1][NEXT_V]  # truncated: λ = 1 still bootstraps (excluded from the bias rows)
            assert ret[i] == pytest.approx(g, abs=1e-5)
            assert term[i] == any(t[DONE] for t in seg[k:])
            i += 1
    assert term.any() and not term.all()


def test_bias_zero_with_an_exact_critic():
    # deterministic chain of T decisions, terminal reward R: V exact = γ^(T-1-t) R ⇒ δ = 0, every G^λ = G^MC
    segs, gamma = {}, 0.99
    for s, (T, R) in enumerate([(5, 1.0), (9, -1.0), (1, 1.0), (30, 1.0)]):
        V = [gamma ** (T - 1 - t) * R for t in range(T)]
        segs[(s, 0)] = [tr(None, V[t], R if t == T - 1 else 0.0, 0.0 if t == T - 1 else V[t + 1], t == T - 1)
                        for t in range(T)]
    st, n_term = lam.lambda_stats(*lam.seg_arrays(segs), gamma, [0.754, 0.6, 0.5])
    assert n_term == 45
    for lmb, (adv, ret, s) in st.items():
        assert np.abs(adv).max() < 1e-6, lmb
        assert abs(s["bias_mean"]) < 1e-6 and s["bias_cal"] < 1e-6 and s["bias_rms"] < 1e-6, (lmb, s)


def test_bias_tracks_critic_miscalibration():
    """Stochastic game (gambler's ruin): exact V^π ⇒ bias_mean ≈ 0 and bias_cal at the MC-noise floor, the same for
    every λ; an overconfident critic (1.3 V^π) ⇒ bias_cal grows as λ falls."""
    rng = np.random.default_rng(2)
    L, p = 5, 0.55
    r = (1 - p) / p
    vpi = lambda x: 2 * (1 - r ** (x + L)) / (1 - r ** (2 * L)) - 1  # noqa: E731
    walks = []
    for _ in range(3000):
        x, xs = 0, [0]
        while abs(x) < L:
            x += 1 if rng.random() < p else -1
            xs.append(x)
        walks.append(xs)
    res = {}
    for scale in (1.0, 1.3):
        segs = {}
        for e, xs in enumerate(walks):
            segs[(e, 0)] = [tr(None, scale * vpi(xs[k]), float(np.sign(xs[k + 1])) if k == len(xs) - 2 else 0.0,
                               0.0 if k == len(xs) - 2 else scale * vpi(xs[k + 1]), k == len(xs) - 2)
                            for k in range(len(xs) - 1)]
        res[scale] = lam.lambda_stats(*lam.seg_arrays(segs), 1.0, [0.754, 0.5])[0]
    for lmb in (0.754, 0.5):
        assert abs(res[1.0][lmb][2]["bias_mean"]) < 0.02 and res[1.0][lmb][2]["bias_cal"] < 0.03
    assert res[1.3][0.5][2]["bias_cal"] > res[1.3][0.754][2]["bias_cal"] > res[1.0][0.754][2]["bias_cal"]


def test_measure_uses_the_trainers_gradient_diagnostics_on_fixed_chunks():
    rng = np.random.default_rng(3)
    packer = Packer(9)
    segs = random_segs(rng, lambda g: rand_blob(packer, g), n_segs=40)
    args = parse_args(["--batch-size", "32", "--micro-batch", "16"])
    torch.manual_seed(0)
    net = Net(9, use_damage=False, d_model=32, n_layers=1, n_heads=4, ffn_mult=2, d_move=16, d_species=16,
              d_move_emb=16, d_item=8, d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)
    with torch.no_grad():  # the policy head's last layer starts at 0 (no trunk policy gradient): a trained-like net
        net.policy_head[2].weight.normal_(0, 0.3)
    before = [p.detach().clone() for p in net.parameters()]
    out = lam.measure(net, segs, packer, args, torch.device("cpu"), False, [0.754, 0.5], 2, chunk_seed=7)
    assert out["chunk"] == 16 and out["pairs"] == 2 and out["n"] >= 64
    # the first pair at the trainer's own λ = the trainer's diagnostic on build_rollout's data, same chunks
    from porygon.v2.model import model_keys
    data = build_rollout(segs, packer, args.gamma, args.gae_lambda)
    keys = model_keys(9) + ["drop_seed"]
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in data.items()}
    dmg = rollout_damage(net, t, keys)
    perm = torch.randperm(out["n"], generator=torch.Generator().manual_seed(7))

    def mb(idx):
        b = {k: v[idx] for k, v in t.items()}
        return {k: (b[k].long() if not b[k].is_floating_point() and b[k].dtype != torch.bool else b[k])
                for k in keys}, dmg, b
    m = Meter()
    gradient_diagnostics(net, [mb(perm[:16]), mb(perm[16:32])], args, False, m)
    assert out["lam"]["0.754"]["cos_pg0"] == pytest.approx(m.means()["diag/grad_cos_policy"], abs=1e-6)
    for k in ("0.754", "0.5"):
        assert all(np.isfinite(out["lam"][k][x]) for x in lam.METRICS), out["lam"][k]
    assert out["lam"]["0.5"]["cos_pg0"] != out["lam"]["0.754"]["cos_pg0"]
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))  # no step


def test_summary_paired_interval():
    rows = [{"lam": {"0.754": {"cos_pg": c}, "0.5": {"cos_pg": c + d}}}
            for c, d in zip(np.linspace(0, 0.1, 40), np.tile([0.01, 0.03], 20))]
    s = lam.summary(rows, [0.754, 0.5], 0.754)["delta_vs_ref"]["0.5"]["cos_pg"]
    sd = np.std(np.tile([0.01, 0.03], 20), ddof=1)
    assert s["mean"] == pytest.approx(0.02) and s["n"] == 40
    assert s["hi"] - s["mean"] == pytest.approx(2.0227 * sd / np.sqrt(40), rel=1e-3)
    assert lam.t95(39) == pytest.approx(2.0227, abs=1e-3)


def test_trainer_args_from_the_checkpoint_and_pool_links(tmp_path):
    rec = ["--collector", "fastenv", "--n-battles", "16", "--pool-battles", "16", "--batch-size", "4096",
           "--n-epochs", "3", "--no-overlap", "--stacked-opp", "--snapshot-cache", "72", "--p-frozen", "0.67",
           "--frozen", "/c/P3.pt", "--kl-anchor", "/c/P3.pt", "--fastenv-workers", "28", "--compile",
           "--run-dir", "/c/run", "--seed", "5"]
    saved = {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "frozen" else v)
             for k, v in vars(parse_args(rec)).items()}  # = train.py checkpoint()'s args
    o = type("O", (), {"ckpt": "/c/ck.pt", "scratch": str(tmp_path / "s"), "seed": 9})()
    ns = lam.trainer_args(saved, [], o)
    assert (ns.batch_size, ns.n_epochs, ns.snapshot_cache, ns.p_frozen, ns.stacked_opp, ns.frozen, ns.compile) == \
        (4096, 3, 72, 0.67, True, [Path("/c/P3.pt")], True)
    assert (ns.fastenv_workers, ns.seed, ns.no_overlap, ns.kl_anchor, ns.resume) == (8, 9, True, None, Path("/c/ck.pt"))
    assert ns.pool_dir == tmp_path / "s" / "pool" and ns.gae_lambda == 0.754
    ns = lam.trainer_args(saved, ["--snapshot-cache", "16", "--fastenv-workers", "0", "--no-compile"], o)
    assert (ns.snapshot_cache, ns.fastenv_workers, ns.compile, ns.batch_size) == (16, 0, False, 4096)
    src = tmp_path / "pool"
    (src / "frozen").mkdir(parents=True)
    for name in ("00002500000.pt", "00045000000.pt", "00047500000.pt", "frozen/P3.pt", "frozen/best-00040000000.pt",
                 "frozen/best-00042500000.pt", "frozen/best-00050000000.pt"):
        (src / name).write_bytes(b"x")
    lam.link_pool(src, ns.pool_dir, 45_004_575)
    got = sorted(str(p.relative_to(ns.pool_dir)) for p in ns.pool_dir.rglob("*.pt"))
    assert got == ["00002500000.pt", "00045000000.pt", "frozen/P3.pt", "frozen/best-00042500000.pt"]
