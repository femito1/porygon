"""Trainer v2 distillation term (--distill): the labelled set (split by game, gen check, reload), the extra CE pass in
`ppo_update` (bit-identical when off or at coef 0, learns a fixed target, skipped in the critic warm-up) and the
log-line fields. CPU, tiny net, synthetic shards in tmp_path (deleted after each test: the laptop disk is full)."""

import copy
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from porygon.v2.model import Net
from porygon.v2.pretrain import is_val
from porygon.v2.spec import aux_fields, fields, n_actions
from porygon.v2.train import HEADS_ONLY, DistillSet, Trainer, distill_metrics, parse_args, ppo_update

GEN = 9
CPU = torch.device("cpu")
TINY = dict(d_model=32, n_layers=1, n_heads=4, ffn_mult=1, d_move=16, d_species=16, d_move_emb=16, d_item=8,
            d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)
LEGAL = [0, 1, 2, 6]  # labelled rows: three switches (identical tokens on zero observations) and one move
TARGET = 6
ILLEGAL = 9


@pytest.fixture
def tmp(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def tiny_net(seed: int = 0) -> Net:
    try:
        import porygon.v2.damage  # noqa: F401
        use_damage = True
    except ImportError:
        use_damage = False
    torch.manual_seed(seed)
    return Net(GEN, use_damage=use_damage, **TINY)


def zero_obs(n: int, legal, gen: int = GEN) -> dict[str, np.ndarray]:
    obs = {k: np.zeros((n, *shape), dtype) for k, (shape, dtype) in fields(gen).items()}
    obs["action_mask"][:, legal] = 1.0
    return obs


def write_shard(path, games, per_game: int = 4, tag: bool = False, gen: int = GEN) -> int:
    """`per_game` rows per game id; target = one-hot of TARGET + 0.5 on an illegal action every other row (must be
    masked out); row 1 has no target and row 3 only illegal mass (both dropped). `tag`: field_num[:, 0] = game id."""
    games = np.asarray(list(games), np.int64)
    n = len(games) * per_game
    obs = zero_obs(n, LEGAL, gen)
    game = np.repeat(games, per_game)
    if tag:
        obs["field_num"][:, 0] = game
    tgt = np.zeros((n, n_actions(gen)), np.float32)
    tgt[:, TARGET] = 1.0
    tgt[::2, ILLEGAL] = 0.5
    tgt[1] = 0.0
    tgt[3] = 0.0
    tgt[3, ILLEGAL] = 1.0
    np.savez(path, **{f"obs/{k}": v for k, v in obs.items()}, target=tgt, z=np.zeros(n, np.float32), game=game)
    return n


def np_state():
    s = np.random.get_state()
    return s[1].copy(), s[2:]


def same(x: float, y: float) -> bool:
    return x == y or (np.isnan(x) and np.isnan(y))


def rollout(n: int = 64, adv: bool = True, seed: int = 0) -> dict[str, np.ndarray]:
    """A ppo_update input: zero observations, four legal actions, no aux targets; adv=False: zero advantages (pg 0)."""
    rng = np.random.default_rng(seed)
    legal = [0, 1, 6, 22]
    data = zero_obs(n, legal)
    for k, (shape, dtype) in aux_fields(GEN).items():
        data[k] = np.full((n, *shape), 0 if k == "aux_opp_target" else -1, dtype)
    data["action"] = rng.choice(legal, n).astype(np.int64)
    data["logp"] = np.full(n, -np.log(len(legal)), np.float32)
    data["value"] = np.zeros(n, np.float32)
    data["drop_seed"] = rng.integers(1, 2**31 - 1, n)
    data["adv"] = rng.normal(size=n).astype(np.float32) if adv else np.zeros(n, np.float32)
    data["ret"] = rng.choice([-1.0, 1.0], n).astype(np.float32)
    data["opp_cls"] = np.full(n, -1, np.int64)
    return data


def test_off_and_monitor_only_are_bit_identical(tmp):
    a = parse_args([])
    assert (a.distill, a.distill_coef, a.distill_batch, a.distill_val_frac, a.distill_reload_every) == \
        (None, 0.0, 512, 0.06, 0)
    write_shard(tmp / "shard-0000.npz", range(40))
    ds = DistillSet(str(tmp / "shard-*.npz"), GEN, 0.3, np.random.default_rng(0))
    args = parse_args(["--batch-size", "32", "--n-epochs", "2"])
    data, ref = rollout(), tiny_net()
    runs = {}
    # no set (old call), distill=None explicitly, and a set at coef 0 (held-out monitor only)
    for name, kw in (("default", {}), ("none", {"distill": None}), ("monitor", {"distill": ds})):
        net = copy.deepcopy(ref)
        opt = torch.optim.AdamW(net.param_groups(0.01), lr=1e-3)
        torch.manual_seed(0)
        np.random.seed(0)
        st = ppo_update(net, opt, data, args, CPU, False, True, **kw)
        runs[name] = (net, st, torch.get_rng_state(), np_state())
    net0, st0, trng0, nrng0 = runs["default"]
    for name in ("none", "monitor"):
        net, st, trng, nrng = runs[name]
        assert all(torch.equal(p, q) for p, q in zip(net0.parameters(), net.parameters())), name
        assert torch.equal(trng, trng0) and np.array_equal(nrng[0], nrng0[0]) and nrng[1] == nrng0[1], name
        assert "loss/distill" not in st
        assert all(same(st[k], st0[k]) for k in st0 if not k.startswith("perf/")), name
    assert not any("distill" in k for k in st0) and not any("distill" in k for k in runs["none"][1])
    mon = runs["monitor"][1]
    for k in ("diag/distill_val_ce", "diag/distill_val_top1", "diag/distill_train_ce", "diag/distill_train_top1"):
        assert np.isfinite(mon[k]), k
    assert runs["monitor"][0].training


def test_distill_learns_the_target(tmp):
    write_shard(tmp / "shard-0000.npz", range(0, 30))
    write_shard(tmp / "shard-0001.npz", range(30, 60))
    ds = DistillSet(str(tmp / "shard-*.npz"), GEN, 0.3, np.random.default_rng(0))
    assert len(ds.val_idx) and len(ds.train_idx)
    args = parse_args(["--batch-size", "32", "--n-epochs", "4", "--distill-coef", "1.0", "--distill-batch", "64"])
    net = tiny_net()
    opt = torch.optim.AdamW(net.param_groups(0.01), lr=1e-2)
    data = rollout(adv=False)  # pg = 0: only the distillation term pulls the policy towards TARGET
    torch.manual_seed(0)
    sts = [ppo_update(net, opt, data, args, CPU, False, i == 0, distill=ds) for i in range(3)]
    start = sts[0]  # measured at the update's starting weights: the zero-init policy head is uniform
    assert start["diag/distill_val_ce"] == pytest.approx(np.log(len(LEGAL)), abs=1e-4)
    assert start["diag/distill_train_ce"] == pytest.approx(np.log(len(LEGAL)), abs=1e-4)
    for st in sts:
        assert all(np.isfinite(v) for k, v in st.items() if k.startswith(("loss/", "diag/distill"))), st
    assert sts[-1]["loss/distill"] < 0.5 * sts[0]["loss/distill"]
    end = distill_metrics(net, ds, CPU, False)
    assert net.training
    assert end["diag/distill_val_top1"] == 1.0 and end["diag/distill_train_top1"] == 1.0
    assert end["diag/distill_val_ce"] < 0.25 * start["diag/distill_val_ce"]


def test_distill_set_split_by_game(tmp):
    games = range(1000, 1060)
    n = write_shard(tmp / "shard-0000.npz", games[:40], tag=True)
    n += write_shard(tmp / "shard-0001.npz", games[30:], tag=True)  # games 1030-1039 span both shards
    ds = DistillSet(str(tmp / "shard-*.npz"), GEN, 0.3, np.random.default_rng(0))
    assert len(ds.train_idx) + len(ds.val_idx) == n - 4  # rows 1 and 3 of each shard: no legal target
    assert (ds.data["target"][:, TARGET] == 1).all() and ds.data["action_mask"].shape[1] == n_actions(GEN)
    g = ds.data["game"]
    train_g, val_g = set(g[ds.train_idx].tolist()), set(g[ds.val_idx].tolist())
    assert train_g and val_g and not train_g & val_g and train_g | val_g == set(games)
    assert val_g == {x for x in games if is_val(np.array([x]), 0.3)[0]}  # = porygon.v2.pretrain's split
    assert (ds.n_games, ds.n_val_games) == (60, len(val_g))
    assert set(ds.monitor["val"]) == set(ds.val_idx) and set(ds.monitor["train"]) == set(ds.train_idx)  # < 4096
    # sampling: train rows only, nonzero int64 drop seeds, the given generator only (no global draws)
    t_state, n_state = torch.get_rng_state(), np_state()
    b = ds.sample(500, np.random.default_rng(1))
    assert torch.equal(t_state, torch.get_rng_state())
    assert np.array_equal(n_state[0], np_state()[0]) and n_state[1] == np_state()[1]
    assert set(b["field_num"][:, 0].astype(np.int64).tolist()) <= train_g and "game" not in b
    assert b["drop_seed"].dtype == np.int64 and (b["drop_seed"] > 0).all() and len(b["target"]) == 500
    b2 = ds.sample(500, np.random.default_rng(1))
    assert np.array_equal(b["drop_seed"], b2["drop_seed"]) and np.array_equal(b["field_num"], b2["field_num"])
    # wrong generation: by shape, and by the data stream's meta.json
    (tmp / "g4").mkdir()
    write_shard(tmp / "g4" / "shard-0000.npz", range(5), gen=4)
    with pytest.raises(AssertionError, match="not gen9"):
        DistillSet(str(tmp / "g4" / "shard-*.npz"), GEN, 0.3)
    (tmp / "meta").mkdir()
    write_shard(tmp / "meta" / "shard-0000.npz", range(5))
    (tmp / "meta" / "meta.json").write_text(json.dumps({"gen": 4}))
    with pytest.raises(AssertionError, match="gen 4 shards"):
        DistillSet(str(tmp / "meta" / "shard-*.npz"), GEN, 0.3)
    with pytest.raises(AssertionError, match="no shards"):
        DistillSet(str(tmp / "none-*.npz"), GEN, 0.3)


def test_heads_only_skips_distill(tmp):
    write_shard(tmp / "shard-0000.npz", range(40))
    ds = DistillSet(str(tmp / "shard-*.npz"), GEN, 0.3, np.random.default_rng(0))
    args = parse_args(["--batch-size", "32", "--n-epochs", "2", "--distill-coef", "1.0", "--distill-batch", "64"])
    net = tiny_net()
    before = {k: p.detach().clone() for k, p in net.named_parameters()}
    opt = torch.optim.AdamW(net.param_groups(0.01), lr=1e-2)
    torch.manual_seed(0)
    st = ppo_update(net, opt, rollout(), args, CPU, False, True, heads_only=True, distill=ds)
    assert "loss/distill" not in st and st["perf/heads_only"] == 1.0
    assert np.isfinite(st["diag/distill_val_ce"])  # the monitor still runs (the starting policy's agreement)
    changed = {k for k, p in net.named_parameters() if not torch.equal(p, before[k])}
    assert changed and all(k.startswith(HEADS_ONLY) for k in changed), sorted(changed)
    assert all(p.requires_grad for p in net.parameters())


def test_reload_and_log_line(tmp):
    write_shard(tmp / "shard-0000.npz", range(100, 120))
    rng = np.random.default_rng(1)
    ds = DistillSet(str(tmp / "shard-*.npz"), GEN, 0.3, rng)
    assert ds.reload() is None  # nothing changed
    write_shard(tmp / "shard-0001.npz", range(120, 140))
    logs = []
    fake = SimpleNamespace(distill=ds, log=logs.append)  # the two Trainer attributes _reload_distill uses
    Trainer._reload_distill(fake)
    new = fake.distill
    assert new is not ds and new.rng is rng and len(new.paths) == 2 and "reloaded" in logs[-1]
    assert len(new.train_idx) + len(new.val_idx) == 2 * (len(ds.train_idx) + len(ds.val_idx))
    (tmp / "shard-0002.npz").write_bytes(b"not a zip")  # e.g. a shard being written
    Trainer._reload_distill(fake)
    assert fake.distill is new and "reload failed" in logs[-1]
    (tmp / "shard-0002.npz").unlink()
    n_logs = len(logs)
    Trainer._reload_distill(fake)  # back to the loaded shards: unchanged, nothing reloaded
    assert fake.distill is new and len(logs) == n_logs
    st = {"perf/steps_per_s": 1.0, "perf/collect_steps_per_s": 1.0, "perf/games_per_s": 1.0, "perf/lr": 1e-4,
          "perf/update_s": 2.0, "loss/distill": 0.25}
    fake = SimpleNamespace(counters={"updates": 3, "steps": 10})
    assert " wr_frozen - ptop - dst 0.250 dval - upd_s 2.0 | " in Trainer._line(fake, 10, st)  # new fields before upd_s
    st["diag/distill_val_ce"] = 1.5
    assert " dst 0.250 dval 1.500 upd_s " in Trainer._line(fake, 10, st)
