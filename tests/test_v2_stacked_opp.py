"""--stacked-opp: the stacked pool-opponent forward (`porygon.v2.stacked.OppStack`), the trainer's batched opponent
call (`Trainer._opp_act_many`) and the fastenv collector's switch (`FastCollector(opp_fn_many=...)`).

Parity: stacked logits = each net's own forward up to fp32 rounding (production gen9 architecture, damage layer on,
row counts 40 / 17 / 1 / 25, nets whose own move-slot trim differs, an ignored cfg difference); a larger trim than a
net's own changes its logits only by rounding; padding / other groups' rows never reach a group's logits. Trainer:
same nets, cache, loads and load accounting as per-item `_opp_act` (same actions from the same seed once the nets
are cached), a different architecture takes the per-net forward, a missing file or a wrong-gen net plays uniformly
and is recorded, stacks follow the snapshot cache. Collector: flag off -> opp_fn per name (old path), on -> one
opp_fn_many call per step. Synthetic observations (structurally valid: present / fainted / absent mons, varying move
slots, tera: porygon.v2.synthetic); with node, also real fastenv observations end to end."""

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from porygon.v2.model import Net, batch_to_tensors, model_keys, save
from porygon.v2.stacked import OppStack, signature
from porygon.v2.synthetic import synthetic_obs as synth_obs
from porygon.v2.train import PoolLoads, Trainer, parse_args

CPU = torch.device("cpu")
PROD = dict(n_layers=4, ffn_mult=1)  # the production gen9 net (d_model 256, 8 heads, damage layer on)
SMALL = dict(d_model=32, n_layers=2, n_heads=4, ffn_mult=1, d_move=16, d_species=16, d_move_emb=16, d_item=8,
             d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)
TOL = 1e-4  # max abs difference allowed on legal-action logits


@pytest.fixture
def tmp(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)  # the laptop disk is nearly full


def own_trim(ob: dict) -> int:
    used = np.flatnonzero((ob["moves"] != 0).any((0, 1)))
    return max(4, int(used.max()) + 1) if len(used) else 4


def make_net(seed: int, gen: int = 9, **cfg) -> Net:
    """A random net whose zero-initialised output layers (policy head, pair bias) are not zero."""
    torch.manual_seed(seed)
    net = Net(gen, **cfg).eval()
    with torch.no_grad():
        net.policy_head[-1].weight.normal_(std=0.5)
        net.policy_head[-1].bias.normal_(std=0.1)
        if hasattr(net, "pair_mlp"):
            net.pair_mlp[-1].weight.normal_(std=0.3)
    return net


@torch.no_grad()
def per_net(net: Net, ob: dict) -> torch.Tensor:
    """The existing per-net path's logits (Trainer._opp_logits)."""
    return net(batch_to_tensors(ob, "cpu", keys=model_keys(net.gen) + ["drop_seed"]), aux=False)["logits"].float()


def legal_diff(ref: torch.Tensor, got: torch.Tensor, ob: dict) -> float:
    legal = torch.from_numpy(ob["action_mask"]) > 0
    assert torch.equal(ref[~legal], got[~legal])  # masked logits: exactly float32 min
    return float((ref - got)[legal].abs().max()) if legal.any() else 0.0


# ---------------------------------------------------------------- OppStack parity

@pytest.fixture(scope="module")
def prod_nets():
    """4 production-size gen9 nets; the last differs in cfg keys without effect on the logits (same signature)."""
    nets = [make_net(s, **PROD) for s in range(3)] + [make_net(3, **PROD, id_dropout=0.0, value_hidden=128)]
    assert len({signature(n) for n in nets}) == 1
    return nets


def test_stacked_matches_per_net(prod_nets):
    rng = np.random.default_rng(1)
    rows, max_moves = [40, 17, 1, 25], [5, 10, 4, 7]
    obs = [synth_obs(9, r, m, rng) for r, m in zip(rows, max_moves)]
    assert [own_trim(o) for o in obs] == [5, 10, 4, 7]  # every net's own trim differs; the union's is 10
    ref = [per_net(n, o) for n, o in zip(prod_nets, obs)]
    stack = OppStack(prod_nets[0])
    out, refills = stack.logits(prod_nets, obs)
    assert refills == 4 and [o.shape for o in out] == [r.shape for r in ref]
    diffs = [legal_diff(r, o, ob) for r, o, ob in zip(ref, out, obs)]
    print(f"stacked vs per-net, max |logit diff| on legal actions per net: {[f'{d:.2e}' for d in diffs]}")
    assert max(diffs) <= TOL
    # the same nets again, in another order: nothing copied, same logits
    perm = [2, 0, 3, 1]
    out2, refills = stack.logits([prod_nets[i] for i in perm], [obs[i] for i in perm])
    assert refills == 0 and all(torch.equal(out2[j], out[i]) for j, i in enumerate(perm))
    # a subset with another row count: net 1 keeps slot 1, net 3 is copied into slot 0; still the per-net logits
    sub = {k: v[:9] for k, v in obs[3].items()}
    out3, refills = stack.logits([prod_nets[3], prod_nets[1]], [sub, obs[1]])
    assert refills == 1 and stack.capacity == 4
    assert max(legal_diff(per_net(prod_nets[3], sub), out3[0], sub), legal_diff(ref[1], out3[1], obs[1])) <= TOL


def test_larger_trim_changes_logits_by_rounding_only(prod_nets):
    """Net.forward trims the move slots to the batch's own k; the stacked forward uses the union's (>= own) k."""
    net, ob = prod_nets[0], synth_obs(9, 24, 5, np.random.default_rng(2))
    assert own_trim(ob) == 5
    own = per_net(net, ob)
    net.cfg["trim_moves"], net.damage.trim = False, False  # k = 24 in the net and in its damage layer
    try:
        full = per_net(net, ob)
    finally:
        net.cfg["trim_moves"], net.damage.trim = True, True
    d = legal_diff(own, full, ob)
    print(f"trim 5 vs 24: max |logit diff| on legal actions {d:.2e}")
    assert d <= 1e-5


def test_rows_do_not_mix(prod_nets):
    """A group's logits do not depend on the other groups' rows or on its padding (same shapes: bitwise equal)."""
    rng = np.random.default_rng(3)
    a, b1, b2 = synth_obs(9, 6, 6, rng), synth_obs(9, 11, 6, rng), synth_obs(9, 11, 6, rng)
    stack = OppStack(prod_nets[0])
    x = stack.logits(prod_nets[:2], [a, b1])[0][0]  # a padded from 6 to 11 rows with copies of its last row
    y = stack.logits(prod_nets[:2], [a, b2])[0][0]
    assert torch.equal(x, y)
    assert legal_diff(per_net(prod_nets[0], a), x, a) <= TOL


@pytest.mark.parametrize("gen,cfg", [(4, SMALL), (9, dict(SMALL, use_damage=False)),
                                     (9, dict(SMALL, qk_norm=False, mask_fainted=False, use_tables=False))])
def test_stacked_matches_per_net_variants(gen, cfg):
    """Other branches of the forward: gen 4 (no tera tokens), no damage layer, no qk-norm / fainted mask / tables."""
    rng = np.random.default_rng(4)
    nets = [make_net(s, gen, **cfg) for s in range(3)]
    obs = [synth_obs(gen, r, m, rng) for r, m in zip([7, 1, 12], [6, 4, 9])]
    out, _ = OppStack(nets[0]).logits(nets, obs)
    assert max(legal_diff(per_net(n, o), x, o) for n, o, x in zip(nets, obs, out)) <= TOL


def test_ignored_cfg_keys_stack_exactly():
    """Nets differing only in IGNORED_CFG keys (id dropout, value head width, static tables) share a stack and keep
    their own logits."""
    rng = np.random.default_rng(8)
    nets = [make_net(60, 9, **SMALL), make_net(61, 9, **dict(SMALL, use_tables=False)),
            make_net(62, 9, **dict(SMALL, id_dropout=0.0, value_hidden=16))]
    assert len({signature(n) for n in nets}) == 1
    obs = [synth_obs(9, r, 7, rng) for r in (5, 3, 8)]
    out, _ = OppStack(nets[0]).logits(nets, obs)
    assert max(legal_diff(per_net(n, o), x, o) for n, o, x in zip(nets, obs, out)) <= TOL


def test_signature_separates_architectures():
    a, b = make_net(0, 9, **SMALL), make_net(1, 9, **SMALL)
    assert signature(a) == signature(b)
    assert signature(make_net(2, 9, **dict(SMALL, ffn_mult=2))) != signature(a)
    assert signature(make_net(3, 4, **SMALL)) != signature(a)
    assert signature(make_net(4, 9, **dict(SMALL, mask_fainted=False))) != signature(a)
    with pytest.raises(AssertionError):
        OppStack(a).bind([a, make_net(5, 9, **dict(SMALL, n_layers=1))])


# ---------------------------------------------------------------- Trainer._opp_act_many

def fake_trainer(tmp: Path, *argv: str, gen: int = 9) -> Trainer:
    """A Trainer without collector or GPU: what `_opp_act` / `_opp_act_many` use."""
    t = object.__new__(Trainer)
    t.args = parse_args(["--format", f"gen{gen}randombattle", "--run-dir", str(tmp / "run"), *argv])
    t.pool_dir = tmp / "run" / "pool"
    t.pool_dir.mkdir(parents=True)
    t.gen, t.device, t.snapshots, t.compile, t.amp, t.pool_loads = gen, CPU, {}, False, False, PoolLoads()
    return t


def test_opp_act_many_matches_opp_act(tmp):
    t = fake_trainer(tmp)
    for i in range(3):
        save(make_net(10 + i, 9, **SMALL), t.pool_dir / f"{100 * (i + 1):011d}.pt")
    save(make_net(13, 9, **dict(SMALL, ffn_mult=2)), t.pool_dir / "frozen" / "p1.pt")  # another architecture
    rng = np.random.default_rng(5)
    names = ["00000000200.pt", "frozen/p1.pt", "00000000100.pt", "00000000300.pt"]
    items = [(name, synth_obs(9, r, m, rng)) for name, r, m in zip(names, [9, 4, 1, 14], [5, 8, 4, 6])]
    acts, st = t._opp_act_many(items)  # loads all four
    assert st == {"opp_fwds": 2, "opp_stacked_fwds": 1, "opp_stacked_nets": 3, "opp_stack_refills": 3,
                  "opp_uniform": 0}
    assert set(t.snapshots) == set(names) and len(t.opp_stacks) == 1
    loads, _ = t.pool_loads.take()
    assert (loads["pool/load_attempts_snapshot"], loads["pool/load_attempts_frozen"]) == (3, 1)
    assert loads["pool/load_failures_snapshot"] == loads["pool/load_failures_frozen"] == 0
    for a, (_, ob) in zip(acts, items):
        legal = ob["action_mask"] > 0
        assert all(legal[r, x] or not legal[r].any() for r, x in enumerate(a))
    # cached nets: the same seed gives the same actions as one _opp_act per item (logits equal up to rounding)
    for seed in range(5):
        torch.manual_seed(seed)
        many, st = t._opp_act_many(items)
        torch.manual_seed(seed)
        one = [t._opp_act(name, ob) for name, ob in items]
        assert all(np.array_equal(x, y) for x, y in zip(many, one))
        assert st["opp_stack_refills"] == 0
    assert t.pool_loads.take()[0]["pool/load_attempts_snapshot"] == 0  # no reload: the same cache as _opp_act


def test_opp_act_many_fallbacks(tmp):
    t = fake_trainer(tmp)
    save(make_net(20, 9, **SMALL), t.pool_dir / "a.pt")
    save(make_net(21, 9, **SMALL), t.pool_dir / "b.pt")
    save(make_net(22, 4, **SMALL), t.pool_dir / "frozen" / "g4.pt")  # a gen-4 net in a gen-9 run
    save(make_net(23, 9, **dict(SMALL, n_layers=1)), t.pool_dir / "c.pt")  # another architecture
    rng = np.random.default_rng(6)
    names = ["a.pt", "missing.pt", "frozen/g4.pt", "c.pt", "b.pt"]
    items = [(name, synth_obs(9, 8, 6, rng)) for name in names]
    for ob in (items[1][1], items[2][1]):
        ob["action_mask"][:] = 0
        ob["action_mask"][:, [0, 6]] = 1  # the failed nets play uniformly over these two
    acts, st = t._opp_act_many(items)
    assert st == {"opp_fwds": 2, "opp_stacked_fwds": 1, "opp_stacked_nets": 2, "opp_stack_refills": 2,
                  "opp_uniform": 2}
    assert set(acts[1].tolist()) <= {0, 6} and set(acts[2].tolist()) <= {0, 6}
    loads, msgs = t.pool_loads.take()
    assert (loads["pool/load_attempts_snapshot"], loads["pool/load_failures_snapshot"]) == (4, 1)
    assert (loads["pool/load_attempts_frozen"], loads["pool/load_failures_frozen"]) == (1, 1)
    assert any("missing.pt: FileNotFoundError" in m for m in msgs)
    assert any("g4.pt: AssertionError('gen 4 snapshot')" in m for m in msgs)
    assert set(t.snapshots) == {"a.pt", "b.pt", "c.pt"}
    # the per-net path's logits for c (alone in its architecture), the stacked ones for a and b
    obs = {name: ob for name, ob in items}
    for name in ("a.pt", "b.pt", "c.pt"):
        ref = per_net(t.snapshots[name], obs[name])
        got = t._opp_logits(t.snapshots[name], obs[name])
        assert legal_diff(ref, got, obs[name]) == 0.0
    (stack,) = t.opp_stacks.values()
    got = stack.logits([t.snapshots["b.pt"], t.snapshots["a.pt"]], [obs["b.pt"], obs["a.pt"]])[0]
    assert max(legal_diff(per_net(t.snapshots[n], obs[n]), g, obs[n]) for n, g in zip(["b.pt", "a.pt"], got)) <= TOL
    # failed loads are retried every call (as _opp_act), never cached
    t._opp_act_many(items)
    loads, _ = t.pool_loads.take()
    assert (loads["pool/load_failures_snapshot"], loads["pool/load_failures_frozen"]) == (1, 1)


def test_stacks_follow_the_snapshot_cache(tmp):
    """Slots of evicted nets are released; a stack without a cached net is dropped; the cache bounds a stack."""
    t = fake_trainer(tmp, "--snapshot-cache", "2")
    for i in range(3):
        save(make_net(30 + i, 9, **SMALL), t.pool_dir / f"x{i}.pt")
    for i in range(2):
        save(make_net(40 + i, 9, **dict(SMALL, ffn_mult=2)), t.pool_dir / f"y{i}.pt")
    rng = np.random.default_rng(7)
    ob = lambda: synth_obs(9, 5, 6, rng)  # noqa: E731
    _, st = t._opp_act_many([("x0.pt", ob()), ("x1.pt", ob())])
    assert st["opp_stacked_fwds"] == 1 and len(t.opp_stacks) == 1
    (sig_x,) = t.opp_stacks
    _, st = t._opp_act_many([("y0.pt", ob()), ("y1.pt", ob())])  # evicts x0, x1
    assert set(t.snapshots) == {"y0.pt", "y1.pt"}
    assert sig_x not in t.opp_stacks and len(t.opp_stacks) == 1  # x's stack held no cached net any more
    # three nets of one architecture with a cache of 2: stacks of at most 2 (and one per-net forward)
    _, st = t._opp_act_many([(f"x{i}.pt", ob()) for i in range(3)])
    assert (st["opp_fwds"], st["opp_stacked_fwds"], st["opp_stacked_nets"]) == (2, 1, 2)
    assert all(s.capacity <= 2 for s in t.opp_stacks.values())
    assert all(s is None or any(s is n for n in t.snapshots.values())
               for stack in t.opp_stacks.values() for s in stack.slots)


def test_stacked_failure_falls_back_to_per_net(tmp, monkeypatch):
    """A stacked forward that raises (e.g. CUDA out of memory) is logged once; this and later steps act per net."""
    import porygon.v2.stacked as stacked_mod
    t = fake_trainer(tmp)
    for i in range(2):
        save(make_net(70 + i, 9, **SMALL), t.pool_dir / f"s{i}.pt")
    rng = np.random.default_rng(9)
    items = [(f"s{i}.pt", synth_obs(9, 6, 6, rng)) for i in range(2)]

    def boom(self, nets, obs_list):
        raise RuntimeError("CUDA out of memory (test)")

    monkeypatch.setattr(stacked_mod.OppStack, "logits", boom)
    acts, st = t._opp_act_many(items)
    assert (st["opp_fwds"], st["opp_stacked_fwds"], st["opp_stack_errors"]) == (2, 0, 1)
    assert all(len(a) == 6 for a in acts) and not t.opp_stacks
    monkeypatch.undo()
    _, st = t._opp_act_many(items)
    assert (st["opp_fwds"], st["opp_stacked_fwds"]) == (2, 0) and "opp_stack_errors" not in st
    _, msgs = t.pool_loads.take()
    assert msgs == ["--stacked-opp: the stacked forward failed (RuntimeError('CUDA out of memory (test)')); per-net "
                    "opponent forwards from now on"]


# ---------------------------------------------------------------- the switch

def test_flag_wiring(tmp, monkeypatch):
    """--stacked-opp off (default): FastCollector gets no opp_fn_many (its old per-name loop); on: the trainer's."""
    import porygon.fastenv_v2 as fv2
    seen = []

    class FakeCollector:
        def __init__(self, *a, **kw):
            seen.append(kw)

    env = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(fv2, "build_env", lambda args, seed=None: SimpleNamespace(env=env, pool_per_worker=0,
                                                                                    restarts=0, close=lambda: None))
    monkeypatch.setattr(fv2, "FastCollector", FakeCollector)
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    small = ["--d-model", "16", "--n-layers", "1", "--n-heads", "2", "--ffn-mult", "1", "--d-move", "8",
             "--model-json", '{"d_species":8,"d_move_emb":8,"d_item":4,"d_ability":4,"d_type":4,"pair_hidden":4,'
                             '"value_hidden":8}', "--no-damage", "--ema-decay", "0"]
    try:
        for i, extra in enumerate([[], ["--stacked-opp"]]):
            args = parse_args(["--collector", "fastenv", "--device", "cpu", "--run-dir", str(tmp / f"r{i}"), *small,
                               *extra])
            tr = Trainer(args)
            tr.log_file.close()
            tr.writer.close()
            if extra:
                assert seen[-1]["opp_fn_many"] == tr._opp_act_many
            else:
                assert seen[-1]["opp_fn_many"] is None and args.stacked_opp is False
    finally:
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision(precision)


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node (fastenv)")


@needs_node
@pytest.mark.parametrize("many", [False, True])
def test_collector_switch(tmp, many):
    """FastCollector: without opp_fn_many, opp_fn per snapshot name and step (unchanged); with it, one call per step
    with every name, opp_fn unused; counters perf/col_opp_fwds (forwards) and perf/col_opp_fwd_nets (names)."""
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    for name in ("snap_a.pt", "snap_b.pt"):
        (tmp / name).write_bytes(b"")
    rng = np.random.default_rng(1)
    single, calls = [], []

    def act(batch):
        mask = batch["action_mask"] > 0
        probs = mask / np.maximum(mask.sum(1, keepdims=True), 1)
        acts = (rng.random(mask.shape) * mask).argmax(1)
        n = len(acts)
        return acts, np.zeros(n), np.zeros(n, np.float32), probs.astype(np.float32)

    def opp(name, batch):
        single.append(name)
        assert "aux_item" not in batch and not batch["drop_seed"].any()
        return (rng.random(batch["action_mask"].shape) * (batch["action_mask"] > 0)).argmax(1)

    def opp_many(items):
        calls.append([name for name, _ in items])
        assert len(set(calls[-1])) == len(items)
        acts = [(rng.random(b["action_mask"].shape) * (b["action_mask"] > 0)).argmax(1) for _, b in items]
        return acts, {"opp_fwds": 1, "opp_test_calls": 1}

    workers = FastWorkers(FastEnv("gen9randombattle", 1, 6, seed=3, encoder="v2", shaping=(0, 0, 0)))
    try:
        col = FastCollector(workers.env, Packer(9), act, opp, seed=0, pool_per_worker=4, pool_dir=tmp,
                            opp_fn_many=opp_many if many else None)
        while len([e for e in col.episodes if e["kind"] == "pool"]) < 4:
            col.step()
        st = col.rollout_stats()
        if many:
            assert not single and calls and max(map(len, calls)) == 2
            assert st["perf/col_opp_fwds"] == st["perf/col_opp_test_calls"] == len(calls)
            assert st["perf/col_opp_fwd_nets"] == sum(map(len, calls))
        else:
            assert single and not calls and "perf/col_opp_fwd_nets" not in st
            assert st["perf/col_opp_fwds"] == len(single)
        assert col.opp_decisions > 0
    finally:
        workers.close()


@needs_node
def test_end_to_end_on_fastenv(tmp, prod_nets):
    """The trainer's _opp_act / _opp_act_many on a fastenv collector (small saved nets: 2 snapshots + a frozen net of
    another architecture); then stacked vs per-net logits of the production nets on the recorded real batches."""
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    t = fake_trainer(tmp)
    save(make_net(50, 9, **SMALL), t.pool_dir / "00000000100.pt")
    save(make_net(51, 9, **SMALL), t.pool_dir / "00000000200.pt")
    save(make_net(52, 9, **dict(SMALL, ffn_mult=2)), t.pool_dir / "frozen" / "p1.pt")
    recorded = []

    def opp_many(items):
        recorded.append(items)
        return t._opp_act_many(items)

    rng = np.random.default_rng(2)

    def act(batch):
        mask = batch["action_mask"] > 0
        probs = mask / np.maximum(mask.sum(1, keepdims=True), 1)
        acts = (rng.random(mask.shape) * mask).argmax(1)
        n = len(acts)
        return acts, np.zeros(n), np.zeros(n, np.float32), probs.astype(np.float32)

    workers = FastWorkers(FastEnv("gen9randombattle", 1, 8, seed=5, encoder="v2", shaping=(0, 0, 0)))
    try:
        col = FastCollector(workers.env, Packer(9), act, t._opp_act, seed=1, pool_per_worker=6, pool_dir=tmp / "run"
                            / "pool", p_frozen=0.4, opp_fn_many=opp_many)
        while len([e for e in col.episodes if e["kind"] == "pool"]) < 6:
            col.step()
        st = col.rollout_stats()
    finally:
        workers.close()
    assert st["perf/col_opp_stacked_fwds"] > 0 and st["perf/col_opp_uniform"] == 0
    assert st["perf/col_opp_fwds"] < st["perf/col_opp_fwd_nets"] == sum(map(len, recorded))
    assert st["perf/col_opp_stack_refills"] == 2  # each snapshot copied into the stack once
    assert {name for items in recorded for name, _ in items} == {"00000000100.pt", "00000000200.pt", "frozen/p1.pt"}
    loads, _ = t.pool_loads.take()
    assert loads["pool/load_failures_snapshot"] == loads["pool/load_failures_frozen"] == 0
    print(f"fastenv: {st['perf/col_opp_fwd_nets']:.0f} opponent batches in {st['perf/col_opp_fwds']:.0f} forwards "
          f"({st['perf/col_opp_stacked_fwds']:.0f} stacked, {st['perf/col_opp_stacked_nets']:.0f} nets)")
    # real observations: stacked production nets vs their own forwards
    multi = sorted(recorded, key=len, reverse=True)[:3]
    assert len(multi[-1]) >= 2
    stack, worst = OppStack(prod_nets[0]), 0.0
    for items in multi:
        obs = [ob for _, ob in items]
        nets = prod_nets[:len(obs)]
        out, _ = stack.logits(nets, obs)
        worst = max([worst] + [legal_diff(per_net(n, o), x, o) for n, o, x in zip(nets, obs, out)])
    print(f"real fastenv batches ({[len(i) for i in multi]} nets): max |logit diff| on legal actions {worst:.2e}")
    assert worst <= TOL


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))


# ---------------------------------------------------------------- --snapshot-cache-cpu (2026-11-09)

def _spy_to(net, calls: list):
    """Record every `net.to(...)` (device moves of the CPU tier) and delegate."""
    orig = net.to
    object.__setattr__(net, "to", lambda *a, **k: (calls.append(str(a[0]) if a else str(k)), orig(*a, **k))[1])


def test_cpu_tier_off_is_unchanged(tmp):
    """Default (no CPU tier): an evicted net is loaded from disk again, as before; no CPU-tier state or stats."""
    t = fake_trainer(tmp, "--snapshot-cache", "2")
    for i in range(3):
        save(make_net(50 + i, 9, **SMALL), t.pool_dir / f"x{i}.pt")
    rng = np.random.default_rng(11)
    for name in ("x0.pt", "x1.pt", "x2.pt", "x0.pt"):
        t._opp_act(name, synth_obs(9, 4, 6, rng))
    loads, _ = t.pool_loads.take()
    assert loads["pool/load_attempts_snapshot"] == 4 and set(t.snapshots) == {"x2.pt", "x0.pt"}
    assert not t.__dict__.get("snapshots_cpu") and not any(k.startswith("pool/cpu_hits") for k in loads)


def test_cpu_tier_brings_evicted_nets_back(tmp):
    """--snapshot-cache-cpu: an evicted net is kept (same object), moved to the CPU at the next step, and comes back
    without a disk load with the same logits; the tier is LRU-bounded; derived caches are dropped on the move."""
    t = fake_trainer(tmp, "--snapshot-cache", "2", "--snapshot-cache-cpu", "2")
    for i in range(4):
        save(make_net(60 + i, 9, **SMALL), t.pool_dir / f"x{i}.pt")
    rng = np.random.default_rng(12)
    ob = synth_obs(9, 5, 6, rng)
    t._opp_act("x0.pt", ob)
    net0, calls = t.snapshots["x0.pt"], []
    _spy_to(net0, calls)
    with torch.no_grad():
        ref = t._opp_logits(net0, ob)
    t._opp_act("x1.pt", ob)
    t._opp_act("x2.pt", ob)  # evicts x0 into the CPU tier (still on the device until the next step)
    assert "x0.pt" in t.snapshots_cpu and calls == []
    t._opp_act("x3.pt", ob)  # flush: x0 moves to the CPU (its cache dropped); x1 is evicted
    assert calls == ["cpu"] and all("_cache" not in m.__dict__ for m in net0.modules())
    assert list(t.snapshots_cpu) == ["x0.pt", "x1.pt"] and set(t.snapshots) == {"x2.pt", "x3.pt"}
    loads, _ = t.pool_loads.take()
    assert loads["pool/load_attempts_snapshot"] == 4
    t._opp_act("x0.pt", ob)  # back from the CPU tier: the same object, no disk load
    assert t.snapshots["x0.pt"] is net0 and calls == ["cpu", str(CPU)]
    loads, _ = t.pool_loads.take()
    assert (loads["pool/load_attempts_snapshot"], loads["pool/cpu_hits_snapshot"]) == (0, 1)
    with torch.no_grad():
        assert torch.equal(t._opp_logits(net0, ob), ref)
    for name in ("x1.pt", "x2.pt", "x3.pt", "x0.pt", "x1.pt"):  # cycling 4 nets: tier never above its bound
        t._opp_act(name, ob)
        assert len(t.snapshots_cpu) <= 2 and len(t.snapshots) <= 2
    assert not (set(t.snapshots) & set(t.snapshots_cpu))


def test_cpu_tier_waits_for_the_step(tmp):
    """In one stacked step a net evicted by a later lookup still plays that step on the device; it moves to the CPU
    only at the next step (before its lookups)."""
    t = fake_trainer(tmp, "--snapshot-cache", "2", "--snapshot-cache-cpu", "4")
    for i in range(3):
        save(make_net(80 + i, 9, **SMALL), t.pool_dir / f"x{i}.pt")
    rng = np.random.default_rng(13)
    t._opp_act("x0.pt", synth_obs(9, 4, 6, rng))
    net0, calls = t.snapshots["x0.pt"], []
    _spy_to(net0, calls)
    acts, st = t._opp_act_many([(f"x{i}.pt", synth_obs(9, 4, 6, rng)) for i in range(3)])  # x0 evicted mid-step
    assert all(len(a) == 4 for a in acts) and st["opp_uniform"] == 0
    assert "x0.pt" in t.snapshots_cpu and calls == []  # not moved while its step was assembled
    t._opp_act_many([("x1.pt", synth_obs(9, 4, 6, rng))])
    assert calls == ["cpu"]


# ---------------------------------------------------------------- frozen nets stacked apart + fused slot fills (2026-11-10)

def _capture_logits(monkeypatch) -> list:
    """Record, per item and in order, the logits `_opp_act_many` samples from (None = uniform)."""
    seen = []
    orig = Trainer._opp_sample  # a staticmethod

    def spy(lg, batch):
        seen.append(None if lg is None else lg.clone())
        return orig(lg, batch)

    monkeypatch.setattr(Trainer, "_opp_sample", staticmethod(spy))
    return seen


def test_frozen_nets_stack_apart(tmp, monkeypatch):
    """Frozen nets of the pool's own architecture get their own stack: two stacked forwards per step (frozen, pool),
    each padded only to its own largest group; the logits are the per-net ones; the same step again copies nothing
    into either stack, and a pool step cannot displace the frozen nets' slots."""
    t = fake_trainer(tmp)
    for i in range(4):
        save(make_net(90 + i, 9, **SMALL), t.pool_dir / f"{100 * (i + 1):011d}.pt")
    save(make_net(95, 9, **SMALL), t.pool_dir / "frozen" / "P1.pt")
    save(make_net(96, 9, **SMALL), t.pool_dir / "frozen" / "best-00000000300.pt")
    rng = np.random.default_rng(21)
    names = ["frozen/P1.pt", "00000000100.pt", "frozen/best-00000000300.pt", "00000000200.pt", "00000000300.pt",
             "00000000400.pt"]
    items = [(n, synth_obs(9, r, 6, rng)) for n, r in zip(names, [60, 3, 55, 1, 7, 2])]
    seen = _capture_logits(monkeypatch)
    acts, st = t._opp_act_many(items)
    assert st == {"opp_fwds": 2, "opp_stacked_fwds": 2, "opp_stacked_nets": 6, "opp_stack_refills": 6,
                  "opp_uniform": 0}
    assert sorted(k[1] for k in t.opp_stacks) == ["frozen", "pool"] and len({k[0] for k in t.opp_stacks}) == 1
    sig = signature(t.snapshots["frozen/P1.pt"])
    assert t.opp_stacks[(sig, "frozen")].capacity == 2 and t.opp_stacks[(sig, "pool")].capacity == 4
    assert all(len(a) == len(ob["action_mask"]) for a, (_, ob) in zip(acts, items))
    assert max(legal_diff(per_net(t.snapshots[n], ob), lg, ob) for (n, ob), lg in zip(items, seen)) <= TOL
    first = [lg.clone() for lg in seen]
    seen.clear()
    _, st = t._opp_act_many(items)
    assert st["opp_stack_refills"] == 0 and st["opp_stacked_fwds"] == 2
    assert all(torch.equal(a, b) for a, b in zip(first, seen))
    _, st = t._opp_act_many([items[i] for i in (3, 1)])  # pool nets only: the frozen stack is untouched
    frozen = t.opp_stacks[(sig, "frozen")]
    assert st["opp_stacked_fwds"] == 1 and {id(s) for s in frozen.slots} == {id(t.snapshots[n]) for n in names[::2][:2]}


def test_fused_fill_equals_copy_loop(prod_nets, monkeypatch):
    """OppStack slot fills: the fused multi-tensor copy writes exactly what the per-tensor copy_ loop writes."""
    import porygon.v2.stacked as stacked_mod
    fused = OppStack(prod_nets[0])
    fused.bind(prod_nets[:3])
    monkeypatch.setattr(stacked_mod, "_FOREACH_COPY", False)
    loop = OppStack(prod_nets[0])
    loop.bind(prod_nets[:3])
    assert fused.buf.keys() == loop.buf.keys() and all(torch.equal(fused.buf[k], loop.buf[k]) for k in fused.buf)
    for kind in ("species", "moves", "items", "abilities"):
        key = f"ent_{kind}"
        if key in fused.buf:
            assert not fused.buf[key][:3, 0].any()  # PAD rows zero


@needs_node
def test_recorded_fastenv_batches_frozen_apart(tmp, monkeypatch):
    """Real fastenv opponent batches (recorded from a collector with frozen nets of the snapshots' architecture, the
    production mix: most pool battles vs the frozen nets) through `_opp_act_many`: two stacked forwards per step, and
    every opponent's logits equal its own per-net forward on the recorded batch."""
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector, FastWorkers
    from porygon.v2.workers import Packer

    t = fake_trainer(tmp)
    for i in range(5):
        save(make_net(110 + i, 9, **SMALL), t.pool_dir / f"{100 * (i + 1):011d}.pt")
    save(make_net(116, 9, **SMALL), t.pool_dir / "frozen" / "P1.pt")
    save(make_net(117, 9, **SMALL), t.pool_dir / "frozen" / "best-00000000500.pt")
    recorded, seen = [], _capture_logits(monkeypatch)

    def opp_many(items):
        recorded.append(items)
        return t._opp_act_many(items)

    rng = np.random.default_rng(3)

    def act(batch):
        mask = batch["action_mask"] > 0
        probs = mask / np.maximum(mask.sum(1, keepdims=True), 1)
        acts = (rng.random(mask.shape) * mask).argmax(1)
        n = len(acts)
        return acts, np.zeros(n), np.zeros(n, np.float32), probs.astype(np.float32)

    workers = FastWorkers(FastEnv("gen9randombattle", 1, 12, seed=7, encoder="v2", shaping=(0, 0, 0)))
    try:
        col = FastCollector(workers.env, Packer(9), act, t._opp_act, seed=2, pool_per_worker=10,
                            pool_dir=tmp / "run" / "pool", p_frozen=0.67, opp_fn_many=opp_many)
        while len([e for e in col.episodes if e["kind"] == "pool"]) < 8:
            col.step()
        st = col.rollout_stats()
    finally:
        workers.close()
    names = {name for items in recorded for name, _ in items}
    assert {"frozen/P1.pt", "frozen/best-00000000500.pt"} <= names and len(names - {"frozen/P1.pt",
                                                                                  "frozen/best-00000000500.pt"}) >= 2
    assert st["perf/col_opp_uniform"] == 0 and st["perf/col_opp_stacked_fwds"] > len(recorded)  # 2 stacks per step
    flat = [(name, ob) for items in recorded for name, ob in items]
    assert len(flat) == len(seen)
    worst = max(legal_diff(per_net(t.snapshots[n], ob), lg, ob) for (n, ob), lg in zip(flat, seen))
    print(f"recorded fastenv steps {len(recorded)}, opponent batches {len(flat)}: max |logit diff| vs per-net "
          f"{worst:.2e}")
    assert worst <= TOL
