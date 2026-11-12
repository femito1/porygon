"""Model v2 on real encoded observations (pickled battles from /tmp/netv2, see .orchestra/netv2/STATUS.md)."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from porygon.v2.encoder import aux_targets, encode
from porygon.v2.model import (
    DEFAULT_CFG, Entity, Net, _hash_uniform, grad_scale, load, obs_to_tensors, save,
)
from porygon.v2.spec import N_OPP_CLASSES, N_TYPES, UNK, VOCAB_SIZES, n_actions

SNAPS = {g: Path(f"/tmp/netv2/snaps_gen{g}randombattle.pkl") for g in (4, 9)}
TINY = dict(d_model=32, n_layers=2, n_heads=4, ffn_mult=2, d_move=16, d_species=16, d_move_emb=16, d_item=8,
            d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)

try:
    import porygon.v2.damage  # noqa: F401
    DAMAGE = [False, True]
except ImportError:
    DAMAGE = [False]


@pytest.fixture(scope="module", params=[4, 9])
def batch(request):
    g = request.param
    if not SNAPS[g].exists():
        pytest.skip(f"{SNAPS[g]} missing")
    snaps = pickle.load(open(SNAPS[g], "rb"))
    rng = np.random.default_rng(0)
    picks = rng.choice(len(snaps), 24, replace=False)
    obs = []
    for i in picks:
        p1, p2 = snaps[i]
        o = encode(p1)
        obs.append(o | aux_targets(p1, p2, o))
    return g, obs, [snaps[i][0] for i in picks]


@pytest.mark.parametrize("use_damage", DAMAGE)
def test_forward_shapes_and_masking(batch, use_damage):
    g, obs, _ = batch
    torch.manual_seed(0)
    net = Net(g, use_damage=use_damage, **TINY).eval()
    t = obs_to_tensors(obs, "cpu")
    assert t["species"].dtype == torch.int64 and t["mon_num"].dtype == torch.float32
    out = net(t)
    B, A = len(obs), n_actions(g)
    assert net.n_actions == A
    assert out["logits"].shape == (B, A) and out["value"].shape == (B,) and out["win_logit"].shape == (B,)
    ax = out["aux"]
    assert ax["item"].shape == (B, 6, VOCAB_SIZES["items"])
    assert ax["ability"].shape == (B, 6, VOCAB_SIZES["abilities"])
    assert ax["tera"].shape == (B, 6, N_TYPES)
    assert ax["moves"].shape == (B, 6, 24)
    assert ax["unseen"].shape == (B, VOCAB_SIZES["species"])
    assert ax["opp_action"].shape == (B, N_OPP_CLASSES)
    mask = t["action_mask"] > 0
    low = torch.finfo(torch.float32).min
    assert torch.all(out["logits"][~mask] == low)
    assert torch.all(out["logits"][mask] > low / 2)
    # zero-init pointer head: uniform over legal actions at init
    p = torch.softmax(out["logits"], -1)
    legal = mask.sum(-1, keepdim=True).float()
    assert torch.allclose(p[mask], (1 / legal).expand_as(p)[mask], atol=1e-6)
    assert torch.allclose(out["value"], torch.tanh(out["win_logit"] / 2))
    assert out["value"].abs().max() <= 1
    for v in [out["logits"], out["win_logit"], *[x for x in ax.values() if x.is_floating_point()]]:
        assert not torch.isnan(v).any()
    # every opponent-action class the self-play target can hit is allowed by the head's mask
    cls = t["aux_opp_class"]
    for b in range(B):
        hit = cls[b][cls[b] >= 0]
        assert ax["opp_action_mask"][b, hit].all(), (b, hit)


def test_policy_head_reads_tokens(batch):
    """After a gradient step the pointer logits are no longer uniform and differ across legal actions."""
    g, obs, _ = batch
    torch.manual_seed(0)
    net = Net(g, use_damage=False, **TINY)
    t = obs_to_tensors(obs, "cpu")
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    for _ in range(3):
        out = net(t)
        lp = torch.log_softmax(out["logits"], -1)
        first = t["action_mask"].argmax(-1)
        loss = -lp.gather(1, first[:, None]).mean() + out["win_logit"].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = net(t)
    lg = out["logits"][t["action_mask"] > 0]
    assert lg.std() > 1e-3


def test_grad_scale_op():
    x = torch.randn(5, 3, requires_grad=True)
    y = grad_scale(x, 0.1)
    assert torch.equal(y, x)
    (y * torch.arange(3.0)).sum().backward()
    assert torch.allclose(x.grad, 0.1 * torch.arange(3.0).expand(5, 3))
    x.grad = None
    grad_scale(x, 1.0).sum().backward()
    assert torch.allclose(x.grad, torch.ones(5, 3))


def test_value_gradient_into_trunk_is_scaled(batch):
    g, obs, _ = batch
    t = obs_to_tensors(obs, "cpu")
    grads = {}
    for s in (1.0, 0.1):
        torch.manual_seed(0)
        net = Net(g, use_damage=False, value_trunk_grad=s, **TINY)
        net(t)["win_logit"].sum().backward()
        trunk = net.blocks[0].qkv.weight.grad
        head = net.value_head[0].weight.grad
        grads[s] = (trunk.clone(), head.clone())
    assert torch.allclose(grads[0.1][0], 0.1 * grads[1.0][0], rtol=1e-4, atol=1e-7)
    assert torch.allclose(grads[0.1][1], grads[1.0][1], rtol=1e-5, atol=1e-7)  # the head itself: unscaled


def test_id_dropout_deterministic(batch):
    g, obs, _ = batch
    torch.manual_seed(0)
    net = Net(g, use_damage=False, id_dropout=0.5, **TINY).eval()
    t = obs_to_tensors(obs, "cpu")
    base = net(t)["value"]
    t["drop_seed"] = torch.zeros(len(obs), dtype=torch.int64)
    assert torch.equal(net(t)["value"], base)  # seed 0 = no dropout
    t["drop_seed"] = torch.arange(1, len(obs) + 1, dtype=torch.int64) * 7919
    a, b = net(t)["value"], net(t)["value"]
    assert torch.equal(a, b) and not torch.allclose(a, base)
    u = _hash_uniform(torch.arange(1, 2001, dtype=torch.int64), 288, 3)
    assert 0.48 < (u < 0.5).float().mean() < 0.52 and 0.0 <= u.min() and u.max() < 1.0
    assert not torch.equal(u, _hash_uniform(torch.arange(1, 2001, dtype=torch.int64), 288, 4))


def test_entity_dropout_keeps_static_features():
    table = torch.randn(10, 3)
    table[:2] = 0
    e = Entity(10, 4, table)
    ids = torch.tensor([[0, 1, 5, 7]])
    drop = torch.tensor([[True, True, True, False]])
    feats = e.all_features()
    x = e(ids, feats, drop)
    assert torch.all(x[0, 0] == 0)  # PAD stays zero
    expect5 = e.emb.weight[UNK] + feats[5]  # id embedding dropped, features of the true id kept
    assert torch.allclose(x[0, 2], expect5)
    assert torch.allclose(x[0, 3], e.emb.weight[7] + feats[7])


def test_save_load_roundtrip(tmp_path, batch):
    g, obs, _ = batch
    torch.manual_seed(1)
    net = Net(g, use_damage=False, **TINY)
    with torch.no_grad():
        net.policy_head[-1].weight.normal_()  # non-trivial policy
    path = save(net, tmp_path / "net.pt", step=7)
    ck = torch.load(path, weights_only=False)
    assert ck["gen"] == g and ck["cfg"]["d_model"] == 32 and ck["step"] == 7
    net2 = load(path)
    assert not net2.training and net2.cfg == net.cfg and net2.gen == g
    t = obs_to_tensors(obs, "cpu")
    with torch.no_grad():
        a, b = net.eval()(t), net2(t)
    assert torch.allclose(a["logits"], b["logits"]) and torch.allclose(a["value"], b["value"])
    assert torch.equal(net.ent_moves.table, net2.ent_moves.table)


def test_policy_probs_and_player(batch):
    from porygon.v2.player import NetPlayer, policy_probs
    g, _, battles = batch
    net = Net(g, use_damage=False, **TINY).eval()
    for battle in battles[:5]:
        probs, mask = policy_probs(net, battle)
        assert probs.shape == mask.shape == (n_actions(g),)
        if mask.any():
            assert abs(probs.sum() - 1) < 1e-6 and np.all(probs[mask == 0] == 0)
    player = NetPlayer(net, battle_format=f"gen{g}randombattle", start_listening=False)
    order = player.choose_move(next(b for b in battles if not b.wait))
    assert order.message.startswith("/choose")


def test_default_param_count():
    net = Net(9, use_damage=False)
    n = sum(p.numel() for p in net.parameters())
    print(f"default v2 params (no damage layer): {n:,}")
    assert net.cfg == {**DEFAULT_CFG, "use_damage": False}
    assert 9_500_000 < n < 11_500_000  # 6.3M of it: the four per-token-type 2x FFNs in 6 layers
    if True in DAMAGE:
        n_d = sum(p.numel() for p in Net(9).parameters())
        print(f"default v2 params (with damage layer): {n_d:,}")
        assert n_d >= n


@pytest.mark.skipif(True not in DAMAGE, reason="porygon.v2.damage not available")
def test_precomputed_damage_is_exact(batch):
    g, obs, _ = batch
    torch.manual_seed(0)
    net = Net(g, **TINY).eval()
    t = obs_to_tensors(obs, "cpu")
    with torch.no_grad():
        a = net(t)
        dmg = net.damage_features(t)
        half = {k: v[: len(obs) // 2] for k, v in dmg.items()}
        b = net({k: v[: len(obs) // 2] for k, v in t.items()}, dmg=half)
    assert torch.allclose(a["win_logit"][: len(obs) // 2], b["win_logit"], atol=1e-6)
    assert torch.allclose(a["aux"]["moves"][: len(obs) // 2], b["aux"]["moves"], atol=1e-6)


def test_feature_cache_follows_parameters(batch):
    g, obs, _ = batch
    torch.manual_seed(0)
    net = Net(g, use_damage=False, **TINY).eval()
    t = obs_to_tensors(obs, "cpu")
    with torch.no_grad():
        a = net(t, aux=False)["win_logit"]
        assert net(t)["aux"] is not None and net(t, aux=False)["aux"] is None
        assert torch.equal(net(t, aux=False)["win_logit"], a)  # cached features: same result
        net.ent_species.feat[0].weight.add_(0.5)
        b = net(t, aux=False)["win_logit"]
    assert not torch.allclose(a, b)
    assert torch.allclose(b, net(t, aux=False)["win_logit"])
    assert torch.allclose(b, net(t)["win_logit"].detach())  # grad-enabled path (no cache) agrees


def test_fainted_mons_are_not_attended(batch):
    """Legal-action logits do not depend on a fainted mon's inputs (its own switch is illegal anyway)."""
    g, obs, _ = batch
    from porygon.v2.model import COL
    torch.manual_seed(0)
    net = Net(g, use_damage=False, **TINY).eval()
    with torch.no_grad():
        for blk in net.blocks:  # make attention matter
            blk.qkv.weight.normal_(std=0.5)
        net.policy_head[-1].weight.normal_()
    hits = 0
    for o in obs:
        fainted = np.flatnonzero((o["mon_num"][:, COL["present"]] > 0) & (o["mon_num"][:, COL["fainted"]] > 0))
        if not len(fainted):
            continue
        hits += 1
        o2 = {k: v.copy() for k, v in o.items()}
        i = fainted[0]
        o2["species"][i] = 5
        o2["moves"][i, :4] = [7, 8, 9, 10]
        o2["stats"][i] *= 2
        with torch.no_grad():
            a, b = net(obs_to_tensors([o], "cpu")), net(obs_to_tensors([o2], "cpu"))
        legal = o["action_mask"] > 0
        if i < 6:
            legal[i] = False  # its own switch logit (legal only in poke-env's "wait" mask, action 0)
        assert torch.allclose(a["logits"][0][legal], b["logits"][0][legal], atol=1e-5)
        assert not torch.allclose(a["win_logit"], b["win_logit"])  # its own token still updates (pooled value)
    if not hits:
        pytest.skip("no fainted mon in the sample")


def test_qk_norm_bounds_attention_logits():
    from porygon.v2.model import Block
    torch.manual_seed(0)
    blk = Block(32, 4, 2, [(0, 3)], qk_norm=True)
    x = torch.randn(2, 3, 32) * 100
    with torch.no_grad():
        blk.qkv.weight.mul_(100)
        q, k, _ = blk.qkv(blk.norm1(x)).view(2, 3, 3, 4, 8).permute(2, 0, 3, 1, 4)
        q, k = blk.q_norm(q), blk.k_norm(k)
        logits = (q @ k.transpose(-1, -2)) / 8 ** 0.5
    assert logits.abs().max() <= 8 ** 0.5 + 1e-4  # |q|, |k| = sqrt(d_head) at unit scale
