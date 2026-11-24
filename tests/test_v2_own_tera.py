"""Spec v2.2a (own pre-Tera visibility, council P1 2026-11-24): v21_view, the per-policy checkpoint flag, the trainer's
per-net opponent / anchor views and the fastenv v2.2a rows. The full parity / byte-identity / leakage checks over 300
streaming battles are scripts/netv2_p1tera/stage0_*."""

import shutil

import numpy as np
import pytest
import torch

from porygon.v2.spec import MON_NUM, fields, v21_view

TINY = dict(d_model=32, n_layers=1, n_heads=4, ffn_mult=1, d_move=16, d_species=16, d_move_emb=16, d_item=8,
            d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)
T = MON_NUM.index("terastallized")


def _obs(n=2):
    o = {k: np.zeros((n, *sh), dt) for k, (sh, dt) in fields(9).items()}
    o["tera"][:, :12] = 7
    o["tera_p"][:, :12, 7] = 1
    o["mon_num"][1, 3, T] = 1  # own slot 3 of row 1 has terastallized
    return o


def test_v21_view_numpy_and_torch():
    o = _obs()
    for x in (o, {k: torch.from_numpy(v) for k, v in o.items()}):
        v = v21_view(x)
        tera, tp = np.asarray(v["tera"]), np.asarray(v["tera_p"])
        assert tera[0, :6].tolist() == [0] * 6 and tera[1, :6].tolist() == [0, 0, 0, 7, 0, 0]
        assert (tera[:, 6:] == 7).all() and tp[1, 3, 7] == 1 and tp[0, :6].sum() == 0 and (tp[:, 6:, 7] == 1).all()
        assert (np.asarray(x["tera"]) == 7).all()  # the input is not modified
        assert all(v[k] is x[k] for k in x if k not in ("tera", "tera_p"))


def test_checkpoint_flag_roundtrip(tmp_path):
    from porygon.v2.model import Net, load, save
    net = Net(9, **TINY)
    save(net, tmp_path / "a.pt")
    assert "own_tera_visible" not in torch.load(tmp_path / "a.pt", weights_only=False)
    assert not getattr(load(tmp_path / "a.pt"), "own_tera_visible", False)
    net.own_tera_visible = True
    save(net, tmp_path / "b.pt", counters={"steps": 1})
    ck = torch.load(tmp_path / "b.pt", weights_only=False)
    assert ck["own_tera_visible"] is True and ck["counters"] == {"steps": 1}
    assert load(tmp_path / "b.pt").own_tera_visible is True


def test_opp_view_and_policy_view():
    from porygon.v2.train import Trainer, policy_view

    class N:  # stands in for a Net (only the flag is read)
        def __init__(self, own):
            if own:
                self.own_tera_visible = True
    b = {"tera": np.ones(3), "tera_p": np.ones(3), "tera_v21": np.zeros(3), "tera_p_v21": np.zeros(3), "x": 1}
    assert Trainer._opp_view(N(False), b)["tera"].sum() == 0 and Trainer._opp_view(None, b)["tera_p"].sum() == 0
    assert Trainer._opp_view(N(True), b) is b
    plain = {"tera": np.ones(3)}
    assert Trainer._opp_view(N(False), plain) is plain  # a v2.1 env: no v2.1 keys, nothing to route
    t = {k: torch.from_numpy(v) for k, v in _obs().items()}
    learner, legacy, new = N(True), N(False), N(True)
    legacy.damage = None  # rollout_damage: no damage layer -> None
    view, dmg = policy_view(learner, legacy, t, list(t), {"d": 1})
    assert dmg is None and view["tera"][0, :6].sum() == 0 and view["tera"][0, 6:].sum() == 42
    assert policy_view(learner, new, t, list(t), {"d": 1}) == (t, {"d": 1})
    assert policy_view(N(False), legacy, t, list(t), {"d": 1}) == (t, {"d": 1})


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_fastenv_v22a_rows_and_v21_keys():
    from porygon.fastenv import FastEnv
    from porygon.fastenv_v2 import FastCollector
    from porygon.v2.workers import Packer
    first = {}
    for own in (False, True):
        env = FastEnv("gen9randombattle", 1, 4, seed=5, encoder="v2", shaping=(0, 0, 0), own_tera=own)
        try:
            col = FastCollector(env, Packer(9), lambda batch: None, seed=0) if own else None
            b = next(iter(col.ready.values() if own else dict(env.poll_all()).values()))
            first[own] = {k[4:]: v.copy() for k, v in b.cols.items() if k.startswith("obs/")}
            if own:
                rows = np.flatnonzero(b.needs_action)
                ob = col._opp_obs([b], [rows], [(0, k) for k in range(len(rows))])
                v = v21_view({k: first[own][k][rows] for k in ("tera", "tera_p", "mon_num")})
                assert np.array_equal(ob["tera_v21"], v["tera"]) and np.array_equal(ob["tera_p_v21"], v["tera_p"])
        finally:
            env.close()
    off, on = first[False], first[True]
    assert (on["tera"][:, :6] > 0).all() and (off["tera"][:, :6] == 0).all()  # turn 1: no one has terastallized
    v = v21_view(on)
    assert all(np.array_equal(v[k], off[k]) for k in off)
