"""porygon.v2.tables / porygon.v2.damage: table shapes, damage-feature shapes and invariants on real observations,
hand-computed values on synthetic positions, and agreement with poke-engine's damage calculator (gen 4 build).

Real observations come from the battle snapshots in /tmp/netv2 (lists of (battle_p1, battle_p2)); those tests are
skipped when the pickles are missing. Single-threaded and small (a few seconds).
"""

import math
import os
import pickle
import random

import numpy as np
import pytest
import torch

from porygon.v2 import encoder as E
from porygon.v2 import tables as T
from porygon.v2.damage import COL, FCOL, N_MOVE_VS, N_PAIR, DamageFeatures
from porygon.v2.spec import INDEX, KM, N_TYPES, TYPE_ID, VOCAB_SIZES

torch.set_num_threads(1)
SNAPS = "/tmp/netv2/snaps_gen{}randombattle.pkl"
N_SNAPS = 60


def batch_of(obs_list: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list])) for k in obs_list[0]}


def snapshots(gen: int, n: int = N_SNAPS):
    path = SNAPS.format(gen)
    if not os.path.exists(path):
        pytest.skip(f"no snapshots at {path}")
    with open(path, "rb") as f:
        snaps = pickle.load(f)
    return snaps[:: max(1, len(snaps) // n)][:n]


# ------------------------------------------------------------------------------------------------ tables

@pytest.mark.parametrize("gen", [4, 9])
def test_static_tables(gen):
    feats = T.static_features(gen)
    assert set(feats) == {"species", "moves", "items", "abilities"}
    for kind, a in feats.items():
        assert a.dtype == np.float32 and a.shape == (VOCAB_SIZES[kind], len(T.FEATURE_NAMES[kind]))
        assert np.isfinite(a).all() and (a[:2] == 0).all()
        assert np.abs(a).max() <= 3.0, kind  # O(1)
    sp, mv = feats["species"], feats["moves"]
    col = lambda kind, n: T.FEATURE_NAMES[kind].index(n)  # noqa: E731
    garchomp = INDEX["species"]["garchomp"]
    assert sp[garchomp, col("species", "type_dragon")] == 1 and sp[garchomp, col("species", "type_ground")] == 1
    assert sp[INDEX["species"]["gible"], col("species", "nfe")] == 1 and sp[garchomp, col("species", "nfe")] == 0
    eq = INDEX["moves"]["earthquake"]
    assert mv[eq, col("moves", "bp")] == pytest.approx(100 / 150) and mv[eq, col("moves", "physical")] == 1
    assert mv[INDEX["moves"]["uturn"], col("moves", "self_switch")] == 1
    assert mv[INDEX["moves"]["stealthrock"], col("moves", "hazard")] == 1
    assert mv[INDEX["moves"]["seismictoss"], col("moves", "fixed_level")] == 1
    raw = T.move_raw(gen)
    assert raw["bp"][INDEX["moves"]["hiddenpower"]] == (70 if gen <= 4 else 60)
    assert raw["hits"][INDEX["moves"]["bulletseed"]] == pytest.approx(3.0 if gen <= 4 else 3.1)
    it, ab = feats["items"], feats["abilities"]
    assert it[INDEX["items"]["choiceband"], col("items", "choiceband")] == 1
    assert it[INDEX["items"]["charcoal"], col("items", "type_fire")] == 1
    imm = col("abilities", "immune")
    assert ab[INDEX["abilities"]["levitate"], imm] == 1
    assert ab[INDEX["abilities"]["lightningrod"], imm] == (0 if gen <= 4 else 1)  # gen 4: redirect only


# ------------------------------------------------------------------------------------------------ real observations

@pytest.mark.parametrize("gen", [4, 9])
def test_real_observations(gen):
    snaps = snapshots(gen)
    obs = [E.encode(b1) for b1, _ in snaps]
    batch = batch_of(obs)
    B = len(obs)
    mod = DamageFeatures(gen)
    with torch.no_grad():
        out = mod(batch)
        full = DamageFeatures(gen, trim=False)(batch)
        one = mod({k: v[3:4] for k, v in batch.items()})
    shapes = {"move_vs": (B, 12, KM, 6, N_MOVE_VS), "active_tera_vs": (B, 4, 6, N_MOVE_VS), "speed": (B, 12, 6),
              "pair": (B, 12, 12, N_PAIR)}
    assert {k: tuple(v.shape) for k, v in out.items()} == shapes
    for k, v in out.items():
        assert v.dtype == torch.float32 and torch.isfinite(v).all(), k
        assert torch.allclose(v, full[k], atol=1e-5), f"trim changes {k}"
        assert torch.allclose(v[3:4], one[k], atol=1e-5), f"{k} depends on the rest of the batch"
    mv = out["move_vs"]
    f0, f1, f2, f3 = mv.unbind(-1)
    assert (f0 >= 0).all() and (f0 <= 4).all() and (f1 >= 0).all() and (f1 <= 2).all()
    assert (f2 >= 0).all() and (f2 <= 1).all() and (f3 <= f0 + 1e-6).all() and (f3 >= 0).all()
    assert f0.amax() > 0.5  # something hits hard
    # zeros: empty / status move slots, absent or fainted attackers and defenders
    moves = batch["moves"].long()
    cat = torch.as_tensor(T.move_raw(gen)["category"])[moves]
    status = (cat != 1) & (cat != 2)
    assert (mv[status.nonzero(as_tuple=True)] == 0).all()
    mn = batch["mon_num"]
    dead = (mn[..., COL["present"]] == 0) | (mn[..., COL["fainted"]] == 1)  # [B,12]
    assert (mv[dead] == 0).all()
    dead_def = torch.cat([dead[:, None, 6:].expand(B, 6, 6), dead[:, None, :6].expand(B, 6, 6)], 1)  # [B,12,6]
    assert (mv.permute(0, 1, 3, 2, 4)[dead_def] == 0).all()
    # speed: P(i before k) + P(k before i) = 1 for live pairs
    sp = out["speed"]
    alive = ~dead
    for i in range(6):
        for k in range(6):
            both = alive[:, i] & alive[:, 6 + k]
            s = sp[:, i, k] + sp[:, 6 + k, i]
            assert torch.allclose(s[both], torch.ones_like(s[both]), atol=1e-5)
            assert (s[~both] == 0).all()
    # pair: zero on the same side, (1) is the transpose of (0), (2) is speed
    pair = out["pair"]
    assert (pair[:, :6, :6] == 0).all() and (pair[:, 6:, 6:] == 0).all()
    assert torch.allclose(pair[..., 1], pair[..., 0].transpose(1, 2))
    assert torch.allclose(pair[:, :6, 6:, 2], sp[:, :6]) and torch.allclose(pair[:, 6:, :6, 2], sp[:, 6:])
    usable = (batch["move_num"][..., 0] > 0) | (batch["move_p"] >= 0.5)
    best = (f0 * usable[..., None]).amax(2)
    assert torch.allclose(pair[:, :6, 6:, 0], best[:, :6]) and torch.allclose(pair[:, 6:, :6, 0], best[:, 6:])
    # tera variant: zeros in gen 4; in gen 9 equal to move_vs for an already terastallized active
    tv = out["active_tera_vs"]
    if gen < 9:
        assert (tv == 0).all()
    else:
        act = mn[:, :6, COL["active"]]
        for b in range(B):
            if act[b].sum() == 0:
                assert (tv[b] == 0).all()
                continue
            i = int(act[b].argmax())
            if mn[b, i, COL["terastallized"]] > 0:
                assert torch.allclose(tv[b], mv[b, i, :4], atol=1e-5)


# ------------------------------------------------------------------------------------------------ synthetic positions

def position(gen: int, atk_types=("normal",), def_types=("normal",), move="earthquake", atk=200.0, dfn=100.0,
             def_hp=(100.0, 300.0), ability_d=(("pressure", 1.0),), item_a=(("", 1.0),), level=100):
    """Our mon 0 (active) attacking their mon 6 (active) with one move; stats hp atk def spa spd spe."""
    o = E.empty(gen)
    for i, types, st in ((0, atk_types, [300, atk, 100, atk, 100, 100]), (6, def_types, [def_hp[1], 100, dfn, 100,
                                                                                         dfn, 90])):
        o["species"][i] = INDEX["species"]["snorlax"]
        o["types"][i] = [TYPE_ID[t] for t in types] + [0] * (2 - len(types))
        n = o["mon_num"][i]
        n[COL["present"]] = n[COL["active"]] = n[COL["revealed"]] = 1
        n[COL["mine"]] = float(i == 0)
        n[COL["level"]] = level / 100
        n[COL["log_weight"]] = math.log10(100) / 3
        o["stats"][i] = st
        o["hp"][i] = [300, 300] if i == 0 else def_hp
        n[COL["hp_frac"]] = o["hp"][i][0] / o["hp"][i][1]
        o["ability"][i, 0], o["ability_p"][i, 0] = INDEX["abilities"]["pressure"], 1.0
    for k, (name, p) in enumerate(ability_d):
        o["ability"][6, k], o["ability_p"][6, k] = INDEX["abilities"][name], p
    for k, (name, p) in enumerate(item_a):
        o["item"][0, k], o["item_p"][0, k] = (INDEX["items"][name] if name else 0), p
    o["moves"][0, 0], o["move_p"][0, 0] = INDEX["moves"][move], 1.0
    o["move_type"][0, 0] = E.move_type_of(gen, move)
    o["move_num"][0, 0] = [1, 1, 1]
    return o


def run(o, gen=9):
    with torch.no_grad():
        return DamageFeatures(gen)(batch_of([o]))


def expected(bp, atk, dfn, mult=1.0, level=100):
    return ((math.floor(2 * level / 5) + 2) * bp * atk / dfn / 50 + 2) * mult


@pytest.mark.parametrize("gen", [4, 9])
def test_synthetic_values(gen):
    base = expected(100, 200, 100)  # earthquake, no STAB, neutral
    f = run(position(gen), gen)["move_vs"][0, 0, 0, 0]
    assert f[0].item() == pytest.approx(0.925 * base / 300, rel=1e-4)
    assert f[1].item() == pytest.approx(min(2.0, 0.925 * base / 100), rel=1e-4)
    assert f[2].item() == 1.0  # every roll (>= 0.85 * 172) KOs 100 HP
    assert f[3].item() == pytest.approx(f[0].item(), rel=1e-6)  # 100% accurate
    # P(KO) counts rolls: KO only from roll r with r / 100 * base >= hp
    hp = 0.9 * base
    f = run(position(gen, def_hp=(hp, 300.0)), gen)["move_vs"][0, 0, 0, 0]
    assert f[2].item() == pytest.approx(sum(math.floor(base * r) / 100 >= hp for r in range(85, 101)) / 16, abs=1 / 16)
    # STAB, super effective, immune
    f = run(position(gen, atk_types=("ground",), def_types=("fire",)), gen)["move_vs"][0, 0, 0, 0]
    assert f[0].item() == pytest.approx(0.925 * base * 3 / 300, rel=1e-4)
    f = run(position(gen, def_types=("flying",)), gen)["move_vs"][0, 0, 0, 0]
    assert f[0].item() < 1e-6 and f[2].item() < 1e-6
    # Levitate with probability 0.5 halves the mean and P(KO)
    f = run(position(gen, ability_d=(("levitate", 0.5), ("pressure", 0.5))), gen)["move_vs"][0, 0, 0, 0]
    assert f[0].item() == pytest.approx(0.5 * 0.925 * base / 300, rel=1e-3)
    assert f[2].item() == pytest.approx(0.5, abs=1e-3)
    # Choice Band with probability 0.5: expected attack x1.25
    f = run(position(gen, item_a=(("choiceband", 0.5), ("leftovers", 0.5))), gen)["move_vs"][0, 0, 0, 0]
    assert f[0].item() == pytest.approx(0.925 * expected(100, 250, 100) / 300, rel=1e-3)
    # burn halves physical damage, reflect halves it again
    o = position(gen)
    o["mon_num"][0, COL["status_brn"]] = 1
    o["side_num"][1, 0] = 1  # reflect on the defender's side
    f = run(o, gen)["move_vs"][0, 0, 0, 0]
    # gen 4 applies burn and screens before the "+2" (as modelled); later gens after it (a ~2 HP difference)
    assert f[0].item() == pytest.approx(0.925 * (expected(100, 100, 100, 0.5) + 1) / 300, rel=0.03)
    # +2 Atk, -1 Def
    o = position(gen)
    o["boosts"][0, 0], o["boosts"][6, 1] = 2, -1
    f = run(o, gen)["move_vs"][0, 0, 0, 0]
    assert f[0].item() == pytest.approx(0.925 * expected(100, 400, 100 * 2 / 3) / 300, rel=1e-3)
    # status move and empty slots: zero
    out = run(position(gen, move="swordsdance"), gen)["move_vs"]
    assert (out == 0).all()
    # seismic toss: the level, all rolls equal
    f = run(position(gen, move="seismictoss", level=80), gen)["move_vs"][0, 0, 0, 0]
    assert f[0].item() == pytest.approx(80 / 300, rel=1e-4) and f[2].item() == 0.0
    # rain boosts water, sun weakens it
    o = position(gen, move="surf")
    o["field_num"][FCOL["weather_raindance"]] = 1
    rain = run(o, gen)["move_vs"][0, 0, 0, 0, 0].item()
    o["field_num"][FCOL["weather_raindance"]], o["field_num"][FCOL["weather_sunnyday"]] = 0, 1
    sun = run(o, gen)["move_vs"][0, 0, 0, 0, 0].item()
    assert rain == pytest.approx(3 * sun, rel=1e-3 if gen >= 5 else 0.03)  # gen 4: weather before the "+2"


def test_tera_and_speed():
    gen = 9
    # tera into the move's type gives STAB in active_tera_vs only
    o = position(gen)
    o["tera"][0] = TYPE_ID["ground"]
    o["tera_p"][0, TYPE_ID["ground"]] = 1
    out = run(o, gen)
    base = expected(100, 200, 100)
    assert out["move_vs"][0, 0, 0, 0, 0].item() == pytest.approx(0.925 * base / 300, rel=1e-4)
    assert out["active_tera_vs"][0, 0, 0, 0].item() == pytest.approx(0.925 * base * 1.5 / 300, rel=1e-4)
    assert N_TYPES == 20
    # speed: 100 vs 90 -> we move first; trick room reverses; scarf with p = 0.5 on a slower foe
    assert out["speed"][0, 0, 0].item() == 1.0 and out["speed"][0, 6, 0].item() == 0.0
    o["field_num"][FCOL["trickroom"]] = 1
    assert run(o, gen)["speed"][0, 0, 0].item() == 0.0
    o["field_num"][FCOL["trickroom"]] = 0
    o["item"][6, :2], o["item_p"][6, :2] = [INDEX["items"]["choicescarf"], INDEX["items"]["leftovers"]], [0.5, 0.5]
    assert run(o, gen)["speed"][0, 0, 0].item() == pytest.approx(0.5)
    o["stats"][6, 5] = 100  # tie without the scarf
    assert run(o, gen)["speed"][0, 0, 0].item() == pytest.approx(0.25)


# ------------------------------------------------------------------------------------------------ vs poke-engine

def test_against_poke_engine_gen4():
    pe = pytest.importorskip("poke_engine")
    from porygon.search import GEN_NUM, convert
    if GEN_NUM != 4:
        pytest.skip("the installed engine build is gen 4")
    raw = T.move_raw(4)
    mod = DamageFeatures(4)
    errs = []
    for b1, _ in snapshots(4, 80):
        a, d = b1.active_pokemon, b1.opponent_active_pokemon
        if a is None or d is None or a.fainted or d.fainted:
            continue
        obs = E.encode(b1)
        try:
            cv = convert(b1, random.Random(0))
            st, ids = pe.State.from_string(cv.state_str), cv.move_ids
        except BaseException as e:  # unknown moves panic in Rust
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            continue
        with torch.no_grad():
            f = mod(batch_of([obs]))["move_vs"][0].numpy()
        act = obs["mon_num"][:, COL["active"]]
        i, k = int(np.flatnonzero(act[:6])[0]), int(np.flatnonzero(act[6:])[0])
        maxhp = st.side_two.pokemon[int(st.side_two.active_index)].maxhp
        for j, eng in enumerate(ids[:4]):
            mid = int(obs["moves"][i, j])
            if raw["category"][mid] not in (1, 2) or raw["max_hits"][mid] > 1:
                continue
            try:
                dmg = pe.calculate_damage(st, eng, "splash", True)[0]
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                continue
            ref = (dmg[0] if raw["fixed"][mid] else 0.925 * dmg[0]) / maxhp if dmg else 0.0
            errs.append(abs(100 * f[i, j, k, 0] - 100 * ref))
    errs = np.array(errs)
    assert len(errs) > 50
    assert errs.mean() < 1.0 and (errs <= 5).mean() > 0.97, (errs.mean(), (errs <= 5).mean())
