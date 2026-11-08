"""porygon.bc: shard loading, masking, split, a tiny overfit, checkpoints that load and play (gen9 and gen4)."""

import json
import logging

import numpy as np
import pytest
import torch

from porygon import bc
from porygon.encoder import OBS_DIM

TINY = dict(d_model=32, n_layers=1, n_heads=4)


def tiny_args(data, out, *extra):
    return ["train", str(data), "--out", str(out), "--d-model", "32", "--n-layers", "1", "--n-heads", "4",
            "--device", "cpu", "--load-workers", "2", "--log-every", "5", "--val-frac", "0.2", *extra]


@pytest.fixture(scope="module")
def gen9_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("gen9")
    bc.main(["synth", str(d), "--n", "1200", "--n-actions", "26", "--games", "60", "--shards", "2"])
    return d


def test_synthetic_shard_contract(gen9_data):
    meta = bc.read_meta([gen9_data])
    assert meta["obs_dim"] == OBS_DIM and meta["n_actions"] == 26
    d = bc.load_shard(bc.list_shards([gen9_data])[0], 0)
    assert d["obs"].dtype == np.float16 and d["obs"].shape[1] == OBS_DIM
    assert d["mask"].dtype == np.int8 and d["target"].dtype == np.float16 and d["z"].dtype == np.float32
    assert (d["mask"][np.arange(len(d["action"])), d["action"]] == 1).all()
    p = bc.prepare(d, bc.Filters())
    assert np.allclose(p["target"].sum(1), 1) and (p["target"][~p["mask"]] == 0).all()
    assert (p["pw"][p["n_legal"] < 2] == 0).all() and (p["vw"] > 0).all()  # value loss on every decision


def test_filters_and_mix():
    d = bc.synth_shard(500, 10, 20, seed=1)
    d["n"] = np.int64(500)
    f = bc.Filters(min_rating=1500, sources={1}, mix={1: 0.5})
    w = f.row_weight(d)
    assert (w[(d["rating"] < 1500) | (d["source"] != 1)] == 0).all()
    assert np.allclose(w[(d["rating"] >= 1500) & (d["source"] == 1)], 0.5)
    table = bc.source_table({"sources": {"run:searchlabels-x": 10}})
    assert bc.source_codes("human", table) == {0, 1, 2, 3} and bc.source_codes("search", table) == {10}
    assert bc.source_codes("run:searchlabels-x", table) == {10} and bc.source_codes("11", table) == {11}
    d["rating"][:10] = -1
    assert (bc.Filters(min_rating=1500).row_weight(d)[:10] == 1).all()  # unknown ratings pass ...
    assert (bc.Filters(drop_unrated=True).row_weight(d)[:10] == 0).all()  # ... unless dropped
    p = bc.prepare(d, bc.Filters(skip_forced=True))
    assert p["pw"][d["forced"][(p["vw"] > 0)].astype(bool)].sum() == 0


def test_search_labels_soft_targets_root_value_and_visits():
    d = bc.synth_shard(200, 26, 10, seed=2)
    d["n"] = np.int64(200)
    rng = np.random.default_rng(0)
    soft = rng.random((200, 26)).astype(np.float32) * d["mask"]  # visit shares, legal actions only
    soft[:, 0] += 0.5  # some mass on an illegal action (switch 0 is never legal here): must be dropped
    d["target"] = soft.astype(np.float16)
    d["action"][:] = -1  # search rows: no human action
    d["root_value"][:5] = np.nan
    p = bc.prepare(d, bc.Filters())
    assert np.allclose(p["target"].sum(1), 1, atol=1e-3) and (p["target"][~p["mask"]] == 0).all()
    assert (p["target"] > 0).sum(1).mean() > 2  # soft, not one-hot
    zr = bc.Filters(value_target="result").value_targets(d)
    zs = bc.Filters(value_target="search").value_targets(d)
    zm = bc.Filters(value_target="mix", value_mix=0.25).value_targets(d)
    assert np.allclose(zr, d["z"]) and np.allclose(zs[5:], 2 * d["root_value"][5:] - 1)
    assert np.allclose(zs[:5], d["z"][:5])  # no root value -> game result
    assert np.allclose(zm[5:], 0.25 * zs[5:] + 0.75 * zr[5:])
    w = bc.Filters(visit_weight_power=1.0).row_weight(d)
    big, small = d["visits"].argmax(), d["visits"].argmin()
    assert w[big] > 1 > w[small] >= 0.1
    logits = torch.zeros(200, 26)
    b = {k: torch.as_tensor(v) for k, v in p.items() if k != "obs"}
    pol, val = bc.losses(logits, torch.zeros(200), b)
    uniform_ce = np.log(p["mask"].sum(1))[p["pw"] > 0].mean()
    assert pol.item() == pytest.approx(uniform_ce, rel=1e-4)  # CE to a distribution under a uniform policy


def test_meta_with_null_n_actions_is_inferred(tmp_path):
    d = bc.synth_shard(50, 26, 4)
    np.savez(tmp_path / "shard-00000.npz", **d)
    (tmp_path / "meta.json").write_text(json.dumps({"n_actions": None, "obs_dim": 4079, "format": "gen9randombattle"}))
    m = bc.read_meta([tmp_path])
    assert m["n_actions"] == 26 and m["obs_dim"] == d["obs"].shape[1]


def test_game_level_split_is_deterministic():
    games = np.repeat(np.arange(2000), 7)
    v = bc.val_bucket(games, 0.1)
    assert (v == bc.val_bucket(games, 0.1)).all()
    per_game = v.reshape(2000, 7)
    assert (per_game.all(1) | ~per_game.any(1)).all()  # no game on both sides
    assert 0.07 < v.mean() < 0.13


def test_legacy_expert_shards(tmp_path):
    from porygon.expert import ShardWriter
    w = ShardWriter(tmp_path, games_per_shard=10)
    turn = bc.turn_index()
    for g, n in enumerate((3, 5)):
        recs = []
        for t in range(n):
            obs = np.zeros(OBS_DIM, np.float16)
            obs[turn] = (t + 1) / 100
            recs.append({"obs": obs, "mask": np.ones(10, np.int8), "prior": np.full(10, 0.1, np.float32),
                         "target": np.eye(10, dtype=np.float32)[t % 10], "ret": np.float32(1 - 2 * g)})
        w.add_game(recs)
    w.flush()
    meta = bc.read_meta([tmp_path])
    assert meta["n_actions"] == 10 and meta["datasets"][0]["legacy"]
    d = bc.load_shard(bc.list_shards([tmp_path])[0], 0)
    assert len(np.unique(d["game"][:3])) == 1 and len(np.unique(d["game"])) == 2  # games recovered from turns
    assert d["z"].tolist() == [1.0] * 3 + [-1.0] * 5 and (d["source"] == bc.DEFAULT_SOURCES["searchlabels"]).all()


@pytest.mark.parametrize("n_actions", [10, 26])
def test_masked_policy_gives_illegal_actions_zero_probability(n_actions):
    model = bc.build_model(OBS_DIM, n_actions, **TINY)
    d = bc.synth_shard(64, n_actions, 4)
    obs = torch.as_tensor(d["obs"]).float()
    mask = torch.as_tensor(d["mask"] > 0)
    for p in model.policy.action_net.parameters():  # move away from the uniform init
        torch.nn.init.normal_(p, std=0.5)
    logits, values = bc.BCNet(model.policy)(obs, mask)
    assert logits.shape == (64, n_actions) and values.shape == (64,)
    probs = bc.masked_log_probs(logits, mask).exp()
    assert (probs[~mask] == 0).all() and torch.allclose(probs.sum(1), torch.ones(64))
    with torch.no_grad():  # same distribution as SB3's (what PolicyPlayer samples from)
        sb3 = model.policy.get_distribution({"observation": obs, "action_mask": mask.to(torch.int8)}).distribution.probs
    assert torch.allclose(probs, sb3, atol=1e-5)


def test_overfit_and_checkpoints(gen9_data, tmp_path):
    out = tmp_path / "run"
    bc.main(tiny_args(gen9_data, out, "--max-steps", "60", "--batch-size", "64", "--lr", "3e-3",
                      "--warmup", "5", "--loader", "ram", "--max-rows", "256"))
    logs = [json.loads(line) for line in (out / "train_log.jsonl").open()]
    train = [r for r in logs if r["type"] == "train"]
    assert train[-1]["policy_ce"] < 0.6 * train[0]["policy_ce"], train  # learns the synthetic rule
    val = [r for r in logs if r["type"] == "val"]
    assert {"val_ce", "val_top1", "val_top1_moves", "val_top1_switches", "val_top1_ge2", "val_value_mse",
            "val_value_logloss", "val_ece"} <= set(val[-1])
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["meta"]["n_actions"] == 26 and cfg["total_steps"] == 60
    for name in ("best_model.zip", "final_model.zip", "last_model.zip", "last_state.pt"):
        assert (out / name).exists()

    from porygon.model import load_policy
    policy = load_policy(out / "final_model.zip")
    assert policy.action_space.n == 26 and policy.observation_space["observation"].shape == (OBS_DIM,)

    # --resume continues from last_state.pt
    args = tiny_args(gen9_data, out, "--max-steps", "70", "--batch-size", "64", "--loader", "ram",
                     "--max-rows", "256", "--resume", str(out))
    i = args.index("--out")
    bc.main(args[:i] + args[i + 2:])  # no --out: continues in the resumed directory
    logs = [json.loads(line) for line in (out / "train_log.jsonl").open()]
    assert max(r["step"] for r in logs) == 70


def test_stream_loader_and_eval(gen9_data, tmp_path, capsys):
    out = tmp_path / "stream"
    bc.main(tiny_args(gen9_data, out, "--epochs", "1", "--batch-size", "128", "--loader", "stream",
                      "--buffer-rows", "300", "--mix", "human=1,search=0.5", "--min-rating", "1200",
                      "--value-target", "mix", "--visit-weight-power", "0.5"))
    assert (out / "final_model.zip").exists()
    capsys.readouterr()
    bc.main(["eval", str(out / "final_model.zip"), str(gen9_data), "--device", "cpu", "--all"])
    m = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert m["val_n"] > 1000 and 0 <= m["val_top1"] <= 1


def gen9_battle():
    from poke_env.battle import Battle

    def mon(name, active, moves, tera):
        return {"ident": f"p1: {name}", "details": f"{name}, L80, M", "condition": "250/250", "active": active,
                "stats": {s: 200 for s in ("atk", "def", "spa", "spd", "spe")}, "moves": moves,
                "baseAbility": "blaze", "item": "leftovers", "pokeball": "pokeball", "ability": "blaze",
                "commanding": False, "reviving": False, "teraType": tera, "terastallized": ""}

    moves = ["flamethrower", "airslash", "roost", "earthquake"]
    req = {"active": [{"moves": [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal", "disabled": False}
                                 for m in moves], "canTerastallize": "Fire"}],
           "side": {"name": "me", "id": "p1", "pokemon": [
               mon("Charizard", True, moves, "Fire"),
               mon("Blissey", False, ["softboiled", "seismictoss", "toxic", "stealthrock"], "Fairy")]},
           "rqid": 3}
    b = Battle("battle-gen9randombattle-1", "me", logging.getLogger("test"), gen=9)
    b.player_role = "p1"
    for line in ["|player|p1|me|", "|player|p2|you|", "|teamsize|p1|2", "|teamsize|p2|6", "|gen|9",
                 "|switch|p1a: Charizard|Charizard, L80, M|250/250",
                 "|switch|p2a: Garchomp|Garchomp, L78, F|100/100", "|turn|1"]:
        b.parse_message(line.split("|"))
    b.parse_request(req)
    return b


def test_checkpoint_plays_gen9_through_policy_player(tmp_path):
    from poke_env.environment import SinglesEnv
    from poke_env.player import BattleOrder

    from porygon.env import PolicyPlayer
    from porygon.model import load_policy

    model = bc.build_model(OBS_DIM, 26, **TINY)
    model.save(tmp_path / "m")
    policy = load_policy(tmp_path / "m.zip")
    battle = gen9_battle()
    mask = np.array(SinglesEnv.get_action_mask(battle))
    assert mask[22:26].all()  # tera variants of the 4 moves are legal
    player = PolicyPlayer(policy, battle_format="gen9randombattle", start_listening=False)
    seen = set()
    for _ in range(60):
        order = player.choose_move(battle)
        assert isinstance(order, BattleOrder)
        seen.add(order.message)
    assert any("terastallize" in m for m in seen) and any("switch" in m for m in seen)


def test_spaces_match_training_env_and_resume_loads(tmp_path):
    """A from-scratch BC checkpoint loads the way `porygon.train --resume` loads it (env spaces of SelfPlayEnv)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from porygon.async_env import AsyncPPO
    from porygon.env import SelfPlayEnv

    for fmt, n in (("gen4randombattle", 10), ("gen9randombattle", 26)):
        probe = SelfPlayEnv(battle_format=fmt, start_listening=False)
        agent = probe.possible_agents[0]
        obs_space, act_space = bc.spaces_for(OBS_DIM, n)
        assert probe.observation_space(agent) == obs_space and probe.action_space(agent) == act_space
    model = bc.build_model(OBS_DIM, 10, **TINY)
    model.save(tmp_path / "m")
    import gymnasium as gym

    class Env(gym.Env):
        observation_space, action_space = bc.spaces_for(OBS_DIM, 10)

        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}

    for algo in (PPO, AsyncPPO):
        m = algo.load(tmp_path / "m.zip", env=DummyVecEnv([Env]), device="cpu")
        assert m.n_steps == bc.PPO_DEFAULTS["n_steps"] and m.gamma == bc.PPO_DEFAULTS["gamma"]
        assert m.policy_kwargs["features_extractor_kwargs"] == TINY


def test_gen4_init_from_checkpoint_and_reference(tmp_path):
    d = tmp_path / "g4"
    bc.main(["synth", str(d), "--n", "600", "--n-actions", "10", "--games", "30"])
    init = bc.build_model(OBS_DIM, 10, **TINY)
    init.save(tmp_path / "init")
    out = tmp_path / "run"
    bc.main(tiny_args(d, out, "--init", str(tmp_path / "init.zip"), "--max-steps", "4", "--batch-size", "64"))
    ref = bc.load_reference(out / "final_model.zip")
    assert not any(p.requires_grad for p in ref.parameters())
    s = bc.synth_shard(16, 10, 2)
    obs = {"observation": torch.as_tensor(s["obs"]).float(), "action_mask": torch.as_tensor(s["mask"])}
    logp, v = bc.policy_log_probs(ref, obs)
    assert torch.allclose(logp.exp().sum(1), torch.ones(16)) and (logp.exp()[obs["action_mask"] == 0] == 0).all()


def test_summarize_duplicate_pairs(tmp_path):
    rows = []
    for pair in range(20):
        for game in (0, 1):
            win = "netA" if (pair % 4 != 0 or game == 0) else "searchB"
            rows.append({"pair": pair, "game": game, "winner": win, "p1": "netA", "p2": "searchB"})
    (tmp_path / "battles.jsonl").write_text("\n".join(map(json.dumps, rows)))
    line = bc.summarize([tmp_path], "^net")
    assert "87.5%" in line and "duplicate pairs: 87.5%" in line and "pairs=20" in line
