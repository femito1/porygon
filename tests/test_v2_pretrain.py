"""v2 pretraining's streaming loader (`--stream-gb`): same training / validation rows, weights and value-row selection
as the in-memory loader, every training row exactly once per epoch across several parts, bounded parts in flight,
and an end-to-end run of `main()` in both modes. CPU, tiny net, synthetic shards in tmp_path (deleted after each
test: the laptop disk is full)."""

import json
import shutil
import sys

import numpy as np
import pytest
import torch

from porygon.v2 import pretrain
from porygon.v2.pretrain import StreamSource, is_val, load_sets
from porygon.v2.spec import fields, n_actions

GEN = 9
TINY = dict(d_model=32, n_layers=1, n_heads=4, ffn_mult=1, d_move=16, d_species=16, d_move_emb=16, d_item=8,
            d_ability=8, d_type=8, pair_hidden=8, value_hidden=32)


@pytest.fixture
def tmp(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def write_shard(path, games, tag0: int, per_game: int = 5, weight=None) -> np.ndarray:
    """`per_game` rows per game id; field_num[:, 0] = a unique row tag (tag0, tag0 + 1, ...); legal actions 0, 1, 6
    with a one-hot target on 6; z alternates per game."""
    games = np.asarray(list(games), np.int64)
    n = len(games) * per_game
    obs = {k: np.zeros((n, *shape), dtype) for k, (shape, dtype) in fields(GEN).items()}
    obs["action_mask"][:, [0, 1, 6]] = 1.0
    tags = np.arange(tag0, tag0 + n)
    obs["field_num"][:, 0] = tags
    tgt = np.zeros((n, n_actions(GEN)), np.float32)
    tgt[:, 6] = 1.0
    game = np.repeat(games, per_game)
    extra = {} if weight is None else {"weight": np.full(n, weight, np.float32)}
    np.savez_compressed(path, **{f"obs/{k}": v for k, v in obs.items()}, target=tgt,
                        z=np.where(game % 2 == 0, 1.0, -1.0).astype(np.float32), game=game, **extra)
    return tags


def make_sets(tmp):
    """Primary: 3 shards, game 7 spans shards 0 and 1; mix A (weight 2.5, a shard with its own weights 0.5); mix B
    (weight 0: validation only)."""
    (tmp / "p").mkdir()
    (tmp / "a").mkdir()
    (tmp / "b").mkdir()
    write_shard(tmp / "p" / "shard-00000.npz", range(0, 8), 0)
    write_shard(tmp / "p" / "shard-00001.npz", [7] + list(range(8, 20)), 1000)
    write_shard(tmp / "p" / "shard-00002.npz", range(20, 40), 2000, per_game=3)
    write_shard(tmp / "a" / "shard-00000.npz", range(100, 130), 3000, weight=0.5)
    write_shard(tmp / "b" / "shard-00000.npz", range(200, 230), 4000)
    return str(tmp / "p" / "shard-*.npz"), [str(tmp / "a" / "shard-*.npz") + ":2.5", str(tmp / "b" / "shard-*.npz")
                                            + ":0"]


def tags_of(d) -> np.ndarray:
    return np.asarray(d["field_num"][:, 0]).astype(np.int64)


def test_stream_rows_match_in_memory(tmp):
    primary, mix = make_sets(tmp)
    data, names, counts = load_sets(primary, mix, GEN)
    val = is_val(data["game"], 0.2)
    train_idx = np.flatnonzero(~val & (data["weight"] > 0))
    s = StreamSource(primary, mix, GEN, 0.2, part_gb=1e-9, seed=0)  # tiny parts: one part per few rows
    assert s.names == names and s.counts == counts and s.n_rows == len(data["z"])
    assert s.passes > 3
    # training rows and their weights
    assert np.array_equal(np.sort(tags_of(data)[train_idx]), np.sort(tags_of(s.assemble(s.train_rows))))
    ref_w = dict(zip(tags_of(data)[train_idx], data["weight"][train_idx]))
    got = s.assemble(s.train_rows)
    assert all(ref_w[t] == w for t, w in zip(tags_of(got), got["weight"]))
    assert set(np.unique(got["weight"])) == {1.0, 1.25}  # primary 1, mix A 0.5 * 2.5; mix B (0) never trains
    # validation sets: same rows per set, every key equal row by row (in-memory order is shard order = ours)
    for i, nm in enumerate(names):
        ref = np.flatnonzero(val & (data["src"] == i))
        mine = s.val_sets[nm]
        assert len(ref) == len(mine) > 0
        for k in data:
            assert np.array_equal(data[k][ref], s.vdata[k][mine]), k


def test_stream_epoch_covers_each_row_once_and_one_value_row_per_game(tmp):
    primary, mix = make_sets(tmp)
    s = StreamSource(primary, mix, GEN, 0.2, part_gb=1e-9, seed=1)
    s.passes = 4  # several parts per epoch
    n_train = len(s.train_rows)
    train_tags = np.sort(tags_of(s.assemble(s.train_rows)))
    bs = 16
    rng = np.random.default_rng(0)
    it = s.batches(bs, torch.device("cpu"), rng, one_per_game=True)
    for epoch in range(2):
        seen, vrows = [], []
        rows_left = n_train
        while rows_left > 0:
            mb = next(it)
            t = mb["field_num"][:, 0].long().numpy()
            seen.append(t)
            vrows.append(t[mb["vsel"].numpy() > 0])
            rows_left -= len(t)
            assert (mb["drop_seed"] > 0).all() and (mb["opp_cls"] == -1).all()
        assert rows_left == 0  # parts end exactly at the epoch boundary
        seen = np.concatenate(seen)
        assert np.array_equal(np.sort(seen), train_tags)  # every training row exactly once
        # exactly one value row per game that has training rows (game 7 spans two shards)
        tag_game = {t: g for t, g in zip(tags_of(s.assemble(s.train_rows)), s.game[s.train_rows])}
        vg = [tag_game[t] for t in np.concatenate(vrows)]
        assert sorted(vg) == sorted(set(tag_game.values()))
    it.close()


def test_stream_bounded_parts_in_flight(tmp, monkeypatch):
    primary, mix = make_sets(tmp)
    s = StreamSource(primary, mix, GEN, 0.2, part_gb=1e-9, seed=2)
    s.passes = 6
    built, consumed = [], []
    orig = s.assemble

    def counting(rows):
        built.append(len(rows))
        return orig(rows)

    monkeypatch.setattr(s, "assemble", counting)
    it = s.batches(10_000, torch.device("cpu"), np.random.default_rng(0), one_per_game=False)  # one batch per part
    for _ in range(4):
        next(it)
        consumed.append(1)
        import time
        time.sleep(0.3)  # give the loader time to run ahead if it could
        assert len(built) <= len(consumed) + 1  # at most one part prefetched beyond the one being consumed
    it.close()


def test_main_runs_in_both_modes(tmp, monkeypatch):
    primary, mix = make_sets(tmp)
    cfg = json.dumps(TINY)
    for mode, extra in (("mem", []), ("stream", ["--stream-gb", "1e-9"]),
                        ("stream1pg", ["--stream-gb", "1e-9", "--value-sample", "one-per-game"])):
        out = tmp / f"out-{mode}"
        argv = ["pretrain", "--shards", primary, "--mix", mix[0], "--mix", mix[1], "--gen", str(GEN), "--out",
                str(out), "--model-json", cfg, "--epochs", "2", "--batch-size", "16", "--warmup", "2",
                "--eval-every", "5", "--val-frac", "0.2", "--device", "cpu", "--no-bf16", *extra]
        monkeypatch.setattr(sys, "argv", argv)
        pretrain.main()
        recs = [json.loads(x) for x in (out / "pretrain.log").read_text().splitlines()]
        data = recs[0]
        assert data["event"] == "data" and ("stream" in data) == (mode != "mem")
        vals = [r for r in recs if r["event"] == "val"]
        assert vals and all(np.isfinite(r["policy_ce"]) for r in vals)
        assert {"val/a/policy_ce", "val/b/policy_ce"} <= set(vals[-1])
        assert (out / "final.pt").exists() and (out / "best.pt").exists()


def mark_shuffled(tmp, dirs=("p", "a", "b")):
    for d in dirs:
        (tmp / d / "meta.json").write_text(json.dumps({"layout": "shuffled"}))


def test_stream_shard_parts_whole_shards_once_per_epoch(tmp, monkeypatch):
    """--stream-parts shards: every training row once per epoch, each part = the training rows of whole shards
    (rows shuffled within it), each shard decompressed once per epoch, value rows as in the row plan."""
    primary, mix = make_sets(tmp)
    mark_shuffled(tmp)
    s = StreamSource(primary, mix, GEN, 0.2, part_gb=1e-9, seed=3, parts="shards")
    n_sh = len(s.shard_train)
    assert n_sh == 4  # primary 3 + mix A 1 (mix B has weight 0: no training rows)
    s.passes = 3  # 4 shards into 3 parts
    loads = []
    orig = pretrain.load_shard
    monkeypatch.setattr(pretrain, "load_shard", lambda path, gen: (loads.append(path), orig(path, gen))[1])
    sid_of = {int(r): int(s.sid[r]) for r in s.train_rows}
    train_of = {int(s.sid[r[0]]): set(r.tolist()) for r in s.shard_train}  # shard id -> its training rows
    plans = s.plans(one_per_game=True)
    for epoch in range(2):
        rows_ep, shards_ep = [], []
        for _ in range(3):
            rows, vs = next(plans)
            shards = {sid_of[int(r)] for r in rows}
            for sh in shards:  # whole shards: all of a touched shard's training rows are in this part
                assert train_of[sh] <= set(rows.tolist())
            shards_ep.append(shards)
            rows_ep.append(rows)
            assert vs is not None and len(vs) == len(rows)
        allr = np.concatenate(rows_ep)
        assert np.array_equal(np.sort(allr), s.train_rows)  # every training row exactly once
        assert sum(len(x) for x in shards_ep) == n_sh  # parts touch disjoint shard sets
    # rows are shuffled within a part (not in shard row order)
    assert any(not np.array_equal(r, np.sort(r)) for r in rows_ep)
    # assembling one epoch's parts decompresses each training shard once
    loads.clear()
    for _ in range(3):
        rows, _ = next(plans)
        s.assemble(rows)
    assert len(loads) == len(set(loads)) == n_sh


def test_stream_shard_parts_batches_cover_epoch(tmp):
    primary, mix = make_sets(tmp)
    mark_shuffled(tmp)
    s = StreamSource(primary, mix, GEN, 0.2, part_gb=1e-9, seed=4, parts="shards")
    s.passes = 2
    train_tags = np.sort(tags_of(s.assemble(s.train_rows)))
    it = s.batches(7, torch.device("cpu"), np.random.default_rng(0), one_per_game=True)
    seen, left = [], len(s.train_rows)
    while left > 0:
        t = next(it)["field_num"][:, 0].long().numpy()
        seen.append(t)
        left -= len(t)
    assert left == 0 and np.array_equal(np.sort(np.concatenate(seen)), train_tags)
    it.close()


def test_stream_shard_parts_refuses_unshuffled(tmp):
    primary, mix = make_sets(tmp)
    with pytest.raises(SystemExit, match="meta.json"):
        StreamSource(primary, mix, GEN, 0.2, part_gb=1e-9, seed=0, parts="shards")
    mark_shuffled(tmp, dirs=("p", "a"))
    (tmp / "b" / "meta.json").write_text(json.dumps({"layout": "source"}))
    with pytest.raises(SystemExit, match="not 'shuffled'"):
        StreamSource(primary, mix, GEN, 0.2, part_gb=1e-9, seed=0, parts="shards")


def test_main_runs_with_shard_parts(tmp, monkeypatch):
    primary, mix = make_sets(tmp)
    mark_shuffled(tmp)
    out = tmp / "out-shards"
    argv = ["pretrain", "--shards", primary, "--mix", mix[0], "--gen", str(GEN), "--out", str(out), "--model-json",
            json.dumps(TINY), "--epochs", "2", "--batch-size", "16", "--warmup", "2", "--eval-every", "5",
            "--val-frac", "0.2", "--device", "cpu", "--no-bf16", "--stream-gb", "1e-9", "--stream-parts", "shards",
            "--value-sample", "one-per-game"]
    monkeypatch.setattr(sys, "argv", argv)
    pretrain.main()
    recs = [json.loads(x) for x in (out / "pretrain.log").read_text().splitlines()]
    assert recs[0]["stream"]["parts"] == "shards" and recs[0]["stream"]["train_shards"] == 4
    assert (out / "final.pt").exists()
