"""Behaviour cloning: supervised training of the policy/value network on human replays and search labels.

    python -m porygon.bc train DATA [DATA ...] --out runs/bc-x [--epochs 4 --batch-size 2048 --lr 5e-4 ...]
    python -m porygon.bc eval  MODEL.zip DATA [DATA ...] [--all]      # val metrics of any checkpoint (e.g. old PPO)
    python -m porygon.bc synth OUT_DIR --n 4000 --n-actions 26        # synthetic shard + meta.json (tests, smoke)
    python -m porygon.bc summarize RUN_DIR [RUN_DIR ...] --ours REGEX # duplicate-pair win rate of a benchmark run

Input (the data stream's supervised format): DATA is a directory `encoded/<format>/<encoder>/` with `meta.json`
(encoder "module:function", obs_dim, n_actions, format, counts, optional "sources": {name: code}) and
`shard-NNNN.npz` files (np.savez_compressed), each with
    obs float16 [N, D]   mask int8 [N, A]   action int16 [N] (-1 unknown)   target float16 [N, A] (policy target)
    z float32 [N] (final result, decider's view, +-1/0)   rating, opp_rating int16 (-1 unknown)   game int32
    turn int16   source uint8   forced uint8   n_legal uint8
    optional (search labels): root_value float32 [N] (search's win expectation in [0, 1]; NaN / -1 = none),
    visits int32 [N] (root visits, for --visit-weight-power).
The older expert-iteration shards (porygon.expert: obs, mask, prior, target, ret) load too: ret -> z, source
"search", no rating; they carry no game id, so games are recovered as runs of non-decreasing turn (one player's
side of one battle; the two sides of a battle can still land on different sides of the split).
Source ids follow scripts/convert_replays.py (showdown-replays 1, holidayougi 2, pokechamp 3, searchlabels 10,
benchmark 11; legacy shards are searchlabels); --sources / --mix take names from meta.json, numeric ids, or the
groups human (0-3) and search (10).

Model: a normal SB3 PPO checkpoint (`porygon.model.load_policy`, `PolicyPlayer`, `porygon.search`,
`porygon.train --resume` all take the output). Built from scratch for any obs_dim / n_actions (gen4 10, gen9 26
with tera 22-25) on spec-only spaces identical to training's (Dict observation Box(D) float32 + action_mask
Box(A) int8, Discrete(A)), with the PPO hyperparameters of porygon.train's defaults, or initialised from a
checkpoint (--init). --policy module:Class picks the policy class (default porygon.model:MaskedPolicy); a class
may define `bc_forward(obs) -> (logits, values)`, otherwise the SB3 actor-critic path is used.

Losses: soft cross-entropy of the masked policy to `target` (renormalised over legal actions; one-hot of
`action` when the target is empty) on decisions with a policy target, n_legal >= --min-legal and (with
--skip-forced) not forced; value loss (MSE of V to z, or --value-loss bce: BCE of (tanh V + 1)/2 to (z + 1)/2) on
ALL kept decisions, times --vf-coef; the value target is the game result z (--value-target result), the
search's root value 2 rv - 1 (search; z where a decision has none) or a mix. Row weights: --mix source weights x
optional rating weights x optional visit weights. Use the default mse for checkpoints that search / PPO will read
(they treat V as a value in [-1, 1]). Arms: human only `--sources human`, search only `--sources search`, both
`--mix human=1,search=W` (source names from meta.json "sources", or numeric codes).

Split: by game, deterministic (a hash of the game id against --val-frac); the validation set is the val-bucket
games of the shards in sorted order up to --val-max decisions (val-bucket games beyond the cap are not trained
on either). Per evaluation: policy CE (on the training subset and on all decisions with a target), top-1
(overall, moves, switches, tera, >= 2 legal), value MSE, log loss, expected calibration error (10 bins).

Data loading (--loader): `gpu` keeps the filtered training rows on the GPU, `ram` in host memory, `stream`
reads shards in a random order through a thread pool into a shuffle buffer of --buffer-rows decisions (default
1M = ~8 GB of float16 observations at D = 4079) and never holds more. `auto` picks gpu / ram / stream by size.
Memory plan for ~20M decisions (D = 4079): obs alone are 20M x 8.2 KB = 163 GB as float16, so stream them
(--mem ~64 GB: buffer + concat copy + prefetch); compressed shards are ~10x smaller, so copy them to node-local
disk first (scripts/slurm_bc.sbatch). Up to ~5M decisions (40 GB) fit an a100's 80 GB with --loader gpu.

Outputs in --out: best_model.zip (lowest val policy CE), final_model.zip, last_model.zip + last_state.pt (for
--resume after --max-hours), train_log.jsonl, config.json (args, data meta, code commit, host).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# shard `source` ids: scripts/convert_replays.py SOURCE_IDS (meta.json "source_ids" / "sources" override them)
DEFAULT_SOURCES = {"other": 0, "showdown-replays": 1, "holidayougi": 2, "pokechamp": 3, "searchlabels": 10,
                   "benchmark": 11}
SOURCE_GROUPS = {"human": ("other", "showdown-replays", "holidayougi", "pokechamp"), "search": ("searchlabels",)}
SMALL_KEYS = ("game", "rating", "source", "z")  # what the counting pass reads
MASK_FILL = -1e9

# porygon.train's PPO defaults, so `porygon.train --resume` on a BC checkpoint behaves like a normal run
# (it rebuilds the rollout buffer for its own --rollout and resets the lr schedule).
PPO_DEFAULTS = dict(n_steps=1024, batch_size=1024, n_epochs=7, gamma=0.9999, gae_lambda=0.754,
                    clip_range=0.0829, clip_range_vf=0.0184, ent_coef=0.01, vf_coef=0.4375, max_grad_norm=0.543)


# ----------------------------------------------------------------------------------------------- data


def list_shards(paths: list[Path]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out += [p] if p.is_file() else sorted(p.glob("shard-*.npz")) or sorted(p.glob("*.npz"))
    if not out:
        raise SystemExit(f"no .npz shards in {paths}")
    return out


def read_meta(paths: list[Path]) -> dict:
    """meta.json of each data directory (inferred from the first shard for legacy ones); all must agree on
    obs_dim and n_actions."""
    metas = []
    for p in map(Path, paths):
        d = p if p.is_dir() else p.parent
        m = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        if not isinstance(m.get("obs_dim"), int) or not isinstance(m.get("n_actions"), int):  # absent / null
            with np.load(list_shards([p])[0]) as z:
                m["obs_dim"], m["n_actions"] = int(z["obs"].shape[1]), int(z["mask"].shape[1])
                m.setdefault("legacy", "ret" in z.files and "z" not in z.files)
        m.setdefault("encoder", "porygon.encoder:embed_battle")
        m["path"] = str(p)
        metas.append(m)
    for m in metas[1:]:
        assert (m["obs_dim"], m["n_actions"]) == (metas[0]["obs_dim"], metas[0]["n_actions"]), \
            f"datasets disagree on obs_dim / n_actions: {metas}"
    return {"obs_dim": metas[0]["obs_dim"], "n_actions": metas[0]["n_actions"],
            "encoder": metas[0]["encoder"], "format": metas[0].get("format"), "datasets": metas}


def turn_index() -> int:
    """Index of min(turn, 100) / 100 in porygon.encoder's field token (legacy game recovery)."""
    from porygon import encoder as E
    return 2 * len(E.WEATHERS) + 1 + 2 * len(E.FIELDS) + 2 * (2 * len(E.SCREENS) + len(E.HAZARDS))


def load_shard(path: Path, shard_id: int, keys: tuple[str, ...] | None = None,
               sources: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    """One shard in the standard field set (legacy expert shards converted). `keys` limits what is
    decompressed (np.load reads npz members lazily)."""
    sources = sources or DEFAULT_SOURCES
    with np.load(path) as f:
        legacy = "ret" in f.files and "z" not in f.files
        want = set(keys) if keys else None

        def get(k):
            return f[k] if (want is None or k in want) else None

        if not legacy:
            out = {k: f[k] for k in f.files if want is None or k in want}
            n = len(f["z"]) if "z" in f.files else len(f["obs"])
            if "game" in out:
                out["game"] = out["game"].astype(np.int64)
            return out | {"n": np.int64(n)}
        n = len(f["ret"])
        need = lambda k: want is None or k in want  # noqa: E731
        out = {"n": np.int64(n)}
        if need("z"):
            out["z"] = f["ret"].astype(np.float32)
        if need("game"):
            obs = f["obs"]
            t = obs[:, turn_index()].astype(np.float32)
            block = np.concatenate([[0], np.cumsum(t[1:] < t[:-1])])
            out["game"] = (np.int64(shard_id + 1) << 32) + block  # never collides with real (int32) ids
            if need("obs"):
                out["obs"] = obs
        for k in ("obs", "mask", "target"):
            if need(k) and k not in out:
                out[k] = get(k)
        if need("action"):
            out["action"] = np.full(n, -1, np.int16)
        for k in ("rating", "opp_rating"):
            if need(k):
                out[k] = np.full(n, -1, np.int16)
        if need("source"):
            out["source"] = np.full(n, sources.get("searchlabels", 10), np.uint8)
        if need("forced"):
            out["forced"] = np.zeros(n, np.uint8)
        if need("turn"):
            out["turn"] = np.zeros(n, np.int16)
        if need("n_legal"):
            out["n_legal"] = (get("mask") > 0).sum(1).astype(np.uint8)
        return out


def val_bucket(game: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Deterministic game-level split: splitmix64(game ^ seed) mod 10^6 < frac * 10^6."""
    x = game.astype(np.uint64) ^ np.uint64(seed * 0x9E3779B97F4A7C15 % 2**64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
    return (x % np.uint64(10**6)) < np.uint64(round(frac * 10**6))


@dataclass
class Filters:
    min_rating: int | None = None
    max_rating: int | None = None
    drop_unrated: bool = False
    sources: set[int] | None = None  # None = all
    mix: dict[int, float] = field(default_factory=dict)  # source code -> loss weight (default 1)
    rating_weight_scale: float = 0.0  # w *= clip(exp((rating - ref) / scale), 0.1, 10) when > 0
    rating_ref: float = 1500.0
    skip_forced: bool = False
    min_legal: int = 2
    label_smoothing: float = 0.0
    target_temperature: float = 1.0
    value_target: str = "result"  # result | search | mix (search: 2 * root_value - 1 where present, else result)
    value_mix: float = 0.5  # mix: value_mix * search + (1 - value_mix) * result
    visit_weight_power: float = 0.0  # > 0: w *= clip((visits / shard mean) ^ power, 0.1, 10) where visits exist

    def row_weight(self, d: dict[str, np.ndarray]) -> np.ndarray:
        """Per-decision weight (0 = dropped) from the row-level filters."""
        n = int(d["n"])
        w = np.ones(n, np.float32)
        r = d["rating"].astype(np.int32) if "rating" in d else np.full(n, -1)
        known = r >= 0
        if self.drop_unrated:
            w[~known] = 0
        if self.min_rating is not None:
            w[known & (r < self.min_rating)] = 0
        if self.max_rating is not None:
            w[known & (r > self.max_rating)] = 0
        src = d["source"].astype(np.int64) if "source" in d else np.zeros(n, np.int64)
        if self.sources is not None:
            w[~np.isin(src, list(self.sources))] = 0
        for code, m in self.mix.items():
            w[src == code] *= m
        if self.rating_weight_scale > 0:
            rw = np.clip(np.exp((r - self.rating_ref) / self.rating_weight_scale), 0.1, 10.0)
            w = np.where(known, w * rw, w).astype(np.float32)
        if self.visit_weight_power > 0 and "visits" in d:
            v = d["visits"].astype(np.float64)
            has = v > 0
            if has.any():
                vw = np.clip((v / v[has].mean()) ** self.visit_weight_power, 0.1, 10.0)
                w = np.where(has, w * vw, w).astype(np.float32)
        return w

    def value_targets(self, d: dict[str, np.ndarray]) -> np.ndarray:
        """The value-head target on the [-1, 1] scale: the game result z, the search's root value (stored in
        [0, 1] as our win expectation -> 2 rv - 1) where a decision has one, or a mix."""
        z = d["z"].astype(np.float32)
        if self.value_target == "result" or "root_value" not in d:
            return z
        rv = d["root_value"].astype(np.float32)
        ok = np.isfinite(rv) & (rv >= 0) & (rv <= 1)
        s = 2 * rv - 1
        lam = 1.0 if self.value_target == "search" else self.value_mix
        return np.where(ok, lam * s + (1 - lam) * z, z).astype(np.float32)


def prepare(d: dict[str, np.ndarray], filt: Filters, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Training arrays for the kept rows (`rows`: boolean selection on top of the filters)."""
    w = filt.row_weight(d)
    z = filt.value_targets(d)
    keep = (w > 0) & np.isfinite(z)
    if rows is not None:
        keep &= rows
    ix = np.flatnonzero(keep)
    mask = d["mask"][ix] > 0
    target = d["target"][ix].astype(np.float32) * mask
    if filt.target_temperature != 1.0:
        target = target ** (1.0 / filt.target_temperature)
    s = target.sum(1)
    action = d["action"][ix].astype(np.int64) if "action" in d else np.full(len(ix), -1)
    fill = (s <= 0) & (action >= 0)
    if fill.any():  # empty target but a known action -> one-hot
        r = np.flatnonzero(fill)
        legal = mask[r, action[r].clip(0, mask.shape[1] - 1)]
        target[r[legal], action[r[legal]]] = 1.0
        s = target.sum(1)
    has_pol = s > 0
    target[has_pol] /= s[has_pol, None]
    n_legal = (d["n_legal"][ix] if "n_legal" in d else mask.sum(1)).astype(np.uint8)
    forced = (d["forced"][ix] if "forced" in d else np.zeros(len(ix))).astype(bool)
    pol_sub = has_pol & (n_legal >= filt.min_legal) & ~(forced & filt.skip_forced)
    return {
        "obs": np.ascontiguousarray(d["obs"][ix], dtype=np.float16),
        "mask": mask, "target": target, "z": np.nan_to_num(z[ix]), "result": np.nan_to_num(d["z"][ix]).astype(np.float32),
        "pw": (w[ix] * pol_sub).astype(np.float32), "vw": w[ix].astype(np.float32),
        "has_pol": has_pol, "n_legal": n_legal,
    }


def concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    parts = [p for p in parts if len(p["z"])]
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def take(d: dict[str, np.ndarray], ix) -> dict[str, np.ndarray]:
    return {k: v[ix] for k, v in d.items()}


def rows(d: dict) -> int:
    return len(d["z"]) if d else 0


class ShardSet:
    """Shards + split + filters: counts in a cheap first pass (small fields only), then train / val rows."""

    def __init__(self, paths: list[Path], filt: Filters, val_frac: float, val_max: int, seed: int = 0,
                 workers: int = 4, max_shards: int | None = None, val_mask: Path | None = None):
        self.shards = list_shards(paths)[:max_shards]
        self.filt, self.val_frac, self.val_max, self.seed, self.workers = filt, val_frac, val_max, seed, workers
        self.val_mask = Path(val_mask) if val_mask else None
        meta = read_meta(paths)
        self.sources = source_table(meta)
        with ThreadPoolExecutor(workers) as ex:
            small = list(ex.map(lambda a: load_shard(a[1], a[0], SMALL_KEYS, self.sources),
                                enumerate(self.shards)))
        self.counts = []  # (train rows, val rows) per shard after the row filters
        for d in small:
            keep = (filt.row_weight(d) > 0) & np.isfinite(d["z"])
            v = self.is_val(len(self.counts), d)
            self.counts.append((int((keep & ~v).sum()), int((keep & v).sum())))
        self.n_total = sum(int(d["n"]) for d in small)
        # the val set: val-bucket games of the shards in order until val_max decisions
        acc, self.val_shards = 0, []
        for i, (_, nv) in enumerate(self.counts):
            if acc >= val_max:
                break
            if nv:
                self.val_shards.append(i)
                acc += nv
        self.n_train = sum(t for t, _ in self.counts)
        self.n_val = min(acc, val_max)

    def is_val(self, i: int, d: dict[str, np.ndarray]) -> np.ndarray:
        """Row-level val selection: the hash split of the game id, or --val-mask DIR/<shard stem>.npy (bool per
        row, e.g. another model's held-out games mapped onto these shards; scripts/paired_val_mask.py)."""
        if self.val_mask is None:
            return val_bucket(d["game"], self.val_frac, self.seed)
        m = np.load(self.val_mask / f"{Path(self.shards[i]).stem}.npy").astype(bool)
        assert len(m) == int(d["n"]), f"val mask of {self.shards[i]}: {len(m)} rows vs {int(d['n'])}"
        return m

    def load(self, i: int, part: str) -> dict[str, np.ndarray]:
        d = load_shard(self.shards[i], i, sources=self.sources)
        v = self.is_val(i, d)
        return prepare(d, self.filt, ~v if part == "train" else v)

    def val_set(self) -> dict[str, np.ndarray]:
        with ThreadPoolExecutor(self.workers) as ex:
            d = concat(list(ex.map(lambda i: self.load(i, "val"), self.val_shards)))
        return take(d, slice(0, self.val_max)) if d else d

    def all_train(self) -> dict[str, np.ndarray]:
        with ThreadPoolExecutor(self.workers) as ex:
            return concat(list(ex.map(lambda i: self.load(i, "train"), range(len(self.shards)))))

    def stream(self, epoch: int, batch_size: int, buffer_rows: int):
        """Shuffled numpy batches: shards in a random order, mixed in a buffer of >= buffer_rows decisions."""
        rng = np.random.default_rng((self.seed, epoch))
        order = [i for i in rng.permutation(len(self.shards)) if self.counts[i][0]]
        carry: list[dict] = []
        with ThreadPoolExecutor(self.workers) as ex:
            pending = deque(ex.submit(self.load, i, "train") for i in order[:2 * self.workers])
            nxt = 2 * self.workers
            while pending:
                carry.append(pending.popleft().result())
                if nxt < len(order):
                    pending.append(ex.submit(self.load, order[nxt], "train"))
                    nxt += 1
                if sum(map(rows, carry)) >= buffer_rows or not pending:
                    buf = concat(carry)
                    n = rows(buf)
                    perm = rng.permutation(n)
                    full = n // batch_size * batch_size
                    for s in range(0, full, batch_size):
                        yield take(buf, perm[s:s + batch_size])
                    carry = [take(buf, perm[full:])] if full < n else []


def ram_batches(data: dict, epoch: int, batch_size: int, seed: int):
    perm = np.random.default_rng((seed, epoch)).permutation(rows(data))
    for s in range(0, len(perm) - batch_size + 1, batch_size):
        yield take(data, np.sort(perm[s:s + batch_size]))  # sorted gather is kinder to the cache


TENSOR_KEYS = ("obs", "mask", "target", "z", "result", "pw", "vw", "has_pol", "n_legal")


def to_torch(b: dict[str, np.ndarray], device: torch.device, pin: bool) -> dict[str, torch.Tensor]:
    out = {}
    for k in TENSOR_KEYS:
        t = torch.from_numpy(np.ascontiguousarray(b[k]))
        if pin:
            t = t.pin_memory()
        out[k] = t.to(device, non_blocking=pin)
    return out


class Prefetch:
    """Runs a batch generator (and host->device copies) in a background thread; close() stops it."""

    def __init__(self, gen, device: torch.device, depth: int = 6):
        self.q: queue.Queue = queue.Queue(depth)
        self.wait, self.stopped = 0.0, threading.Event()
        pin = device.type == "cuda"

        def put(x) -> bool:
            while not self.stopped.is_set():
                try:
                    self.q.put(x, timeout=0.5)
                    return True
                except queue.Full:
                    pass
            return False

        def run():
            try:
                for b in gen:
                    if not put(to_torch(b, device, pin)):
                        break
            except BaseException as e:  # surfaced in the training thread
                put(e)
            finally:
                gen.close()
            put(None)

        threading.Thread(target=run, daemon=True).start()

    def __iter__(self):
        while True:
            t = time.perf_counter()
            b = self.q.get()
            self.wait += time.perf_counter() - t
            if b is None:
                return
            if isinstance(b, BaseException):
                raise b
            yield b

    def close(self) -> None:
        self.stopped.set()


# ---------------------------------------------------------------------------------------------- model


def import_obj(spec: str):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def spaces_for(obs_dim: int, n_actions: int):
    """Exactly the spaces SelfPlayEnv gives SB3 (poke-env wraps the Box in a Dict with an int8 action mask)."""
    from gymnasium import spaces
    obs = spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
        "action_mask": spaces.Box(0, 1, shape=(n_actions,), dtype=np.int8),
    })
    return obs, spaces.Discrete(n_actions)


def build_model(obs_dim: int, n_actions: int, policy: str = "porygon.model:MaskedPolicy", d_model: int = 256,
                n_layers: int = 3, n_heads: int = 8, policy_kwargs: dict | None = None, device: str = "cpu"):
    """A fresh PPO (porygon.train's settings) on spec-only spaces: no Showdown, works for any generation."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from porygon.train import thesis_lr

    obs_space, act_space = spaces_for(obs_dim, n_actions)

    class SpecEnv(gym.Env):
        observation_space, action_space = obs_space, act_space

        def reset(self, *, seed=None, options=None):
            return obs_space.sample(), {}

        def step(self, action):
            raise NotImplementedError("spec-only env (porygon.bc)")

    kw = dict(policy_kwargs or {})  # a "features_extractor_kwargs" entry replaces d_model / n_layers / n_heads
    kw.setdefault("features_extractor_kwargs", dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads))
    if kw["features_extractor_kwargs"] is None:
        del kw["features_extractor_kwargs"]
    return PPO(import_obj(policy), DummyVecEnv([SpecEnv]), learning_rate=thesis_lr, device=device, verbose=0,
               policy_kwargs=kw, **PPO_DEFAULTS)


def load_model(path: str | Path, device: str = "cpu"):
    from stable_baselines3 import PPO
    return PPO.load(path, device=device)


class BCNet(nn.Module):
    """policy -> (logits, values) on raw tensors, sharing the policy's parameters (so it can be compiled)."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.policy
        o = {"observation": obs, "action_mask": mask}
        if hasattr(p, "bc_forward"):
            return p.bc_forward(o)
        feats = p.extract_features(o)
        if p.share_features_extractor:
            latent_pi, latent_vf = p.mlp_extractor(feats)
        else:
            latent_pi, latent_vf = p.mlp_extractor.forward_actor(feats[0]), p.mlp_extractor.forward_critic(feats[1])
        return p.action_net(latent_pi), p.value_net(latent_vf).squeeze(-1)


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float().masked_fill(~mask, MASK_FILL), -1)


def policy_log_probs(policy, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(masked log-probs [B, A], values [B]) of any checkpoint's policy on an SB3-style observation dict
    {"observation": float [B, D], "action_mask": [B, A]}. Illegal actions get ~-1e9 (probability exactly 0), so
    KL(pi || ref) = sum pi * (log pi - ref) over legal actions is finite. Same distribution SB3 samples from."""
    mask = obs["action_mask"] > 0
    logits, values = BCNet(policy)(obs["observation"].float(), mask)
    return masked_log_probs(logits, mask), values.float()


def load_reference(path: str | Path, device: str = "cpu"):
    """A frozen copy of a checkpoint's policy (eval mode, no gradients), e.g. the BC network as a KL anchor for RL:
        ref = load_reference("runs/bc-.../best_model.zip", "cuda")
        with torch.no_grad(): ref_logp, _ = policy_log_probs(ref, obs)"""
    from porygon.model import load_policy
    policy = load_policy(path, device)
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


def value_prob(v: torch.Tensor, kind: str) -> torch.Tensor:
    """P(win) implied by the value output."""
    return torch.sigmoid(2 * v) if kind == "bce" else ((v + 1) / 2).clamp(1e-4, 1 - 1e-4)


def losses(logits, values, b, label_smoothing: float = 0.0, value_loss: str = "mse"):
    mask = b["mask"]
    logp = masked_log_probs(logits, mask)
    t = b["target"]
    if label_smoothing > 0:
        t = (1 - label_smoothing) * t + label_smoothing * mask / mask.sum(1, keepdim=True).clamp_min(1)
    ce = -torch.where(mask, t * logp, torch.zeros_like(logp)).sum(1)
    pw, vw = b["pw"], b["vw"]
    pol = (pw * ce).sum() / pw.sum().clamp_min(1e-8)
    v, z = values.float(), b["z"]
    if value_loss == "bce":
        vl = F.binary_cross_entropy_with_logits(2 * v, ((z + 1) / 2).clamp(0, 1), reduction="none")
    else:
        vl = (v - z) ** 2
    val = (vw * vl).sum() / vw.sum().clamp_min(1e-8)
    return pol, val


@torch.no_grad()
def evaluate(net: BCNet, data: dict[str, np.ndarray], device: torch.device, batch_size: int = 4096,
             amp: torch.dtype | None = None, value_loss: str = "mse") -> dict[str, float]:
    """Unweighted validation metrics."""
    net.eval()
    acc = defaultdict(float)
    bins = np.zeros((10, 3))  # n, sum p, sum y
    for s in range(0, rows(data), batch_size):
        b = to_torch(take(data, slice(s, s + batch_size)), device, False)
        with torch.autocast(device.type, dtype=amp or torch.bfloat16, enabled=amp is not None):
            logits, v = net(b["obs"].float(), b["mask"])
        logp = masked_log_probs(logits, b["mask"])
        t = b["target"]
        ce = -torch.where(b["mask"], t * logp, torch.zeros_like(logp)).sum(1)
        hp, sub = b["has_pol"], b["pw"] > 0
        tgt = t.argmax(1)
        hit = (logp.argmax(1) == tgt).float()
        groups = {"": hp, "_moves": hp & (tgt >= 6), "_switches": hp & (tgt < 6), "_tera": hp & (tgt >= 22),
                  "_ge2": hp & (b["n_legal"] >= 2)}
        for g, m in groups.items():
            acc[f"n{g}"] += m.sum().item()
            acc[f"hit{g}"] += hit[m].sum().item()
        acc["ce_all"] += ce[hp].sum().item()
        acc["ce"] += ce[sub].sum().item()
        acc["n_sub"] += sub.sum().item()
        ent = -(logp.exp() * logp.clamp_min(-50)).sum(1)
        acc["entropy"] += ent[sub].sum().item()
        v = v.float()
        z = b["z"]
        p = value_prob(v, value_loss)
        y = ((z + 1) / 2).clamp(0, 1)
        acc["vmse"] += ((torch.tanh(v) if value_loss == "bce" else v) - z).pow(2).sum().item()
        acc["vlog"] += F.binary_cross_entropy(p, y, reduction="sum").item()
        acc["vlog_result"] += F.binary_cross_entropy(p, ((b["result"] + 1) / 2).clamp(0, 1), reduction="sum").item()
        acc["nv"] += len(z)
        k = (p * 10).long().clamp(0, 9).cpu().numpy()
        np.add.at(bins, k, np.stack([np.ones(len(k)), p.cpu().numpy(), y.cpu().numpy()], 1))
    net.train()
    nz = lambda k: max(acc[k], 1)  # noqa: E731
    out = {"val_ce": acc["ce"] / nz("n_sub"), "val_ce_all": acc["ce_all"] / nz("n"),
           "val_entropy": acc["entropy"] / nz("n_sub"), "val_n": int(acc["nv"]), "val_n_policy": int(acc["n"])}
    for g in ("", "_moves", "_switches", "_tera", "_ge2"):
        if acc[f"n{g}"]:
            out[f"val_top1{g}"] = acc[f"hit{g}"] / acc[f"n{g}"]
    out |= {"val_value_mse": acc["vmse"] / nz("nv"), "val_value_logloss": acc["vlog"] / nz("nv"),
            "val_value_logloss_result": acc["vlog_result"] / nz("nv"),
            "val_ece": float(np.abs(bins[:, 1] - bins[:, 2]).sum() / max(bins[:, 0].sum(), 1)),
            "val_calibration": [[int(n), round(sp / n, 3), round(sy / n, 3)] for n, sp, sy in bins if n]}
    return out


# --------------------------------------------------------------------------------------------- training


def lr_at(step: int, total: int, args) -> float:
    if step < args.warmup:
        return args.lr * (0.01 + 0.99 * step / max(args.warmup, 1))
    prog = min(1.0, (step - args.warmup) / max(total - args.warmup, 1))
    f = args.lr_floor
    if args.schedule == "cosine":
        return args.lr * (f + (1 - f) * 0.5 * (1 + math.cos(math.pi * prog)))
    if args.schedule == "linear":
        return args.lr * (f + (1 - f) * (1 - prog))
    return args.lr


def param_groups(policy, args) -> list[dict]:
    """No weight decay on biases, norms and embeddings; embeddings at lr x --embedding-lr-mult."""
    emb = {id(m.weight) for m in policy.modules() if isinstance(m, nn.Embedding)}
    groups = {"decay": [], "no_decay": [], "emb": []}
    for _, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        groups["emb" if id(p) in emb else "decay" if p.ndim >= 2 else "no_decay"].append(p)
    return [{"params": groups["decay"], "weight_decay": args.weight_decay, "lr_mult": 1.0},
            {"params": groups["no_decay"], "weight_decay": 0.0, "lr_mult": 1.0},
            {"params": groups["emb"], "weight_decay": 0.0, "lr_mult": args.embedding_lr_mult}]


def code_commit() -> str:
    if os.environ.get("PORYGON_COMMIT"):
        return os.environ["PORYGON_COMMIT"]
    src = Path(__file__).resolve().parents[2]
    if (src / "COMMIT").exists():
        return (src / "COMMIT").read_text().strip()
    try:
        return subprocess.run(["git", "-C", str(src), "describe", "--always", "--dirty"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def source_table(meta: dict) -> dict[str, int]:
    """name -> id: the converter's static table, then meta.json's ids, then its per-run names ("run:<dir>")."""
    m = meta["datasets"][0] if "datasets" in meta else meta
    return DEFAULT_SOURCES | m.get("source_ids", {}) | m.get("sources", {})


def source_codes(name: str, table: dict[str, int]) -> set[int]:
    if name.isdigit():
        return {int(name)}
    if name in table:
        return {table[name]}
    if name in SOURCE_GROUPS:
        return {table[n] for n in SOURCE_GROUPS[name] if n in table}
    raise SystemExit(f"unknown source {name!r}: names {sorted(table)}, groups {sorted(SOURCE_GROUPS)}, or ids")


def make_filters(args, sources: dict[str, int]) -> Filters:
    mix = {}
    for item in filter(None, (args.mix or "").split(",")):
        k, v = item.split("=")
        mix |= {c: float(v) for c in source_codes(k, sources)}
    codes = set().union(*(source_codes(x, sources) for x in args.sources.split(","))) if args.sources else None
    return Filters(min_rating=args.min_rating, max_rating=args.max_rating, drop_unrated=args.drop_unrated,
                   sources=codes, mix=mix,
                   rating_weight_scale=args.rating_weight_scale, rating_ref=args.rating_ref,
                   skip_forced=args.skip_forced, min_legal=args.min_legal, label_smoothing=args.label_smoothing,
                   target_temperature=args.target_temperature, value_target=args.value_target,
                   value_mix=args.value_mix, visit_weight_power=args.visit_weight_power)


def amp_dtype(args, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or args.amp == "off":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else None


def log_line(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def jsonable(v):
    return str(v) if isinstance(v, Path) else [jsonable(x) for x in v] if isinstance(v, list) else v


def train(args) -> dict:
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out or args.resume or Path("runs") / time.strftime("bc-%Y%m%d-%H%M%S"))
    args.out = out
    out.mkdir(parents=True, exist_ok=True)
    meta = read_meta(args.data)
    filt = make_filters(args, source_table(meta))
    t0 = time.time()
    data = ShardSet(args.data, filt, args.val_frac, args.val_max, args.seed, args.load_workers, args.max_shards,
                    args.val_mask)
    print(f"{len(data.shards)} shards, {data.n_total:,} decisions: {data.n_train:,} train / {data.n_val:,} val "
          f"after filters ({time.time() - t0:.0f}s); obs_dim {meta['obs_dim']}, {meta['n_actions']} actions",
          flush=True)
    val = data.val_set()

    resume = Path(args.resume) if args.resume else None
    state = torch.load(resume / "last_state.pt", weights_only=False) if resume else None
    if state:
        model = load_model(resume / "last_model.zip", args.device)
    elif args.init:
        model = load_model(args.init, args.device)
    else:
        model = build_model(meta["obs_dim"], meta["n_actions"], args.policy, args.d_model, args.n_layers,
                            args.n_heads, json.loads(args.policy_kwargs) if args.policy_kwargs else None,
                            args.device)
    obs_space = model.policy.observation_space
    assert obs_space["observation"].shape[0] == meta["obs_dim"] and model.action_space.n == meta["n_actions"], \
        f"model spaces {obs_space} / {model.action_space} vs data {meta['obs_dim']} / {meta['n_actions']}"
    policy = model.policy
    policy.set_training_mode(True)
    net = BCNet(policy)
    fwd = torch.compile(net) if args.compile else net
    amp = amp_dtype(args, device)
    opt = torch.optim.AdamW(param_groups(policy, args), lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    n_params = sum(p.numel() for p in policy.parameters())

    # loader
    obs_bytes = data.n_train * meta["obs_dim"] * 2
    loader = args.loader
    if loader == "auto":
        gpu_mem = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
        loader = ("gpu" if obs_bytes < 0.45 * gpu_mem else
                  "ram" if obs_bytes < args.ram_gb * 2**30 else "stream")
    resident = None
    if loader in ("ram", "gpu"):
        t = time.time()
        resident = data.all_train()
        if args.max_rows:
            resident = take(resident, slice(0, args.max_rows))
        print(f"loaded {rows(resident):,} training decisions into {loader} ({time.time() - t:.0f}s)", flush=True)
        if loader == "gpu":
            resident = to_torch(resident, device, False)
    n_train = rows(resident) if resident is not None else data.n_train
    steps_per_epoch = max(1, n_train // args.batch_size)
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    if args.warmup < 0:
        args.warmup = min(1000, total // 20)

    config = {"args": {k: jsonable(v) for k, v in vars(args).items()}, "meta": meta, "commit": code_commit(),
              "n_params": n_params, "loader": loader, "n_train": n_train, "n_val": rows(val),
              "steps_per_epoch": steps_per_epoch, "total_steps": total, "host": socket.gethostname(),
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    log = out / "train_log.jsonl"
    print(f"model {type(policy).__name__} {n_params:,} params on {device} (amp {amp}, compile {args.compile}); "
          f"loader {loader}; {steps_per_epoch:,} steps/epoch x {args.epochs} = {total:,} steps", flush=True)

    step, epoch, best = 0, 0, float("inf")
    if state:
        opt.load_state_dict(state["optimizer"])
        step, epoch, best = state["step"], state["epoch"] + int(state["epoch_done"]), state["best"]
        print(f"resumed at step {step:,}, epoch {epoch}", flush=True)
    else:
        m = evaluate(net, val, device, args.eval_batch_size, amp, args.value_loss)
        log_line(log, {"type": "val", "step": 0, "epoch": 0} | m)
        print(f"init: val ce {m['val_ce']:.4f} top-1 {m.get('val_top1', 0):.1%} value mse {m['val_value_mse']:.4f}",
              flush=True)
    deadline = t0 + args.max_hours * 3600 if args.max_hours else None

    def checkpoint(epoch_done: bool) -> None:
        nonlocal best
        m = evaluate(net, val, device, args.eval_batch_size, amp, args.value_loss)
        rec = {"type": "val", "step": step, "epoch": epoch + int(epoch_done), "hours": (time.time() - t0) / 3600} | m
        log_line(log, rec)
        print(f"[step {step:,} epoch {rec['epoch']}] val ce {m['val_ce']:.4f} (all {m['val_ce_all']:.4f}) top-1 "
              f"{m.get('val_top1', 0):.1%} (moves {m.get('val_top1_moves', 0):.1%}, switches "
              f"{m.get('val_top1_switches', 0):.1%}, >=2 legal {m.get('val_top1_ge2', 0):.1%}) value mse "
              f"{m['val_value_mse']:.4f} logloss {m['val_value_logloss']:.4f} ece {m['val_ece']:.3f}", flush=True)
        if m["val_ce"] < best:
            best = m["val_ce"]
            model.save(out / "best_model")
        model.save(out / "last_model")
        torch.save({"optimizer": opt.state_dict(), "step": step, "epoch": epoch, "epoch_done": epoch_done,
                    "best": best}, out / "last_state.pt")

    stopped = False
    while step < total and not stopped:
        if loader == "gpu":
            prefetch = None
            perm = torch.randperm(n_train, device=device,
                                  generator=torch.Generator(device).manual_seed(args.seed * 1000 + epoch))
            batches = ({k: v[perm[s:s + args.batch_size]] for k, v in resident.items()}
                       for s in range(0, n_train - args.batch_size + 1, args.batch_size))
        else:
            gen = (ram_batches(resident, epoch, args.batch_size, args.seed) if loader == "ram"
                   else data.stream(epoch, args.batch_size, args.buffer_rows))
            prefetch = batches = Prefetch(gen, device)
        sums, n_log, t_log, wait0 = torch.zeros(3, device=device), 0, time.perf_counter(), 0.0
        completed, n_batches = True, 0
        for b in batches:
            n_batches += 1
            lr = lr_at(step, total, args)
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]
            with torch.autocast(device.type, dtype=amp or torch.bfloat16, enabled=amp is not None):
                logits, values = fwd(b["obs"].float(), b["mask"])
            pol, vl = losses(logits, values, b, filt.label_smoothing, args.value_loss)
            loss = pol + args.vf_coef * vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            opt.step()
            sums += torch.stack([pol.detach(), vl.detach(), gn.detach()])
            step += 1
            n_log += 1
            if step % args.log_every == 0:
                s = (sums / n_log).tolist()
                dt = time.perf_counter() - t_log
                wait = prefetch.wait - wait0 if prefetch else 0.0
                rec = {"type": "train", "step": step, "epoch": epoch, "lr": lr, "policy_ce": s[0],
                       "value_loss": s[1], "grad_norm": s[2], "samples_per_s": n_log * args.batch_size / dt,
                       "data_wait": wait / dt}
                log_line(log, rec)
                print(f"step {step:,} lr {lr:.2e} ce {s[0]:.4f} v {s[1]:.4f} |g| {s[2]:.2f} "
                      f"{rec['samples_per_s']:,.0f} samples/s (data wait {rec['data_wait']:.0%})", flush=True)
                sums.zero_()
                n_log, t_log = 0, time.perf_counter()
                wait0 = prefetch.wait if prefetch else 0.0
            if args.eval_every and step % args.eval_every == 0:
                checkpoint(False)
            if step >= total:
                completed = False
                break
            if deadline and time.time() > deadline:
                completed, stopped = False, True
                break
        if prefetch:
            prefetch.close()
        if not n_batches:
            print("an epoch produced no batch (dataset smaller than --batch-size?)", flush=True)
            break
        checkpoint(completed)
        epoch += int(completed)
    model.save(out / "final_model")
    if stopped:
        print(f"stopped by --max-hours at step {step:,}: --resume {out} continues", flush=True)
    print(f"done: {step:,} steps in {(time.time() - t0) / 3600:.2f} h; best val ce {best:.4f}; saved {out}/",
          flush=True)
    return {"best_val_ce": best, "steps": step}


def eval_cmd(args) -> None:
    device = torch.device(args.device)
    meta = read_meta(args.data)
    filt = make_filters(args, source_table(meta))
    data = ShardSet(args.data, filt, 1.0 if args.all else args.val_frac, args.val_max, args.seed,
                    args.load_workers, args.max_shards, None if args.all else args.val_mask)
    model = load_model(args.model, args.device)
    m = evaluate(BCNet(model.policy), data.val_set(), device, args.eval_batch_size, amp_dtype(args, device),
                 args.value_loss)
    print(json.dumps(m))


# ------------------------------------------------------------------------------------------- synthetic


def synth_shard(n: int, n_actions: int = 26, n_games: int = 20, seed: int = 0) -> dict[str, np.ndarray]:
    """Random decisions in the shard format with in-range categorical ids and a learnable policy
    (the target action is a fixed function of the observation)."""
    from porygon import encoder as E
    rng = np.random.default_rng(seed)
    obs = (rng.standard_normal((n, E.OBS_DIM)) * 0.3).astype(np.float32)
    mons = obs[:, E.FIELD_DIM:].reshape(n, 12, E.MON_DIM)
    kinds = ["species", "items", "abilities"] + ["moves"] * E.N_MOVES
    for j, kind in enumerate(kinds):
        mons[..., j] = rng.integers(0, E.VOCAB_SIZES[kind], (n, 12))
    mons[..., E.N_CAT:E.N_CAT + E.MOVEPOOL] = rng.integers(0, E.VOCAB_SIZES["moves"], (n, 12, E.MOVEPOOL))
    flag = E.N_CAT + E.MOVEPOOL + E.ACTIVE_FLAG
    mons[..., flag] = 0
    mons[:, 0, flag] = 1  # own mon 0 is active
    mask = np.zeros((n, n_actions), np.int8)
    mask[:, 1:10] = rng.random((n, 9)) < 0.6
    if n_actions > 10:
        mask[:, 22:26] = mask[:, 6:10] * (rng.random((n, 1)) < 0.5)
    forced = rng.random(n) < 0.1
    mask[forced, 6:] = 0
    mask[mask.sum(1) == 0, 6] = 1
    score = obs[:, :n_actions] * 10 + mask * 0  # learnable: argmax over legal actions of a slice of the field
    action = np.where(mask > 0, score, -np.inf).argmax(1).astype(np.int16)
    target = np.zeros((n, n_actions), np.float16)
    target[np.arange(n), action] = 1
    game = np.sort(rng.integers(0, n_games, n)).astype(np.int32)
    z = np.where(rng.random(n_games) < 0.5, 1.0, -1.0)[game].astype(np.float32)
    rating = rng.integers(1000, 2000, n_games)[game].astype(np.int16)
    return {"obs": obs.astype(np.float16), "mask": mask, "action": action, "target": target, "z": z,
            "rating": rating, "opp_rating": rating, "game": game, "turn": rng.integers(1, 40, n).astype(np.int16),
            "source": np.where(rng.random(n) < 0.3, 10, 1).astype(np.uint8), "forced": forced.astype(np.uint8),
            "n_legal": mask.sum(1).astype(np.uint8),
            "root_value": np.clip((z + 1) / 2 * 0.6 + 0.2 + rng.normal(0, 0.05, n), 0, 1).astype(np.float32),
            "visits": rng.integers(100, 5000, n).astype(np.int32)}


def synth_cmd(args) -> None:
    from porygon import encoder as E
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per = -(-args.n // args.shards)
    for i in range(args.shards):
        d = synth_shard(per, args.n_actions, max(2, args.games // args.shards), args.seed + i)
        d["game"] = d["game"] + i * args.games
        np.savez_compressed(out / f"shard-{i:04d}.npz", **d)
    fmt = "gen9randombattle" if args.n_actions == 26 else "gen4randombattle"
    (out / "meta.json").write_text(json.dumps({
        "encoder": "porygon.encoder:embed_battle", "obs_dim": E.OBS_DIM, "n_actions": args.n_actions,
        "format": fmt, "sources": {"showdown-replays": 1, "searchlabels": 10}, "source_ids": DEFAULT_SOURCES,
        "counts": {"decisions": per * args.shards}, "synthetic": True}))
    print(f"wrote {args.shards} synthetic shard(s), {per * args.shards} decisions, to {out}")


# ------------------------------------------------------------------------------------------ benchmarks


def summarize(dirs: list[Path], ours: str) -> str:
    """Win rate of the players matching `ours` over duplicate-battle logs (battles.jsonl): per game
    (binomial 95%) and by pair means (95% from their SD; pairs never span directories)."""
    pat = re.compile(ours)
    games, means = [], []
    for d in map(Path, dirs):
        by = defaultdict(dict)
        for line in (d / "battles.jsonl").open():
            b = json.loads(line)
            if not (pat.search(b["p1"]) or pat.search(b["p2"])):
                continue
            s = 0.5 if not b["winner"] else float(bool(pat.search(b["winner"])))
            games.append(s)
            by[b["pair"]][b["game"]] = s
        means += [sum(p.values()) / 2 for p in by.values() if len(p) == 2]
    if not games:
        return f"no battles with a player matching {ours!r} in {dirs}"
    w = sum(games) / len(games)
    line = f"{w:.1%} ± {1.96 * math.sqrt(w * (1 - w) / len(games)):.1%} per game (n={len(games)})"
    if len(means) > 1:
        m = sum(means) / len(means)
        sd = math.sqrt(sum((x - m) ** 2 for x in means) / (len(means) - 1))
        line += f"  |  duplicate pairs: {m:.1%} ± {1.96 * sd / math.sqrt(len(means)):.1%} (pairs={len(means)})"
    return f"{ours}: {line}"


# ------------------------------------------------------------------------------------------------- cli


def add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("data", nargs="+", type=Path, help="dataset directories (meta.json + shard-*.npz) or shards")
    p.add_argument("--val-frac", type=float, default=0.02, help="share of games in the validation bucket")
    p.add_argument("--val-max", type=int, default=200_000, help="validation decisions kept")
    p.add_argument("--val-mask", type=Path, help="DIR of <shard stem>.npy bool row masks: the held-out set "
                   "(replaces the hash split; --val-frac is then unused)")
    p.add_argument("--min-rating", type=int)
    p.add_argument("--max-rating", type=int)
    p.add_argument("--drop-unrated", action="store_true", help="drop decisions with unknown rating (-1)")
    p.add_argument("--sources", help="keep only these sources: groups human / search, meta.json names, or ids")
    p.add_argument("--mix", help="per-source loss weights, e.g. human=1,search=0.5 (0 drops a source)")
    p.add_argument("--rating-weight-scale", type=float, default=0.0,
                   help="> 0: weight decisions by clip(exp((rating - ref) / scale), 0.1, 10)")
    p.add_argument("--rating-ref", type=float, default=1500)
    p.add_argument("--skip-forced", action="store_true", help="no policy loss on forced decisions")
    p.add_argument("--min-legal", type=int, default=2, help="policy loss only with at least this many legal actions")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--target-temperature", type=float, default=1.0, help="target ~ target^(1/T) (soft targets)")
    p.add_argument("--value-loss", choices=["mse", "bce"], default="mse")
    p.add_argument("--value-target", choices=["result", "search", "mix"], default="result",
                   help="value-head target: game result z, search root_value (2 rv - 1; z where absent), or a mix")
    p.add_argument("--value-mix", type=float, default=0.5, help="mix: weight of the search root value")
    p.add_argument("--visit-weight-power", type=float, default=0.0,
                   help="> 0: weight decisions by clip((visits / shard mean)^p, 0.1, 10) where `visits` exists")
    p.add_argument("--max-shards", type=int)
    p.add_argument("--load-workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", choices=["auto", "off"], default="auto", help="bf16 autocast on GPUs that have it")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="supervised training on shards")
    add_data_args(t)
    t.add_argument("--out", type=Path, help="run directory (default: the --resume dir, else runs/bc-<date>)")
    t.add_argument("--init", help="initialise from this PPO checkpoint (.zip) instead of from scratch")
    t.add_argument("--resume", help="continue the run in this directory (last_model.zip + last_state.pt)")
    t.add_argument("--policy", default="porygon.model:MaskedPolicy", help="policy class, module:Class")
    t.add_argument("--policy-kwargs", help="JSON merged into policy_kwargs (e.g. net_arch)")
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--n-layers", type=int, default=3)
    t.add_argument("--n-heads", type=int, default=8)
    t.add_argument("--epochs", type=float, default=4)
    t.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0: --epochs)")
    t.add_argument("--max-hours", type=float, help="save and stop after this long (then --resume)")
    t.add_argument("--max-rows", type=int, help="ram/gpu loader: train on the first N decisions only")
    t.add_argument("--batch-size", type=int, default=2048)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--lr-floor", type=float, default=0.1, help="final lr as a fraction of --lr")
    t.add_argument("--schedule", choices=["cosine", "linear", "constant"], default="cosine")
    t.add_argument("--warmup", type=int, default=-1, help="linear warmup steps (default min(1000, 5%%))")
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--embedding-lr-mult", type=float, default=1.0)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--vf-coef", type=float, default=0.5)
    t.add_argument("--loader", choices=["auto", "stream", "ram", "gpu"], default="auto")
    t.add_argument("--ram-gb", type=float, default=24, help="auto: largest dataset (obs GB) held in RAM")
    t.add_argument("--buffer-rows", type=int, default=1_000_000, help="stream: shuffle buffer (decisions)")
    t.add_argument("--compile", action="store_true", help="torch.compile the network")
    t.add_argument("--log-every", type=int, default=200)
    t.add_argument("--eval-every", type=int, default=0, help="also evaluate + checkpoint every N steps")
    e = sub.add_parser("eval", help="validation metrics of a checkpoint")
    e.add_argument("model")
    add_data_args(e)
    e.add_argument("--all", action="store_true", help="every decision, not only the validation games")
    s = sub.add_parser("synth", help="write synthetic shards")
    s.add_argument("out")
    s.add_argument("--n", type=int, default=4000)
    s.add_argument("--n-actions", type=int, default=26)
    s.add_argument("--games", type=int, default=40)
    s.add_argument("--shards", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("summarize", help="duplicate-pair win rate from battles.jsonl")
    m.add_argument("dirs", nargs="+", type=Path)
    m.add_argument("--ours", required=True, help="regex for our player's name")
    args = p.parse_args(argv)
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_cmd(args)
    elif args.cmd == "synth":
        synth_cmd(args)
    else:
        print(summarize(args.dirs, args.ours))


if __name__ == "__main__":
    sys.exit(main())
