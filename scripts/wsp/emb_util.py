"""Helpers for the emb (categorical-embedding) analysis: data path, extracted checkpoints, token counts, game bootstrap."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from porygon.encoder import INDEX, VOCAB_SIZES
from wsp_common import split_obs

WSP = Path("/vol/bitbucket/fr1226/porygon-ws-wsp")
OUT = WSP / "out/emb"
TABLES = ("species", "items", "abilities", "moves")
NAMES = {k: {i: n for n, i in v.items()} | {0: "<pad>", 1: "<unk>"} for k, v in INDEX.items()}
ADAM_PER_ENV_STEP = 7 * 39 / 39936  # 7 epochs x 39 minibatches of 1024 per 39,936-step rollout


def data_dir() -> Path:
    loc = Path("/tmp/wsp-674320/data")
    return loc if (loc / "decisions.npz").exists() else WSP / "data"


def load_data():
    z = np.load(data_dir() / "decisions.npz")
    return {k: z[k] for k in z.files}


def ck(tag: str) -> dict:
    return torch.load(OUT / "ckpt" / f"{tag}.pt", map_location="cpu", weights_only=False)


def emb(c: dict, table: str) -> np.ndarray:
    return c["params"][f"features_extractor.{table}.weight"].numpy()


def token_counts(d: dict) -> dict:
    """Occurrences per logged decision of each id, by role. own = our 6 mons, opp = opponent's 6 slots."""
    s = split_obs(d["obs"])
    cats, pool = s["cats"], s["pool"]
    n = len(cats)
    out = {}
    for t, col in (("species", 0), ("items", 1), ("abilities", 2)):
        out[t] = {"own": np.bincount(cats[:, :6, col].ravel(), minlength=VOCAB_SIZES[t]) / n,
                  "opp": np.bincount(cats[:, 6:, col].ravel(), minlength=VOCAB_SIZES[t]) / n}
    V = VOCAB_SIZES["moves"]
    out["moves"] = {"own": np.bincount(cats[:, :6, 3:].ravel(), minlength=V) / n,
                    "opp": np.bincount(cats[:, 6:, 3:].ravel(), minlength=V) / n,
                    "pool_own": np.bincount(pool[:, :6].ravel(), minlength=V) / n,
                    "pool_opp": np.bincount(pool[:, 6:].ravel(), minlength=V) / n}
    for t in out:
        out[t]["all"] = sum(v for k, v in out[t].items())
        for r in out[t].values():
            r[0] = 0.0  # padding never trains
    return out


def game_boot(per_row: np.ndarray, game: np.ndarray, n_boot: int = 1000, seed: int = 0):
    """Mean of a per-decision statistic with a 95% interval from resampling games."""
    ug, inv = np.unique(game, return_inverse=True)
    s = np.bincount(inv, weights=per_row, minlength=len(ug))
    c = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(m, lo, hi, p=3):
    return f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"
