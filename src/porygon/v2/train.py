"""Self-play PPO for model v2 (custom loop; SB3 cannot carry the dict observation or the aux targets).

    scripts/start_servers.sh 8
    python -m porygon.v2.train --format gen9randombattle --ports 8000-8007 --n-battles 48 --pool-battles 24 \
        --run-dir runs/v2-gen9 --total-steps 300_000_000

Data: battle worker processes (`porygon.v2.workers`): self-play battles (both sides are samples) and pool battles
(the learner vs a frozen v2 snapshot from --pool-dir or a baseline bot). Collection runs in a background thread with
a copy of the network refreshed before every rollout; by default it keeps running during the PPO update (data at most
one update old; the stored log-probs keep the importance ratio exact), `--no-overlap` pauses it instead.
`--actor-stream` (off by default): the collector thread runs on a high-priority CUDA stream and acts from two
alternating weight copies (`porygon.v2.actor_buffers`; refreshed while it acts), so under overlap its forwards do not
queue behind the update's kernels on the default stream; on CPU only the double buffer.

`--collector fastenv` collects from fastenv instead (`porygon.fastenv_v2`: Showdown's simulator in-process in Node
workers, the v2 encoder in JS at parity with `porygon.v2.encoder`; no servers or poke-env workers; `--ports` is then
only needed for the in-training evaluation). With `--no-overlap` it is strictly on-policy (end-of-rollout bootstrap,
no carry-over of transitions acted by the previous weights); with overlap use `--decoupled`. `--stacked-opp` (off by
default): the pool opponents acting in one fastenv collector step share one forward when they have the same
architecture (`porygon.v2.stacked`: weights stacked, rows folded per net; logits equal the per-net forward's up to
float rounding), instead of one launch-bound forward per opponent.

Loss (per minibatch): clipped PPO surrogate (advantages normalised per minibatch) − ent_coef · entropy
+ zero_avoid_coef · (−mean log π over legal actions: Jaxcalibur's KL to uniform) + vf_coef · BCE(win prob,
clip((λ-return + 1) / 2, 0, 1)) (no value clipping) + aux_coef · Σ aux heads (CE item / ability / tera, BCE per
candidate move, BCE unseen species / 6, soft CE of the opponent's policy mapped to our frame).

Options for runs that start from a supervised net (gen9 G9, 2026-11-09): `--kl-anchor CKPT` adds
β(t) · KL(π_anchor ‖ π) over legal actions (kickstarting-style forward KL to a frozen net, e.g. the human-BC init;
β(t) = --kl-anchor-coef · max(0, 1 − steps / --kl-anchor-decay-steps), anchor log-probs from one no-grad pass per
rollout); `--critic-warmup-steps N`: until N env steps only the value and aux heads train (policy and trunk frozen,
so the init's policy acts unchanged while the critic fits its games); `--frozen CKPT...` + `--p-frozen p`: pool
battles play a frozen net (copied to pool_dir/frozen/) with probability p, `--frozen-best` adds each new best_model
there (anti-cycling); `--guard-entropy-rel R`: entropy guard at R × the mean entropy of this process's first 5
updates; collapse guards `--guard-entropy-min` / `--guard-ptop-max` (entropy too low, or too many choices with a top
probability ≥ 0.9 — `ptop` in the log, over decisions with ≥ 2 legal actions — for 5 updates).

Distillation (2026-11-09): `--distill GLOB` loads v2 supervised shards (`porygon.v2.pretrain`'s format; rows without
target mass on a legal action dropped) into CPU memory once, split by game with `pretrain.is_val` (--distill-val-frac;
the held-out games are only monitored). With --distill-coef C > 0 every PPO minibatch adds one pass of C · CE(target
renormalised over legal actions, π) on --distill-batch train rows (id dropout on; not during the critic warm-up),
logged as loss/distill. Every --grad-ratio-every updates, at the update's starting weights, CE and top-1 agreement on
≤ 4096 fixed held-out / train rows are logged (diag/distill_{val,train}_{ce,top1}): C = 0 monitors without changing
training. `--distill-reload-every U`: every U updates re-glob and reload if a shard was added, removed or rewritten.

lr: warmup–stable–decay on absolute env steps: linear warmup over --warmup-steps, constant --lr, and (only with
--cooldown-start) cosine down to --final-lr-frac · lr at --total-steps. Resuming keeps the schedule.
`--lr-schedule power --lr-horizon H` (Wang's anneal; default `wsd` = the above, unchanged): after the warm-up the lr
is multiplied by (8x + 1)^(-1.5), x = (steps - warmup) / (H - warmup) clamped to [0, 1] (lr / 27 from H on, flat);
H is independent of --total-steps. The optional cosine cooldown multiplies on top (a cooled branch of a power run).

Outputs in --run-dir: train.log (one line per update), tb/ (TensorBoard), checkpoints/ckpt_<steps>.pt every
--checkpoint-every steps, final_model.pt (on exit, SIGINT / SIGTERM included), best_model.pt (best eval score:
vs --anchor if given, else vs SimpleHeuristicsPlayer), pool snapshots every --snapshot-every steps.

`--eval-async`: the evaluation games run in a CPU subprocess (`porygon.v2.train_eval`) on a snapshot of the live net
(eval/snap_<steps>.pt) while training continues, one at a time (an eval point is skipped while the previous one
runs); its result is handled at the first update after it finished, like a synchronous eval of the snapshot (log
line, TensorBoard, best_model.pt and the guard at the snapshot's steps). Subprocess output: eval/eval.log.

Pool health, every update: the realised opponent mix of the finished games (mix/*, `mix` in the log), the pool-net
loads (pool/load_*, `plf` = failures), and a guard with a "pool:" reason: a frozen net fails to load, > 5% of >= 20
snapshot loads fail in one update, or pool battles are configured but none finished for --pool-alarm-updates updates
in a row (also POOL_ALARM). A frozen/best-*.pt rotated out by --frozen-best mid-battle is replaced by the newest best.
"""

import argparse
import asyncio
import copy
import glob
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from porygon.env import local_server
from porygon.v2.model import (
    COL, DEFAULT_CFG, Net, batch_to_tensors, load, model_keys, net_from_checkpoint, save,
)
from porygon.v2.spec import VOCAB_SIZES, n_actions
from porygon.v2.train_eval import EvalJob, copy_checkpoint, launch, stop_process, tail
from porygon.v2.workers import (
    ACT, DONE, LOGP, NEXT_V, OBS, OPP_C, OPP_T, REW, SEED, VAL, BattleWorkers, Collector, Packer, WorkerSpec, gen_of,
)


def parse_ports(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = map(int, spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(p) for p in spec.split(",")]


POWER_K, POWER_P = 8.0, 1.5  # --lr-schedule power: (K x + 1)^(-P) (Wang 2024: peak / 27 at x = 1)


def power_decay(step: int, warmup: int, horizon: int) -> float:
    """--lr-schedule power's factor: 1 until the end of the warm-up, (8x + 1)^(-1.5) with x = (step - warmup) /
    (horizon - warmup) clamped to [0, 1], so flat at 1/27 from `horizon` on."""
    x = min(1.0, max(0.0, (step - warmup) / max(1, horizon - warmup)))
    return (POWER_K * x + 1.0) ** -POWER_P


def wsd_lr(step: int, lr: float, warmup: int, cooldown_start: int | None, total: int, final_frac: float,
           schedule: str = "wsd", horizon: int | None = None) -> float:
    f = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
    if schedule == "power":  # multiplies the warm-up and (below) the cosine cooldown; "wsd" skips it (unchanged)
        f *= power_decay(step, warmup, horizon)
    if cooldown_start is not None and step > cooldown_start:
        x = min(1.0, (step - cooldown_start) / max(1, total - cooldown_start))
        f *= final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * x))
    return lr * f


# ---------------------------------------------------------------- rollout

def build_rollout(segments: dict, packer: Packer, gamma: float, lam: float) -> dict[str, np.ndarray]:
    """Segments {(worker, side): [transition, ...]} -> arrays incl. advantages / λ-returns (truncated GAE: each
    transition carries its own bootstrap value, so a segment's last transition needs nothing from the next one)."""
    rows = [t for seg in segments.values() for t in seg]
    n = len(rows)
    data = packer.unpack([t[OBS] for t in rows])
    data["action"] = np.array([t[ACT] for t in rows], np.int64)
    data["logp"] = np.array([t[LOGP] for t in rows], np.float32)
    data["value"] = np.array([t[VAL] for t in rows], np.float32)
    data["drop_seed"] = np.array([t[SEED] for t in rows], np.int64)
    reward = np.array([t[REW] for t in rows], np.float32)
    next_v = np.array([t[NEXT_V] for t in rows], np.float32)
    done = np.array([t[DONE] for t in rows], bool)
    opp_t = data["aux_opp_target"]  # spec 2.1: zeros from the encoder (= masked); self-play fills it here
    for i, t in enumerate(rows):
        if t[OPP_T] is not None:
            opp_t[i] = t[OPP_T]
    data["opp_cls"] = np.array([t[OPP_C] for t in rows], np.int64)
    last = np.zeros(n, bool)
    i = 0
    for seg in segments.values():
        i += len(seg)
        last[i - 1] = True
    delta = reward + gamma * next_v * ~done - data["value"]
    carry = gamma * lam * (~done & ~last)
    adv = np.zeros(n, np.float32)
    g = 0.0
    for t in range(n - 1, -1, -1):
        g = delta[t] + carry[t] * g
        adv[t] = g
    data["adv"], data["ret"] = adv, adv + data["value"]
    data["reward"], data["done"] = reward, done
    return data


# ---------------------------------------------------------------- losses

class Meter:
    def __init__(self):
        self.s: dict[str, float] = {}
        self.n: dict[str, float] = {}

    def add(self, k: str, v, n=1.0) -> None:
        v = float(v)
        self.s[k] = self.s.get(k, 0.0) + v * n
        self.n[k] = self.n.get(k, 0.0) + n

    def add_ratio(self, k: str, hits, total) -> None:
        total = float(total)
        if total > 0:
            self.s[k] = self.s.get(k, 0.0) + float(hits)
            self.n[k] = self.n.get(k, 0.0) + total

    def means(self) -> dict[str, float]:
        return {k: self.s[k] / self.n[k] for k in self.s if self.n[k] > 0}


def _ce(logits: torch.Tensor, target: torch.Tensor):
    """CE over [..., V] logits with targets [...] (-1 = none) -> (loss, correct mask over the valid entries, valid)."""
    l, t = logits.flatten(0, -2), target.flatten().long()
    ok = t >= 0
    if not ok.any():
        return logits.sum() * 0.0, None, ok
    return F.cross_entropy(l[ok], t[ok]), l[ok].argmax(-1) == t[ok], ok


def aux_losses(out: dict, mb: dict, m: Meter | None) -> torch.Tensor:
    ax = out["aux"]
    mon = mb["mon_num"][:, 6:].float()
    losses = []

    loss, hit, ok = _ce(ax["item"], mb["aux_item"])
    losses.append(loss)
    if m is not None:
        m.add("aux/item_loss", loss.item())
        if hit is not None:
            hidden = (mon[..., COL["item_known"]] == 0).flatten()[ok]
            m.add_ratio("aux/item_acc_hidden", (hit & hidden).sum().item(), hidden.sum().item())

    loss, hit, ok = _ce(ax["ability"], mb["aux_ability"])
    losses.append(loss)
    if m is not None:
        m.add("aux/ability_loss", loss.item())
        if hit is not None:
            hidden = (mon[..., COL["ability_known"]] == 0).flatten()[ok]
            m.add_ratio("aux/ability_acc", hit.sum().item(), hit.numel())
            m.add_ratio("aux/ability_acc_hidden", (hit & hidden).sum().item(), hidden.sum().item())

    loss, hit, ok = _ce(ax["tera"], mb["aux_tera"])
    losses.append(loss)
    if m is not None and hit is not None:
        m.add("aux/tera_loss", loss.item())
        hidden = (mon[..., COL["terastallized"]] == 0).flatten()[ok]
        m.add_ratio("aux/tera_acc_hidden", (hit & hidden).sum().item(), hidden.sum().item())

    tgt = mb["aux_move_in_set"].float()
    ok = tgt >= 0
    if ok.any():
        loss = F.binary_cross_entropy_with_logits(ax["moves"][ok], tgt[ok])
        losses.append(loss)
        if m is not None:
            m.add("aux/moves_loss", loss.item())
            cand = ok & (mb["move_num"][:, 6:, :, 0] == 0)
            hit = ((ax["moves"] > 0).float() == tgt) & cand
            m.add_ratio("aux/moves_acc_cand", hit.sum().item(), cand.sum().item())

    ids = mb["aux_unseen"].long()
    multi = torch.zeros(ids.shape[0], VOCAB_SIZES["species"] + 1, device=ids.device)
    multi.scatter_(1, torch.where(ids >= 0, ids, VOCAB_SIZES["species"]), 1.0)
    multi = multi[:, :-1]
    loss = F.binary_cross_entropy_with_logits(ax["unseen"], multi, reduction="sum") / (6 * ids.shape[0])
    losses.append(loss)
    if m is not None:
        m.add("aux/unseen_loss", loss.item())
        n_true = multi.sum(1)
        top = ax["unseen"].topk(6, -1).indices
        hits = multi.gather(1, top).sum(1)
        has = n_true > 0
        if has.any():
            m.add_ratio("aux/unseen_recall6", hits[has].sum().item(), n_true[has].sum().item())

    valid = mb["aux_opp_target"].sum(-1) > 0  # zero vector = no target
    if valid.any():
        tgt = mb["aux_opp_target"][valid].float()
        raw = ax["opp_action_raw"][valid]
        mask = ax["opp_action_mask"][valid] | (tgt > 0)
        lsm = torch.log_softmax(raw.masked_fill(~mask, -1e9), -1)
        loss = -(tgt * lsm).sum(-1).mean()
        losses.append(loss)
        if m is not None:
            m.add("aux/opp_action_loss", loss.item())
    if m is not None:
        c = mb["opp_cls"]
        has = c >= 0
        if has.any():
            pred = ax["opp_action"][has].argmax(-1)
            m.add_ratio("aux/opp_action_acc", (pred == c[has]).sum().item(), has.sum().item())
    return torch.stack(losses).sum()


def ppo_losses(out: dict, mb: dict, args, normalized: bool = False) -> dict[str, torch.Tensor]:
    mask = mb["action_mask"] > 0
    logp_all = torch.log_softmax(out["logits"].float(), -1)
    logp_safe = torch.where(mask, logp_all, torch.zeros_like(logp_all))
    probs = torch.where(mask, logp_all.exp(), torch.zeros_like(logp_all))
    logp = logp_all.gather(1, mb["action"].long()[:, None]).squeeze(1)
    # decoupled PPO (Hilton et al. 2022): clip against the proximal policy (the weights at update start, "logp_prox")
    # and weight each sample by the truncated behaviour ratio pi_prox / pi_behaviour, so off-policy (overlapped) data is
    # not clipped before the first step; without it, the behaviour log-probs are the proximal policy
    decoupled = "logp_prox" in mb
    log_ratio = logp - (mb["logp_prox"] if decoupled else mb["logp"])
    ratio = log_ratio.exp()
    adv = mb["adv"]
    if not normalized:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) if adv.numel() > 1 else adv
    if args.adv_clip > 0:  # heavy-tailed per-sample advantages: cap the normalised value
        adv = adv.clamp(-args.adv_clip, args.adv_clip)
    surr = torch.min(ratio * adv, ratio.clamp(1 - args.clip_range, 1 + args.clip_range) * adv)
    if decoupled:
        w = torch.exp(mb["logp_prox"] - mb["logp"]).clamp(max=args.decoupled_rho)
        pg = -(w * surr).mean()
    else:
        pg = -surr.mean()
    ent = -(probs * logp_safe).sum(-1).mean()
    za = -(logp_safe.sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    target = ((mb["ret"] + 1) / 2).clamp(0, 1)
    vf = F.binary_cross_entropy_with_logits(out["win_logit"].float(), target)
    if "logp_anchor" in mb:  # forward KL(anchor || pi) over legal actions (anchor log-probs precomputed, masked)
        la = mb["logp_anchor"]
        kla = (torch.where(mask, la.exp() * (la - logp_all), torch.zeros_like(la))).sum(-1).mean()
    else:
        kla = torch.zeros((), device=logp_all.device)
    with torch.no_grad():
        kl = ((ratio - 1) - log_ratio).mean()
        clip = ((ratio - 1).abs() > args.clip_range).float().mean()
        r_mean, r_std = ratio.mean(), ratio.std() if ratio.numel() > 1 else torch.zeros_like(ratio.mean())
        # sharpness: share of real choices (>= 2 legal actions) whose top action has probability >= 0.9
        multi = mask.sum(-1) >= 2
        ptop = ((probs.max(-1).values >= 0.9) & multi).float().sum() / multi.float().sum().clamp(min=1.0)
    return {"pg": pg, "ent": ent, "za": za, "vf": vf, "kl": kl, "clip": clip, "ratio_mean": r_mean, "ratio_std": r_std,
            "kla": kla, "ptop": ptop}


def _trunk_grad(loss: torch.Tensor, params: list) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(grads, params)])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))


def gradient_diagnostics(net: Net, batches: list[dict], args, amp: bool, m: Meter) -> None:
    """At the update's starting parameters, on two disjoint minibatches: trunk gradient norms of the weighted value
    loss and of the policy loss (MissingNo.Zero's ratio alarm), and the cosine between the two minibatches' trunk
    gradients for each loss (gradient noise: ~0 means the minibatch gradient is mostly noise)."""
    trunk = net.trunk_parameters()
    g = []
    for obs, dmg, mb in batches:
        with torch.autocast(obs["mon_num"].device.type, dtype=torch.bfloat16, enabled=amp):
            out = net(obs, dmg=dmg, aux=False)
        L = ppo_losses(out, mb, args)
        g.append((_trunk_grad(L["pg"], trunk), _trunk_grad(args.vf_coef * L["vf"], trunk)))
    gp, gv = g[0][0].norm().item(), g[0][1].norm().item()
    m.add("diag/trunk_grad_policy", gp)
    m.add("diag/trunk_grad_value", gv)
    m.add("diag/trunk_grad_ratio", gv / max(gp, 1e-12))
    if len(g) > 1:
        m.add("diag/grad_cos_policy", _cos(g[0][0], g[1][0]))
        m.add("diag/grad_cos_value", _cos(g[0][1], g[1][1]))


def rollout_damage(net: Net, tensors: dict, keys: list[str], chunk: int = 4096) -> dict | None:
    """Damage features of the whole rollout in large chunks, once per update (they depend on the observation only;
    float32 on the device, ~28 KB per transition)."""
    if net.damage is None:
        return None
    parts = []
    n = len(tensors["action"])
    for s in range(0, n, chunk):
        obs = {k: tensors[k][s:s + chunk] for k in keys}
        parts.append(net.damage_features(obs))
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def anchor_logp(anchor: Net, tensors: dict, keys: list[str], dmg_all: dict | None, device, chunk: int = 4096):
    """[N, A] log-probs of the frozen anchor net on the rollout (eval mode, every id visible, no grad)."""
    parts = []
    n = len(tensors["action"])
    with torch.no_grad():
        for c0 in range(0, n, chunk):
            sl = slice(c0, c0 + chunk)
            o = {k: (tensors[k][sl].long() if not tensors[k].is_floating_point() and tensors[k].dtype != torch.bool
                     else tensors[k][sl]) for k in keys if k != "drop_seed"}
            out = anchor(o, dmg=None if dmg_all is None else {k: v[sl] for k, v in dmg_all.items()}, aux=False)
            parts.append(torch.log_softmax(out["logits"].float(), -1))
    return torch.cat(parts)


# ---------------------------------------------------------------- distillation (--distill)

class DistillSet:
    """Labelled positions in CPU memory from v2 supervised shards (the format `porygon.v2.pretrain.load_shard` reads):
    the model fields, `target` and `game` of every row with target mass on a legal action, split by game with
    `pretrain.is_val` (train rows are sampled for the loss, held-out rows only monitored). `rng`: the generator
    `sample` draws from (never the global ones)."""

    MONITOR = 4096  # fixed rows per split for diag/distill_*

    def __init__(self, pattern: str, gen: int, val_frac: float, rng: np.random.Generator | None = None):
        from porygon.v2.pretrain import is_val, load_shard  # not at module level: pretrain imports this module
        self.pattern, self.gen, self.val_frac = pattern, gen, val_frac
        self.rng = np.random.default_rng() if rng is None else rng
        self.paths = sorted(glob.glob(pattern))
        assert self.paths, f"--distill {pattern}: no shards match"
        self.signature = self._signature(self.paths)
        for meta in {Path(p).with_name("meta.json") for p in self.paths}:  # the data stream's per-directory meta
            g = json.loads(meta.read_text()).get("gen") if meta.exists() else None
            assert g is None or int(g) == gen, f"{meta}: gen {g} shards, trainer gen {gen}"
        keys, parts = model_keys(gen) + ["target", "game"], []
        for p in self.paths:
            d = load_shard(p, gen)
            assert d["action_mask"].shape[1:] == d["target"].shape[1:] == (n_actions(gen),), \
                f"{p}: action_mask {d['action_mask'].shape}, target {d['target'].shape}: not gen{gen} shards"
            keep = (d["target"] * (d["action_mask"] > 0)).sum(-1) > 0  # all-zero (or illegal-only) target = none
            parts.append({k: d[k][keep] for k in keys})
        self.data = {k: np.concatenate([q.pop(k) for q in parts]) for k in keys}
        games, inv = np.unique(self.data["game"], return_inverse=True)
        val_game = is_val(games, val_frac)
        self.n_games, self.n_val_games = len(games), int(val_game.sum())
        self.train_idx, self.val_idx = np.flatnonzero(~val_game[inv]), np.flatnonzero(val_game[inv])
        assert len(self.train_idx), f"--distill {pattern}: no training rows"
        pick = np.random.default_rng(0)  # fixed monitoring rows: comparable across updates
        self.monitor = {s: np.sort(pick.choice(ix, min(self.MONITOR, len(ix)), replace=False))
                        for s, ix in (("val", self.val_idx), ("train", self.train_idx))}

    @staticmethod
    def _signature(paths: list[str]) -> tuple:
        return tuple((p, s.st_size, s.st_mtime_ns) for p, s in ((p, os.stat(p)) for p in paths))

    def summary(self) -> str:
        return (f"{self.pattern}: {len(self.paths)} shards, {len(self.train_idx):,} train / {len(self.val_idx):,} "
                f"held-out rows, {self.n_games:,} games ({self.n_val_games:,} held out)")

    def rows(self, idx: np.ndarray) -> dict[str, np.ndarray]:
        return {k: v[idx] for k, v in self.data.items() if k != "game"}

    def sample(self, n: int, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        """n rows drawn uniformly (with replacement) from the train split, with random nonzero `drop_seed`s (id
        dropout as in pretraining)."""
        rng = self.rng if rng is None else rng
        b = self.rows(self.train_idx[rng.integers(0, len(self.train_idx), n)])
        b["drop_seed"] = rng.integers(1, 2**31 - 1, n)
        return b

    def reload(self) -> "DistillSet | None":
        """Re-glob: a freshly loaded set (same generator) if a shard was added, removed or rewritten, else None.
        Raises on a bad shard (the trainer then keeps the current set)."""
        if self._signature(sorted(glob.glob(self.pattern))) == self.signature:
            return None
        return DistillSet(self.pattern, self.gen, self.val_frac, self.rng)


def distill_ce(logits: torch.Tensor, mask: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row: CE(target renormalised over the legal actions, log π) and top-1 agreement with the target's argmax
    (`porygon.v2.pretrain.losses`' policy term, unweighted); rows without target mass on a legal action are left out."""
    mask = mask > 0
    logp = torch.log_softmax(logits.float(), -1)
    tgt = target * mask
    has = tgt.sum(-1) > 0
    t = tgt[has] / tgt[has].sum(-1, keepdim=True)
    ce = -(t * torch.where(mask[has], logp[has], torch.zeros_like(logp[has]))).sum(-1)
    return ce, logits[has].argmax(-1) == tgt[has].argmax(-1)


def distill_metrics(net: Net, distill: DistillSet, device, amp: bool) -> dict[str, float]:
    """diag/distill_{val,train}_{ce,top1} on the set's fixed monitoring rows (eval mode, every id visible, no grad;
    the net is left in train mode)."""
    st = {}
    net.eval()
    with torch.no_grad():
        for split, rows in distill.monitor.items():
            if not len(rows):
                continue
            b = batch_to_tensors(distill.rows(rows), device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = net(b, aux=False)
            ce, hit = distill_ce(out["logits"], b["action_mask"], b["target"])
            if ce.numel():
                st[f"diag/distill_{split}_ce"] = ce.mean().item()
                st[f"diag/distill_{split}_top1"] = hit.float().mean().item()
    net.train()
    return st


HEADS_ONLY = ("value_head.", "aux_")  # what trains during --critic-warmup-steps


def ppo_update(net: Net, opt, data: dict, args, device, amp: bool, measure_ratio: bool, anchor: Net | None = None,
               anchor_coef: float = 0.0, heads_only: bool = False,
               distill: DistillSet | None = None) -> dict[str, float]:
    keys = model_keys(net.gen) + ["drop_seed"]
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in data.items()}
    n = len(data["action"])
    m = Meter()
    t0 = time.time()
    dmg_all = rollout_damage(net, tensors, keys)
    if getattr(args, "decoupled", False):  # proximal log-probs: one no-grad pass at the update's starting weights
        net.train()  # same id-dropout masks (hashed from drop_seed) as the gradient passes
        parts = []
        with torch.no_grad():
            for c0 in range(0, n, 4096):
                sl = slice(c0, c0 + 4096)
                o = {k: (tensors[k][sl].long() if not tensors[k].is_floating_point() and tensors[k].dtype != torch.bool
                         else tensors[k][sl]) for k in keys}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out = net(o, dmg=None if dmg_all is None else {k: v[sl] for k, v in dmg_all.items()}, aux=False)
                lp = torch.log_softmax(out["logits"].float(), -1)
                parts.append(lp.gather(1, tensors["action"][sl].long()[:, None]).squeeze(1))
        tensors["logp_prox"] = torch.cat(parts)
        rho = torch.exp(tensors["logp_prox"] - tensors["logp"])
        m.add("diag/behaviour_ratio_mean", rho.mean().item())
        m.add("diag/behaviour_ratio_sd", rho.std().item())
        m.add("diag/behaviour_off_clip", ((rho - 1).abs() > args.clip_range).float().mean().item())
    m.add("perf/damage_s", time.time() - t0)
    if anchor is not None:
        tensors["logp_anchor"] = anchor_logp(anchor, tensors, keys, dmg_all, device)
    net.train()
    frozen_now = []
    if heads_only:  # critic warm-up: policy and trunk frozen (no grad -> AdamW skips them: no decay, no moments)
        for name, prm in net.named_parameters():
            if prm.requires_grad and not name.startswith(HEADS_ONLY):
                prm.requires_grad_(False)
                frozen_now.append(prm)
    if distill is not None and measure_ratio:  # labelled-set agreement at the update's starting weights
        for k, v in distill_metrics(net, distill, device, amp).items():
            m.add(k, v)
    distill_on = distill is not None and args.distill_coef > 0 and args.distill_batch > 0 and not heads_only

    def minibatch(idx):
        mb = {k: v[idx] for k, v in tensors.items()}
        obs = {k: (mb[k].long() if not mb[k].is_floating_point() and mb[k].dtype != torch.bool else mb[k])
               for k in keys}
        return obs, (None if dmg_all is None else {k: v[idx] for k, v in dmg_all.items()}), mb

    if measure_ratio and not heads_only:  # (trunk frozen during the critic warm-up)
        perm = torch.randperm(n, device=device)
        bs = min(args.micro_batch or args.batch_size, args.batch_size, n // 2)  # GPU memory: one chunk each
        if bs >= 2:
            gradient_diagnostics(net, [minibatch(perm[:bs]), minibatch(perm[bs:2 * bs])], args, amp, m)
    for epoch in range(args.n_epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            if len(idx) < 2:
                continue
            # gradient accumulation over --micro-batch chunks (same update as one big minibatch; advantages are
            # normalised over the whole minibatch first): lets 4096-sample minibatches fit a 16 GB GPU
            adv_all = tensors["adv"][idx]
            adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8) if len(idx) > 1 else adv_all
            mbs = args.micro_batch if args.micro_batch and args.micro_batch < len(idx) else len(idx)
            opt.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            for c0 in range(0, len(idx), mbs):
                sub = idx[c0:c0 + mbs]
                w = len(sub) / len(idx)
                obs, dmg, mb = minibatch(sub)
                mb["adv"] = adv_all[c0:c0 + mbs]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    out = net(obs, dmg=dmg)
                L = ppo_losses(out, mb, args, normalized=True)
                if epoch == 0 and s == 0 and c0 == 0:  # before any gradient step: how off-policy is the data?
                    m.add("diag/first_ratio_mean", L["ratio_mean"].item())
                    m.add("diag/first_ratio_std", L["ratio_std"].item())
                    m.add("diag/first_clip", L["clip"].item())
                    m.add("diag/first_kl", L["kl"].item())
                aux = aux_losses(out, mb, m if epoch == 0 else None)
                if heads_only:
                    total = args.vf_coef * L["vf"] + args.aux_coef * aux
                else:
                    total = (L["pg"] - args.ent_coef * L["ent"] + args.zero_avoid_coef * L["za"]
                             + args.vf_coef * L["vf"] + args.aux_coef * aux + anchor_coef * L["kla"])
                (total * w).backward()
                for k in ("pg", "ent", "za", "vf", "kl", "clip", "kla", "ptop"):
                    sums[k] = sums.get(k, 0.0) + w * L[k].detach()
                sums["aux"] = sums.get("aux", 0.0) + w * aux.detach()
                sums["total"] = sums.get("total", 0.0) + w * total.detach()
            if distill_on:  # --distill: one extra pass on labelled train rows, added to this step's gradient
                db = batch_to_tensors(distill.sample(args.distill_batch), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    dout = net(db, aux=False)
                ce = distill_ce(dout["logits"], db["action_mask"], db["target"])[0]
                if ce.numel():
                    ce = ce.mean()
                    (args.distill_coef * ce).backward()
                    m.add("loss/distill", ce.item())
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            for k in ("pg", "ent", "za", "vf", "kl", "clip", "ptop"):
                m.add(f"loss/{k}", float(sums[k]))
            if anchor is not None:
                m.add("loss/kl_anchor", float(sums["kla"]))
            m.add("loss/aux", float(sums["aux"]))
            m.add("loss/total", float(sums["total"]))
            m.add("diag/grad_norm", gn.item())
    for prm in frozen_now:
        prm.requires_grad_(True)
    st = m.means()
    if anchor is not None:
        st["perf/kl_anchor_coef"] = anchor_coef
    st["perf/heads_only"] = float(heads_only)
    v, ret = data["value"], data["ret"]
    var = ret.var()
    st["diag/explained_variance"] = float(1 - (ret - v).var() / var) if var > 0 else float("nan")
    st["diag/value_mean"] = float(v.mean())
    return st


# ---------------------------------------------------------------- evaluation

def anchor_player(path: Path, device, common: dict):
    from porygon.v2.model import is_v2_checkpoint, load
    from porygon.v2.player import NetPlayer
    if is_v2_checkpoint(path):
        return NetPlayer(load(path, device), **common)
    from porygon.env import PolicyPlayer
    from porygon.model import load_policy
    return PolicyPlayer(load_policy(path, str(device)), **common)


class Evaluator:
    """Long-lived players (one websocket each) reused across evaluations."""

    def __init__(self, net: Net, battle_format: str, port: int, anchor: Path | None, device, concurrency: int = 16):
        from poke_env.player import MaxBasePowerPlayer, SimpleHeuristicsPlayer
        from porygon.v2.player import NetPlayer
        common = dict(battle_format=battle_format, server_configuration=local_server(port),
                      max_concurrent_battles=concurrency, log_level=40)
        self.player = NetPlayer(net, **common)
        self.opponents = {"heuristic": SimpleHeuristicsPlayer(**common), "max_power": MaxBasePowerPlayer(**common)}
        if anchor is not None:
            self.opponents = {"anchor": anchor_player(anchor, device, common)} | self.opponents

    def run(self, n_battles: int) -> dict[str, float]:
        results = {}
        for name, opp in self.opponents.items():
            won, done = self.player.n_won_battles, self.player.n_finished_battles
            asyncio.run(self.player.battle_against(opp, n_battles=n_battles))
            results[name] = (self.player.n_won_battles - won) / max(1, self.player.n_finished_battles - done)
            self.player.reset_battles()
            opp.reset_battles()
        return results


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--format", default="gen9randombattle", help="gen4randombattle | gen9randombattle")
    a("--ports", default="8000", help="Showdown server ports, e.g. 8000-8007")
    a("--n-battles", type=int, default=16, help="concurrent self-play battles (two learner sides each)")
    a("--pool-battles", type=int, default=0, help="concurrent learner-vs-pool battles")
    a("--pool-dir", type=Path, help="opponent pool (v2 .pt snapshots; default RUN_DIR/pool)")
    a("--p-baseline", type=float, default=0.2, help="share of pool battles vs baseline bots")
    a("--snapshot-every", type=int, default=2_500_000)
    a("--total-steps", type=int, default=300_000_000)
    a("--rollout", type=int, default=40_960, help="transitions per PPO update")
    a("--batch-size", type=int, default=1024)
    a("--micro-batch", type=int, default=2048, help="gradient-accumulation chunk of a minibatch (GPU memory)")
    a("--n-epochs", type=int, default=7)
    a("--gamma", type=float, default=0.9999)
    a("--gae-lambda", type=float, default=0.754)
    a("--clip-range", type=float, default=0.1)
    a("--guard-anchor-drop", type=float, default=0.0, help="stop when the eval score is this far below its best twice")
    a("--guard-entropy-max", type=float, default=0.0, help="stop when the policy entropy exceeds this for 5 updates")
    a("--guard-pg-mean", type=int, default=0,
      help="stop when the PPO surrogate pg averaged over the last N updates of this process is > 0 (0 = off)")
    a("--guard-entropy-min", type=float, default=0.0,
      help="collapse guard: stop when the policy entropy is below this for 5 updates")
    a("--guard-ptop-max", type=float, default=0.0,
      help="collapse guard: stop when the share of choices (>= 2 legal actions) with top probability >= 0.9 exceeds "
           "this for 5 updates")
    a("--guard-entropy-rel", type=float, default=0.0,
      help="entropy guard at this multiple of the mean entropy of this process's first 5 updates (if no max)")
    a("--kl-anchor", type=Path, help="frozen v2 net: add beta(t) * KL(anchor || pi) to the loss")
    a("--kl-anchor-coef", type=float, default=0.0, help="beta at step 0 (linear decay to 0)")
    a("--kl-anchor-decay-steps", type=int, default=50_000_000, help="env steps (lineage counter) at which beta = 0")
    a("--critic-warmup-steps", type=int, default=0, help="until this many env steps only value + aux heads train")
    a("--frozen", type=Path, nargs="*", default=[], help="frozen opponents for pool battles (copied to pool/frozen)")
    a("--p-frozen", type=float, default=0.0, help="share of pool battles vs a frozen opponent")
    a("--frozen-best", action="store_true", help="each new best_model.pt also becomes a frozen opponent")
    a("--distill", help="glob of v2 supervised shards (porygon.v2.pretrain format): distillation term + monitor")
    a("--distill-coef", type=float, default=0.0, help="weight of CE(target, pi) per minibatch (0 = monitor only)")
    a("--distill-batch", type=int, default=512, help="labelled train rows per PPO minibatch")
    a("--distill-val-frac", type=float, default=0.06, help="held-out share of the --distill games (monitored only)")
    a("--distill-reload-every", type=int, default=0, help="re-glob and reload --distill every N updates (0 = never)")
    a("--decoupled", action="store_true", help="decoupled PPO: clip vs the update-start policy, truncated IS weights")
    a("--decoupled-rho", type=float, default=2.0, help="decoupled PPO: truncation of pi_prox / pi_behaviour")
    a("--ent-coef", type=float, default=0.01)
    a("--zero-avoid-coef", type=float, default=0.003)
    a("--vf-coef", type=float, default=0.5)
    a("--aux-coef", type=float, default=0.1)
    a("--max-grad-norm", type=float, default=0.5)
    a("--lr", type=float, default=1e-4)
    a("--warmup-steps", type=int, default=1_000_000)
    a("--cooldown-start", type=int, default=None, help="start of the cosine cooldown (env steps); none by default")
    a("--final-lr-frac", type=float, default=0.1)
    a("--lr-schedule", choices=("wsd", "power"), default="wsd",
      help="wsd: warm-up, constant, optional cosine cooldown (default); power: x (8x+1)^-1.5 after the warm-up "
           "over --lr-horizon steps (flat at lr/27 after it), times the optional cooldown")
    a("--lr-horizon", type=int, default=None,
      help="--lr-schedule power: the planned horizon in absolute env steps (independent of --total-steps)")
    a("--weight-decay", type=float, default=0.01)
    a("--adv-clip", type=float, default=0.0, help="clip per-sample normalised advantages to ±C (0 = off)")
    a("--ema-decay", type=float, default=0.999,
      help="EMA of the weights per update (0 = off); written to ema_model.pt with every checkpoint and at the end")
    a("--shaping", type=float, nargs=3, metavar=("FAINTED", "HP", "STATUS"), default=(0.0, 0.0, 0.0),
      help="potential-based shaping weights (v1 used 0.05 0.05 0); default sparse ±1 only")
    a("--eval-every", type=int, default=2_500_000)
    a("--eval-battles", type=int, default=200, help="per opponent; 0 disables evaluation")
    a("--eval-async", action="store_true",
      help="play the evaluation in a CPU subprocess on a snapshot while training continues (porygon.v2.train_eval)")
    a("--eval-timeout", type=float, default=3600.0,
      help="--eval-async: kill an eval subprocess that has run this many seconds without a result (0 = never)")
    a("--anchor", type=Path, help="also evaluate vs this net (v2 .pt, or v1 .zip / pool .pt)")
    a("--checkpoint-every", type=int, default=5_000_000)
    a("--run-dir", type=Path, default=Path("runs") / ("v2-" + time.strftime("%Y%m%d-%H%M%S")))
    a("--resume", type=Path, help="trainer checkpoint (.pt) to continue from (model, optimizer, step counters)")
    a("--reset-optimizer", action="store_true", help="with --resume: fresh optimizer state")
    a("--max-hours", type=float, help="stop (and save) after this much wall-clock time")
    a("--max-updates", type=int, help="stop after this many updates in this process (tests)")
    a("--no-overlap", action="store_true", help="pause battles during the PPO update")
    a("--actor-stream", action="store_true",
      help="the collector thread on a high-priority CUDA stream, acting from double-buffered weights refreshed while "
           "it acts (porygon.v2.actor_buffers), so overlapped forwards do not queue behind the update; CPU: the "
           "double buffer only (default off: one actor copy, every kernel on the default stream)")
    a("--pool-alarm-updates", type=int, default=5,
      help="pool battles configured but no pool episode finished for this many updates in a row: log, write "
           "POOL_ALARM and stop with a 'pool:' guard (0 = off); a silently missing opponent mix (R1's hung pool "
           "workers) must not recur")
    from porygon.fastenv_v2 import add_args as fastenv_args
    fastenv_args(a)  # --collector showdown|fastenv, --fastenv-workers / -battles / -pool-share
    a("--grad-ratio-every", type=int, default=10,
      help="every N updates: trunk value/policy gradient ratio and the minibatch-to-minibatch gradient cosines")
    a("--stall-seconds", type=float, default=900.0, help="restart a battle worker silent this long")
    a("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a("--threads", type=int, default=1, help="torch CPU threads in the trainer process")
    a("--no-bf16", action="store_true", help="no bf16 autocast on CUDA")
    a("--compile", action=argparse.BooleanOptionalAction, default=None,
      help="torch.compile the damage layer (default: on CUDA); falls back to eager if compilation fails")
    a("--snapshot-cache", type=int, default=8, help="pool snapshot opponents kept on the device")
    a("--snapshot-cache-cpu", type=int, default=0,
      help="pool snapshot opponents evicted from the device cache kept in CPU memory (LRU; 0 = off): a net found "
           "there moves back to the device (milliseconds) instead of a load from disk (~0.1-0.3 s per net)")
    a("--stacked-opp", action="store_true",
      help="--collector fastenv: the pool opponents of one collector step share one forward when they have the same "
           "architecture (weights stacked, porygon.v2.stacked; ~one forward's kernel launches instead of one "
           "forward per opponent; logits equal up to float rounding); others and failed loads as without it")
    a("--seed", type=int, default=None)
    # model (ignored with --resume: the checkpoint's cfg is used)
    a("--d-model", type=int, default=DEFAULT_CFG["d_model"])
    a("--n-layers", type=int, default=DEFAULT_CFG["n_layers"])
    a("--n-heads", type=int, default=DEFAULT_CFG["n_heads"])
    a("--ffn-mult", type=int, default=DEFAULT_CFG["ffn_mult"])
    a("--d-move", type=int, default=DEFAULT_CFG["d_move"])
    a("--value-trunk-grad", type=float, default=DEFAULT_CFG["value_trunk_grad"])
    a("--id-dropout", type=float, default=DEFAULT_CFG["id_dropout"])
    a("--no-damage", action="store_true", help="model without the damage layer (porygon.v2.damage)")
    a("--no-tables", action="store_true", help="id embeddings only (no static-feature MLPs)")
    a("--model-json", default=None, help='other Net cfg keys as JSON, e.g. \'{"d_species": 64}\'')
    args = p.parse_args(argv)
    if args.lr_schedule == "power" and (args.lr_horizon is None or args.lr_horizon <= args.warmup_steps):
        p.error("--lr-schedule power needs --lr-horizon > --warmup-steps")
    return args


def lr_schedule_desc(args) -> str:
    """The lr schedule for the startup log line."""
    s = f"lr schedule: {args.lr_schedule}"
    if args.lr_schedule == "power":
        s += f" H={args.lr_horizon}"
    s += f", lr {args.lr:g}, warm-up {args.warmup_steps}"
    if args.cooldown_start is not None:
        s += f", cosine cooldown {args.cooldown_start} -> {args.total_steps} to x{args.final_lr_frac:g}"
    return s


def lr_at(args, step: int) -> float:
    """The lr at absolute env step `step` for these args (what the trainer sets before each update)."""
    return wsd_lr(step, args.lr, args.warmup_steps, args.cooldown_start, args.total_steps, args.final_lr_frac,
                  args.lr_schedule, args.lr_horizon)


def model_cfg(args) -> dict:
    return dict(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, ffn_mult=args.ffn_mult,
                d_move=args.d_move, value_trunk_grad=args.value_trunk_grad, id_dropout=args.id_dropout,
                use_damage=not args.no_damage, use_tables=not args.no_tables, **json.loads(args.model_json or "{}"))


class EMA:
    """Exponential moving average of the parameters (buffers copied), updated once per PPO update. The effective
    decay is min(decay, (1 + n) / (10 + n)) so early averages are not dominated by the random init."""

    def __init__(self, net: Net, decay: float):
        self.decay, self.n = decay, 0
        self.net = copy.deepcopy(net).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net: Net) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        torch._foreach_lerp_(list(self.net.parameters()), [p.detach() for p in net.parameters()], 1.0 - d)
        for b, src in zip(self.net.buffers(), net.buffers()):
            b.copy_(src)

    def state_dict(self) -> dict:
        return {"n": self.n, "decay": self.decay, "state_dict": self.net.state_dict()}

    def load_state_dict(self, st: dict) -> None:
        self.n = st["n"]
        self.net.load_state_dict(st["state_dict"], strict=False)


class PoolLoads:
    """Pool-opponent net loads of `Trainer._opp_act` (collector thread) per kind, "frozen" (names under frozen/,
    best-*.pt included) or "snapshot": attempts (= cache misses), failures, and messages queued once per net for the
    main loop's log. `take()` (main loop): the counts since its previous call and the queued messages."""

    KINDS = ("frozen", "snapshot")

    def __init__(self):
        self.lock = threading.Lock()
        self.attempts = dict.fromkeys(self.KINDS, 0)
        self.failures = dict.fromkeys(self.KINDS, 0)
        self.cpu_hits = dict.fromkeys(self.KINDS, 0)  # --snapshot-cache-cpu: nets brought back from CPU memory
        self._taken = (dict(self.attempts), dict(self.failures))
        self._taken_cpu = dict(self.cpu_hits)
        self._noted: set = set()
        self._messages: list[str] = []

    @staticmethod
    def kind(name: str) -> str:
        return "frozen" if name.startswith("frozen/") else "snapshot"

    def note(self, key, msg: str) -> None:
        """Queue `msg` for the log unless one was queued for `key` before."""
        with self.lock:
            if key not in self._noted:
                self._noted.add(key)
                self._messages.append(msg)

    def record(self, name: str, path, error: BaseException | None = None) -> None:
        kind = self.kind(name)
        with self.lock:
            self.attempts[kind] += 1
            self.failures[kind] += error is not None
        if error is not None:
            self.note(("failed", name), f"pool load FAILED {kind} {path}: {error!r}")

    def record_cpu_hit(self, name: str) -> None:
        with self.lock:
            self.cpu_hits[self.kind(name)] += 1

    def take(self) -> tuple[dict[str, float], list[str]]:
        with self.lock:
            (a0, f0), now = self._taken, (dict(self.attempts), dict(self.failures))
            c0, cnow = self._taken_cpu, dict(self.cpu_hits)
            self._taken, self._taken_cpu, msgs, self._messages = now, cnow, self._messages, []
        st = {}
        for k in self.KINDS:
            st[f"pool/load_attempts_{k}"] = float(now[0][k] - a0[k])
            st[f"pool/load_failures_{k}"] = float(now[1][k] - f0[k])
            if cnow[k] or c0[k]:  # only with --snapshot-cache-cpu (unchanged stats otherwise)
                st[f"pool/cpu_hits_{k}"] = float(cnow[k] - c0[k])
        return st, msgs


class Trainer:
    def __init__(self, args):
        self.args = args
        self.gen = gen_of(args.format)
        self.run_dir = args.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.pool_dir = args.pool_dir or self.run_dir / "pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.log_file = open(self.run_dir / "train.log", "a", buffering=1)
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        ports = parse_ports(args.ports)
        shaping = dict(zip(("fainted_value", "hp_value", "status_value"), args.shaping))
        if args.collector == "fastenv":
            from porygon.fastenv_v2 import build_env
            self.workers = build_env(args, args.seed)
        else:
            specs = ([WorkerSpec("selfplay", ports[i % len(ports)], args.format, shaping) for i in range(args.n_battles)]
                     + [WorkerSpec("pool", ports[::-1][i % len(ports)], args.format, shaping, str(self.pool_dir),
                                   args.p_baseline, p_frozen=args.p_frozen) for i in range(args.pool_battles)])
            self.workers = BattleWorkers(specs)  # before CUDA init (forkserver anyway)

        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision("high")
        self.device = torch.device(args.device)
        self.amp = self.device.type == "cuda" and not args.no_bf16
        self.counters = {"steps": 0, "updates": 0, "games": 0, "best": -1.0}
        ck = None
        if args.resume:
            ck = torch.load(args.resume, map_location="cpu", weights_only=False)
            self.net = net_from_checkpoint(ck)
            self.counters.update(ck.get("counters", {}))
        else:
            self.net = Net(self.gen, **model_cfg(args))
            save(self.net, self.run_dir / "init_model.pt", counters=dict(self.counters))  # step-0 weights (drift analyses)
        assert self.net.gen == self.gen, f"model gen {self.net.gen} vs format {args.format}"
        self.net.to(self.device).train()
        self.opt = torch.optim.AdamW(self.net.param_groups(args.weight_decay), lr=args.lr, betas=(0.9, 0.999),
                                     eps=1e-5, fused=self.device.type == "cuda")
        if ck is not None and "optimizer" in ck and not args.reset_optimizer:
            self.opt.load_state_dict(ck["optimizer"])
        self.actor_buffers = None
        if getattr(args, "actor_stream", False):
            from porygon.v2.actor_buffers import ActorBuffers
            self.actor_buffers = ActorBuffers(self.net, self.device)  # two acting copies (+ the actor stream on CUDA)
            self.actor = self.actor_buffers.nets[0]  # holds the compiled damage layer both copies and _opp_net share
            self.actor_lock = self.actor_buffers.lock  # _act's version counts and _take_versions under the flip's lock
        else:
            self.actor = copy.deepcopy(self.net).eval()
            self.actor_lock = threading.Lock()
        if args.frozen:  # fixed opponents for the whole run (anti-cycling); the trainer loads them by relative name
            fdir = self.pool_dir / "frozen"
            fdir.mkdir(parents=True, exist_ok=True)
            for f in args.frozen:
                dst = fdir / f.name
                if not dst.exists():
                    tmp = fdir / (f.name + ".tmp")
                    shutil.copyfile(f, tmp)
                    os.replace(tmp, dst)
        self.anchor = None
        if args.kl_anchor is not None:
            self.anchor = load(args.kl_anchor, self.device).eval()
            for prm in self.anchor.parameters():
                prm.requires_grad_(False)
            assert self.anchor.gen == self.gen, f"anchor gen {self.anchor.gen}"
        self.distill = None
        if args.distill:
            self.distill = DistillSet(args.distill, self.gen, args.distill_val_frac, np.random.default_rng(args.seed))
        self.ema = EMA(self.net, args.ema_decay) if args.ema_decay > 0 else None
        if self.ema is not None and ck is not None and "ema" in ck:
            self.ema.load_state_dict(ck["ema"])
        self.packer = Packer(self.gen)
        self.compile = (self.device.type == "cuda") if args.compile is None else args.compile
        if self.compile:
            self.net.compile_damage()
            self.actor.compile_damage()
            if self.actor_buffers is not None:  # the other copy shares the compiled parameter-free layer (no recompile)
                for k in ("_damage_layer_untrimmed", "_damage_compiled"):
                    if k in self.actor.__dict__:
                        object.__setattr__(self.actor_buffers.nets[1], k, self.actor.__dict__[k])
        self.snapshots: dict[str, Net] = {}  # pool snapshot opponents on the device (LRU)
        self.pool_loads = PoolLoads()  # their loads (attempts / failures), checked after every update
        self.opp_stacks: dict = {}  # --stacked-opp: architecture signature -> porygon.v2.stacked.OppStack
        if args.collector == "fastenv":
            from porygon.fastenv_v2 import FastCollector
            self.collector = FastCollector(self.workers.env, self.packer, self._act, self._opp_act,
                                           id_dropout=self.net.cfg["id_dropout"] > 0, seed=args.seed,
                                           pool_per_worker=self.workers.pool_per_worker, pool_dir=self.pool_dir,
                                           p_baseline=args.p_baseline, p_frozen=args.p_frozen,
                                           on_policy=args.no_overlap,
                                           opp_fn_many=self._opp_act_many if args.stacked_opp else None)
        else:
            if args.stacked_opp:
                print("warning: --stacked-opp only applies to --collector fastenv; ignored", flush=True)
            self.collector = Collector(self.workers, self.packer, self._act, self._opp_act,
                                       id_dropout=self.net.cfg["id_dropout"] > 0, stall_seconds=args.stall_seconds,
                                       seed=args.seed)
        self.requests = threading.Semaphore(0)
        self.rollouts: queue.Queue = queue.Queue()
        self.evaluator = None
        self.eval_job: EvalJob | None = None  # --eval-async: the running eval subprocess
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.run_dir / "tb"))
        n_params = sum(p.numel() for p in self.net.parameters())
        self.log(f"v2 trainer: {args.format} gen{self.gen}, collector {args.collector}, {args.n_battles} self-play + "
                 f"{args.pool_battles} pool "
                 f"battles on ports {ports}, params={n_params:,}, cfg={self.net.cfg}, device={self.device}, "
                 f"amp={self.amp}, overlap={not args.no_overlap}, resume={args.resume}, "
                 f"steps={self.counters['steps']:,}, {lr_schedule_desc(args)}")
        if self.distill is not None:
            self.log(f"distill: {self.distill.summary()}; coef {args.distill_coef}, batch {args.distill_batch}, "
                     f"reload every {args.distill_reload_every} updates (0 = never)")
        if args.stacked_opp and args.collector == "fastenv":
            self.log("stacked-opp: same-architecture pool opponents share one forward per collector step "
                     "(perf/col_opp_fwds = forwards, perf/col_opp_fwd_nets = opponents served)")
        if self.actor_buffers is not None:
            self.log(self.actor_buffers.describe())
            if not args.no_overlap and not args.decoupled:
                self.log("warning: --actor-stream overlaps collection with the update without --decoupled: the actor "
                         "acts with weights up to one update old (use --decoupled with overlap)")

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.log_file.write(line + "\n")

    @torch.no_grad()
    def _act(self, batch: dict):
        t = batch_to_tensors(batch, self.device, keys=model_keys(self.gen) + ["drop_seed"])
        buffers = getattr(self, "actor_buffers", None)
        if buffers is None:
            with self.actor_lock, torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                out = self.actor(t, aux=False)
                v = getattr(self, "actor_version", 0)
                if not hasattr(self, "version_counts"):
                    self.version_counts = {}
                self.version_counts[v] = self.version_counts.get(v, 0) + len(batch["action_mask"])
        else:  # --actor-stream: the front copy and its version, under the buffers' lock (= self.actor_lock)
            with buffers.acquire() as (net, v), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                                               enabled=self.amp):
                out = net(t, aux=False)
                if not hasattr(self, "version_counts"):
                    self.version_counts = {}
                self.version_counts[v] = self.version_counts.get(v, 0) + len(batch["action_mask"])
        logp_all = torch.log_softmax(out["logits"].float(), -1)
        probs = logp_all.exp()
        a = torch.multinomial(probs, 1).squeeze(1)
        return (a.cpu().numpy(), logp_all.gather(1, a[:, None]).squeeze(1).cpu().numpy(),
                out["value"].float().cpu().numpy(), probs.cpu().numpy())

    @torch.no_grad()
    def _opp_act(self, name: str, batch: dict) -> np.ndarray:
        """Sampled actions of pool snapshot `name` (cached on the device; uniform over legal actions if it cannot
        be loaded). Collector thread: every load (cache miss) and failure is only recorded in `self.pool_loads`;
        the main loop logs the failures and trips the pool guard after each update (`_pool_alarm`)."""
        self._flush_cpu()
        net = self._opp_net(name)
        return self._opp_sample(None if net is None else self._opp_logits(net, batch), batch)

    def _opp_net(self, name: str) -> Net | None:
        """`_opp_act`'s net for `name`: the cached one (moved to most recently used), else (--snapshot-cache-cpu)
        the one kept in CPU memory, moved back to the device, else loaded (recorded in `self.pool_loads`); when the
        device cache is full its least recently used net is evicted (to the CPU tier if on); None if it cannot be
        loaded."""
        net = self.snapshots.pop(name, None)
        if net is None:
            name = self._rotated(name)
            net = self.snapshots.pop(name, None)
        if net is None:
            cpu = self.__dict__.setdefault("snapshots_cpu", {})
            net = cpu.pop(name, None)
            if net is not None:  # the same object goes back (OppStack slots are matched by identity)
                self.__dict__.setdefault("pending_cpu", {}).pop(id(net), None)
                net.to(self.device)
                self.pool_loads.record_cpu_hit(name)
            else:
                path = self.pool_dir / name
                try:
                    net = load(path, self.device)
                    assert net.gen == self.gen, f"gen {net.gen} snapshot"
                    if self.compile:  # same parameter-free layer: reuse the actor's compiled function (no recompiles)
                        object.__setattr__(net, "_damage_compiled", getattr(self.actor, "_damage_compiled", None))
                except Exception as e:
                    self.pool_loads.record(name, path, e)
                    net = None
                else:
                    self.pool_loads.record(name, path)
            if len(self.snapshots) >= self.args.snapshot_cache:
                self._evict(next(iter(self.snapshots)))
        if net is not None:
            self.snapshots[name] = net  # most recently used last
        return net

    def _evict(self, name: str) -> None:
        """Drop `name` from the device cache; with --snapshot-cache-cpu keep it in the CPU tier (LRU, bounded). The
        move to the CPU waits for the next collector step (`_flush_cpu`): a net evicted while its step is still
        being assembled may be in that step's forward."""
        net = self.snapshots.pop(name)
        cap = getattr(self.args, "snapshot_cache_cpu", 0)
        if cap > 0:
            cpu = self.__dict__.setdefault("snapshots_cpu", {})
            cpu[name] = net
            self.__dict__.setdefault("pending_cpu", {})[id(net)] = net
            while len(cpu) > cap:
                old = cpu.pop(next(iter(cpu)))
                self.pending_cpu.pop(id(old), None)

    def _flush_cpu(self) -> None:
        """Move the nets evicted during earlier collector steps to the CPU (called before a step's lookups)."""
        pending = self.__dict__.get("pending_cpu")
        if pending:
            for net in list(pending.values()):
                net.to("cpu")
                for m in net.modules():  # derived device tensors (Entity.all_features' no-grad cache): recomputed later
                    m.__dict__.pop("_cache", None)
                    m.__dict__.pop("_cache_key", None)
            pending.clear()

    @torch.no_grad()
    def _opp_logits(self, net: Net, batch: dict) -> torch.Tensor:
        """One pool net's float32 CPU logits on its decision batch (one forward)."""
        t = batch_to_tensors(batch, self.device, keys=model_keys(self.gen) + ["drop_seed"])
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            return net(t, aux=False)["logits"].float().cpu()

    @staticmethod
    def _opp_sample(logits: torch.Tensor | None, batch: dict) -> np.ndarray:
        """Actions sampled from the logits (None: uniform over the legal actions; no legal action: uniform)."""
        mask = torch.from_numpy(batch["action_mask"]) > 0
        if logits is None:
            logits = torch.where(mask, 0.0, torch.finfo(torch.float32).min)
        probs = torch.softmax(logits, -1)
        probs = torch.where(mask.any(-1, keepdim=True), probs, torch.full_like(probs, 1.0 / probs.shape[-1]))
        return torch.multinomial(probs, 1).squeeze(1).numpy()

    @torch.no_grad()
    def _opp_act_many(self, items: list[tuple[str, dict]]) -> tuple[list[np.ndarray], dict[str, float]]:
        """--stacked-opp (fastenv collector): `_opp_act` for all the pool opponents of one collector step, items =
        [(name, batch), ...] with distinct names. Nets are looked up / loaded / evicted / recorded exactly as by
        `_opp_act` called per item in order; the loaded nets with the same architecture (`porygon.v2.stacked.
        signature`: gen, cfg, shapes) and the same kind (frozen/* apart from pool snapshots) share one stacked
        forward (`OppStack`, cached per architecture and kind: slots refilled only for nets new to it, released when a
        net leaves the snapshot cache), the rest take `_opp_logits`, a net that cannot be loaded plays uniformly.
        Then each item samples as `_opp_act` does, in order (same distributions; the CPU generator's draws differ
        from per-item calls only around a cache miss, since loads now precede all draws). Returns the actions per
        item and counters for the collector's perf/col_*: opp_fwds (device forwards, a stacked one counts once),
        opp_stacked_fwds / opp_stacked_nets, opp_stack_refills, opp_uniform. A stacked forward that raises (e.g. out
        of memory) is noted in the log once and the run continues with per-net forwards (opp_stack_errors)."""
        from porygon.v2.stacked import OppStack, signature
        stacks = self.__dict__.setdefault("opp_stacks", {})  # (signature, kind) -> OppStack
        self._flush_cpu()
        nets = [self._opp_net(name) for name, _ in items]
        # one stacked forward per architecture AND opponent kind: a stacked forward pads every net's rows to its
        # largest group, and the frozen nets (--p-frozen of the pool battles shared by ~2 nets) have ~10-50x the rows
        # of a pool snapshot, so stacked together most of the forward was padding, growing with the pool; apart, the
        # frozen stack also keeps its slots (no refills from snapshots displacing them). Logits are per net either way.
        groups: dict[tuple, list[int]] = {}
        for i, net in enumerate(nets):
            if net is not None:
                kind = "frozen" if items[i][0].startswith("frozen/") else "pool"
                groups.setdefault((signature(net), kind), []).append(i)
        logits: list[torch.Tensor | None] = [None] * len(items)
        st = dict.fromkeys(("opp_fwds", "opp_stacked_fwds", "opp_stacked_nets", "opp_stack_refills"), 0)
        st["opp_uniform"] = sum(net is None for net in nets)
        cap = max(2, self.args.snapshot_cache)  # nets per stacked forward (bounds an OppStack's slots)
        for key, idx in groups.items():
            for c in range(0, len(idx), cap):
                chunk = idx[c:c + cap]
                if len(chunk) > 1 and not self.__dict__.get("opp_stack_off"):
                    try:
                        stack = stacks.get(key)
                        if stack is None:
                            stack = stacks[key] = OppStack(nets[chunk[0]])
                        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                            out, refills = stack.logits([nets[i] for i in chunk], [items[i][1] for i in chunk])
                    except Exception as e:  # the speed-up must not stop training: per-net forwards from now on
                        self.__dict__["opp_stack_off"] = True
                        stacks.clear()
                        st["opp_stack_errors"] = 1
                        self.pool_loads.note(("stacked-opp",), f"--stacked-opp: the stacked forward failed ({e!r:.300}"
                                                               "); per-net opponent forwards from now on")
                    else:
                        for i, lg in zip(chunk, out):
                            logits[i] = lg
                        st["opp_fwds"] += 1
                        st["opp_stacked_fwds"] += 1
                        st["opp_stacked_nets"] += len(chunk)
                        st["opp_stack_refills"] += refills
                        continue
                for i in chunk:
                    logits[i] = self._opp_logits(nets[i], items[i][1])
                    st["opp_fwds"] += 1
        live = {id(net) for net in self.snapshots.values()}
        for key in [k for k, stack in stacks.items() if not stack.release(live)]:
            del stacks[key]  # none of its nets is cached any more: free its weights
        return [self._opp_sample(lg, batch) for lg, (_, batch) in zip(logits, items)], st

    def _rotated(self, name: str) -> str:
        """`name`, unless it is a frozen/best-*.pt that --frozen-best replaced while a battle still plays it (the file
        gone, a newer best there): then that newest best, noted once per name (a rotation, not a load failure)."""
        if not name.startswith("frozen/best-") or (self.pool_dir / name).exists():
            return name
        newest = max((self.pool_dir / "frozen").glob("best-*.pt"), default=None)
        if newest is None or newest.name <= name[len("frozen/"):]:
            return name  # nothing newer: a real failure
        new = "frozen/" + newest.name
        self.pool_loads.note(("rotated", name), f"pool opponent {name} was replaced by --frozen-best mid-battle; "
                                                f"playing {new} instead")
        return new

    def _refresh_actor(self) -> None:
        buffers = getattr(self, "actor_buffers", None)
        if buffers is not None:  # --actor-stream: into the back copy while the actor acts on the front one, then flip
            buffers.refresh(self.net, self.counters["updates"])
            with self.actor_lock:
                self.actor_version = self.counters["updates"]  # (_act counts the version of the copy it acted with)
            return
        with self.actor_lock:
            self.actor.load_state_dict(self.net.state_dict())
            self.actor_version = self.counters["updates"]  # the acting weights = the learner after this many updates

    def _collect_on_actor_stream(self, *run_args) -> None:
        """--actor-stream: the collector thread's target. The current CUDA stream is per thread, so every GPU op of
        this thread (actor and pool-opponent forwards, pool-net loads and device moves, host-to-device copies) runs on
        the high-priority actor stream while the update stays on the default stream. CPU: the plain collector loop."""
        if self.actor_buffers.stream is not None:
            torch.cuda.set_stream(self.actor_buffers.stream)
        self.collector.run(*run_args)

    def _take_versions(self) -> dict[int, int]:
        """Decisions acted per actor version since the last call (≈ the rollout just taken)."""
        with self.actor_lock:
            vc, self.version_counts = getattr(self, "version_counts", {}), {}
        return vc

    def _get_rollout(self):
        while not self.stop.is_set():
            try:
                item = self.rollouts.get(timeout=1.0)
            except queue.Empty:
                continue
            if isinstance(item, BaseException):
                raise item
            return item
        return None

    def checkpoint(self, path: Path) -> None:
        args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(self.args).items()}
        extra = {} if self.ema is None else {"ema": self.ema.state_dict()}
        save(self.net, path, optimizer=self.opt.state_dict(), counters=dict(self.counters), args=args, **extra)
        if self.ema is not None:  # deployable average (live net still drives play, eval and the pool)
            save(self.ema.net, self.run_dir / "ema_model.pt", counters=dict(self.counters), ema_updates=self.ema.n,
                 ema_decay=self.ema.decay)

    def evaluate(self) -> None:
        a = self.args
        if self.evaluator is None:
            self.evaluator = Evaluator(self.net, a.format, parse_ports(a.ports)[0], a.anchor, self.device)
        self.net.eval()
        start = time.time()
        res = self.evaluator.run(a.eval_battles)
        self.net.train()
        self._after_eval(res, self.counters["steps"], time.time() - start,
                         lambda path, **extra: save(self.net, path, **extra), self.counters)

    def _after_eval(self, res: dict[str, float], steps: int, seconds: float, put, counters: dict,
                    note: str = "") -> bool:
        """Everything a finished evaluation does, synchronous or async: TensorBoard and the log line at the evaluated
        weights' `steps`, best_model.pt (+ --frozen-best) on a new best score, the anchor-drop guard. `put(path,
        **extra)` writes the evaluated weights; `counters`: the trainer counters they were taken at. Returns whether
        the score is a new best."""
        a = self.args
        for k, v in res.items():
            self.writer.add_scalar(f"eval/win_rate_{k}", v, steps)
        self.writer.add_scalar("eval/seconds", seconds, steps)
        score = res["anchor" if "anchor" in res else "heuristic"]
        self.log(f"[eval @ {steps:,}] " + "  ".join(f"{k}={v:.1%}" for k, v in res.items())
                 + f"  ({seconds:.0f} s){note}")
        best = score > self.counters["best"]
        if best:
            self.counters["best"] = score
            put(self.run_dir / "best_model.pt", counters=dict(counters, best=score), eval=res)
            if a.frozen_best:  # a new name each time (the trainer caches snapshot nets by name); older bests removed
                fdir = self.pool_dir / "frozen"
                fdir.mkdir(parents=True, exist_ok=True)
                name = f"best-{steps:011d}.pt"
                put(fdir / name)
                for old in fdir.glob("best-*.pt"):
                    if old.name != name:
                        old.unlink(missing_ok=True)
        # guard: the eval score fell more than --guard-anchor-drop below its best for 2 evals in a row -> stop
        if a.guard_anchor_drop > 0 and score < self.counters["best"] - a.guard_anchor_drop:
            self.guard_evals = getattr(self, "guard_evals", 0) + 1
            if self.guard_evals >= 2:
                self.trip_guard(f"eval score {score:.1%} < best {self.counters['best']:.1%} - "
                                f"{a.guard_anchor_drop:.0%} for 2 evals")
        else:
            self.guard_evals = 0
        return best

    def _eval_command(self, snap: Path, out: Path, steps: int) -> list[str]:
        a = self.args
        cmd = [sys.executable, "-m", "porygon.v2.train_eval", str(snap), "--format", a.format,
               "--port", str(parse_ports(a.ports)[0]), "--battles", str(a.eval_battles), "--steps", str(steps),
               "--out", str(out)]
        return cmd + (["--anchor", str(a.anchor)] if a.anchor is not None else [])

    def start_eval(self) -> None:
        """--eval-async, at an eval point: snapshot the live net (eval/snap_<steps>.pt) and start the eval subprocess
        on it, unless the previous one is still running (then this eval point is skipped)."""
        steps = self.counters["steps"]
        if self.stop.is_set():
            return
        if self.eval_job is not None:
            job = self.eval_job
            self.log(f"eval @ {steps:,} skipped: previous still running (eval @ {job.steps:,}, "
                     f"{time.time() - job.started:.0f} s)")
            return
        d = self.run_dir / "eval"
        snap, result = d / f"snap_{steps:011d}.pt", d / f"result_{steps:011d}.json"
        save(self.net, snap, counters=dict(self.counters))
        result.unlink(missing_ok=True)  # an earlier process may have evaluated these steps
        proc = launch(self._eval_command(snap, result, steps), d / "eval.log")
        self.eval_job = EvalJob(steps, snap, result, dict(self.counters), proc, d / "eval.log")
        self.log(f"eval @ {steps:,} started (async, pid {proc.pid})")

    def poll_eval(self) -> None:
        """--eval-async, every update (non-blocking; nothing to do without a running eval): once the subprocess has
        written its result, handle it as an eval of the snapshot's weights at the snapshot's steps; an exit without
        a result is logged with the tail of eval.log and leaves the best score and the guard alone. Snapshots that
        are not the current best are deleted."""
        job = self.eval_job
        if job is None:
            return
        rc = job.proc.poll()
        if rc is None and not job.result.exists():
            if self.args.eval_timeout > 0 and time.time() - job.started > self.args.eval_timeout:
                self.eval_job = None
                rc = stop_process(job.proc)
                job.snap.unlink(missing_ok=True)
                self.log(f"eval @ {job.steps:,} timed out after {time.time() - job.started:.0f} s (exit code {rc}); "
                         f"killed — the next eval point starts a new one; {job.log} ends:\n{tail(job.log)}")
            return  # still playing
        if rc is None:  # the result is written and the process exits right after it
            try:
                rc = job.proc.wait(5)
            except subprocess.TimeoutExpired:
                rc = stop_process(job.proc)
        self.eval_job = None
        try:
            out = json.loads(job.result.read_text())
            res, seconds = {k: float(v) for k, v in out["res"].items()}, float(out["seconds"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            job.snap.unlink(missing_ok=True)
            self.log(f"eval @ {job.steps:,} failed: exit code {rc}, no result ({e!r:.200}); "
                     f"{job.log} ends:\n{tail(job.log)}")
            return
        best = self._after_eval(res, job.steps, seconds, lambda path, **extra: copy_checkpoint(job.snap, path, **extra),
                                job.counters, note=f" (async, finished at {self.counters['steps']:,})")
        if best:  # the previous best's snapshot (and any left by an earlier process) is no longer needed
            for f in job.snap.parent.glob("snap_*.pt"):
                if f != job.snap:
                    f.unlink(missing_ok=True)
        else:
            job.snap.unlink(missing_ok=True)

    def stop_eval(self, timeout: float = 10.0) -> None:
        """Trainer exit: terminate a running eval subprocess (SIGTERM, up to `timeout` s, then SIGKILL) and delete its
        snapshot. Never raises."""
        job, self.eval_job = self.eval_job, None
        if job is None:
            return
        try:
            rc = stop_process(job.proc, timeout)
            job.snap.unlink(missing_ok=True)
            self.log(f"eval @ {job.steps:,} stopped at exit (exit code {rc})")
        except Exception as e:
            self.log(f"eval @ {job.steps:,}: could not stop the subprocess ({e!r})")

    def _reload_distill(self) -> None:
        """--distill-reload-every: pick up added / rewritten shards; on any error keep training on the current set."""
        try:
            new = self.distill.reload()
        except Exception as e:  # e.g. a shard being written right now: retried at the next reload
            self.log(f"distill: reload failed ({e!r:.300}); keeping the current set")
            return
        if new is not None:
            self.distill = new
            self.log(f"distill: reloaded {new.summary()}")

    def _pool_alarm(self, eps: list[dict], st: dict) -> None:
        """Pool health after each update (main loop), into `st`: the realised opponent mix of the rollout's finished
        games (mix/selfplay, mix/frozen, mix/snapshot, mix/baseline = heuristic + max_power; perf/pool_episode_share)
        and the pool-net loads of `_opp_act` since the last update (pool/load_{attempts,failures}_{frozen,snapshot};
        lineage totals in the counters; each failing net logged once). Then the pool guard, one trip with a "pool: "
        reason: no pool game finished for --pool-alarm-updates updates in a row although pool battles are configured
        (also writes POOL_ALARM), any frozen-net load failure, or > 5% failed snapshot loads (>= 20 attempts)."""
        a = self.args
        n_pool = sum(e["kind"] == "pool" for e in eps)
        st["perf/pool_episode_share"] = n_pool / max(len(eps), 1)
        if eps:
            mix = ["selfplay" if e["kind"] != "pool" else e["opp"] if e["opp"] in ("frozen", "snapshot")
                   else "baseline" for e in eps]
            for k in ("selfplay", "frozen", "snapshot", "baseline"):
                st[f"mix/{k}"] = mix.count(k) / len(eps)
        loads, msgs = self.pool_loads.take()
        st.update(loads)
        for k, v in loads.items():
            key = k.replace("/", "_")
            self.counters[key] = self.counters.get(key, 0) + int(v)
        for msg in msgs:
            self.log(msg)
        why = []
        if loads["pool/load_failures_frozen"] > 0:
            why.append(f"{loads['pool/load_failures_frozen']:.0f} frozen-net load failure(s) in the last update")
        tried, failed = loads["pool/load_attempts_snapshot"], loads["pool/load_failures_snapshot"]
        if tried >= 20 and failed / tried > 0.05:
            why.append(f"{failed:.0f} of {tried:.0f} snapshot loads failed in the last update (> 5%)")
        expected = a.pool_battles > 0 and getattr(self.workers, "pool_per_worker", 1) != 0
        if expected and a.pool_alarm_updates > 0:
            self.pool_dry = 0 if n_pool else getattr(self, "pool_dry", 0) + 1
            if self.pool_dry >= a.pool_alarm_updates:
                msg = (f"POOL ALARM: {a.pool_battles} pool battles configured but no pool episode finished in "
                       f"{self.pool_dry} updates in a row (worker restarts {self.workers.restarts})")
                self.log(msg)
                (self.run_dir / "POOL_ALARM").write_text(msg + "\n")
                why.append(msg)
        if why:
            self.trip_guard("pool: " + "; ".join(why))

    def trip_guard(self, why: str) -> None:
        """Stop training (final_model.pt is saved as usual) and leave GUARD_TRIPPED in the run dir."""
        self.log(f"GUARD TRIPPED: {why}; stopping (best model: best_model.pt)")
        (self.run_dir / "GUARD_TRIPPED").write_text(why + "\n")
        self.stop.set()

    def run(self) -> None:
        a, c = self.args, self.counters
        overlap = not a.no_overlap
        target = self.collector.run if getattr(self, "actor_buffers", None) is None else self._collect_on_actor_stream
        thread = threading.Thread(target=target, args=(a.rollout, self.requests, self.rollouts, self.stop),
                                  daemon=True)
        thread.start()
        deadline = time.time() + a.max_hours * 3600 if a.max_hours else None
        nxt = lambda every: (c["steps"] // every + 1) * every  # noqa: E731
        next_ckpt, next_snap, next_eval = nxt(a.checkpoint_every), nxt(a.snapshot_every), nxt(a.eval_every)
        updates_here = 0
        last = time.time()
        if overlap:
            self.requests.release()
        try:
            while c["steps"] < a.total_steps and not self.stop.is_set():
                if deadline and time.time() > deadline:
                    self.log("max hours reached")
                    break
                if a.max_updates is not None and updates_here >= a.max_updates:
                    break
                self._refresh_actor()
                if not overlap:
                    self.requests.release()
                item = self._get_rollout()
                if item is None:
                    break
                if overlap:
                    self.requests.release()
                segs, eps, collect_s = item
                vc = self._take_versions()  # policy lag: learner updates done minus the version that acted
                lag_n = sum(vc.values())
                lag_mean = sum((c["updates"] - v) * k for v, k in vc.items()) / lag_n if lag_n else float("nan")
                lag_max = max((c["updates"] - v for v in vc), default=-1)
                now = time.time()
                wall, last = now - last, now
                t0 = time.time()
                data = build_rollout(segs, self.packer, a.gamma, a.gae_lambda)
                n = len(data["action"])
                c["steps"] += n
                c["games"] += len(eps)
                lr = lr_at(a, c["steps"])
                for g in self.opt.param_groups:
                    g["lr"] = lr
                measure = a.grad_ratio_every > 0 and c["updates"] % a.grad_ratio_every == 0
                beta = (a.kl_anchor_coef * max(0.0, 1.0 - c["steps"] / max(1, a.kl_anchor_decay_steps))
                        if self.anchor is not None else 0.0)
                use_anchor = self.anchor is not None and (beta > 0 or measure)  # after the decay: monitor only
                heads_only = c["steps"] < a.critic_warmup_steps
                st = ppo_update(self.net, self.opt, data, a, self.device, self.amp, measure,
                                anchor=self.anchor if use_anchor else None, anchor_coef=beta, heads_only=heads_only,
                                distill=self.distill)
                if self.ema is not None:
                    self.ema.update(self.net)
                c["updates"] += 1
                updates_here += 1
                st.update(self._episode_stats(eps, wall))
                self._pool_alarm(eps, st)  # opponent mix, pool-net loads, pool guard
                st.update(getattr(self.collector, "last_stats", {}))  # fastenv: stale / drained shares, drain time
                st["diag/policy_lag_mean"], st["diag/policy_lag_max"] = lag_mean, float(lag_max)
                st["perf/steps_per_s"] = n / max(wall, 1e-9)
                st["perf/collect_steps_per_s"] = n / max(collect_s, 1e-9)  # while collecting (overlaps updates)
                st["perf/update_s"] = time.time() - t0
                st["perf/lr"] = lr
                st["perf/worker_restarts"] = self.workers.restarts
                st["perf/opp_target_frac"] = float((data["aux_opp_target"].sum(-1) > 0).mean())
                st["perf/forward_batches"] = self.collector.forward_batches
                st["perf/snapshot_decisions"] = self.collector.opp_decisions
                for k, v in st.items():
                    self.writer.add_scalar(k, v, c["steps"])
                self.log(self._line(n, st))
                # guard: entropy above --guard-entropy-max (or --guard-entropy-rel x the first 5 updates' mean) for 5
                # updates in a row (drift towards uniform) -> stop
                if a.guard_entropy_rel > 0 and not a.guard_entropy_max and updates_here <= 5:
                    self.ent_first = getattr(self, "ent_first", []) + [st.get("loss/ent", 0.0)]
                    if updates_here == 5:
                        self.ent_cap = a.guard_entropy_rel * float(np.mean(self.ent_first))
                        self.log(f"entropy guard: cap {self.ent_cap:.3f} = {a.guard_entropy_rel} x "
                                 f"{np.mean(self.ent_first):.3f} (first 5 updates)")
                cap = a.guard_entropy_max or getattr(self, "ent_cap", 0.0)
                if cap > 0 and updates_here > 5:
                    self.ent_bad = getattr(self, "ent_bad", 0) + 1 if st.get("loss/ent", 0.0) > cap else 0
                    if self.ent_bad >= 5:
                        self.trip_guard(f"entropy > {cap:.3f} for 5 updates")
                # guard: the PPO surrogate (before each minibatch's own step) positive on average over the last N
                # updates = the updates no longer follow the advantages (R1's failure signature)
                if a.guard_pg_mean > 0 and not heads_only and "loss/pg" in st:
                    self.pg_hist = (getattr(self, "pg_hist", []) + [st["loss/pg"]])[-a.guard_pg_mean:]
                    if len(self.pg_hist) == a.guard_pg_mean and float(np.mean(self.pg_hist)) > 0:
                        self.trip_guard(f"pg mean {np.mean(self.pg_hist):+.5f} > 0 over {a.guard_pg_mean} updates")
                # collapse guards (two-sided entropy control): entropy below --guard-entropy-min, or the share of
                # near-deterministic choices above --guard-ptop-max, for 5 updates in a row; not during the warm-up
                if updates_here > 5 and not heads_only:
                    if a.guard_entropy_min > 0:
                        low = st.get("loss/ent", 1e9) < a.guard_entropy_min
                        self.ent_low = getattr(self, "ent_low", 0) + 1 if low else 0
                        if self.ent_low >= 5:
                            self.trip_guard(f"collapse: entropy < {a.guard_entropy_min} for 5 updates")
                    if a.guard_ptop_max > 0:
                        sharp = st.get("loss/ptop", 0.0) > a.guard_ptop_max
                        self.ptop_bad = getattr(self, "ptop_bad", 0) + 1 if sharp else 0
                        if self.ptop_bad >= 5:
                            self.trip_guard(f"collapse: P(top >= 0.9) > {a.guard_ptop_max} for 5 updates")
                self.poll_eval()  # --eval-async: a finished eval subprocess (no-op otherwise)
                if c["steps"] >= next_snap:
                    next_snap = nxt(a.snapshot_every)
                    save(self.net, self.pool_dir / f"{c['steps']:011d}.pt")
                if c["steps"] >= next_ckpt:
                    next_ckpt = nxt(a.checkpoint_every)
                    self.checkpoint(self.run_dir / "checkpoints" / f"ckpt_{c['steps']:011d}.pt")
                if a.eval_battles > 0 and c["steps"] >= next_eval:
                    next_eval = nxt(a.eval_every)
                    if a.eval_async:
                        self.start_eval()
                    else:
                        self.evaluate()
                if (self.distill is not None and a.distill_reload_every > 0
                        and updates_here % a.distill_reload_every == 0):
                    self._reload_distill()
        finally:
            self.stop.set()
            thread.join(30)  # the collector notices `stop` within ~1 s unless a forward is running
            try:
                self.poll_eval()  # --eval-async: a result written since the last update still counts
            except Exception as e:
                self.log(f"eval: handling the last result failed ({e!r})")
            self.checkpoint(self.run_dir / "final_model.pt")
            self.log(f"saved final_model.pt at {c['steps']:,} steps")
            self.stop_eval()  # --eval-async: terminate a running eval subprocess
            self.workers.close()
            self.writer.close()

    @staticmethod
    def _episode_stats(eps: list[dict], wall: float) -> dict[str, float]:
        st = {"perf/games_per_s": len(eps) / max(wall, 1e-9)}
        if not eps:
            return st
        st["episode/turns"] = float(np.mean([e["turns"] for e in eps]))
        st["episode/decisions"] = float(np.mean([d for e in eps for d in e["decisions"]]))
        for label in ("heuristic", "max_power", "snapshot", "frozen"):
            res = [e["result"][0] for e in eps if e["kind"] == "pool" and e["opp"] == label]
            if res:
                st[f"pool/win_rate_{label}"] = float(np.mean([r == 1 for r in res]))
                st[f"pool/games_{label}"] = float(len(res))
        return st

    def _line(self, n: int, st: dict) -> str:
        c = self.counters
        g = lambda k, f="{:.3f}": f.format(st[k]) if k in st else "-"  # noqa: E731
        aux = " ".join(f"{k.split('/')[1]}={st[k]:.3f}" for k in sorted(st) if k.startswith("aux/"))
        plf = (f"{st['pool/load_failures_frozen'] + st['pool/load_failures_snapshot']:.0f}"
               if "pool/load_failures_frozen" in st else "-")
        mix = ("/".join(f"{st['mix/' + k]:.2f}" for k in ("selfplay", "frozen", "snapshot", "baseline"))
               if "mix/selfplay" in st else "-")
        return (f"upd {c['updates']} steps {c['steps']:,} (+{n}) sps {st['perf/steps_per_s']:.0f} "
                f"collect_sps {st['perf/collect_steps_per_s']:.0f} "
                f"games/s {st['perf/games_per_s']:.2f} lr {st['perf/lr']:.2e} pg {g('loss/pg', '{:+.4f}')} "
                f"vf {g('loss/vf', '{:.4f}')} ent {g('loss/ent')} za {g('loss/za')} kl {g('loss/kl', '{:.4f}')} "
                f"clip {g('loss/clip')} ev {g('diag/explained_variance')} gn {g('diag/grad_norm')} "
                f"ratio {g('diag/trunk_grad_ratio')} cos_pg {g('diag/grad_cos_policy')} "
                f"cos_vf {g('diag/grad_cos_value')} r0 {g('diag/first_ratio_mean')}±{g('diag/first_ratio_std')} "
                f"clip0 {g('diag/first_clip')} beh_off {g('diag/behaviour_off_clip')} lag {g('diag/policy_lag_mean', '{:.2f}')}/{g('diag/policy_lag_max', '{:.0f}')} "
                + (f"stale {st['perf/fastenv_stale_frac']:.3f} " if "perf/fastenv_stale_frac" in st else "")
                + f"len {g('episode/decisions', '{:.1f}')} kla {g('loss/kl_anchor', '{:.4f}')} "
                f"beta {g('perf/kl_anchor_coef', '{:.3f}')} warm {g('perf/heads_only', '{:.0f}')} "
                f"wr_frozen {g('pool/win_rate_frozen', '{:.2f}')} ptop {g('loss/ptop')} plf {plf} mix {mix} "
                f"dst {g('loss/distill')} dval {g('diag/distill_val_ce')} "
                f"upd_s {st['perf/update_s']:.1f} | {aux}")


def main(argv=None):
    args = parse_args(argv)
    trainer = Trainer(args)

    def handler(signum, frame):
        if trainer.stop.is_set():
            if signum == signal.SIGINT:
                raise KeyboardInterrupt  # second Ctrl-C: give up waiting
            return  # repeated SIGTERM (Slurm + the job script): keep saving
        trainer.log(f"signal {signum}: saving and exiting after the current step")
        trainer.stop.set()
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    trainer.run()


if __name__ == "__main__":
    main()
