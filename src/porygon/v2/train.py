"""Self-play PPO for model v2 (custom loop; SB3 cannot carry the dict observation or the aux targets).

    scripts/start_servers.sh 8
    python -m porygon.v2.train --format gen9randombattle --ports 8000-8007 --n-battles 48 --pool-battles 24 \
        --run-dir runs/v2-gen9 --total-steps 300_000_000

Data: battle worker processes (`porygon.v2.workers`): self-play battles (both sides are samples) and pool battles
(the learner vs a frozen v2 snapshot from --pool-dir or a baseline bot). Collection runs in a background thread with
a copy of the network refreshed before every rollout; by default it keeps running during the PPO update (data at most
one update old; the stored log-probs keep the importance ratio exact), `--no-overlap` pauses it instead.

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
updates.

lr: warmup–stable–decay on absolute env steps: linear warmup over --warmup-steps, constant --lr, and (only with
--cooldown-start) cosine down to --final-lr-frac · lr at --total-steps. Resuming keeps the schedule.

Outputs in --run-dir: train.log (one line per update), tb/ (TensorBoard), checkpoints/ckpt_<steps>.pt every
--checkpoint-every steps, final_model.pt (on exit, SIGINT / SIGTERM included), best_model.pt (best eval score:
vs --anchor if given, else vs SimpleHeuristicsPlayer), pool snapshots every --snapshot-every steps.
"""

import argparse
import asyncio
import copy
import json
import math
import os
import queue
import shutil
import signal
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
from porygon.v2.spec import VOCAB_SIZES
from porygon.v2.workers import (
    ACT, DONE, LOGP, NEXT_V, OBS, OPP_C, OPP_T, REW, SEED, VAL, BattleWorkers, Collector, Packer, WorkerSpec, gen_of,
)


def parse_ports(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = map(int, spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(p) for p in spec.split(",")]


def wsd_lr(step: int, lr: float, warmup: int, cooldown_start: int | None, total: int, final_frac: float) -> float:
    f = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
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
    return {"pg": pg, "ent": ent, "za": za, "vf": vf, "kl": kl, "clip": clip, "ratio_mean": r_mean, "ratio_std": r_std,
            "kla": kla}


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


HEADS_ONLY = ("value_head.", "aux_")  # what trains during --critic-warmup-steps


def ppo_update(net: Net, opt, data: dict, args, device, amp: bool, measure_ratio: bool, anchor: Net | None = None,
               anchor_coef: float = 0.0, heads_only: bool = False) -> dict[str, float]:
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
                for k in ("pg", "ent", "za", "vf", "kl", "clip", "kla"):
                    sums[k] = sums.get(k, 0.0) + w * L[k].detach()
                sums["aux"] = sums.get("aux", 0.0) + w * aux.detach()
                sums["total"] = sums.get("total", 0.0) + w * total.detach()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            for k in ("pg", "ent", "za", "vf", "kl", "clip"):
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
    a("--guard-entropy-rel", type=float, default=0.0,
      help="entropy guard at this multiple of the mean entropy of this process's first 5 updates (if no max)")
    a("--kl-anchor", type=Path, help="frozen v2 net: add beta(t) * KL(anchor || pi) to the loss")
    a("--kl-anchor-coef", type=float, default=0.0, help="beta at step 0 (linear decay to 0)")
    a("--kl-anchor-decay-steps", type=int, default=50_000_000, help="env steps (lineage counter) at which beta = 0")
    a("--critic-warmup-steps", type=int, default=0, help="until this many env steps only value + aux heads train")
    a("--frozen", type=Path, nargs="*", default=[], help="frozen opponents for pool battles (copied to pool/frozen)")
    a("--p-frozen", type=float, default=0.0, help="share of pool battles vs a frozen opponent")
    a("--frozen-best", action="store_true", help="each new best_model.pt also becomes a frozen opponent")
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
    a("--weight-decay", type=float, default=0.01)
    a("--adv-clip", type=float, default=0.0, help="clip per-sample normalised advantages to ±C (0 = off)")
    a("--ema-decay", type=float, default=0.999,
      help="EMA of the weights per update (0 = off); written to ema_model.pt with every checkpoint and at the end")
    a("--shaping", type=float, nargs=3, metavar=("FAINTED", "HP", "STATUS"), default=(0.0, 0.0, 0.0),
      help="potential-based shaping weights (v1 used 0.05 0.05 0); default sparse ±1 only")
    a("--eval-every", type=int, default=2_500_000)
    a("--eval-battles", type=int, default=200, help="per opponent; 0 disables evaluation")
    a("--anchor", type=Path, help="also evaluate vs this net (v2 .pt, or v1 .zip / pool .pt)")
    a("--checkpoint-every", type=int, default=5_000_000)
    a("--run-dir", type=Path, default=Path("runs") / ("v2-" + time.strftime("%Y%m%d-%H%M%S")))
    a("--resume", type=Path, help="trainer checkpoint (.pt) to continue from (model, optimizer, step counters)")
    a("--reset-optimizer", action="store_true", help="with --resume: fresh optimizer state")
    a("--max-hours", type=float, help="stop (and save) after this much wall-clock time")
    a("--max-updates", type=int, help="stop after this many updates in this process (tests)")
    a("--no-overlap", action="store_true", help="pause battles during the PPO update")
    a("--grad-ratio-every", type=int, default=10,
      help="every N updates: trunk value/policy gradient ratio and the minibatch-to-minibatch gradient cosines")
    a("--stall-seconds", type=float, default=900.0, help="restart a battle worker silent this long")
    a("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a("--threads", type=int, default=1, help="torch CPU threads in the trainer process")
    a("--no-bf16", action="store_true", help="no bf16 autocast on CUDA")
    a("--compile", action=argparse.BooleanOptionalAction, default=None,
      help="torch.compile the damage layer (default: on CUDA); falls back to eager if compilation fails")
    a("--snapshot-cache", type=int, default=8, help="pool snapshot opponents kept on the device")
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
    return p.parse_args(argv)


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
        self.ema = EMA(self.net, args.ema_decay) if args.ema_decay > 0 else None
        if self.ema is not None and ck is not None and "ema" in ck:
            self.ema.load_state_dict(ck["ema"])
        self.packer = Packer(self.gen)
        self.compile = (self.device.type == "cuda") if args.compile is None else args.compile
        if self.compile:
            self.net.compile_damage()
            self.actor.compile_damage()
        self.snapshots: dict[str, Net] = {}  # pool snapshot opponents on the device (LRU)
        self.collector = Collector(self.workers, self.packer, self._act, self._opp_act,
                                   id_dropout=self.net.cfg["id_dropout"] > 0, stall_seconds=args.stall_seconds,
                                   seed=args.seed)
        self.requests = threading.Semaphore(0)
        self.rollouts: queue.Queue = queue.Queue()
        self.evaluator = None
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(str(self.run_dir / "tb"))
        n_params = sum(p.numel() for p in self.net.parameters())
        self.log(f"v2 trainer: {args.format} gen{self.gen}, {args.n_battles} self-play + {args.pool_battles} pool "
                 f"battles on ports {ports}, params={n_params:,}, cfg={self.net.cfg}, device={self.device}, "
                 f"amp={self.amp}, overlap={not args.no_overlap}, resume={args.resume}, steps={self.counters['steps']:,}")

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.log_file.write(line + "\n")

    @torch.no_grad()
    def _act(self, batch: dict):
        t = batch_to_tensors(batch, self.device, keys=model_keys(self.gen) + ["drop_seed"])
        with self.actor_lock, torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            out = self.actor(t, aux=False)
            v = getattr(self, "actor_version", 0)
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
        be loaded)."""
        net = self.snapshots.pop(name, None)
        if net is None:
            try:
                net = load(self.pool_dir / name, self.device)
                assert net.gen == self.gen, f"gen {net.gen} snapshot"
                if self.compile:  # same parameter-free layer: reuse the actor's compiled function (no recompiles)
                    object.__setattr__(net, "_damage_compiled", getattr(self.actor, "_damage_compiled", None))
            except Exception as e:
                print(f"pool snapshot {name}: {e!r}; playing uniformly", file=sys.stderr)
                net = None
            if len(self.snapshots) >= self.args.snapshot_cache:
                self.snapshots.pop(next(iter(self.snapshots)))
        mask = torch.from_numpy(batch["action_mask"]) > 0
        if net is None:
            logits = torch.where(mask, 0.0, torch.finfo(torch.float32).min)
        else:
            self.snapshots[name] = net  # most recently used last
            t = batch_to_tensors(batch, self.device, keys=model_keys(self.gen) + ["drop_seed"])
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                logits = net(t, aux=False)["logits"].float().cpu()
        probs = torch.softmax(logits, -1)
        probs = torch.where(mask.any(-1, keepdim=True), probs, torch.full_like(probs, 1.0 / probs.shape[-1]))
        return torch.multinomial(probs, 1).squeeze(1).numpy()

    def _refresh_actor(self) -> None:
        with self.actor_lock:
            self.actor.load_state_dict(self.net.state_dict())
            self.actor_version = self.counters["updates"]  # the acting weights = the learner after this many updates

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
        steps = self.counters["steps"]
        for k, v in res.items():
            self.writer.add_scalar(f"eval/win_rate_{k}", v, steps)
        self.writer.add_scalar("eval/seconds", time.time() - start, steps)
        score = res["anchor" if "anchor" in res else "heuristic"]
        self.log(f"[eval @ {steps:,}] " + "  ".join(f"{k}={v:.1%}" for k, v in res.items())
                 + f"  ({time.time() - start:.0f} s)")
        if score > self.counters["best"]:
            self.counters["best"] = score
            save(self.net, self.run_dir / "best_model.pt", counters=dict(self.counters), eval=res)
            if a.frozen_best:  # a new name each time (the trainer caches snapshot nets by name); older bests removed
                fdir = self.pool_dir / "frozen"
                fdir.mkdir(parents=True, exist_ok=True)
                name = f"best-{steps:011d}.pt"
                save(self.net, fdir / name)
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

    def trip_guard(self, why: str) -> None:
        """Stop training (final_model.pt is saved as usual) and leave GUARD_TRIPPED in the run dir."""
        self.log(f"GUARD TRIPPED: {why}; stopping (best model: best_model.pt)")
        (self.run_dir / "GUARD_TRIPPED").write_text(why + "\n")
        self.stop.set()

    def run(self) -> None:
        a, c = self.args, self.counters
        overlap = not a.no_overlap
        thread = threading.Thread(target=self.collector.run, args=(a.rollout, self.requests, self.rollouts, self.stop),
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
                lr = wsd_lr(c["steps"], a.lr, a.warmup_steps, a.cooldown_start, a.total_steps, a.final_lr_frac)
                for g in self.opt.param_groups:
                    g["lr"] = lr
                measure = a.grad_ratio_every > 0 and c["updates"] % a.grad_ratio_every == 0
                beta = (a.kl_anchor_coef * max(0.0, 1.0 - c["steps"] / max(1, a.kl_anchor_decay_steps))
                        if self.anchor is not None else 0.0)
                use_anchor = self.anchor is not None and (beta > 0 or measure)  # after the decay: monitor only
                heads_only = c["steps"] < a.critic_warmup_steps
                st = ppo_update(self.net, self.opt, data, a, self.device, self.amp, measure,
                                anchor=self.anchor if use_anchor else None, anchor_coef=beta, heads_only=heads_only)
                if self.ema is not None:
                    self.ema.update(self.net)
                c["updates"] += 1
                updates_here += 1
                st.update(self._episode_stats(eps, wall))
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
                if c["steps"] >= next_snap:
                    next_snap = nxt(a.snapshot_every)
                    save(self.net, self.pool_dir / f"{c['steps']:011d}.pt")
                if c["steps"] >= next_ckpt:
                    next_ckpt = nxt(a.checkpoint_every)
                    self.checkpoint(self.run_dir / "checkpoints" / f"ckpt_{c['steps']:011d}.pt")
                if a.eval_battles > 0 and c["steps"] >= next_eval:
                    next_eval = nxt(a.eval_every)
                    self.evaluate()
        finally:
            self.stop.set()
            thread.join(30)  # the collector notices `stop` within ~1 s unless a forward is running
            self.checkpoint(self.run_dir / "final_model.pt")
            self.log(f"saved final_model.pt at {c['steps']:,} steps")
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
        return (f"upd {c['updates']} steps {c['steps']:,} (+{n}) sps {st['perf/steps_per_s']:.0f} "
                f"collect_sps {st['perf/collect_steps_per_s']:.0f} "
                f"games/s {st['perf/games_per_s']:.2f} lr {st['perf/lr']:.2e} pg {g('loss/pg', '{:+.4f}')} "
                f"vf {g('loss/vf', '{:.4f}')} ent {g('loss/ent')} za {g('loss/za')} kl {g('loss/kl', '{:.4f}')} "
                f"clip {g('loss/clip')} ev {g('diag/explained_variance')} gn {g('diag/grad_norm')} "
                f"ratio {g('diag/trunk_grad_ratio')} cos_pg {g('diag/grad_cos_policy')} "
                f"cos_vf {g('diag/grad_cos_value')} r0 {g('diag/first_ratio_mean')}±{g('diag/first_ratio_std')} "
                f"clip0 {g('diag/first_clip')} beh_off {g('diag/behaviour_off_clip')} lag {g('diag/policy_lag_mean', '{:.2f}')}/{g('diag/policy_lag_max', '{:.0f}')} "
                f"len {g('episode/decisions', '{:.1f}')} kla {g('loss/kl_anchor', '{:.4f}')} "
                f"beta {g('perf/kl_anchor_coef', '{:.3f}')} warm {g('perf/heads_only', '{:.0f}')} "
                f"wr_frozen {g('pool/win_rate_frozen', '{:.2f}')} "
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
