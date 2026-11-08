"""Supervised pretraining of model v2 on logged games (search labels, human replays): the dense policy target that
self-play RL lacks (wsp: cross-entropy to the search's choice carries ~40x the consistent gradient of the PPO loss).

    python -m porygon.v2.pretrain --shards 'DIR/shard-*.npz' --gen 9 --out runs/pre9 [--init CKPT] [--epochs 4]

Shards (CONTRACTS.md, "v2 supervised shards"): `np.savez` files with `obs/<field>` (spec.fields(gen), spec dtypes),
`aux/<field>` (spec.aux_fields(gen); `aux_opp_target` = one-hot of the opponent's actual action or zeros),
`target` f32 [N, A] (policy target over poke-env actions: the search's visit distribution or a one-hot of the
human action; all zero = no policy target), `z` f32 [N] (final result for the side deciding: +1 / -1 / 0), `game`
i64 [N] (game id: train / validation split by game, never by decision), optional `weight` f32 [N].

Loss: policy CE(target, softmax over legal actions) + value_coef * BCE(win prob, (z + 1) / 2) + aux_coef * the RL
trainer's aux losses (`porygon.v2.train.aux_losses`). The result is a normal v2 checkpoint: load it with
`porygon.v2.model.load`, or start RL from it with `porygon.v2.train --resume CKPT --reset-optimizer`.
"""

import argparse
import glob
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from porygon.v2.model import Net, batch_to_tensors, load, save
from porygon.v2.spec import SPEC_VERSION, aux_fields, fields
from porygon.v2.train import Meter, aux_losses


def load_shard(path: str, gen: int) -> dict[str, np.ndarray]:
    z = np.load(path)
    out = {}
    for k in fields(gen):
        out[k] = z[f"obs/{k}"]
    for k, (shape, dtype) in aux_fields(gen).items():
        key = f"aux/{k}"
        n = len(out["action_mask"])
        out[k] = z[key] if key in z.files else np.full((n, *shape), -1 if k != "aux_opp_target" else 0, dtype)
    out["target"] = z["target"].astype(np.float32)
    out["z"] = z["z"].astype(np.float32)
    out["game"] = z["game"].astype(np.int64)
    out["weight"] = z["weight"].astype(np.float32) if "weight" in z.files else np.ones(len(out["z"]), np.float32)
    return out


def is_val(game: np.ndarray, frac: float) -> np.ndarray:
    """Deterministic split by game id (crc32 of the id), so every run and shard agrees."""
    h = np.array([zlib.crc32(int(g).to_bytes(8, "little", signed=True)) for g in game], dtype=np.uint64)
    return (h % 10_000) < frac * 10_000


def losses(net: Net, mb: dict, args, m: Meter | None) -> torch.Tensor:
    out = net(mb, aux=args.aux_coef > 0)
    mask = mb["action_mask"] > 0
    logp = torch.log_softmax(out["logits"].float(), -1)
    tgt = mb["target"] * mask
    has = tgt.sum(-1) > 0
    w = mb["weight"]
    pol = torch.zeros((), device=logp.device)
    if has.any():
        t = tgt[has] / tgt[has].sum(-1, keepdim=True)
        ce = -(t * torch.where(mask[has], logp[has], torch.zeros_like(logp[has]))).sum(-1)
        pol = (ce * w[has]).sum() / w[has].sum()
    val = (F.binary_cross_entropy_with_logits(out["win_logit"].float(), ((mb["z"] + 1) / 2).clamp(0, 1),
                                              reduction="none") * w).sum() / w.sum()
    loss = pol + args.value_coef * val
    if args.aux_coef > 0:
        loss = loss + args.aux_coef * aux_losses(out, mb, m)
    if m is not None:
        m.add("policy_ce", pol.item())
        m.add("value_bce", val.item())
        if has.any():
            agree = (out["logits"][has].argmax(-1) == tgt[has].argmax(-1)).float()
            m.add_ratio("top1_vs_target_argmax", agree.sum().item(), agree.numel())
    return loss


def batches(data: dict, idx: np.ndarray, bs: int, device, rng: np.random.Generator | None):
    if rng is not None:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        mb = batch_to_tensors({k: v[j] for k, v in data.items() if k != "game"}, device)
        mb["opp_cls"] = torch.full((len(j),), -1, dtype=torch.long, device=device)  # aux metric: no labels here
        if rng is not None:  # id dropout during training (model hashes a per-sample seed; 0 = off)
            mb["drop_seed"] = torch.from_numpy(rng.integers(1, 2**31 - 1, len(j))).to(device)
        yield mb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True, help="glob of shard npz files")
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--init", type=Path, help="start from this v2 checkpoint (else a fresh Net)")
    p.add_argument("--model-json", default="{}", help="Net config overrides for a fresh Net")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000, help="optimizer steps")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--aux-coef", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2000, help="optimizer steps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    paths = sorted(glob.glob(args.shards))
    assert paths, f"no shards match {args.shards}"
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "pretrain.log", "a", buffering=1)

    def emit(rec: dict) -> None:
        line = json.dumps(rec)
        print(line, flush=True)
        log.write(line + "\n")

    # every shard in memory once (shards are compressed npz; a v2 decision is ~12 KB uncompressed)
    parts = [load_shard(q, args.gen) for q in paths]
    data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    del parts
    val = is_val(data["game"], args.val_frac)
    train_idx, val_idx = np.flatnonzero(~val), np.flatnonzero(val)
    emit({"event": "data", "shards": len(paths), "decisions": int(len(val)), "train": int(len(train_idx)),
          "val": int(len(val_idx)), "games": int(len(np.unique(data["game"]))), "spec": SPEC_VERSION})

    net = load(args.init, device) if args.init else Net(args.gen, **json.loads(args.model_json)).to(device)
    assert net.gen == args.gen, f"checkpoint is gen {net.gen}"
    net.train()
    opt = torch.optim.AdamW(net.param_groups(args.weight_decay), lr=args.lr, betas=(0.9, 0.999))
    total = int(math.ceil(len(train_idx) / args.batch_size) * args.epochs)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        x = (step - args.warmup) / max(total - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(x, 1.0))))

    @torch.no_grad()
    def evaluate(step: int) -> float:
        net.eval()
        m = Meter()
        for mb in batches(data, val_idx, 2048, device, None):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                losses(net, mb, args, m)
        net.train()
        res = m.means()
        emit({"event": "val", "step": step, **{k: round(v, 5) for k, v in res.items()}})
        return res.get("policy_ce", 0.0) + args.value_coef * res.get("value_bce", 0.0)

    step, best, start = 0, float("inf"), time.time()
    m = Meter()
    while step < total:
        for mb in batches(data, train_idx, args.batch_size, device, rng):
            if step >= total:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = losses(net, mb, args, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()
            m.add("grad_norm", gn.item())
            step += 1
            if step % 200 == 0:
                emit({"event": "train", "step": step, "lr": lr_at(step), "samples_per_s":
                      round(step * args.batch_size / (time.time() - start), 1),
                      **{k: round(v, 5) for k, v in m.means().items()}})
                m = Meter()
            if step % args.eval_every == 0 or step == total:
                score = evaluate(step)
                save(net, args.out / "last.pt", pretrain_step=step)
                if score < best:
                    best = score
                    save(net, args.out / "best.pt", pretrain_step=step, val_score=score)
    save(net, args.out / "final.pt", pretrain_step=step)


if __name__ == "__main__":
    main()
