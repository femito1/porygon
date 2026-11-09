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

Options added for the value refit and search-label arms (2026-11-09):
- `--mix GLOB:WEIGHT` (repeatable): more shard sets (e.g. search labels) with a per-sample weight multiplier; every
  set is split by game the same way and reported separately on validation (`val/<name>/...`); weight 0 = report only.
- `--value-sample one-per-game`: the value loss sees one random position per game per epoch (AlphaGo's fix for a value
  net that memorises game outcomes: all ~47 decisions of a game share one label); `all` (default) = every position.
- `--train-only PREFIXES` / `--reinit PREFIXES` (comma lists of parameter-name prefixes, e.g. `value_head`): freeze
  everything else / re-initialise those modules first (refit a head on a fixed trunk; the policy stays bit-identical).
- `--select total|value|policy`: what best.pt minimises on the primary set's validation split.
- `--no-bf16`: fp32 (+TF32 matmuls) instead of bf16 autocast.
- `--stream-gb G` (2026-11-09): streaming loader for sets that do not fit in memory (a v2 decision is ~12 KB in
  RAM, ~140 B compressed). Only a row index (shard, row, game) and the validation rows stay in memory; every epoch
  the training rows are permuted and cut into P = ceil(train bytes / G GiB) equal parts, and a background thread
  assembles each part (decompresses every shard, gathers the part's rows in permutation order) while the GPU trains
  on the previous one: each part is a uniform random subset of the training rows and every row is seen exactly once
  per epoch, as with the in-memory loader (whose exact order it does not reproduce). Memory ≈ 2 parts + validation.
  Weight-0 sets are never read beyond their validation rows. 0 (default) = everything in memory, as before.
"""

import argparse
import glob
import json
import math
import queue
import threading
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


def load_sets(primary: str, mix: list[str], gen: int) -> tuple[dict[str, np.ndarray], list[str], list[int]]:
    """Primary glob + `GLOB:WEIGHT` mixes -> concatenated arrays with `src` (set index) and weights scaled per set."""
    specs = [(primary, 1.0)] + [(m.rsplit(":", 1)[0], float(m.rsplit(":", 1)[1])) for m in mix]
    parts, names, counts = [], [], []
    for i, (g, wt) in enumerate(specs):
        paths = sorted(glob.glob(g))
        assert paths, f"no shards match {g}"
        for q in paths:
            d = load_shard(q, gen)
            d["weight"] = d["weight"] * wt
            d["src"] = np.full(len(d["z"]), i, np.int64)
            parts.append(d)
        names.append("primary" if i == 0 else Path(g).parent.name or f"mix{i}")
        counts.append(len(paths))
    data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    return data, names, counts


def is_val(game: np.ndarray, frac: float) -> np.ndarray:
    """Deterministic split by game id (crc32 of the id), so every run and shard agrees."""
    h = np.array([zlib.crc32(int(g).to_bytes(8, "little", signed=True)) for g in game], dtype=np.uint64)
    return (h % 10_000) < frac * 10_000


class StreamSource:
    """`--stream-gb`: the same rows, split, weights and validation sets as `load_sets` + `is_val`, without holding the
    training rows in memory. Per row only (shard, row in shard, game, set, weight) is kept; validation rows are
    assembled once (`vdata`, `val_sets`: indices into it per set name, as in the in-memory path)."""

    def __init__(self, primary: str, mix: list[str], gen: int, val_frac: float, part_gb: float, seed: int):
        self.gen = gen
        specs = [(primary, 1.0)] + [(m.rsplit(":", 1)[0], float(m.rsplit(":", 1)[1])) for m in mix]
        self.paths, sid, row, game, src, weight = [], [], [], [], [], []
        self.names, self.counts = [], []
        for i, (g, wt) in enumerate(specs):
            paths = sorted(glob.glob(g))
            assert paths, f"no shards match {g}"
            for q in paths:
                with np.load(q) as z:  # only the small per-row arrays are decompressed here
                    gm = z["game"].astype(np.int64)
                    w = z["weight"].astype(np.float32) if "weight" in z.files else np.ones(len(gm), np.float32)
                n = len(gm)
                sid.append(np.full(n, len(self.paths), np.int32))
                row.append(np.arange(n, dtype=np.int32))
                game.append(gm)
                src.append(np.full(n, i, np.int64))
                weight.append(w * wt)
                self.paths.append(q)
            self.names.append("primary" if i == 0 else Path(g).parent.name or f"mix{i}")
            self.counts.append(len(paths))
        self.sid, self.row, self.game = np.concatenate(sid), np.concatenate(row), np.concatenate(game)
        self.src, self.weight = np.concatenate(src), np.concatenate(weight)
        val = is_val(self.game, val_frac)
        self.n_rows = len(val)
        self.train_rows = np.flatnonzero(~val & (self.weight > 0))
        self.game_codes = np.unique(self.game, return_inverse=True)[1]
        val_rows = np.flatnonzero(val)
        self.vdata = self.assemble(val_rows)
        self.val_sets = {nm: np.flatnonzero(self.vdata["src"] == i) for i, nm in enumerate(self.names)}
        row_bytes = sum(v.dtype.itemsize * int(np.prod(v.shape[1:])) for v in self.vdata.values())
        self.row_bytes = row_bytes
        self.passes = max(1, math.ceil(len(self.train_rows) * row_bytes / (part_gb * 2**30)))
        self.rng = np.random.default_rng([seed, 7_919])  # loader thread's own stream (permutations, value rows)

    def assemble(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Arrays of `rows` (global row ids) in that order, as `load_sets` would give them (weights with the set
        multiplier, `src`); each shard is decompressed once."""
        s_of = self.sid[rows]
        order = np.argsort(s_of, kind="stable")
        shards, starts = np.unique(s_of[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        out = None
        for s, a, b in zip(shards, starts, ends):
            pos = order[a:b]  # output positions of this shard's rows
            d = load_shard(self.paths[s], self.gen)
            r = self.row[rows[pos]]
            if out is None:
                out = {k: np.empty((len(rows), *v.shape[1:]), v.dtype) for k, v in d.items()}
            for k, v in d.items():
                out[k][pos] = v[r]
            del d
        if out is None:  # no rows: empty arrays with the right shapes from the first shard
            d = load_shard(self.paths[0], self.gen)
            out = {k: v[:0].copy() for k, v in d.items()}
        out["weight"] = self.weight[rows]
        out["src"] = self.src[rows]
        return out

    def plans(self, one_per_game: bool):
        """Endless: per epoch a permutation of the training rows cut into `passes` equal parts -> (rows in
        permutation order, their `vsel` or None: one random training position per game per epoch carries the value
        loss). Cheap: nothing is decompressed here."""
        while True:
            vsel = None
            if one_per_game:
                sh = self.rng.permutation(self.train_rows)
                first = sh[np.unique(self.game_codes[sh], return_index=True)[1]]
                vsel = np.zeros(self.n_rows, np.float32)
                vsel[first] = 1.0
            perm = self.rng.permutation(self.train_rows)
            for rows in np.array_split(perm, self.passes):
                yield rows, (vsel[rows] if vsel is not None else None)

    def batches(self, bs: int, device, rng: np.random.Generator, one_per_game: bool, prefetch: int = 1):
        """Endless training minibatches (as `batches(..., rng, vsel)` yields them) from parts assembled by a
        background thread; at most `prefetch` + 1 parts are in memory (a slot is taken before a part is built)."""
        slots = threading.Semaphore(prefetch + 1)
        q: queue.Queue = queue.Queue()
        stop = threading.Event()

        def work():
            try:
                for rows, vs in self.plans(one_per_game):
                    while not slots.acquire(timeout=1.0):
                        if stop.is_set():
                            return
                    if stop.is_set():
                        return
                    part = self.assemble(rows)
                    if vs is not None:
                        part["vsel"] = vs
                    q.put(part)
                    del part
            except BaseException as e:  # surfaced in the training thread
                q.put(e)

        t = threading.Thread(target=work, name="stream-loader", daemon=True)
        t.start()
        try:
            while True:
                part = q.get()
                if isinstance(part, BaseException):
                    raise part
                n = len(part["z"])
                for i in range(0, n, bs):
                    sl = slice(i, i + bs)
                    mb = batch_to_tensors({k: v[sl] for k, v in part.items() if k not in ("game", "src", "vsel")},
                                          device)
                    if "vsel" in part:
                        mb["vsel"] = torch.from_numpy(part["vsel"][sl]).to(device)
                    m = len(part["z"][sl])
                    mb["opp_cls"] = torch.full((m,), -1, dtype=torch.long, device=device)
                    mb["drop_seed"] = torch.from_numpy(rng.integers(1, 2**31 - 1, m)).to(device)
                    yield mb
                del part
                slots.release()
        finally:
            stop.set()


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
        pol = (ce * w[has]).sum() / w[has].sum().clamp(min=1e-8)
    vw = w * mb["vsel"] if "vsel" in mb else w  # --value-sample one-per-game: value loss on the selected rows only
    vb = F.binary_cross_entropy_with_logits(out["win_logit"].float(), ((mb["z"] + 1) / 2).clamp(0, 1),
                                            reduction="none")
    val = (vb * vw).sum() / vw.sum().clamp(min=1e-8)
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


def batches(data: dict, idx: np.ndarray, bs: int, device, rng: np.random.Generator | None,
            vsel: np.ndarray | None = None):
    if rng is not None:
        idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        mb = batch_to_tensors({k: v[j] for k, v in data.items() if k not in ("game", "src")}, device)
        if vsel is not None:
            mb["vsel"] = torch.from_numpy(vsel[j]).to(device)
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
    p.add_argument("--mix", action="append", default=[], help="GLOB:WEIGHT, another shard set (repeatable)")
    p.add_argument("--value-sample", choices=("all", "one-per-game"), default="all")
    p.add_argument("--train-only", default="", help="comma list of parameter-name prefixes to train (rest frozen)")
    p.add_argument("--reinit", default="", help="comma list of module-name prefixes to re-initialise first")
    p.add_argument("--select", choices=("total", "value", "policy"), default="total")
    p.add_argument("--no-bf16", action="store_true", help="fp32 (+TF32) instead of bf16 autocast")
    p.add_argument("--stream-gb", type=float, default=0.0,
                   help="streaming loader with parts of about this many GiB (0 = every shard in memory)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_bf16
    torch.set_float32_matmul_precision("high")
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "pretrain.log", "a", buffering=1)

    def emit(rec: dict) -> None:
        line = json.dumps(rec)
        print(line, flush=True)
        log.write(line + "\n")

    stream = None
    if args.stream_gb > 0:  # only a row index + the validation rows in memory; `data` = the validation rows
        stream = StreamSource(args.shards, args.mix, args.gen, args.val_frac, args.stream_gb, args.seed)
        data, names, counts = stream.vdata, stream.names, stream.counts
        train_idx, val_sets = stream.train_rows, stream.val_sets
        n_dec, n_games = stream.n_rows, len(np.unique(stream.game))
        extra = {"stream": {"passes": stream.passes, "row_bytes": stream.row_bytes,
                            "part_gib": round(len(train_idx) * stream.row_bytes / stream.passes / 2**30, 2)}}
    else:
        # every shard in memory once (shards are compressed npz; a v2 decision is ~12 KB uncompressed)
        data, names, counts = load_sets(args.shards, args.mix, args.gen)
        val = is_val(data["game"], args.val_frac)
        # training rows: weight > 0 only (a weight-0 set is reported on validation, never trained on)
        train_idx = np.flatnonzero(~val & (data["weight"] > 0))
        val_sets = {nm: np.flatnonzero(val & (data["src"] == i)) for i, nm in enumerate(names)}
        n_dec, n_games, extra = len(val), len(np.unique(data["game"])), {}
    emit({"event": "data", "shards": counts, "sets": names, "decisions": int(n_dec), "train": int(len(train_idx)),
          "val": {k: int(len(v)) for k, v in val_sets.items()}, "games": int(n_games), **extra,
          "spec": SPEC_VERSION, "args": {k: str(v) for k, v in vars(args).items()}})

    net = load(args.init, device) if args.init else Net(args.gen, **json.loads(args.model_json)).to(device)
    assert net.gen == args.gen, f"checkpoint is gen {net.gen}"
    if args.reinit:
        pre = tuple(x for x in args.reinit.split(",") if x)
        n_re = 0
        for name, mod in net.named_modules():
            if name.startswith(pre) and hasattr(mod, "reset_parameters") and name:
                mod.reset_parameters()
                n_re += 1
        assert n_re, f"--reinit {args.reinit}: no module matched"
        emit({"event": "reinit", "modules": n_re})
    if args.train_only:
        pre = tuple(x for x in args.train_only.split(",") if x)
        for name, prm in net.named_parameters():
            prm.requires_grad_(name.startswith(pre))
        n_tr = sum(prm.numel() for prm in net.parameters() if prm.requires_grad)
        assert n_tr, f"--train-only {args.train_only}: nothing to train"
        emit({"event": "train_only", "params": n_tr})
    net.train()
    groups = [{**g, "params": [q for q in g["params"] if q.requires_grad]} for g in net.param_groups(args.weight_decay)]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=args.lr, betas=(0.9, 0.999))
    total = int(math.ceil(len(train_idx) / args.batch_size) * args.epochs)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        x = (step - args.warmup) / max(total - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(x, 1.0))))

    @torch.no_grad()
    def evaluate(step: int) -> float:
        net.eval()
        rec, res = {"event": "val", "step": step}, {}
        for nm, idx in val_sets.items():
            if not len(idx):
                continue
            m = Meter()
            for mb in batches(data, idx, 2048, device, None):
                mb["weight"] = torch.ones_like(mb["weight"])  # report every set unweighted
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    losses(net, mb, args, m)
            r = m.means()
            if nm == "primary":
                res = r
                rec.update({k: round(v, 5) for k, v in r.items()})
            else:
                rec.update({f"val/{nm}/{k}": round(v, 5) for k, v in r.items() if not k.startswith("aux/")})
        net.train()
        emit(rec)
        pol, vb = res.get("policy_ce", 0.0), res.get("value_bce", 0.0)
        return {"total": pol + args.value_coef * vb, "value": vb, "policy": pol}[args.select]

    step, best, start = 0, float("inf"), time.time()
    m = Meter()
    if args.init:  # the starting point is a candidate too (e.g. a refit that only makes things worse)
        best = evaluate(0)
        save(net, args.out / "best.pt", pretrain_step=0, val_score=best)

    def train_step(mb: dict) -> None:
        nonlocal step, best, m
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = losses(net, mb, args, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([q for q in net.parameters() if q.requires_grad], args.max_grad_norm)
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

    if stream is not None:
        it = stream.batches(args.batch_size, device, rng, args.value_sample == "one-per-game")
        for mb in it:
            if step >= total:
                break
            train_step(mb)
        it.close()
    else:
        game_codes = np.unique(data["game"], return_inverse=True)[1] if args.value_sample == "one-per-game" else None
        while step < total:
            vsel = None
            if game_codes is not None:  # one random training position per game this epoch carries the value loss
                sh = rng.permutation(train_idx)
                first = sh[np.unique(game_codes[sh], return_index=True)[1]]
                vsel = np.zeros(len(data["z"]), np.float32)
                vsel[first] = 1.0
            for mb in batches(data, train_idx, args.batch_size, device, rng, vsel):
                if step >= total:
                    break
                train_step(mb)
    save(net, args.out / "final.pt", pretrain_step=step)


if __name__ == "__main__":
    main()
