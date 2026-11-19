"""`porygon.v2.pretrain` (streaming path, fresh Net) with an exact resume: the bigger-net BC arm on the GPU Dojo VM
(docs/netv2/BIGNET-DESIGN.md), where the VM may be stopped mid-run.

    PYTHONPATH=SRC python -u pretrain_resumable.py <the pretrain.py arguments> [--state-every 1000]
        [--keep-steps S1,S2] [--stop-after N]

SRC = the BC code of record (BC-P4 ran ws/netv2 745c479 = `$C/src-p2b`). Everything that decides the trained
weights is the imported pretrain.py code (StreamSource, losses, the lr schedule and step below are copied from its
main() line for line); what this file adds:
- `OUT/state.pt` (atomic; the previous one kept as `state.prev.pt`) every `--state-every` steps (a multiple of 200,
  so the 200-step train meter is empty): net, AdamW state, step, best, the training-thread numpy generator (id-dropout
  seeds), torch RNG states, elapsed training seconds, the arguments. A restart with the same arguments resumes from it:
  the loader's own generator is re-seeded exactly as in the first start and its epoch plans are replayed without
  decompressing anything (`plans()` is cheap), the first part that is not finished is assembled and its first
  `skip` minibatches dropped, so the minibatch sequence, value rows and dropout seeds continue as if never stopped
  (bit-identical on CPU: tests in `check_resume.sh`; on a GPU only up to non-deterministic kernels).
- `--keep-steps`: a copy of the net at those steps (`keep-SSSSSSS.pt`, a normal v2 checkpoint) for a matched-cost
  reading.
- `--stop-after N`: exit with code 75 right after the state save at step N (tests only).
Without a state file and without the new flags the run is pretrain.py's: same log lines, same best/last/final files.
"""

import argparse
import json
import math
import os
import sys
import threading
import queue
import time
from pathlib import Path

import numpy as np
import torch

from porygon.v2.model import Net, batch_to_tensors, save
from porygon.v2.pretrain import StreamSource, batches, losses
from porygon.v2.spec import SPEC_VERSION
from porygon.v2.train import Meter

# arguments that must be identical between the first start and a resume
SAME = ("shards", "gen", "out", "model_json", "epochs", "batch_size", "lr", "warmup", "weight_decay", "value_coef",
        "aux_coef", "val_frac", "max_grad_norm", "eval_every", "seed", "value_sample", "select", "no_bf16",
        "stream_gb", "stream_parts", "state_every", "keep_steps")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True)
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-json", default="{}")
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--aux-coef", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--value-sample", choices=("all", "one-per-game"), default="all")
    p.add_argument("--select", choices=("total", "value", "policy"), default="total")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--stream-gb", type=float, required=True)
    p.add_argument("--stream-parts", choices=("rows", "shards"), default="rows")
    p.add_argument("--state-every", type=int, default=1000)
    p.add_argument("--keep-steps", default="")
    p.add_argument("--stop-after", type=int, default=0)
    a = p.parse_args()
    assert a.state_every > 0 and a.state_every % 200 == 0, "--state-every must be a multiple of 200"
    return a


def resumed_batches(stream: StreamSource, bs: int, device, rng, one_per_game: bool, skip: int, prefetch: int = 1):
    """StreamSource.batches, starting after `skip` minibatches: plans whose minibatches all lie before the resume
    point are drawn (same loader-generator calls) but not assembled; the first unfinished part is assembled and its
    first minibatches dropped. With skip = 0 it yields exactly what StreamSource.batches yields."""
    slots = threading.Semaphore(prefetch + 1)
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def work():
        left = skip
        try:
            for rows, vs in stream.plans(one_per_game):
                nb = math.ceil(len(rows) / bs)
                if left >= nb:
                    left -= nb
                    continue
                while not slots.acquire(timeout=1.0):
                    if stop.is_set():
                        return
                if stop.is_set():
                    return
                part = stream.assemble(rows)
                if vs is not None:
                    part["vsel"] = vs
                q.put((part, left))
                left = 0
                del part
        except BaseException as e:
            q.put((e, 0))

    t = threading.Thread(target=work, name="stream-loader", daemon=True)
    t.start()
    try:
        while True:
            part, first = q.get()
            if isinstance(part, BaseException):
                raise part
            n = len(part["z"])
            for i in range(first * bs, n, bs):  # identical to StreamSource.batches' inner loop
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


def main():
    args = parse()
    # ---- pretrain.main(), streaming path, fresh Net (order of every RNG call unchanged) ----
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

    state_path, prev_path = args.out / "state.pt", args.out / "state.prev.pt"
    st = None
    for cand in (state_path, prev_path):
        if cand.exists():
            try:
                st = torch.load(cand, map_location="cpu", weights_only=False)
                break
            except Exception as e:  # a torn file: fall back to the previous state
                print(f"warning: cannot read {cand}: {e}", file=sys.stderr)
    if st is not None:
        mine = {k: str(getattr(args, k)) for k in SAME}
        diff = {k: (st["args"].get(k), v) for k, v in mine.items() if st["args"].get(k) != v}
        assert not diff, f"resume with different arguments: {diff}"

    stream = StreamSource(args.shards, [], args.gen, args.val_frac, args.stream_gb, args.seed,
                          parts=args.stream_parts)
    data, names, counts = stream.vdata, stream.names, stream.counts
    train_idx, val_sets = stream.train_rows, stream.val_sets
    n_dec, n_games = stream.n_rows, len(np.unique(stream.game))
    extra = {"stream": {"passes": stream.passes, "row_bytes": stream.row_bytes, "parts": stream.parts,
                        "train_shards": len(stream.shard_train),
                        "part_gib": round(len(train_idx) * stream.row_bytes / stream.passes / 2**30, 2)}}
    if st is None:
        emit({"event": "data", "shards": counts, "sets": names, "decisions": int(n_dec), "train": int(len(train_idx)),
              "val": {k: int(len(v)) for k, v in val_sets.items()}, "games": int(n_games), **extra,
              "spec": SPEC_VERSION, "args": {k: str(v) for k, v in vars(args).items()}})

    net = Net(args.gen, **json.loads(args.model_json)).to(device)
    assert net.gen == args.gen
    net.train()
    groups = [{**g, "params": [q for q in g["params"] if q.requires_grad]} for g in net.param_groups(args.weight_decay)]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=args.lr, betas=(0.9, 0.999))
    total = int(math.ceil(len(train_idx) / args.batch_size) * args.epochs)
    keep = {int(x) for x in args.keep_steps.split(",") if x}

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
                mb["weight"] = torch.ones_like(mb["weight"])
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
    if st is not None:
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        step, best = int(st["step"]), float(st["best"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
        start = time.time() - float(st["elapsed"])
        emit({"event": "resume", "step": step, "best": best, "elapsed_s": round(float(st["elapsed"]), 1),
              "total": total})

    def save_state() -> None:
        tmp = args.out / "state.pt.tmp"
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "elapsed": time.time() - start, "args": {k: str(getattr(args, k)) for k in SAME},
                    "cfg": dict(net.cfg)}, tmp)
        if state_path.exists():
            os.replace(state_path, prev_path)
        os.replace(tmp, state_path)

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
        # ---- additions (no effect on the weights) ----
        if step in keep:
            save(net, args.out / f"keep-{step:07d}.pt", pretrain_step=step)
        if step % args.state_every == 0 and step < total:
            save_state()
            if args.stop_after and step >= args.stop_after:
                emit({"event": "stop_after", "step": step})
                os._exit(75)

    if step < total:
        if st is None:
            it = stream.batches(args.batch_size, device, rng, args.value_sample == "one-per-game")
        else:
            it = resumed_batches(stream, args.batch_size, device, rng, args.value_sample == "one-per-game", step)
        for mb in it:
            if step >= total:
                break
            train_step(mb)
        it.close()
    save(net, args.out / "final.pt", pretrain_step=step)


if __name__ == "__main__":
    main()
