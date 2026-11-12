"""Time (and profile) one v2 PPO update and batched actor inference on a GPU, without battles.

    python scripts/profile_v2_update.py FIXTURE_DIR [--n 40000] [--epochs 3] [--batch 4096] [--profile] [--model-json ...]

The rollout is the fixture's real observations (`porygon.v2.parity` decisions.npz, obs/ and aux/ fields) tiled to
--n transitions, with random actions / log-probs / advantages / returns: the update's cost does not depend on the
values, only on shapes and on which move slots are occupied.
"""

import argparse
import json
import time

import numpy as np
import torch

from porygon.v2.model import Net, batch_to_tensors, model_keys
from porygon.v2.spec import aux_fields
from porygon.v2.train import parse_args, ppo_update


def synthetic_rollout(fixture: str, n: int, gen: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    z = np.load(f"{fixture}/decisions.npz")
    m = len(z["battle"])
    idx = rng.integers(0, m, n)
    data = {k[4:]: z[k][idx] for k in z.files if k.startswith("obs/")}
    for k, (shape, dtype) in aux_fields(gen).items():
        data[k] = z[f"aux/{k}"][idx] if f"aux/{k}" in z.files else np.zeros((n, *shape), dtype)
    mask = data["action_mask"] > 0
    data["action"] = np.array([rng.choice(np.flatnonzero(r)) if r.any() else 0 for r in mask], np.int64)
    data["logp"] = np.log(1.0 / np.maximum(mask.sum(1), 1)).astype(np.float32)
    data["value"] = rng.uniform(-1, 1, n).astype(np.float32)
    data["adv"] = rng.normal(0, 1, n).astype(np.float32)
    data["ret"] = np.clip(data["value"] + data["adv"] * 0.3, -1, 1).astype(np.float32)
    data["drop_seed"] = rng.integers(1, 2**31 - 1, n).astype(np.int64)
    data["opp_cls"] = np.full(n, -1, np.int64)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fixture")
    p.add_argument("--gen", type=int, default=4)
    p.add_argument("--n", type=int, default=40_000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--updates", type=int, default=3)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--model-json", default="{}")
    p.add_argument("--actor-batch", type=int, default=96)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    dev = torch.device(a.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(0)
    fmt = f"gen{a.gen}randombattle"
    args = parse_args(["--format", fmt, "--n-epochs", str(a.epochs), "--batch-size", str(a.batch), "--no-bf16"])
    net = Net(a.gen, **json.loads(a.model_json)).to(dev).train()
    if a.compile:
        net.compile_damage()
    opt = torch.optim.AdamW(net.param_groups(args.weight_decay), lr=args.lr, eps=1e-5, fused=dev.type == "cuda")
    data = synthetic_rollout(a.fixture, a.n, a.gen, rng)
    occ = (data["move_p"] > 0).sum(-1)
    print(f"params {sum(p.numel() for p in net.parameters()):,}; move slots occupied per mon: max {occ.max()}, "
          f"mean {occ.mean():.2f}; device {torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu'}")

    def timed(fn):
        sync()
        t = time.time()
        r = fn()
        sync()
        return time.time() - t, r

    for u in range(1 if a.profile else a.updates):
        s, st = timed(lambda: ppo_update(net, opt, data, args, dev, False, u == 1))
        print(f"update {u}: {s:.2f} s for {a.n} transitions x {a.epochs} epochs / batch {a.batch} "
              f"(damage {st.get('perf/damage_s', float('nan')):.2f} s)", flush=True)
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            timed(lambda: ppo_update(net, opt, data, args, dev, False, False))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    # actor: batched inference as the collector does it
    net.eval()
    keys = model_keys(a.gen) + ["drop_seed"]
    batch = {k: v[:a.actor_batch] for k, v in data.items() if k in keys}
    with torch.no_grad():
        for _ in range(3):
            net(batch_to_tensors(batch, dev, keys=keys), aux=False)
        s, _ = timed(lambda: [net(batch_to_tensors(batch, dev, keys=keys), aux=False) for _ in range(20)])
    print(f"actor forward B={a.actor_batch}: {1e3 * s / 20:.1f} ms -> {a.actor_batch * 20 / s:,.0f} obs/s single-stream")


if __name__ == "__main__":
    main()
