"""Micro-benchmark of --stacked-opp: one fastenv collector step's pool opponents through the trainer's per-net path
(`Trainer._opp_act` once per net) vs the stacked path (one `Trainer._opp_act_many` call), on CPU or CUDA.

    python scripts/bench_stacked_opp.py [NET.pt ...] [--n-nets 4] [--rows 40] [--iters 200] [--device cuda]

Nets: the given v2 .pt files (pool snapshots, frozen nets, checkpoints; at least two of one architecture), else
--n-nets random production-size gen9 nets (--model-json, default {"n_layers": 4, "ffn_mult": 1}; zero-initialised
output layers randomised). Rows: --rows per net (one value, or one per net), synthetic observations
(porygon.v2.synthetic) or real ones from a parity fixture (--fixture DIR with decisions.npz). As in the trainer: fp32
(no autocast), --precision high (TF32 matmuls on Ampere, `torch.set_float32_matmul_precision`), the damage layer
torch.compile'd on CUDA unless --no-compile (one compiled function shared by all nets), --threads 1.

Per iteration (the two paths alternate; CUDA synchronised around each): per-net = for each net H2D, forward, D2H,
CPU sampling; stacked = the same through one call. Prints the mean / median ms per step and the speed-up, the aten
op count per step (dispatcher ops minus views, ~ kernel launches) and on CUDA the profiler's kernel-launch count,
and the max |logit difference| on legal actions between the per-net and the stacked forward (also at --precision
highest on CUDA: TF32 rounding differs between the two GEMM shapes)."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from porygon.v2.model import Net, load
from porygon.v2.stacked import OppStack, signature
from porygon.v2.synthetic import synthetic_obs
from porygon.v2.train import PoolLoads, Trainer, parse_args

VIEWS = {"view", "_unsafe_view", "expand", "slice", "select", "permute", "transpose", "t", "unsqueeze", "squeeze",
         "alias", "as_strided", "detach", "unflatten", "split", "split_with_sizes", "unbind", "lift_fresh",
         "_reshape_alias", "reshape", "flatten", "view_as", "expand_as", "narrow", "contiguous"}


def random_net(seed: int, gen: int, cfg: dict) -> Net:
    torch.manual_seed(seed)
    net = Net(gen, **cfg).eval()
    with torch.no_grad():
        net.policy_head[-1].weight.normal_(std=0.5)
        if hasattr(net, "pair_mlp"):
            net.pair_mlp[-1].weight.normal_(std=0.3)
    return net


def fixture_obs(fixture: str, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    z = np.load(Path(fixture) / "decisions.npz")
    idx = rng.integers(0, len(z["battle"]), n)
    ob = {k[4:]: z[k][idx] for k in z.files if k.startswith("obs/")}
    ob["drop_seed"] = np.zeros(n, np.int64)
    return ob


def count_ops(fn) -> int:
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func.__name__.split(".")[0] not in VIEWS:
                Count.n += 1
            return func(*args, **(kwargs or {}))

    with Count():
        fn()
    return Count.n


def count_launches(fn) -> int:
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages() if "LaunchKernel" in e.key)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("nets", nargs="*", type=Path, help="v2 .pt files (default: random production-size nets)")
    p.add_argument("--n-nets", type=int, default=4)
    p.add_argument("--gen", type=int, default=9, help="random nets' generation")
    p.add_argument("--model-json", default='{"n_layers": 4, "ffn_mult": 1}', help="random nets' Net cfg")
    p.add_argument("--rows", type=int, nargs="+", default=[40], help="decision rows per net (one value or one per net)")
    p.add_argument("--max-moves", type=int, default=8, help="synthetic obs: opponent candidate move slots")
    p.add_argument("--fixture", help="real observations: a porygon.v2.parity fixture directory")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", default="high", choices=("highest", "high", "medium"))
    p.add_argument("--no-compile", action="store_true", help="eager damage layer on CUDA too")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    torch.set_num_threads(a.threads)
    torch.set_float32_matmul_precision(a.precision)
    dev = torch.device(a.device)
    cuda = dev.type == "cuda"
    if a.nets:
        nets = [load(f, dev) for f in a.nets]
        names = [f.name for f in a.nets]
    else:
        cfg = json.loads(a.model_json)
        nets = [random_net(a.seed + i, a.gen, cfg).to(dev) for i in range(a.n_nets)]
        names = [f"random{i}.pt" for i in range(a.n_nets)]
    gen = nets[0].gen
    assert all(n.gen == gen for n in nets), "nets of different generations"
    compiled = cuda and not a.no_compile
    if compiled:  # the trainer: the actor's compiled damage layer, reused by every pool net
        nets[0].compile_damage()
        for n in nets[1:]:
            object.__setattr__(n, "_damage_compiled", nets[0]._damage_compiled)
    sigs = [signature(n) for n in nets]
    groups = len(set(sigs))
    rows = a.rows * len(nets) if len(a.rows) == 1 else a.rows
    assert len(rows) == len(nets), "--rows: one value or one per net"
    rng = np.random.default_rng(a.seed)
    obs = [fixture_obs(a.fixture, r, rng) if a.fixture else synthetic_obs(gen, r, a.max_moves, rng) for r in rows]
    items = list(zip(names, obs))

    # a trainer with only what _opp_act / _opp_act_many use; the nets already in its snapshot cache
    t = object.__new__(Trainer)
    t.args = parse_args(["--format", f"gen{gen}randombattle", "--snapshot-cache", str(max(8, len(nets)))])
    t.gen, t.device, t.compile, t.amp, t.pool_loads, t.pool_dir = gen, dev, compiled, False, PoolLoads(), Path(".")
    t.snapshots = dict(zip(names, nets))

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def per_net():
        for name, ob in items:
            t._opp_act(name, ob)

    def stacked():
        t._opp_act_many(items)

    where = torch.cuda.get_device_name(dev) if cuda else f"{a.threads} thread(s)"
    print(f"{len(nets)} nets ({groups} architecture(s)), rows {rows}, device {dev} ({where}), precision {a.precision}, "
          f"damage compiled {compiled}, torch {torch.__version__}", flush=True)
    t0 = time.perf_counter()
    for _ in range(a.warmup):
        per_net()
        stacked()
    sync()
    print(f"warm-up {time.perf_counter() - t0:.1f} s", flush=True)
    times = {"per-net": [], "stacked": []}
    for _ in range(a.iters):
        for name, fn in (("per-net", per_net), ("stacked", stacked)):
            sync()
            t1 = time.perf_counter()
            fn()
            sync()
            times[name].append((time.perf_counter() - t1) * 1e3)
    for name, ts in times.items():
        print(f"{name:8s} {statistics.mean(ts):8.2f} ms mean  {statistics.median(ts):8.2f} ms median per step "
              f"({len(nets)} nets)")
    print(f"speed-up (mean) {statistics.mean(times['per-net']) / statistics.mean(times['stacked']):.2f}x")
    _, st = t._opp_act_many(items)
    mb = [f"{s.slot_bytes / 2 ** 20:.1f} MiB x {s.capacity}" for s in t.opp_stacks.values()]
    print(f"stacked call: {st}; stacked weights per architecture: {mb}")
    try:  # (a dispatch mode may not trace through a compiled function)
        with torch.no_grad():
            print(f"aten ops per step: per-net {count_ops(per_net)}, stacked {count_ops(stacked)}")
    except Exception as e:
        print(f"aten ops per step: n/a ({e!r:.200})")
    if cuda:
        print(f"CUDA kernel launches per step: per-net {count_launches(per_net)}, stacked {count_launches(stacked)}")

    def max_diff() -> float:
        worst = 0.0
        for sig in set(sigs):
            idx = [i for i, s in enumerate(sigs) if s == sig]
            if len(idx) < 2:
                continue
            out, _ = OppStack(nets[idx[0]]).logits([nets[i] for i in idx], [obs[i] for i in idx])
            for i, got in zip(idx, out):
                legal = torch.from_numpy(obs[i]["action_mask"]) > 0
                ref = t._opp_logits(nets[i], obs[i])
                assert torch.equal(ref[~legal], got[~legal])
                if legal.any():
                    worst = max(worst, float((ref - got)[legal].abs().max()))
        return worst

    print(f"max |logit diff| on legal actions, per-net vs stacked: {max_diff():.3e} (precision {a.precision})")
    if cuda and a.precision != "highest":
        torch.set_float32_matmul_precision("highest")
        print(f"max |logit diff| on legal actions, per-net vs stacked: {max_diff():.3e} (precision highest)")


if __name__ == "__main__":
    sys.exit(main())
