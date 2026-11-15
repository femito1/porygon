"""G9t's binding GAE-λ pre-check: gradient signal and critic bias per λ on the SAME fixed-weight rollouts.

    python scripts/netv2_lambda_noise.py --ckpt CKPT --pool-src POOL --scratch DIR --seed S --out OUT.json \
        [--n-rollouts 40 --burn-in 1 --lambdas 0.754,0.6,0.5 --pairs 8] [-- TRAINER OVERRIDES]
    python scripts/netv2_lambda_noise.py --synthetic --out /tmp/x.json      # numpy only: no torch, no fastenv

Trainer settings = the checkpoint's own saved `args` (train.py `checkpoint()`), over parse_args' defaults, then
--fastenv-workers 8, then any trainer flags after `--`; forced: --resume CKPT, --run-dir/--pool-dir under --scratch,
--no-overlap (strictly on-policy fastenv collection with the end-of-rollout bootstrap), no --actor-stream, no
--kl-anchor / --distill / EMA / evals (they do not touch the policy gradient). The pool is --pool-src symlinked into
the scratch pool: snapshots <= --pool-max-steps (default: the checkpoint's steps) and frozen/ (best-*.pt only the
newest <= that). Weights stay fixed (no optimizer step; a parameter checksum is compared at the end).

Per rollout (after --burn-in discarded ones), from the trainer's own build_rollout (λ = --gae-lambda, asserted
bit-identical to `gae` below), for every λ: A^λ and G^λ = A^λ + V from the SAME rewards / values / bootstraps.
  cos_pg, cos_vf, ratio: train.gradient_diagnostics (the logged cos_pg / cos_vf / ratio) on --pairs fixed disjoint
      pairs of chunks of the logged size min(micro_batch, batch_size, n/2) = 2048, the same chunks for every λ;
      cos_pg = the mean over the pairs, cos_pg0 = the first pair only (one logged measurement).
  R = the rows whose segment ends in a terminal transition: their λ = 1 return G^MC is the realised discounted
      outcome, unbiased for V^π (truncated segments bootstrap at λ = 1 too, so they are left out of the bias terms).
  bias_mean = mean_R(G^λ − G^MC)            (signed lean of the critic's target on V, reward units, terminal ±1)
  bias_cal  = sqrt(Σ_b w_b m_b²), m_b = mean over the V-decile b of R of (G^λ − G^MC), w_b = its share: E[G^MC | s]
      = V^π(s), so m_b is the target's bias given the critic's own prediction (V's miscalibration inherited by the
      target); 0 in expectation when V = V^π, plus an MC-noise floor shared by all λ.
  bias_rms  = sqrt(mean_R (G^λ − G^MC)²)    (informational: it also grows with 1 − λ when V is exact)
  adv_corr  = Pearson corr_R(A^λ, A^1); adv_sd = sd(A^λ) over all rows; ev = the trainer's ev on G^λ.
Summary: per λ the mean over rollouts, and paired Δ(λ − --ref) per rollout with a 95% t interval.
"""

import argparse, json, math, sys, threading
from pathlib import Path

import numpy as np

PATHS = ("run_dir", "pool_dir", "kl_anchor", "anchor", "resume")
METRICS = ("cos_pg", "cos_pg0", "cos_vf", "ratio", "bias_mean", "bias_cal", "bias_rms", "adv_corr", "adv_sd", "ev")


def seg_arrays(segs: dict):
    """reward, value, next_v, done, last over the rows in build_rollout's order."""
    from porygon.v2.workers import DONE, NEXT_V, REW, VAL
    rows = [t for seg in segs.values() for t in seg]
    last = np.zeros(len(rows), bool)
    last[np.cumsum([len(s) for s in segs.values()]) - 1] = True
    f = lambda k: np.array([t[k] for t in rows], np.float32)  # noqa: E731
    return f(REW), f(VAL), f(NEXT_V), np.array([t[DONE] for t in rows], bool), last


def gae(reward, value, next_v, done, last, gamma: float, lam: float):
    """train.build_rollout's recursion, the same float operations (so bit-identical at the same λ)."""
    delta = reward + gamma * next_v * ~done - value
    carry = gamma * lam * (~done & ~last)
    adv = np.zeros(len(reward), np.float32)
    g = 0.0
    for t in range(len(reward) - 1, -1, -1):
        g = delta[t] + carry[t] * g
        adv[t] = g
    return adv, adv + value


def terminal_rows(done, last):
    """Rows from which the segment runs on to a terminal transition (exact Monte-Carlo returns)."""
    term = np.zeros(len(done), bool)
    nxt = False
    for t in range(len(done) - 1, -1, -1):
        nxt = bool(done[t]) or (not last[t] and nxt)
        term[t] = nxt
    return term


def lambda_stats(reward, value, next_v, done, last, gamma, lams, nbins=10):
    """{λ: (adv, ret, stats)} for lams + λ = 1 (the reference for the bias terms)."""
    out = {lam: gae(reward, value, next_v, done, last, gamma, lam) for lam in sorted(set(lams) | {1.0})}
    R = terminal_rows(done, last)
    adv1, mc = out[1.0][0][R], out[1.0][1][R]
    vR = value[R].astype(np.float64)
    edges = np.quantile(vR, np.linspace(0, 1, nbins + 1)[1:-1]) if R.any() else []
    b = np.searchsorted(edges, vR, side="right")
    res = {}
    for lam, (adv, ret) in out.items():
        d = (ret[R] - mc).astype(np.float64)
        cal = sum((b == k).mean() * d[b == k].mean() ** 2 for k in range(nbins) if (b == k).any()) if R.any() else 0.0
        a, a1 = adv[R].astype(np.float64), adv1.astype(np.float64)
        corr = float(np.corrcoef(a, a1)[0, 1]) if R.sum() > 1 and a.std() > 0 and a1.std() > 0 else 1.0
        var = ret.var()
        res[lam] = (adv, ret, {"bias_mean": float(d.mean()) if R.any() else 0.0, "bias_cal": float(math.sqrt(cal)),
                               "bias_rms": float(np.sqrt((d ** 2).mean())) if R.any() else 0.0, "adv_corr": corr,
                               "adv_sd": float(adv.std()),
                               "ev": float(1 - (ret - value).var() / var) if var > 0 else float("nan")})
    return res, int(R.sum())


def t95(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:  # Cornish-Fisher (error < 1e-3 for df >= 5)
        z = 1.959963984540054
        return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2) + \
            (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3)


def summary(rows: list[dict], lams, ref: float) -> dict:
    out = {"level": {}, "delta_vs_ref": {}}
    for lam in lams:
        key = f"{lam:g}"
        out["level"][key] = {m: float(np.mean([r["lam"][key][m] for r in rows])) for m in METRICS
                             if all(m in r["lam"][key] for r in rows)}
        if lam == ref:
            continue
        out["delta_vs_ref"][key] = {}
        for m in out["level"][key]:
            d = np.array([r["lam"][key][m] - r["lam"][f"{ref:g}"][m] for r in rows])
            sd = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
            h = t95(len(d) - 1) * sd / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            out["delta_vs_ref"][key][m] = {"mean": float(d.mean()), "lo": float(d.mean() - h),
                                           "hi": float(d.mean() + h), "sd": sd, "n": len(d)}
    return out


def measure(net, segs, packer, args, device, amp, lams, pairs: int, chunk_seed: int) -> dict:
    """One rollout: the trainer's build_rollout, every λ's targets, the trainer's gradient diagnostics per λ."""
    import torch
    from porygon.v2 import train as T
    from porygon.v2.model import model_keys
    data = T.build_rollout(segs, packer, args.gamma, args.gae_lambda)
    arr = seg_arrays(segs)
    a0, r0 = gae(*arr, args.gamma, args.gae_lambda)
    assert np.array_equal(a0, data["adv"]) and np.array_equal(r0, data["ret"]), "gae != the trainer's build_rollout"
    st, n_term = lambda_stats(*arr, args.gamma, lams)
    n = len(data["action"])
    keys = model_keys(net.gen) + ["drop_seed"]
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in data.items()}
    dmg = T.rollout_damage(net, tensors, keys)
    net.train()  # as ppo_update (the id-dropout masks are hashed from drop_seed: the same for every λ)
    bs = min(args.micro_batch or args.batch_size, args.batch_size, n // 2)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(chunk_seed)).to(device)
    k_pairs = min(pairs, n // (2 * bs))

    def mb(idx):  # = ppo_update's minibatch()
        b = {k: v[idx] for k, v in tensors.items()}
        obs = {k: (b[k].long() if not b[k].is_floating_point() and b[k].dtype != torch.bool else b[k]) for k in keys}
        return obs, (None if dmg is None else {k: v[idx] for k, v in dmg.items()}), b

    out = {"n": n, "n_term": n_term, "chunk": bs, "pairs": k_pairs, "lam": {}}
    for lam in lams:
        adv, ret, s = st[lam]
        tensors["adv"], tensors["ret"] = torch.from_numpy(adv).to(device), torch.from_numpy(ret).to(device)
        ms = []
        for k in range(k_pairs):
            m = T.Meter()
            T.gradient_diagnostics(net, [mb(perm[2 * k * bs:(2 * k + 1) * bs]),
                                         mb(perm[(2 * k + 1) * bs:(2 * k + 2) * bs])], args, amp, m)
            ms.append(m.means())
        g = lambda key: [x[key] for x in ms]  # noqa: E731
        out["lam"][f"{lam:g}"] = dict(s, cos_pg=float(np.mean(g("diag/grad_cos_policy"))),
                                      cos_pg0=g("diag/grad_cos_policy")[0],
                                      cos_vf=float(np.mean(g("diag/grad_cos_value"))),
                                      ratio=float(np.mean(g("diag/trunk_grad_ratio"))))
    return out


def trainer_args(saved: dict, extra: list[str], o):
    from porygon.v2.train import parse_args
    ns, default = parse_args([]), vars(parse_args([]))
    for k, v in saved.items():  # (checkpoint() stores Paths as str)
        if k in default:
            setattr(ns, k, [Path(x) for x in v] if k == "frozen" else Path(v) if k in PATHS and v is not None else v)
    ns.fastenv_workers = 8
    given = {t[2:].split("=")[0].replace("-", "_") for t in extra if t.startswith("--")}
    given |= {g[3:] for g in given if g.startswith("no_") and g[3:] in default}
    for k, v in vars(parse_args(extra)).items():
        if k in given:
            setattr(ns, k, v)
    ns.resume, ns.run_dir, ns.seed = Path(o.ckpt), Path(o.scratch), o.seed
    ns.pool_dir = ns.run_dir / "pool"
    ns.no_overlap, ns.actor_stream, ns.kl_anchor, ns.distill = True, False, None, None
    ns.eval_battles, ns.eval_async, ns.ema_decay = 0, False, 0.0
    return ns


def link_pool(src: Path, dst: Path, max_steps: int) -> None:
    (dst / "frozen").mkdir(parents=True, exist_ok=False)  # a fresh scratch dir only
    for p in src.glob("*.pt"):
        if p.stem.isdigit() and int(p.stem) <= max_steps:
            (dst / p.name).symlink_to(p.resolve())
    best = [p for p in (src / "frozen").glob("best-*.pt") if int(p.stem[5:]) <= max_steps]
    for p in [p for p in (src / "frozen").glob("*.pt") if not p.name.startswith("best-")] + sorted(best)[-1:]:
        (dst / "frozen" / p.name).symlink_to(p.resolve())


def real(o, extra, lams) -> dict:
    import torch
    from porygon.v2 import train as T
    ck = torch.load(o.ckpt, map_location="cpu", weights_only=False)
    steps, args = int(ck.get("counters", {}).get("steps", 0)), trainer_args(ck.get("args", {}), extra, o)
    del ck
    assert args.collector == "fastenv", args.collector
    link_pool(Path(o.pool_src), args.pool_dir, o.pool_max_steps or steps)
    tr = T.Trainer(args)
    chk = lambda: float(sum(p.detach().double().sum() for p in tr.net.parameters()))  # noqa: E731
    c0 = chk()
    meta = dict(ckpt=str(o.ckpt), steps=steps, lambdas=lams, ref=o.ref, seed=o.seed, burn_in=o.burn_in,
                trainer_args={k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if k == "frozen" else v)
                              for k, v in vars(args).items()})
    if o.ref != args.gae_lambda:
        print(f"warning: --ref {o.ref} != the checkpoint's --gae-lambda {args.gae_lambda}", flush=True)
    tr._refresh_actor()
    thread = threading.Thread(target=tr.collector.run, args=(args.rollout, tr.requests, tr.rollouts, tr.stop),
                              daemon=True)
    thread.start()
    rows = []
    try:
        for i in range(o.burn_in + o.n_rollouts):
            tr.requests.release()
            segs, eps, collect_s = tr._get_rollout()
            if i < o.burn_in:
                continue
            r = measure(tr.net, segs, tr.packer, args, tr.device, tr.amp, lams, o.pairs, o.seed * 1000 + i)
            loads, msgs = tr.pool_loads.take()  # pool-net loads (failures = uniform-random opponents) + the mix
            for msg in msgs:
                print(msg, flush=True)
            mix = [e["opp"] if e["kind"] == "pool" else "selfplay" for e in eps]
            rows.append(dict(r, i=i, games=len(eps), collect_s=collect_s, loads=loads,
                             mix={k: mix.count(k) for k in sorted(set(mix))}))
            Path(o.out).write_text(json.dumps({"meta": meta, "rollouts": rows}, indent=1))
            print(f"rollout {i}: n {r['n']} " + " ".join(
                f"λ{k} cos {v['cos_pg']:+.4f} cal {v['bias_cal']:.4f}" for k, v in r["lam"].items()), flush=True)
    finally:
        tr.stop.set()
        thread.join(30)
        tr.workers.close()
        tr.writer.close()
    assert chk() == c0, "weights changed"
    return {"meta": meta, "rollouts": rows}


def synthetic(o, lams) -> dict:
    """Fake rollouts (a ±1 random-walk game; V = 1.15 × the exact value (overconfident) + noise sd 0.1): the λ /
    bias / summary math end to end; no gradients."""
    rng = np.random.default_rng(o.seed)
    L, p, gamma = 6, 0.55, 0.9999
    r = (1 - p) / p  # gambler's ruin: P(+L before -L | x) = (1 - r^(x+L)) / (1 - r^(2L)); V^π = 2P - 1 (γ ≈ 1)
    v_of = lambda x: 1.15 * (2 * (1 - r ** (x + L)) / (1 - r ** (2 * L)) - 1) + rng.normal(0, 0.1)  # noqa: E731
    rows = []
    for i in range(o.n_rollouts):
        rew, val, nv, dn, la = [], [], [], [], []
        for _ in range(200):  # segments: one episode each, every 5th cut mid-game
            x, cut = 0, rng.random() < 0.2
            while True:
                x2 = x + (1 if rng.random() < p else -1)
                end = abs(x2) == L
                val.append(v_of(x)); rew.append(float(np.sign(x2)) if end else 0.0)
                nv.append(0.0 if end else v_of(x2)); dn.append(end); la.append(False)
                x = x2
                if end or (cut and rng.random() < 0.1):
                    la[-1] = True
                    break
        arr = [np.array(a, np.float32) for a in (rew, val, nv)] + [np.array(dn), np.array(la)]
        st, n_term = lambda_stats(*arr, gamma, lams)
        rows.append({"i": i, "n": len(rew), "n_term": n_term, "lam": {f"{k:g}": st[k][2] for k in lams}})
    return {"meta": {"synthetic": True, "lambdas": lams, "ref": o.ref, "seed": o.seed}, "rollouts": rows}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = p.add_argument
    a("--ckpt"), a("--pool-src"), a("--scratch"), a("--out", required=True)
    a("--seed", type=int, required=True)
    a("--n-rollouts", type=int, default=40), a("--burn-in", type=int, default=1)
    a("--lambdas", default="0.754,0.6,0.5"), a("--ref", type=float, default=0.754)
    a("--pairs", type=int, default=8), a("--pool-max-steps", type=int, default=0)
    a("--synthetic", action="store_true")
    o = p.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    lams = [float(x) for x in o.lambdas.split(",")]
    assert o.ref in lams, "--ref must be one of --lambdas"
    if not o.synthetic and not (o.ckpt and o.pool_src and o.scratch):
        p.error("--ckpt, --pool-src and --scratch are required (or --synthetic)")
    res = synthetic(o, lams) if o.synthetic else real(o, extra, lams)
    res["summary"] = summary(res["rollouts"], lams, o.ref)
    Path(o.out).write_text(json.dumps(res, indent=1))
    for k, d in res["summary"]["delta_vs_ref"].items():
        print(f"Δ(λ {k} − {o.ref:g}) over {len(res['rollouts'])} rollouts: " + "  ".join(
            f"{m} {v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for m, v in d.items()), flush=True)


if __name__ == "__main__":
    main()
