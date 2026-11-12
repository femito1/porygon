"""V3 I1 inference bench, pre-registered in `.orchestra/fastenv/V3-I1-PREREG.md` (ws/fastenv
docs/netsearch/V3-I1-PREREG.md). No games, no Rust: can one inference process on the T4 sustain >= 35,300 leaves/s
(10k leaves/decision for 3 battles at eta 0.85) at B <= 384 under slot-like CPU contention, with parity?

  i1    leaf observations from random walks below P2's roots (v3leaf.LeafEncoder), parity of E16 / G32 / G16 vs the
        eager fp32 reference E32, server-loop throughput per configuration and B, uncontended then contended
        (11 engine-search burners + 3 spin burners, killed by PID). Writes i1.json, i1-rows.jsonl.
  burn  one contention process: `burn mcts STATES_JSON READY_FILE` or `burn spin READY_FILE`.

    PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$C/src NETV2_SRC=<frozen netv2 src> \
      $PY scripts/fastenv/v3_i1.py i1 --roots RUN/p2/p2-roots.pkl --net NET --out OUT
"""
import argparse
import json
import os
import pickle
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIGS = ("E32", "E16", "G32", "G16")
RULE_G = 35_300.0  # 3 x 10,000 / 0.85 = 35,294, registered rounded up
RULE_BMAX = 384


# ---------------------------------------------------------------- leaves

def make_leaves(roots, walks: int, seed: int):
    from v3_gates import prep
    ST, NS, S, R, L = prep()
    rng = random.Random(seed)
    leaves, skipped, errors = [], 0, 0
    states = []
    for r, (root, convs) in enumerate(roots):
        tera_ok = not root.used_tera
        for det in range(min(4, len(convs))):
            enc = L.LeafEncoder(root, convs[det])
            states.append(enc.state.to_string())
            for _ in range(walks):
                st, turns, ok = enc.state, 0, True
                for _ply in range(rng.choice((1, 2, 3))):
                    if L.terminal(st) is not None:
                        break
                    s1, s2 = st.side_one, st.side_two
                    c1, _ = L.side_options(s1, s2, tera_ok)
                    c2, _ = L.side_options(s2, s1, tera_ok)
                    i, j = rng.randrange(len(c1)), rng.randrange(len(c2))
                    try:
                        br = L.pe.generate_instructions(st, L.engine_choice(c1[i]), L.engine_choice(c2[j]))
                    except BaseException as ex:
                        if isinstance(ex, (KeyboardInterrupt, SystemExit)):
                            raise
                        br = []
                    if not br:
                        ok = False
                        break
                    p = np.array([b.percentage for b in br], dtype=np.float64)
                    k = int(np.searchsorted(np.cumsum(p / p.sum()), rng.random() * 0.999999))
                    st = st.apply_instructions(br[k])
                    turns += int(c1[i] != "none" and c2[j] != "none")
                if not ok:
                    errors += 1
                    continue
                if L.terminal(st) is not None:
                    skipped += 1
                    continue
                leaves.append(enc.encode(st, turns=turns))
    return leaves, states, skipped, errors


# ---------------------------------------------------------------- runners

class Runner:
    """One configuration at one batch size: static device inputs, `run()` -> (win_logit, logits, opp, opp_mask)."""

    def __init__(self, net_path: str, cfg: str, B: int, sample: dict, dev: str):
        import torch
        from v3_gates import load_net
        self.torch, self.B, self.dev = torch, B, dev
        self.net = net = load_net(net_path, dev)
        self.fp16 = cfg.endswith("16")
        self.graph = cfg.startswith("G") and dev == "cuda"
        if cfg.startswith("G"):
            from porygon.v2.damage import DamageFeatures
            net.cfg["trim_moves"] = False
            if net.damage is not None:
                net.damage = DamageFeatures(9, trim=False).to(dev)
        self.inp = {k: torch.empty((B, *v.shape[1:]), dtype=v.dtype, device=dev) for k, v in sample.items()}
        self.load({k: v[:1].expand(B, *v.shape[1:]) for k, v in sample.items()})
        self.out = None
        if self.graph:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._fwd()
            torch.cuda.current_stream().wait_stream(s)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self.out = self._fwd()
        else:
            for _ in range(2):
                self._fwd()
        if dev == "cuda":
            torch.cuda.synchronize()

    def _fwd(self):
        torch = self.torch
        with torch.inference_mode(), torch.autocast(self.dev, dtype=torch.float16, enabled=self.fp16,
                                                     cache_enabled=False):
            o = self.net(self.inp, aux=True)
            return o["win_logit"], o["logits"], o["aux"]["opp_action"], o["aux"]["opp_action_mask"]

    def load(self, host: dict):
        for k, v in host.items():
            self.inp[k].copy_(v, non_blocking=True)

    def run(self):
        if self.graph:
            self.g.replay()
            return self.out
        return self._fwd()

    def close(self):
        torch = self.torch
        self.g = self.out = self.net = self.inp = None
        if self.dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def host_batches(leaves, B: int, n: int, pin: bool):
    from porygon.v2.model import obs_to_tensors
    out = []
    for b in range(n):
        idx = [(b * B + i) % len(leaves) for i in range(B)]
        t = obs_to_tensors([leaves[i] for i in idx], "cpu")
        out.append({k: (v.pin_memory() if pin else v) for k, v in t.items()})
    return out


def outputs(net_path, cfg, B, leaves, dev):
    """All leaves through one configuration at batch B (last batch padded by repetition)."""
    hb = host_batches(leaves[:B], B, 1, False)[0]
    r = Runner(net_path, cfg, B, hb, dev)
    res = {"P": [], "logits": [], "opp": [], "opp_mask": []}
    from porygon.v2.model import obs_to_tensors
    torch = r.torch
    for s in range(0, len(leaves), B):
        chunk = leaves[s:s + B]
        n = len(chunk)
        chunk = chunk + [chunk[0]] * (B - n)
        r.load(obs_to_tensors(chunk, "cpu"))
        w, lg, op, om = r.run()
        res["P"].append(torch.sigmoid(w.double()).cpu().numpy()[:n])
        res["logits"].append(lg.double().cpu().numpy()[:n])
        res["opp"].append(op.double().cpu().numpy()[:n])
        res["opp_mask"].append(om.cpu().numpy()[:n])
    r.close()
    return {k: np.concatenate(v) for k, v in res.items()}


def parity(ref, x, mask, fp16: bool) -> dict:
    dP = np.abs(x["P"] - ref["P"])
    legal = mask > 0
    om = ref["opp_mask"].astype(bool)
    finite = bool(np.isfinite(x["P"]).all() and np.isfinite(x["logits"][legal]).all()
                  and np.isfinite(x["opp"][om]).all())
    dl = np.abs(x["logits"] - ref["logits"])[legal]
    do = np.abs(x["opp"] - ref["opp"])[om]
    multi = legal.sum(1) >= 2
    omulti = om.sum(1) >= 2

    def top1(a, m):
        return np.where(m, a, -np.inf).argmax(1)
    flips = float(np.mean(top1(x["logits"], legal)[multi] != top1(ref["logits"], legal)[multi])) if multi.any() else 0.0
    oflips = float(np.mean(top1(x["opp"], om)[omulti] != top1(ref["opp"], om)[omulti])) if omulti.any() else 0.0

    def softmax(a, m):
        z = np.where(m, a, -np.inf)
        z = z - z.max(1, keepdims=True)
        e = np.where(m, np.exp(z), 0.0)
        return e / np.maximum(e.sum(1, keepdims=True), 1e-300)
    tv = 0.5 * np.abs(softmax(x["logits"], legal) - softmax(ref["logits"], legal)).sum(1)[multi]
    res = {"n": int(len(dP)), "mean_abs_dp": float(dP.mean()), "max_abs_dp": float(dP.max()),
           "q50": float(np.quantile(dP, 0.5)), "q99": float(np.quantile(dP, 0.99)),
           "max_abs_dlogit": float(dl.max()) if dl.size else 0.0, "max_abs_dopp": float(do.max()) if do.size else 0.0,
           "policy_top1_flips": flips, "opp_top1_flips": oflips, "n_multi": int(multi.sum()),
           "n_opp_multi": int(omulti.sum()), "policy_tv_q50": float(np.quantile(tv, 0.5)) if tv.size else 0.0,
           "policy_tv_q99": float(np.quantile(tv, 0.99)) if tv.size else 0.0, "finite": finite}
    if fp16:
        res["pass"] = bool(finite and res["mean_abs_dp"] <= 0.002 and res["q99"] <= 0.01 and flips <= 0.005
                           and oflips <= 0.005)
    else:
        res["pass"] = bool(finite and res["max_abs_dp"] <= 1e-4 and res["max_abs_dlogit"] <= 1e-3
                           and res["max_abs_dopp"] <= 1e-3)
    return res


def throughput(net_path, cfg, B, leaves, dev, secs: float, warm: float) -> dict:
    hbs = host_batches(leaves, B, 4, dev == "cuda")
    r = Runner(net_path, cfg, B, hbs[0], dev)
    torch = r.torch
    pinned = None
    i = n = 0
    t_end_warm = time.perf_counter() + warm
    t0 = None
    while True:
        r.load(hbs[i % len(hbs)])
        o = r.run()
        if pinned is None:
            pinned = [torch.empty(t.shape, dtype=t.dtype, pin_memory=(dev == "cuda")) for t in o[:3]]
        for p, t in zip(pinned, o[:3]):
            p.copy_(t, non_blocking=True)
        if dev == "cuda":
            torch.cuda.current_stream().synchronize()
        i += 1
        now = time.perf_counter()
        if t0 is None:
            if now >= t_end_warm:
                t0, n = now, 0
            continue
        n += B
        if now - t0 >= secs:
            break
    rate = n / (time.perf_counter() - t0)
    r.close()
    return {"rate": rate, "batches": i}


# ---------------------------------------------------------------- contention

def burn(argv) -> int:
    kind = argv[0]
    if kind == "spin":
        Path(argv[1]).write_text("ok")
        x = 0
        while True:
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
    import poke_engine as pe
    states = json.loads(Path(argv[1]).read_text())
    k = int(argv[3])
    first = True
    while True:
        st = pe.State.from_string(states[k % len(states)])
        pe.monte_carlo_tree_search(st, duration_ms=1000)
        if first:
            Path(argv[2]).write_text("ok")
            first = False
        k += 1


def start_burners(out: Path, states_file: Path, n_mcts: int, n_spin: int, timeout: float):
    procs = []
    ready = []
    for k in range(n_mcts + n_spin):
        rf = out / f"burn-ready-{k}"
        rf.unlink(missing_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "burn"]
        cmd += (["mcts", str(states_file), str(rf), str(k)] if k < n_mcts else ["spin", str(rf)])
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(out / f"burn-{k}.log", "w")))
        ready.append(rf)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(f.exists() for f in ready):
            break
        if any(p.poll() is not None for p in procs):
            break
        time.sleep(1)
    ok = all(f.exists() for f in ready) and all(p.poll() is None for p in procs)
    (out / "burn-pids.txt").write_text(" ".join(str(p.pid) for p in procs) + "\n")
    return procs, ok, time.time() - t0


def stop_burners(procs):
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    time.sleep(2)
    for p in procs:
        if p.poll() is None:
            p.kill()
    for p in procs:
        p.wait()


# ---------------------------------------------------------------- main

def i1(a) -> int:
    import torch
    torch.set_num_threads(1)
    from v3_gates import prep
    prep()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    roots = pickle.load(open(a.roots, "rb"))[:a.n_roots]
    leaves, states, skipped, errors = make_leaves(roots, a.walks, a.seed)
    states_file = out / "burn-states.json"
    states_file.write_text(json.dumps(states))
    target = len(roots) * 4 * a.walks
    print(f"{len(leaves)} leaves of {target} (terminal {skipped}, engine errors {errors}), "
          f"{time.time() - t_start:.0f} s", flush=True)
    rows = []
    res = {"leaves": len(leaves), "target_leaves": target, "terminal_skipped": skipped, "engine_errors": errors,
           "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None}
    void = len(leaves) < 0.9 * target

    # parity at B 256 vs the eager fp32 reference (+ E32 batch invariance at B 1024)
    pb = a.parity_b
    ref = outputs(a.net, "E32", pb, leaves, dev)
    mask = np.stack([lf["action_mask"] for lf in leaves])
    par = {"E32": {"pass": True, "reference": True}}
    try:
        par["E32@1024"] = parity(ref, outputs(a.net, "E32", 1024, leaves, dev), mask, False)
        par["E32@1024"].pop("pass")
    except Exception as ex:  # information only
        par["E32@1024"] = {"error": repr(ex)[:300]}
    for cfg in CONFIGS[1:]:
        try:
            par[cfg] = parity(ref, outputs(a.net, cfg, pb, leaves, dev), mask, cfg.endswith("16"))
        except Exception as ex:
            par[cfg] = {"pass": False, "error": repr(ex)[:300]}
        print(f"parity {cfg}: {json.dumps(par[cfg])}", flush=True)
    res["parity"] = par

    batches = [int(x) for x in a.batches.split(",")]

    def sweep(cond):
        tab = {}
        for cfg in CONFIGS:
            tab[cfg] = {}
            for B in batches:
                try:
                    t = throughput(a.net, cfg, B, leaves, dev, a.secs, a.warm)
                except Exception as ex:
                    t = {"rate": 0.0, "error": repr(ex)[:300]}
                tab[cfg][B] = t
                rows.append({"cond": cond, "cfg": cfg, "B": B} | t)
                print(f"{cond} {cfg} B {B}: {t}", flush=True)
        return tab

    res["throughput_u"] = sweep("u")
    procs, ok, waited = start_burners(out, states_file, a.mcts, a.spin, a.burn_timeout)
    res["burners"] = {"mcts": a.mcts, "spin": a.spin, "ready": ok, "ready_seconds": waited,
                      "pids": [p.pid for p in procs]}
    try:
        res["throughput_c"] = sweep("c") if ok else {}
    finally:
        stop_burners(procs)
    if not ok:
        void = True
        print("burners not ready: contended sweep not run (void)", flush=True)

    # rule
    cands = []
    for cfg in CONFIGS:
        if not par.get(cfg, {}).get("pass"):
            continue
        for B in batches:
            g = res.get("throughput_c", {}).get(cfg, {}).get(B, {}).get("rate", 0.0)
            if B <= RULE_BMAX and g >= RULE_G:
                cands.append((cfg, B, g))
    best_any = max(((cfg, B, t.get("rate", 0.0)) for cfg, tb in res.get("throughput_c", {}).items()
                    if par.get(cfg, {}).get("pass") for B, t in tb.items() if B <= RULE_BMAX),
                   key=lambda x: x[2], default=None)
    named = None
    if cands:
        fp32 = [c for c in cands if c[0].endswith("32")]
        pool = fp32 if fp32 else cands
        cfg = max(pool, key=lambda c: c[2])[0]
        Bmin = min(c[1] for c in cands if c[0] == cfg)
        g = next(c[2] for c in cands if c[0] == cfg and c[1] == Bmin)
        named = {"cfg": cfg, "B": Bmin, "rate": g, "projected_leaves_per_decision": 0.85 * g / 3}
    if best_any:
        best_any = {"cfg": best_any[0], "B": best_any[1], "rate": best_any[2],
                    "projected_leaves_per_decision": 0.85 * best_any[2] / 3}
    e32c = res.get("throughput_c", {}).get("E32", {})
    if ok and not any(t.get("rate", 0.0) > 0 for t in e32c.values()):
        void = True  # the reference failed to run under contention
    res |= {"named": named, "best_parity_passing_contended": best_any, "pass": bool(cands) and not void,
            "void": void, "rule": f"PASS iff a parity-passing config has contended G >= {RULE_G:.0f} leaves/s at "
                                  f"B <= {RULE_BMAX}", "seconds": time.time() - t_start}
    with open(out / "i1-rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out / "i1.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("leaves", "pass", "void", "named", "best_parity_passing_contended")}))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "burn":
        return burn(sys.argv[2:])
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("i1")
    a.add_argument("--roots", required=True)
    a.add_argument("--net", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--seed", type=int, default=7313)
    a.add_argument("--n-roots", type=int, default=30)
    a.add_argument("--walks", type=int, default=50)
    a.add_argument("--parity-b", type=int, default=256)
    a.add_argument("--batches", default="128,192,256,384,512,1024")
    a.add_argument("--secs", type=float, default=8.0)
    a.add_argument("--warm", type=float, default=2.0)
    a.add_argument("--mcts", type=int, default=11)
    a.add_argument("--spin", type=int, default=3)
    a.add_argument("--burn-timeout", type=float, default=900.0)
    args = p.parse_args()
    return i1(args)


if __name__ == "__main__":
    sys.exit(main())
