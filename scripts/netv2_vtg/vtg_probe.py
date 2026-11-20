"""Value-trunk-gradient probe (measurement only; netv2 VTG item 1, 2026-11-20).

At a checkpoint's weights (no update is ever applied), collects R fresh strictly on-policy self-play rollouts with the
fastenv collector (G9q's collector settings, no pool battles) and, on every PPO minibatch of --batch-size rows (G9q:
4096, adv normalised over the minibatch, gradient accumulated over --micro-batch chunks with weight len(chunk) / len(mb),
exactly as porygon.v2.train.ppo_update's first epoch), computes the gradient of each loss term separately:

    pg   = clipped surrogate              (coef 1)
    ent  = - ent_coef * entropy
    za   = zero_avoid_coef * za
    vf   = vf_coef * BCE(win, (ret+1)/2)  (value_trunk_grad of the checkpoint's cfg applies inside the net)
    aux  = aux_coef * sum of aux heads
    POL  = pg + ent + za                  (everything that reaches the trunk through the policy head)
    TOT  = POL + vf + aux                 (+ anchor_coef * kla: 0 at these steps, beta decayed at 50M; reported)

Per parameter group (trunk sub-modules, TRUNK, heads, ALL) and minibatch: norms of every term before clipping and
after clip_grad_norm_(all params, max_grad_norm) (one coefficient c = min(1, mgn / (|TOT|_all + 1e-6)) scales every
term), the ratios |vf| / |POL|, |aux| / |POL|, |vf| / |pg| and cos(vf, POL). Signal estimate: |E g| from the
cross-minibatch dot products (sum over pairs i != j of g_i . g_j / (n (n-1)), also restricted to pairs from
different rollouts), since a single 4096-row policy gradient is mostly noise (G9q logs cos_pg ~ 0).

Run with src-24c633f (G9q's src) first on PYTHONPATH and PORYGON_PS as G9q; writes OUT/<tag>.json + OUT/<tag>.txt.
"""
import argparse
import copy
import json
import math
import queue
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from porygon.fastenv_v2 import FastCollector, build_env
from porygon.v2.model import batch_to_tensors, model_keys, net_from_checkpoint
from porygon.v2.train import aux_losses, build_rollout, ppo_losses, rollout_damage
from porygon.v2.workers import Packer, gen_of

TERMS = ("pg", "ent", "za", "vf", "aux")
DERIVED = ("POL", "TOT")
SIGNAL_TERMS = ("POL", "vf", "aux")


def group_of(name: str) -> str:
    if name.startswith("policy_head."):
        return "H:policy"
    if name.startswith("value_head."):
        return "H:value"
    if name.startswith("aux_"):
        return "H:aux"
    if name.startswith(("ent_", "type_emb", "token_type_emb")):
        return "embed"
    if name.startswith("blocks."):
        return "block" + name.split(".")[1]
    if name.startswith(("move_mlp", "move_token", "tera_token")):
        return "move_enc"
    if name.startswith(("mon_mlp", "absent")):
        return "mon_enc"
    if name.startswith("field_mlp"):
        return "field_enc"
    if name.startswith("pair_mlp"):
        return "pair_mlp"
    if name.startswith("norm."):
        return "final_norm"
    return "other:" + name.split(".")[0]


def q(xs, p):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path, help="weights to probe")
    ap.add_argument("--recipe", required=True, type=Path, help="checkpoint whose stored args give the recipe (G9q)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--rollout", type=int, default=None, help="transitions per rollout (default: the recipe's)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--battles", type=int, default=None, help="fastenv battles per worker (default: the recipe's)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    p = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(p.seed)
    np.random.seed(p.seed)
    torch.set_float32_matmul_precision("high")  # as the trainer (TF32 matmuls, fp32 otherwise; G9q --no-bf16)
    dev = torch.device(p.device)

    rk = torch.load(p.recipe, map_location="cpu", weights_only=False)
    ra = dict(rk["args"])
    args = Namespace(**ra)
    assert args.collector == "fastenv" and args.no_overlap and args.no_bf16, "probe assumes G9q's fp32 on-policy fastenv"
    assert not getattr(args, "decoupled", False) and not getattr(args, "distill", None)
    # collector: G9q's format / shaping / battles per worker, self-play only (no pool, no frozen, no bots)
    args.pool_battles, args.fastenv_pool_share, args.p_frozen, args.p_baseline = 0, 0.0, 0.0, 0.0
    args.fastenv_workers = p.workers
    if p.battles:
        args.fastenv_battles = p.battles
    rollout = p.rollout or args.rollout
    steps_ck = int(rk.get("counters", {}).get("steps", -1))
    beta = (args.kl_anchor_coef * max(0.0, 1.0 - steps_ck / max(1, args.kl_anchor_decay_steps))
            if args.kl_anchor else 0.0)
    assert beta == 0.0, f"KL-anchor coefficient {beta} != 0 at the recipe's steps: the probe does not model it"

    ck = torch.load(p.ckpt, map_location="cpu", weights_only=False)
    net = net_from_checkpoint(ck).to(dev).train()
    vtg = float(net.cfg["value_trunk_grad"])
    gen = net.gen
    assert gen == gen_of(args.format)
    actor = copy.deepcopy(net).eval()
    names = [n for n, _ in net.named_parameters()]
    params = [q_ for _, q_ in net.named_parameters()]
    groups = sorted({group_of(n) for n in names})
    gidx = {g: [i for i, n in enumerate(names) if group_of(n) == g] for g in groups}
    gidx["TRUNK"] = [i for i, n in enumerate(names) if not group_of(n).startswith("H:")]
    gidx["ALL"] = list(range(len(names)))
    order = [g for g in groups if not g.startswith("H:")] + ["TRUNK"] + [g for g in groups if g.startswith("H:")] + ["ALL"]
    gsize = {g: int(sum(params[i].numel() for i in gidx[g])) for g in order}
    damage_params = sum(q_.numel() for n, q_ in net.named_parameters() if n.startswith("damage."))
    assert damage_params == 0, "trainable damage params: rollout_damage once per rollout would not be exact"

    coefs = {"ent_coef": args.ent_coef, "zero_avoid_coef": args.zero_avoid_coef, "vf_coef": args.vf_coef,
             "aux_coef": args.aux_coef, "max_grad_norm": args.max_grad_norm, "clip_range": args.clip_range,
             "adv_clip": args.adv_clip, "batch_size": args.batch_size, "micro_batch": args.micro_batch,
             "gamma": args.gamma, "gae_lambda": args.gae_lambda, "n_epochs": args.n_epochs, "kl_anchor_beta": beta,
             "value_trunk_grad": vtg}
    print(f"[{p.tag}] ckpt {p.ckpt} (steps {ck.get('counters', {}).get('steps')}), recipe {p.recipe} "
          f"(steps {steps_ck}); coefs {coefs}; groups {gsize}", flush=True)

    @torch.no_grad()
    def act(batch: dict):  # = porygon.v2.train.Trainer._act (fp32, one actor copy of the probed weights)
        t = batch_to_tensors(batch, dev, keys=model_keys(gen) + ["drop_seed"])
        out = actor(t, aux=False)
        logp_all = torch.log_softmax(out["logits"].float(), -1)
        probs = logp_all.exp()
        a = torch.multinomial(probs, 1).squeeze(1)
        return (a.cpu().numpy(), logp_all.gather(1, a[:, None]).squeeze(1).cpu().numpy(),
                out["value"].float().cpu().numpy(), probs.cpu().numpy())

    workers = build_env(args, p.seed)
    packer = Packer(gen)
    col = FastCollector(workers.env, packer, act, None, id_dropout=net.cfg["id_dropout"] > 0, seed=p.seed,
                        pool_per_worker=0, pool_dir=None, p_baseline=0.0, p_frozen=0.0, on_policy=True)
    req, outq, stop = threading.Semaphore(0), queue.Queue(), threading.Event()
    th = threading.Thread(target=col.run, args=(rollout, req, outq, stop), daemon=True)
    th.start()
    keys = model_keys(gen) + ["drop_seed"]

    recs = []  # per minibatch: {group: {term: norm}} + extras
    flat_sums = []  # per rollout: {term: CPU vector = sum over its minibatches of the flattened all-param gradient}
    sq = {}    # (group, term) -> sum of squared minibatch norms
    offs = np.cumsum([0] + [x.numel() for x in params])
    gflat = {g: torch.cat([torch.arange(int(offs[i]), int(offs[i + 1])) for i in gidx[g]]) for g in order}
    nmb_r = []
    rollout_info = []
    try:
        for r in range(p.rollouts):
            req.release()
            item = outq.get(timeout=3 * 3600)
            if isinstance(item, BaseException):
                raise item
            segs, eps, collect_s = item
            data = build_rollout(segs, packer, args.gamma, args.gae_lambda)
            n = len(data["action"])
            rollout_info.append({"transitions": n, "games": len(eps), "collect_s": round(collect_s, 1)})
            print(f"[{p.tag}] rollout {r}: {n} transitions, {len(eps)} games, {collect_s:.0f} s", flush=True)
            tensors = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in data.items()}
            with torch.no_grad():
                dmg_all = rollout_damage(net, tensors, keys)
            perm = torch.randperm(n, device=dev)
            k_r = 0
            cur = {t: torch.zeros(int(offs[-1]), device=dev) for t in SIGNAL_TERMS}
            for s in range(0, n - args.batch_size + 1, args.batch_size):  # full minibatches only
                idx = perm[s:s + args.batch_size]
                adv_all = tensors["adv"][idx]
                adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
                mbs = args.micro_batch if args.micro_batch and args.micro_batch < len(idx) else len(idx)
                acc = {t: [torch.zeros_like(x) for x in params] for t in TERMS}
                extra = {"ratio_mean": 0.0, "kl": 0.0, "vf_loss": 0.0, "pg_loss": 0.0, "ent": 0.0}
                for c0 in range(0, len(idx), mbs):
                    sub = idx[c0:c0 + mbs]
                    w = len(sub) / len(idx)
                    mb = {k: v[sub] for k, v in tensors.items()}
                    obs = {k: (mb[k].long() if not mb[k].is_floating_point() and mb[k].dtype != torch.bool else mb[k])
                           for k in keys}
                    dmg = None if dmg_all is None else {k: v[sub] for k, v in dmg_all.items()}
                    mb["adv"] = adv_all[c0:c0 + mbs]
                    out = net(obs, dmg=dmg)
                    L = ppo_losses(out, mb, args, normalized=True)
                    aux = aux_losses(out, mb, None)
                    term = {"pg": L["pg"], "ent": -args.ent_coef * L["ent"], "za": args.zero_avoid_coef * L["za"],
                            "vf": args.vf_coef * L["vf"], "aux": args.aux_coef * aux}
                    for j, t in enumerate(TERMS):
                        gs = torch.autograd.grad(w * term[t], params, retain_graph=j < len(TERMS) - 1,
                                                 allow_unused=True)
                        for a_, g_ in zip(acc[t], gs):
                            if g_ is not None:
                                a_.add_(g_.float())
                    for k_, v_ in (("ratio_mean", L["ratio_mean"]), ("kl", L["kl"]), ("vf_loss", L["vf"]),
                                   ("pg_loss", L["pg"]), ("ent", L["ent"])):
                        extra[k_] += w * float(v_)
                    del out, L, aux, term
                acc["POL"] = [a + b + c for a, b, c in zip(acc["pg"], acc["ent"], acc["za"])]
                acc["TOT"] = [a + b + c for a, b, c in zip(acc["POL"], acc["vf"], acc["aux"])]
                tot_all = torch.sqrt(sum((g * g).sum() for g in acc["TOT"])).item()
                clip_c = min(1.0, args.max_grad_norm / (tot_all + 1e-6))
                rec = {"rollout": r, "clip_coef": clip_c, "tot_norm_all": tot_all, **extra, "g": {}}
                for gname in order:
                    ii = gidx[gname]
                    vec = {t: torch.cat([acc[t][i].flatten() for i in ii]) for t in TERMS + DERIVED}
                    nr = {t: vec[t].norm().item() for t in vec}
                    nr["cos_vf_POL"] = float((vec["vf"] @ vec["POL"]) / (vec["vf"].norm() * vec["POL"].norm()).clamp(min=1e-30))
                    nr["cos_aux_POL"] = float((vec["aux"] @ vec["POL"]) / (vec["aux"].norm() * vec["POL"].norm()).clamp(min=1e-30))
                    rec["g"][gname] = nr
                    for t in SIGNAL_TERMS:
                        sq[(gname, t)] = sq.get((gname, t), 0.0) + nr[t] ** 2
                for t in SIGNAL_TERMS:
                    cur[t] += torch.cat([g_.flatten() for g_ in acc[t]])
                recs.append(rec)
                k_r += 1
                tr = rec["g"]["TRUNK"]
                print(f"[{p.tag}] mb {len(recs)} (rollout {r}): trunk |POL| {tr['POL']:.4g} |pg| {tr['pg']:.4g} "
                      f"|vf| {tr['vf']:.4g} |aux| {tr['aux']:.4g} vf/POL {tr['vf'] / tr['POL']:.4f} "
                      f"aux/POL {tr['aux'] / tr['POL']:.4f} |TOT|all {tot_all:.4g} clip {clip_c:.3f} "
                      f"ratio {extra['ratio_mean']:.4f}", flush=True)
            nmb_r.append(k_r)
            flat_sums.append({t: v.cpu() for t, v in cur.items()})
            del cur
            del tensors, dmg_all, data, segs
            torch.cuda.empty_cache() if dev.type == "cuda" else None
    finally:
        stop.set()
        try:
            workers.close()
        except Exception as e:  # noqa: BLE001
            print(f"workers.close: {e!r}", flush=True)

    n = len(recs)
    R = len(nmb_r)
    summary = {}
    for gname in order:
        rows = [rec["g"][gname] for rec in recs]
        cs = [rec["clip_coef"] for rec in recs]
        S = {}
        stats = {}
        for t in TERMS + DERIVED:
            pre = [x[t] for x in rows]
            post = [x[t] * c for x, c in zip(rows, cs)]
            stats[t] = {"pre_med": q(pre, 50), "pre_q1": q(pre, 25), "pre_q3": q(pre, 75),
                        "post_med": q(post, 50), "post_q1": q(post, 25), "post_q3": q(post, 75)}
        for num, den in (("vf", "POL"), ("aux", "POL"), ("vf", "pg"), ("aux", "vf")):
            rr = [x[num] / x[den] if x[den] > 0 else float("nan") for x in rows]
            stats[f"{num}/{den}"] = ({"med": q(rr, 50), "q1": q(rr, 25), "q3": q(rr, 75)} if all(v == v for v in rr)
                                     else {"med": float("nan"), "q1": float("nan"), "q3": float("nan")})
        for c_ in ("cos_vf_POL", "cos_aux_POL"):
            stats[c_] = {"med": q([x[c_] for x in rows], 50), "q1": q([x[c_] for x in rows], 25),
                         "q3": q([x[c_] for x in rows], 75)}
        nr_ = np.asarray(nmb_r, dtype=np.float64)
        grams = {}
        for t in SIGNAL_TERMS:  # |E g|^2 from rollout sums: unbiased over pairs of different rollouts (iid at fixed weights)
            M = torch.stack([fs[t].index_select(0, gflat[gname]).double() for fs in flat_sums])  # R x P
            G = (M @ M.T).numpy()
            grams[t] = G
            mean_sq = sq[(gname, t)] / n

            def est(keep):
                Gk, nk = G[np.ix_(keep, keep)], nr_[keep]
                num = Gk.sum() - np.trace(Gk)
                den = nk.sum() ** 2 - (nk ** 2).sum()
                return num / den if den > 0 else float("nan")
            allk = list(range(R))
            m2 = est(allk)
            jk = [est([j for j in allk if j != i]) for i in allk] if R >= 3 else []
            se = math.sqrt((R - 1) / R * sum((x - np.mean(jk)) ** 2 for x in jk)) if jk else float("nan")
            S[t] = {"signal_sq": m2, "signal_sq_jk_se": se, "signal": math.sqrt(max(m2, 0.0)),
                    "rms_minibatch": math.sqrt(mean_sq), "coherence": m2 / mean_sq if mean_sq > 0 else float("nan"),
                    "jk_leave_one_out": jk}
        # signal ratio |E vf| / |E POL| with a leave-one-rollout-out jackknife (only meaningful if |E POL|^2 is resolved)
        def ratio(keep, num="vf"):
            vals = []
            for t in (num, "POL"):
                G, nk = grams[t][np.ix_(keep, keep)], nr_[keep]
                vals.append((G.sum() - np.trace(G)) / (nk.sum() ** 2 - (nk ** 2).sum()))
            return math.sqrt(vals[0] / vals[1]) if vals[0] > 0 and vals[1] > 0 else float("nan")
        for num in ("vf", "aux"):
            allk = list(range(R))
            r0 = ratio(allk, num)
            jk = [ratio([j for j in allk if j != i], num) for i in allk] if R >= 3 else []
            ok = [x for x in jk if x == x]
            se = math.sqrt((R - 1) / R * sum((x - np.mean(ok)) ** 2 for x in ok)) if len(ok) == len(jk) and jk else float("nan")
            S[f"signal_{num}/POL"] = {"est": r0, "jk_se": se, "jk": jk}
        stats["signal"] = S
        summary[gname] = stats
    res = {"tag": p.tag, "ckpt": str(p.ckpt), "ckpt_steps": ck.get("counters", {}).get("steps"), "recipe": str(p.recipe),
           "recipe_steps": steps_ck, "coefs": coefs, "rollouts": rollout_info, "minibatches": n,
           "minibatches_per_rollout": nmb_r, "group_params": gsize, "seed": p.seed, "workers": p.workers,
           "battles_per_worker": args.fastenv_battles, "rollout_target": rollout,
           "clip_coef": {"med": q([x["clip_coef"] for x in recs], 50), "q1": q([x["clip_coef"] for x in recs], 25),
                         "q3": q([x["clip_coef"] for x in recs], 75),
                         "share_clipped": float(np.mean([x["clip_coef"] < 1 for x in recs]))},
           "tot_norm_all": {"med": q([x["tot_norm_all"] for x in recs], 50)},
           "ratio_mean": {"med": q([x["ratio_mean"] for x in recs], 50), "min": min(x["ratio_mean"] for x in recs),
                          "max": max(x["ratio_mean"] for x in recs)},
           "vf_loss_med": q([x["vf_loss"] for x in recs], 50), "ent_med": q([x["ent"] for x in recs], 50),
           "summary": summary, "per_minibatch": recs, "wall_s": round(time.time() - t_start, 1)}
    p.out.mkdir(parents=True, exist_ok=True)
    (p.out / f"{p.tag}.json").write_text(json.dumps(res, indent=1))
    lines = [f"== {p.tag}: ckpt {p.ckpt} (steps {res['ckpt_steps']}), value_trunk_grad {vtg}",
             f"recipe {p.recipe} (steps {steps_ck}): vf_coef {args.vf_coef}, ent_coef {args.ent_coef}, za {args.zero_avoid_coef}, "
             f"aux_coef {args.aux_coef}, max_grad_norm {args.max_grad_norm}, clip {args.clip_range}, batch {args.batch_size} "
             f"(micro {args.micro_batch}), KL-anchor beta {beta}",
             f"data: {R} fresh on-policy self-play rollouts {[x['transitions'] for x in rollout_info]} transitions, "
             f"{n} minibatches {nmb_r}; importance ratio mean med {res['ratio_mean']['med']:.5f} "
             f"[{res['ratio_mean']['min']:.5f}, {res['ratio_mean']['max']:.5f}]",
             f"clip coefficient c = min(1, {args.max_grad_norm} / |TOT|all): median {res['clip_coef']['med']:.3f} "
             f"IQR [{res['clip_coef']['q1']:.3f}, {res['clip_coef']['q3']:.3f}], clipped in "
             f"{100 * res['clip_coef']['share_clipped']:.0f}% of minibatches (|TOT|all median {res['tot_norm_all']['med']:.3f}); "
             f"post-clip norms = c x pre-clip, so every ratio is the same before and after clipping",
             "",
             "norms: median [IQR] over minibatches; POL = pg - ent_coef H + za; vf = vf_coef BCE; aux = aux_coef sum",
             f"{'group':<11} {'params':>9} | {'|POL| pre':>22} | {'|vf| pre':>22} | {'|aux| pre':>22} | "
             f"{'vf/POL':>22} | {'aux/POL':>22} | {'vf/pg':>8} | {'cos(vf,POL)':>11} | {'|POL| post':>9} {'|vf| post':>9} {'|aux| post':>9}"]
    f3 = lambda d, a, b, c: f"{d[a]:.3g} [{d[b]:.3g}, {d[c]:.3g}]"  # noqa: E731
    for gname in order:
        s = summary[gname]
        lines.append(f"{gname:<11} {gsize[gname]:>9,} | {f3(s['POL'], 'pre_med', 'pre_q1', 'pre_q3'):>22} | "
                     f"{f3(s['vf'], 'pre_med', 'pre_q1', 'pre_q3'):>22} | {f3(s['aux'], 'pre_med', 'pre_q1', 'pre_q3'):>22} | "
                     f"{f3(s['vf/POL'], 'med', 'q1', 'q3'):>22} | {f3(s['aux/POL'], 'med', 'q1', 'q3'):>22} | "
                     f"{s['vf/pg']['med']:>8.4g} | {s['cos_vf_POL']['med']:>11.4f} | {s['POL']['post_med']:>9.3g} "
                     f"{s['vf']['post_med']:>9.3g} {s['aux']['post_med']:>9.3g}")
    lines += ["", "signal (secondary): |E g|^2 estimated from dot products of rollout-summed gradients of DIFFERENT "
              "rollouts (unbiased at fixed weights), +- leave-one-rollout-out jackknife SE; coherence = |E g|^2 / E|g_mb|^2 "
              "(1 = every minibatch points the same way, ~0 = noise). 'unresolved' = estimate < 2 SE."]
    for gname in [g for g in order if not g.startswith("H:")]:
        S = summary[gname]["signal"]
        parts = []
        for t in SIGNAL_TERMS:
            m2, se = S[t]["signal_sq"], S[t]["signal_sq_jk_se"]
            res_ = "" if (se == se and m2 > 2 * se) else " unresolved"
            parts.append(f"|E {t}|^2 {m2:.3g} +- {se:.2g} (coh {S[t]['coherence']:.4f}{res_})")
        rv, ra = S["signal_vf/POL"], S["signal_aux/POL"]
        lines.append(f"{gname:<11} " + " | ".join(parts) + f" => |E vf|/|E POL| {rv['est']:.3g} +- {rv['jk_se']:.2g}, "
                     f"|E aux|/|E POL| {ra['est']:.3g} +- {ra['jk_se']:.2g}")
    lines.append(f"wall {res['wall_s']:.0f} s")
    (p.out / f"{p.tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
