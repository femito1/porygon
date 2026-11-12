"""dyn: parameter movement and Adam state over the gen4 PPO lineage (and the 600M->640M embedding-lr arms).

For each checkpoint (sorted by steps), per parameter group:
  norm |θ|, Adam step count, lr, ρ = Σm²/Σv (pooled over weights with v>0; pure iid gradient noise gives
  (1-β1)/(1+β1) = 0.0526, a constant gradient gives 1), per-step Adam update RMS per weight
  lr·√mean(m̂²/(√v̂+ε)²), n_active (weights with v>0);
  vs the previous checkpoint: |Δθ|, |Δθ|/|θ|, K = Adam steps between, cos(Δ_prev, Δ), and
  R = |Δθ| / (lr_rms · √K · √n_active): the size of the move relative to a random walk of K Adam steps whose
  per-weight steps are ~lr in size (R ≈ 1 random walk of unit steps, R ≈ √K ≈ 185 pure drift for a 5M gap,
  R < 1 restoring force or steps smaller than lr).
Also the Gram matrix (cosines) of all consecutive Δs in the 300M-600M window, per group (float32, on CPU).

    python scripts/wsp/dyn_params.py OUT_DIR [--arms]
"""
import json
import math
import sys

import numpy as np
import torch

from dyn_ckpts import arms, lineage, read

B1 = 0.9


def opt_names(sd):
    """Unique parameter names in optimizer order (state dict minus the pi_/vf_ duplicates of the shared extractor)."""
    return [k for k in sd if not k.startswith(("pi_features_extractor.", "vf_features_extractor."))]


def group_of(name):
    n = name.replace("features_extractor.", "")
    for e in ("species", "items", "abilities", "moves"):
        if n == f"{e}.weight":
            return f"emb.{e}"
    if n == "pos":
        return "pos"
    for p in ("move_proj", "mon_proj", "field_proj", "action_move_proj"):
        if n.startswith(p + "."):
            return p
    if n.startswith("encoder.layers."):
        i = n.split(".")[2]
        if "self_attn" in n:
            return f"L{i}.attn"
        if "linear" in n:
            return f"L{i}.mlp"
        return f"L{i}.norm"
    if n.startswith("norm."):
        return "final_norm"
    if n.startswith("mlp_extractor."):
        return "value_mlp"
    if n.startswith("value_net."):
        return "value_head"
    if n.startswith("action_net.score"):
        return "policy_head"
    return "other"


AGG = {"all": lambda g: True, "embeddings": lambda g: g.startswith("emb."),
       "non_embedding": lambda g: not g.startswith("emb.")}


def flatten(sd, opt, names):
    """-> {group: (θ, m, v)} flat float64 tensors; step; lr."""
    st, out = opt["state"], {}
    step = None
    for i, n in enumerate(names):
        g = group_of(n)
        if g == "other":
            continue
        th = sd[n].double().flatten()
        s = st.get(i)
        if s is not None:
            m, v = s["exp_avg"].double().flatten(), s["exp_avg_sq"].double().flatten()
            step = float(s["step"])
        else:
            m = v = torch.zeros_like(th)
        out.setdefault(g, []).append((th, m, v))
    flat = {g: tuple(torch.cat([t[j] for t in ts]) for j in range(3)) for g, ts in out.items()}
    base = list(flat)
    for a, f in AGG.items():
        gs = [g for g in base if f(g)]
        flat[a] = tuple(torch.cat([flat[g][j] for g in gs]) for j in range(3))
    return flat, step, opt["param_groups"][0]["lr"]


def stats(flat, step, lr):
    bc1, bc2 = 1 - B1 ** step, 1 - 0.999 ** step
    r = {}
    for g, (th, m, v) in flat.items():
        act = v > 0
        na = int(act.sum())
        mh, vh = m / bc1, v / bc2
        u = (mh / (vh.sqrt() + 1e-5))[act]
        r[g] = dict(n=th.numel(), n_active=na, norm=float(th.norm()),
                    rho=float((m[act] ** 2).sum() / v[act].sum()) if na else float("nan"),
                    rho_mean=float((m[act] ** 2 / v[act]).mean()) if na else float("nan"),
                    upd_rms=float(lr * u.pow(2).mean().sqrt()) if na else 0.0,
                    v_rms=float(v[act].mean().sqrt()) if na else 0.0)
    return r


def run(ckpts, out_path, gram_window=None, ref_steps=()):
    rows, prev, prevd, deltas, refs = [], None, {}, [], {}
    names = None
    for step_n, path in ckpts:
        try:
            sd, opt, n = read(path)
        except Exception as e:  # noqa: BLE001
            print("skip", path, e, flush=True)
            continue
        names = names or opt_names(sd)
        flat, step, lr = flatten(sd, opt, names)
        row = dict(steps=n, path=path, adam_step=step, lr=lr, groups=stats(flat, step, lr))
        th = {g: flat[g][0] for g in flat}
        if prev is not None:
            K = step - prev["adam_step"]
            lr_rms = math.sqrt((lr ** 2 + prev["lr"] ** 2) / 2)
            row["K"] = K
            for g in th:
                d = th[g] - prev["th"][g]
                gs = row["groups"][g]
                na = max(gs["n_active"], prev["groups"][g]["n_active"], 1)
                gs["d_norm"] = float(d.norm())
                gs["d_rel"] = float(d.norm() / prev["th"][g].norm())
                gs["R"] = float(d.norm() / (lr_rms * math.sqrt(max(K, 1)) * math.sqrt(na)))
                if g in prevd and prevd[g].norm() > 0 and d.norm() > 0:
                    gs["cos_prev"] = float(d @ prevd[g] / (d.norm() * prevd[g].norm()))
                prevd[g] = d
            if gram_window and gram_window[0] <= prev["steps"] and n <= gram_window[1]:
                deltas.append((prev["steps"], n, {g: (th[g] - prev["th"][g]).float() for g in th}))
        for s in ref_steps:
            if s == n:
                refs[s] = th
        for s, ref in refs.items():
            for g in th:
                row["groups"][g][f"dist_{s}"] = float((th[g] - ref[g]).norm())
        rows.append(row)
        print(n, f"lr {lr:.3g} step {step:.0f}", " ".join(
            f"{g}:{row['groups'][g].get('d_rel', 0):.2e}/{row['groups'][g].get('cos_prev', float('nan')):+.2f}"
            for g in ("all", "embeddings", "non_embedding")), flush=True)
        prev = dict(steps=n, adam_step=step, lr=lr, th=th, groups=row["groups"])
    res = {"rows": rows}
    if deltas:
        gram = {}
        for g in deltas[0][2]:
            D = torch.stack([d[2][g] for d in deltas])
            nn = D.norm(dim=1, keepdim=True).clamp(min=1e-30)
            gram[g] = ((D / nn) @ (D / nn).T).tolist()
        res["gram"] = {"intervals": [(a, b) for a, b, _ in deltas], "cos": gram}
    json.dump(res, open(out_path, "w"))


def main():
    out = sys.argv[1]
    if "--arms" in sys.argv:
        lin = dict(lineage())
        start = [(600002347, lin[600002347])]
        for arm, ck in arms().items():
            run(start + ck, f"{out}/params_arm_{arm}.json", ref_steps=(600002347,))
        return
    run(lineage(), f"{out}/params_lineage.json", gram_window=(297_000_000, 601_000_000),
        ref_steps=(21261312, 300028848, 400029852, 500036720, 600002347))


if __name__ == "__main__":
    main()
