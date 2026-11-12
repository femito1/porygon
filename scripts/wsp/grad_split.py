"""Value vs policy gradient into the shared trunk (MissingNo.Zero hypothesis: value gradient ~50x the policy's).

On real logged positions, reproduce PPO's loss terms as SB3 computes them at the first epoch of an update
(ratio = 1, so no policy clipping and no value clipping yet):
  L_pi  = -mean(A_norm * log pi(a|s)),      a ~ pi (on-policy action), A normalised per minibatch (SB3 default)
  L_v   = vf_coef * mean((G - V)^2)          vf_coef 0.4375
  L_ent = -ent_coef * mean(H)                ent_coef 0.01
G = Monte-Carlo shaped return: Phi_end - Phi_t (gamma 0.9999 ~ 1), Phi_t = logged (V+Phi) - V_net, Phi_end from
the last logged position of the game plus the +-1 victory term (approximation, see report).
A: two variants, (mc) A = G - V; (gae-like) A = G - V shrunk toward 0 by lambda-weighted noise is not
identifiable here, so we also report the policy-gradient norm per unit advantage (A_norm ~ N(0,1) i.i.d. random
signs = the noise floor, and A_norm = +1 on the search's argmax action = a coherent "teacher" signal).
Per parameter group: ||g|| of each term (minibatch 1024), cosine between independent minibatches (signal
consistency), and the share of the trunk update each term would get under Adam's sign-like normalisation.

    python scripts/wsp/grad_split.py 500M data out/grad --batches 24
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wsp_common import load, load_data, split_obs

VF, ENT = 0.4375, 0.01


def groups(policy):
    g = {}
    for n, p in policy.named_parameters():
        if n.startswith(("pi_features_extractor", "vf_features_extractor")):
            continue
        if n.startswith("features_extractor."):
            k = n.split(".")[1]
            k = {"species": "emb", "items": "emb", "abilities": "emb", "moves": "emb"}.get(k, k)
            if k == "encoder":
                k = "enc.L" + n.split(".")[3]
            g.setdefault("trunk:" + k, []).append((n, p))
        elif n.startswith("mlp_extractor.value_net") or n.startswith("value_net"):
            g.setdefault("head:value", []).append((n, p))
        else:
            g.setdefault("head:policy", []).append((n, p))
    return g


def phi_end(data, games):
    """Phi at the end of each game (from the last logged position of the logged player) incl. victory."""
    sp = split_obs(data["obs"].astype(np.float32))
    nums = sp["nums"]
    hp, fnt = nums[..., 0], nums[..., 1]
    last = {}
    for i, g in enumerate(data["game"]):
        last[g] = i
    out = {}
    for g, i in last.items():
        won = data["won"][i]
        own_hp, opp_hp = hp[i, :6].sum(), hp[i, 6:].sum()
        own_f, opp_f = fnt[i, :6].sum(), fnt[i, 6:].sum()
        if won == 1:   # opponent ends with all 6 fainted, our survivors keep roughly their last HP
            phi = 1 + 0.05 * own_hp - 0.05 * own_f + 0.05 * 6
        elif won == 0:
            phi = -1 - 0.05 * opp_hp + 0.05 * opp_f - 0.05 * 6
        else:
            phi = 0.0
        out[g] = phi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pol = load(a.ckpt, a.device)
    pol.train(False)
    d = load_data(a.data)
    N = len(d["obs"])
    x_all = torch.as_tensor(d["obs"].astype(np.float32))
    m_all = torch.as_tensor(d["mask"].astype(np.int64))
    # V of this checkpoint on every position -> Phi_t = logged(V+Phi) - V_500M; returns G
    with torch.no_grad():
        V500 = np.concatenate([load("500M", a.device).predict_values(
            {"observation": x_all[s:s + 4096].to(a.device), "action_mask": m_all[s:s + 4096].to(a.device)}
        ).squeeze(-1).cpu().numpy() for s in range(0, N, 4096)])
    phi_t = d["value"] - V500
    pe = phi_end(d, None)
    G = np.array([pe[g] for g in d["game"]]) - phi_t
    # GAE(gamma 0.9999, lambda 0.754) advantages of the PLAYED action along each game's logged decisions
    # (rows are in game order in the file): r_t = Phi_{t+1} - Phi_t, last step r = Phi_end - Phi_T
    gae = np.zeros(N, np.float32)
    rows_by_game = {}
    for i, g in enumerate(d["game"]):
        rows_by_game.setdefault(g, []).append(i)
    for g, rows in rows_by_game.items():
        last = 0.0
        for j in range(len(rows) - 1, -1, -1):
            i = rows[j]
            if j == len(rows) - 1:
                r, vnext = pe[g] - phi_t[i], 0.0
            else:
                r, vnext = phi_t[rows[j + 1]] - phi_t[i], V500[rows[j + 1]]
            delta = r + 0.9999 * vnext - V500[i]
            last = delta + 0.9999 * 0.754 * last
            gae[i] = last
    print("mean(G - V) =", float(np.mean(G - V500)), " rmse", float(np.sqrt(np.mean((G - V500) ** 2))),
          " GAE std", float(gae.std()), " corr(GAE, G-V)", float(np.corrcoef(gae, G - V500)[0, 1]))
    grp = groups(pol)
    rng = np.random.default_rng(0)
    terms = ["value", "value_c", "pi_gae", "pi_mc", "pi_noise", "pi_teacher", "entropy"]
    norms = {t: {k: [] for k in grp} for t in terms}
    flat = {t: {k: [] for k in grp} for t in terms}
    stats = []
    for b in range(a.batches):
        idx = rng.choice(N, a.bs, replace=False)
        obs = {"observation": x_all[idx].to(a.device), "action_mask": m_all[idx].to(a.device)}
        dist = pol.get_distribution(obs)
        with torch.no_grad():
            act = dist.sample()
        values, logp, ent = pol.evaluate_actions(obs, act)
        values = values.squeeze(-1)
        Gt = torch.as_tensor(G[idx], dtype=torch.float32, device=a.device)
        A = (Gt - values).detach()
        A_mc = (A - A.mean()) / (A.std() + 1e-8)
        A_noise = torch.as_tensor(rng.standard_normal(a.bs), dtype=torch.float32, device=a.device)
        teacher = torch.as_tensor(d["shares"][idx].argmax(1), device=a.device)
        _, logp_t, _ = pol.evaluate_actions(obs, teacher)
        err = Gt - values
        A_gae = torch.as_tensor(gae[idx], device=a.device)
        A_gae = (A_gae - A_gae.mean()) / (A_gae.std() + 1e-8)
        played = torch.as_tensor(d["action"][idx].astype(np.int64), device=a.device)
        _, logp_p, _ = pol.evaluate_actions(obs, played)
        losses = {"value": VF * (err ** 2).mean(), "value_c": VF * ((err - err.mean().detach()) ** 2).mean(),
                  "pi_gae": -(A_gae * logp_p).mean(), "pi_mc": -(A_mc * logp).mean(),
                  "pi_noise": -(A_noise * logp).mean(), "pi_teacher": -logp_t.mean() * 1.0,
                  "entropy": -ENT * ent.mean()}
        stats.append({"v_rmse": float(((Gt - values) ** 2).mean().sqrt()), "A_std_raw": float(A.std()),
                      "entropy": float(ent.mean())})
        for t, L in losses.items():
            pol.zero_grad(set_to_none=True)
            L.backward(retain_graph=True)
            for k, ps in grp.items():
                gs = [p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=a.device)
                      for _, p in ps]
                gv = torch.cat(gs)
                norms[t][k].append(float(gv.norm()))
                if b < 8:
                    flat[t][k].append(gv.cpu())
    res = {"stats": {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}, "groups": {}}
    for k in grp:
        row = {"n_params": int(sum(p.numel() for _, p in grp[k]))}
        for t in terms:
            row[f"|g_{t}|"] = float(np.mean(norms[t][k]))
            fs = flat[t][k]
            cos = [float(torch.nn.functional.cosine_similarity(fs[i], fs[j], 0)) for i in range(len(fs))
                   for j in range(i + 1, len(fs))]
            row[f"cos_{t}"] = float(np.mean(cos)) if cos else float("nan")
        res["groups"][k] = row
    (out / f"grad_split_{Path(a.ckpt).stem if '/' in a.ckpt else a.ckpt}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["stats"]))
    hdr = f"{'group':22s} {'params':>8s} " + " ".join(f"{'|g_' + t + '|':>13s} {'cos':>6s}" for t in terms)
    print(hdr)
    for k, r in res["groups"].items():
        print(f"{k:22s} {r['n_params']:8d} " + " ".join(f"{r[f'|g_{t}|']:13.3e} {r[f'cos_{t}']:6.3f}" for t in terms))


if __name__ == "__main__":
    main()
