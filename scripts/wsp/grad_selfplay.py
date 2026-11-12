"""PPO gradient anatomy on ON-POLICY self-play rollouts (selfplay_log.py), exactly as SB3 builds a rollout buffer:
rewards r_t = Φ(s_{t+1}) − Φ(s_t) (last step: Φ_end − Φ_T, incl. ±1), GAE(γ 0.9999, λ 0.754) with the network's
own V, returns R = A + V. First-epoch loss terms (ratio 1, value clip inactive), minibatch 1024:
  policy  −mean(A_norm · log π(a)),  value 0.4375·mean((R − V)²),  entropy −0.01·mean(H).
Reports per parameter group: ‖g‖, minibatch-cosine (signal share), signal ≈ ‖g‖·√cos; then the anatomy of the
policy-gradient noise: per-sample trunk-gradient energy ‖g_i‖² = A_i²·‖∇log π(a_i)‖² (exact per-sample backward on
a subsample), its share by decision type (forced switch, voluntary switch, move), by π(a) bucket, by |A| bucket,
and the concentration (share of energy in the top 1% / 10% of samples). Also: the value's explained variance.

    python scripts/wsp/grad_selfplay.py 500M out/selfplay/JOB out/grad --batches 16 --per-sample 1500
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from grad_split import groups
from wsp_common import load

VF, ENT, GAMMA, LAM = 0.4375, 0.01, 0.9999, 0.754


def load_rollouts(d):
    cols, off = {}, 0
    for f in sorted(glob.glob(f"{d}/p*/rollouts-*.npz")):
        z = np.load(f)
        n_games = len(z["phi_end"])
        for k in ("obs", "mask", "action", "logp", "ent", "value", "phi", "turn"):
            cols.setdefault(k, []).append(z[k])
        cols.setdefault("game", []).append(z["game"] + off)
        cols.setdefault("phi_end", []).append(z["phi_end"])
        off += n_games
    return {k: np.concatenate(v) for k, v in cols.items()}


def gae(r):
    N = len(r["obs"])
    adv = np.zeros(N, np.float32)
    rows = {}
    for i, g in enumerate(r["game"]):
        rows.setdefault(g, []).append(i)  # rows of one (game, side) are in decision order
    for g, idx in rows.items():
        last = 0.0
        for j in range(len(idx) - 1, -1, -1):
            i = idx[j]
            if j == len(idx) - 1:
                rew, vn = r["phi_end"][g] - r["phi"][i], 0.0
            else:
                rew, vn = r["phi"][idx[j + 1]] - r["phi"][i], r["value"][idx[j + 1]]
            delta = rew + GAMMA * vn - r["value"][i]
            last = delta + GAMMA * LAM * last
            adv[i] = last
    return adv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("rollouts")
    ap.add_argument("out")
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--per-sample", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = load_rollouts(a.rollouts)
    N = len(r["obs"])
    adv = gae(r)
    ret = adv + r["value"]
    ev = 1 - np.var(ret - r["value"]) / np.var(ret)
    pol = load(a.ckpt)
    pol.train(False)
    grp = groups(pol)
    trunk = [p for k, ps in grp.items() if k.startswith("trunk") for _, p in ps]
    x_all = torch.as_tensor(r["obs"].astype(np.float32))
    m_all = torch.as_tensor(r["mask"].astype(np.int64))
    act_all = torch.as_tensor(r["action"].astype(np.int64))
    rng = np.random.default_rng(0)
    terms = ["policy", "value", "entropy"]
    norms = {t: {k: [] for k in grp} for t in terms}
    flat = {t: {k: [] for k in grp} for t in terms}
    for b in range(a.batches):
        idx = rng.choice(N, a.bs, replace=False)
        obs = {"observation": x_all[idx], "action_mask": m_all[idx]}
        values, logp, ent = pol.evaluate_actions(obs, act_all[idx])
        values = values.squeeze(-1)
        A = torch.as_tensor(adv[idx])
        A = (A - A.mean()) / (A.std() + 1e-8)
        R = torch.as_tensor(ret[idx])
        losses = {"policy": -(A * logp).mean(), "value": VF * ((R - values) ** 2).mean(), "entropy": -ENT * ent.mean()}
        for t, L in losses.items():
            pol.zero_grad(set_to_none=True)
            L.backward(retain_graph=True)
            for k, ps in grp.items():
                gv = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel()) for _, p in ps])
                norms[t][k].append(float(gv.norm()))
                if b < 8:
                    flat[t][k].append(gv)
    res = {"n_decisions": int(N), "n_games_sides": int(len(r["phi_end"])), "explained_variance": float(ev),
           "adv_std": float(adv.std()), "entropy_mean": float(r["ent"].mean()), "groups": {}}
    for k in grp:
        row = {}
        for t in terms:
            fs = flat[t][k]
            cos = [float(torch.nn.functional.cosine_similarity(fs[i], fs[j], 0)) for i in range(len(fs))
                   for j in range(i + 1, len(fs))]
            row[t] = {"norm": float(np.mean(norms[t][k])), "cos": float(np.mean(cos)),
                      "signal": float(np.mean(norms[t][k]) * np.sqrt(max(np.mean(cos), 0)))}
        res["groups"][k] = row
    # per-sample anatomy of the policy term (normalised advantages over the whole rollout set)
    A_all = (adv - adv.mean()) / adv.std()
    sub = rng.choice(N, a.per_sample, replace=False)
    e = np.zeros(len(sub))
    for j, i in enumerate(sub):
        obs = {"observation": x_all[i:i + 1], "action_mask": m_all[i:i + 1]}
        _, logp, _ = pol.evaluate_actions(obs, act_all[i:i + 1])
        pol.zero_grad(set_to_none=True)
        logp.sum().backward()
        g2 = sum(float((p.grad ** 2).sum()) for p in trunk if p.grad is not None)
        e[j] = A_all[i] ** 2 * g2
    mask = r["mask"][sub]
    act = r["action"][sub]
    forced = mask[:, 6:10].sum(1) == 0
    is_switch = act < 6
    pa = np.exp(r["logp"][sub])
    absA = np.abs(A_all[sub])
    tot = e.sum()
    def share(sel):
        return {"frac_samples": float(sel.mean()), "frac_energy": float(e[sel].sum() / tot)}
    anat = {"forced switch": share(forced), "voluntary switch": share(~forced & is_switch),
            "move": share(~is_switch)}
    for lo, hi in [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        anat[f"pi(a) in [{lo},{hi})"] = share((pa >= lo) & (pa < hi))
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 1e9)]:
        anat[f"|A_norm| in [{lo},{hi})"] = share((absA >= lo) & (absA < hi))
    srt = np.sort(e)[::-1]
    anat["top 1% samples energy"] = float(srt[: max(1, len(e) // 100)].sum() / tot)
    anat["top 10% samples energy"] = float(srt[: max(1, len(e) // 10)].sum() / tot)
    gn = e / np.maximum(A_all[sub] ** 2, 1e-12)
    anat["corr(log ||grad logpi||^2, log pi(a))"] = float(np.corrcoef(np.log(gn + 1e-12), np.log(pa + 1e-12))[0, 1])
    res["policy_noise_anatomy"] = anat
    (out / f"grad_selfplay_{a.ckpt}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "groups"}, indent=1))
    print(f"{'group':22s} " + " ".join(f"{t + ' |g|':>12s} {'cos':>6s} {'signal':>8s}" for t in terms))
    for k, row in res["groups"].items():
        print(f"{k:22s} " + " ".join(f"{row[t]['norm']:12.3e} {row[t]['cos']:6.3f} {row[t]['signal']:8.3f}" for t in terms))


if __name__ == "__main__":
    main()
