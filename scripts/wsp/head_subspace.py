"""Which part of the final residual stream do the policy and value heads read? (MissingNo.Zero hypothesis, part 2)

For each token role (field, own active, own bench, opp active, opp bench, move tokens) take the final latent
(after the last LayerNorm) on real positions, its covariance C (256x256), and the input weights each head
applies to that token: policy = ActionHead.score.0 (candidate block for own-mon/move tokens, context block for
the field token); value = mlp_extractor.value_net.0 block of that token.
Reported per role and head:
  - alignment = tr(W C Wᵀ) / (||W||²_F · tr(C) / d): 1 = the head reads directions of average variance, >1 the
    high-variance directions, <1 it reads low-variance (nearly ignored) directions
  - effective rank (participation ratio) of W's singular values and of C's eigenvalues
  - overlap of the top-k principal subspaces of W_pi·C^(1/2) and W_v·C^(1/2) (mean squared cosine of principal
    angles; random k-dim subspaces in d=256: k/256)
  - fraction of the latent variance that lies in the top-16 PCs, and how much of each head's read energy those
    PCs carry
    python scripts/wsp/head_subspace.py 500M data out/grad
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from wsp_common import Net, load, load_data, split_obs
from porygon.encoder import ACTIVE_FLAG


def pr(s):
    s = np.asarray(s, float)
    return float(s.sum() ** 2 / (s ** 2).sum())


def main():
    ck, data_dir, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    pol = load(ck)
    net = Net(pol)
    d = load_data(data_dir)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["obs"]), 8000, replace=False)
    x = d["obs"][idx].astype(np.float32)
    sp = split_obs(x)
    nums = sp["nums"]
    lat = []
    with torch.no_grad():
        for s in range(0, len(x), 1000):
            c = {}
            net.forward(torch.as_tensor(x[s:s + 1000]), torch.as_tensor(d["mask"][idx[s:s + 1000]]), cache=c)
            lat.append(c["final"].numpy())
    lat = np.concatenate(lat)  # B,17,256
    own_active = nums[:, :6, ACTIVE_FLAG] > 0.5
    opp_active = nums[:, 6:, -2] > 0.5  # is_opp_active flag (second-last mon numeric)
    roles = {"field": lat[:, 0]}
    roles["own active"] = lat[:, 1:7][own_active]
    roles["own bench"] = lat[:, 1:7][~own_active & (nums[:, :6, 1] < 0.5)]
    roles["opp active"] = lat[:, 7:13][opp_active]
    roles["opp bench"] = lat[:, 7:13][~opp_active]
    legal_moves = d["mask"][idx][:, 6:10] > 0
    roles["move (legal)"] = lat[:, 13:17][legal_moves]
    Wa = pol.action_net.score[0].weight.detach().numpy()       # 256 x 512: [cand | field ctx]
    Wv = pol.mlp_extractor.value_net[0].weight.detach().numpy().reshape(256, 17, 256)
    res = {}
    for name, Z in roles.items():
        C = np.cov(Z.T)
        ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
        if name == "field":
            Wp, tok = Wa[:, 256:], 0
        else:
            Wp = Wa[:, :256]
            tok = {"own active": 1, "own bench": 2, "opp active": 7, "opp bench": 8, "move (legal)": 13}[name]
        Wval = Wv[:, tok]  # representative slot's block (slot blocks differ: report the slot-averaged too)
        if name.startswith("own"):
            Wval_all = Wv[:, 1:7].mean(1)
        elif name.startswith("opp"):
            Wval_all = Wv[:, 7:13].mean(1)
        elif name.startswith("move"):
            Wval_all = Wv[:, 13:17].mean(1)
        else:
            Wval_all = Wval
        def align(W):
            return float(np.trace(W @ C @ W.T) / ((W ** 2).sum() * np.trace(C) / C.shape[0]))
        # read energy per PC
        evals, evecs = np.linalg.eigh(C)
        evecs = evecs[:, ::-1]
        def pc_share(W, k=16):
            e = ((W @ evecs) ** 2) * np.clip(evals[::-1], 0, None)[None]
            return float(e[:, :k].sum() / e.sum())
        Ch = evecs @ np.diag(np.sqrt(np.clip(evals[::-1], 0, None))) @ evecs.T
        def top(W, k):
            u, s, vt = np.linalg.svd(W @ Ch, full_matrices=False)
            return vt[:k].T
        k = 16
        ov = float((np.linalg.svd(top(Wp, k).T @ top(Wval, k), compute_uv=False) ** 2).mean())
        res[name] = {
            "n": int(len(Z)), "var_total": float(np.trace(C)), "latent_PR": pr(ev),
            "var_top16": float(ev[:16].sum() / ev.sum()),
            "pi_align": align(Wp), "v_align": align(Wval), "v_align_slotavg": align(Wval_all),
            "pi_W_PR": pr(np.linalg.svd(Wp, compute_uv=False) ** 2),
            "v_W_PR": pr(np.linalg.svd(Wval, compute_uv=False) ** 2),
            "pi_energy_top16PC": pc_share(Wp), "v_energy_top16PC": pc_share(Wval),
            "pi_v_overlap_top16": ov, "random_overlap": k / 256,
            "pi_read_norm": float(np.sqrt(np.trace(Wp @ C @ Wp.T))),
            "v_read_norm": float(np.sqrt(np.trace(Wval @ C @ Wval.T))),
        }
    (out / f"head_subspace_{ck}.json").write_text(json.dumps(res, indent=1))
    keys = ["n", "latent_PR", "var_top16", "pi_align", "v_align", "pi_W_PR", "v_W_PR", "pi_energy_top16PC",
            "v_energy_top16PC", "pi_v_overlap_top16", "pi_read_norm", "v_read_norm"]
    print(f"{'role':14s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
    for n, r in res.items():
        print(f"{n:14s} " + " ".join(f"{r[k]:12.3f}" for k in keys))


if __name__ == "__main__":
    main()
