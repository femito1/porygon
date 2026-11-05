"""Function-preserving rescale of a PPO checkpoint's embeddings, so Adam can train them.

Reading the 500M network's weights showed that its species / move / item / ability embeddings never trained:
nn.Embedding starts at N(0, 1) while every linear layer starts at ~0.03, and Adam moves each weight by about
the learning rate per step whatever its scale, so in 500M steps the embeddings moved ~0.4% (the layers that read
them 20-35%) and still predict nothing (types, base stats, move power: no better than random codes). The network
works around it by memorising the random codes downstream.

The embeddings only feed linear layers (species / items / abilities: mon_proj.0; moves: move_proj.0 and, as the
movepool mean, mon_proj.0), so dividing them by K and multiplying the input columns that read them by K leaves the
network's function unchanged, while Adam's steps become K times larger relative to the embeddings. Adam's moments
are rescaled to match (theta / K: m * K, v * K^2; a column * K: m / K, v / K^2), so the next update is unchanged too.

    python scripts/rescale_embeddings.py runs/<run>/final_model.zip out.zip --factor 30
"""

import argparse

import numpy as np
import torch
from stable_baselines3 import PPO

from porygon import encoder as E


def mon_proj_columns(fe) -> dict[str, list[int]]:
    """Input columns of mon_proj.0 that read each embedding (layout of BattleTransformer.forward's `mon`)."""
    s, i, a, m = fe.species.embedding_dim, fe.items.embedding_dim, fe.abilities.embedding_dim, fe.moves.embedding_dim
    o = s + i + a + E.N_MOVES * m  # species, item, ability, the 4 projected moves (not embeddings), then the pool mean
    return {"species": list(range(0, s)), "items": list(range(s, s + i)), "abilities": list(range(s + i, s + i + a)),
            "moves": list(range(o, o + m))}


def rescale(model: PPO, k: float) -> None:
    pol = model.policy
    fe = pol.features_extractor
    opt = pol.optimizer
    state = opt.state

    def scale_param(p: torch.nn.Parameter, f: float, cols: list[int] | None = None) -> None:
        """p (or its columns) *= f, and Adam's moments to match (grad scales by 1 / f)."""
        with torch.no_grad():
            if cols is None:
                p.mul_(f)
            else:
                p[:, cols] *= f
        st = state.get(p)
        if st:
            for key, power in (("exp_avg", 1), ("exp_avg_sq", 2)):
                if cols is None:
                    st[key].mul_(f ** -power)
                else:
                    st[key][:, cols] *= f ** -power
            if "max_exp_avg_sq" in st:
                (st["max_exp_avg_sq"] if cols is None else st["max_exp_avg_sq"][:, cols]).mul_(f ** -2)

    cols = mon_proj_columns(fe)
    for name in ("species", "items", "abilities", "moves"):
        scale_param(getattr(fe, name).weight, 1 / k)
        scale_param(fe.mon_proj[0].weight, k, cols[name])
    scale_param(fe.move_proj[0].weight, k, list(range(fe.moves.embedding_dim)))  # [move embedding, move numerics]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--factor", type=float, default=30.0)
    p.add_argument("--check", help="decisions .npz (obs, mask) to verify the function is unchanged on")
    args = p.parse_args()
    model = PPO.load(args.src, device="cpu")
    if args.check:
        d = np.load(args.check)
        obs = {"observation": torch.as_tensor(d["obs"][:2000].astype(np.float32)),
               "action_mask": torch.as_tensor(d["mask"][:2000].astype(np.float32))}

        def outputs():
            with torch.no_grad():
                f = model.policy.extract_features(obs)
                return model.policy.action_net(f).numpy(), model.policy.predict_values(obs).numpy()
        before = outputs()
    rescale(model, args.factor)
    if args.check:
        after = outputs()
        print(f"max |logit change| {np.abs(after[0] - before[0]).max():.2e}, "
              f"max |value change| {np.abs(after[1] - before[1]).max():.2e}")
    model.save(args.dst)
    fe = model.policy.features_extractor
    print("embedding rms now", {n: round(float(getattr(fe, n).weight.pow(2).mean().sqrt()), 4)
                                for n in ("species", "items", "abilities", "moves")})


if __name__ == "__main__":
    main()
