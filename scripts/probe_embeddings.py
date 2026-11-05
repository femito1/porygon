"""Do a checkpoint's token embeddings know anything? (the embedding-training experiment's first readout)

For each checkpoint: how far each embedding table moved from a reference checkpoint (relative norm), and how well
ridge regression on the embeddings predicts known facts, cross-validated, against random codes of the same size:
species -> base stats, types, random-battle level; moves -> base power, status or not, priority, type.
The 500M network's embeddings score like random codes on all of these (they never left their init).

    python scripts/probe_embeddings.py REF.zip CKPT.zip [CKPT.zip ...]
"""

import argparse

import numpy as np
from poke_env.data import GenData

from porygon.encoder import INDEX
from porygon.inference import library
from porygon.model import load_policy

G = GenData.from_gen(4)


def cv_r2(X: np.ndarray, y: np.ndarray, k: int = 5, lam: float = 1.0) -> float:
    idx = np.random.default_rng(0).permutation(len(y))
    ss = st = 0.0
    for f in np.array_split(idx, k):
        tr = np.setdiff1d(idx, f)
        mu, ym = X[tr].mean(0), y[tr].mean()
        A = X[tr] - mu
        w = np.linalg.solve(A.T @ A + lam * np.eye(X.shape[1]), A.T @ (y[tr] - ym))
        e = (X[f] - mu) @ w + ym - y[f]
        ss += e @ e
        st += ((y[f] - ym) ** 2).sum()
    return 1 - ss / st


def targets():
    lib = library()
    sp = [s for s in lib if s in INDEX["species"] and s in G.pokedex]
    mv = sorted({m for s in lib for st in lib[s][1] for m in st.moves if m in INDEX["moves"] and m in G.moves})
    ys = {"species": (sp, {f"base {k}": [G.pokedex[s]["baseStats"][k] for s in sp] for k in ("hp", "atk", "spa", "spe")}
                      | {f"{t} type": [t in G.pokedex[s]["types"] for s in sp] for t in ("Water", "Steel", "Flying")}
                      | {"level": [lib[s][0] for s in sp]}),
          "moves": (mv, {"base power": [G.moves[m]["basePower"] for m in mv],
                         "status move": [G.moves[m]["category"] == "Status" for m in mv],
                         "priority": [G.moves[m]["priority"] for m in mv],
                         "Fire type": [G.moves[m]["type"] == "Fire" for m in mv]})}
    return {k: (ids, {n: np.asarray(v, float) for n, v in t.items()}) for k, (ids, t) in ys.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ref")
    p.add_argument("ckpts", nargs="+")
    args = p.parse_args()
    ref = load_policy(args.ref).features_extractor
    tg = targets()
    rng = np.random.default_rng(1)
    base = {k: np.mean([cv_r2(rng.standard_normal((len(ids), getattr(ref, k).embedding_dim)), y)
                        for ids, ys in [tg[k]] for y in ys.values() for _ in range(3)]) for k in tg}
    print("random-code baseline mean R2:", {k: round(float(v), 3) for k, v in base.items()})
    for c in args.ckpts:
        fe = load_policy(c).features_extractor
        moved = {n: float(np.linalg.norm(getattr(fe, n).weight.detach() - getattr(ref, n).weight.detach())
                          / np.linalg.norm(getattr(ref, n).weight.detach()))
                 for n in ("species", "moves", "items", "abilities")}
        print(f"{c}\n  moved vs ref: " + "  ".join(f"{n} {v:.3%}" for n, v in moved.items()))
        for k, (ids, ys) in tg.items():
            E = getattr(fe, k).weight.detach().numpy()[[INDEX[k][i] for i in ids]]
            print(f"  {k}: " + "  ".join(f"{n} {cv_r2(E, y):+.3f}" for n, y in ys.items()))


if __name__ == "__main__":
    main()
