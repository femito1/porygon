"""What information the embeddings hold, and what only they could carry (analysis 5 of the emb brief).

A. ridge probes (5-fold CV R^2) of known properties from the embedding rows of several checkpoints, from the
   drift vectors (e_600 - e_2.5, e_x300 - e_600) and from random N(0,1) codes (mean of 10 draws).
B. redundancy: how many species / moves the observation's numerics alone identify (empirical, from the logged
   observations: distinct ids per numeric signature).
C. identity in the network: multinomial logistic probes (train/test split by GAME) of species id / item / ability
   from (i) the numerics alone, (ii) the mon token after mon_proj, (iii) the same token with that table
   mean-replaced, (iv) the final-layer token of that mon; also opponent-pool facts from the pool-mean input.

    python scripts/wsp/emb_probe.py
"""
import copy
import json

import numpy as np
import torch
from poke_env.data import GenData

from emb_util import OUT, TABLES, ck, emb, load_data, token_counts
from porygon.encoder import INDEX
from porygon.inference import library
from wsp_common import GLORIA, Net, load, split_obs

torch.set_num_threads(2)
G = GenData.from_gen(4)
SPECIES_NUMS = list(range(2, 15)) + list(range(15, 33)) + [121]
MOVE_STATIC = [0, 1, 2] + list(range(5, 26)) + list(range(29, 44))
R = {}


def cv_r2(X, y, k=5, lam=1.0, seed=0):
    idx = np.random.default_rng(seed).permutation(len(y))
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
    types = sorted({t for s in sp for t in G.pokedex[s]["types"]})
    ys = {"species": (sp, {**{f"base {k}": [G.pokedex[s]["baseStats"][k] for s in sp] for k in ("hp", "atk", "def", "spa", "spd", "spe")},
                           **{f"type {t}": [t in G.pokedex[s]["types"] for s in sp] for t in types},
                           "level": [lib[s][0] for s in sp], "log weight": [np.log(G.pokedex[s]["weightkg"]) for s in sp]}),
          "moves": (mv, {"base power": [G.moves[m]["basePower"] for m in mv],
                         "accuracy": [100 if G.moves[m]["accuracy"] is True else G.moves[m]["accuracy"] for m in mv],
                         "priority": [G.moves[m]["priority"] for m in mv],
                         **{f"cat {c}": [G.moves[m]["category"] == c for m in mv] for c in ("Physical", "Special", "Status")},
                         **{f"type {t}": [G.moves[m]["type"] == t for m in mv] for t in sorted({G.moves[m]["type"] for m in mv})}})}
    it = sorted({st.item for s in lib for st in lib[s][1] if st.item in INDEX["items"]})
    ys["items"] = (it, {"choice": [i.startswith("choice") for i in it], "berry": [i.endswith("berry") for i in it],
                        "usage (log)": [np.log(1 + sum(st.n for s in lib for st in lib[s][1] if st.item == i)) for i in it]})
    ab = sorted({st.ability for s in lib for st in lib[s][1] if st.ability in INDEX["abilities"]})
    ys["abilities"] = (ab, {"usage (log)": [np.log(1 + sum(st.n for s in lib for st in lib[s][1] if st.ability == a)) for a in ab]})
    return {k: (ids, {n: np.asarray(v, float) for n, v in t.items()}) for k, (ids, t) in ys.items()}


LAMS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def summarize(ys, X):
    """Features z-scored per dimension (drift vectors are tiny); best ridge lambda per target from a grid; the
    random-code baseline goes through the identical procedure."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    out = {}
    for n, y in ys.items():
        if y.std() == 0 or (set(np.unique(y)) <= {0, 1} and y.sum() < 4):
            continue
        out[n] = max(cv_r2(X, y, lam=l) for l in LAMS)
    groups = {}
    for n, v in out.items():
        groups.setdefault(n.split(" ")[0] if n.startswith(("type", "cat", "base")) else n, []).append(v)
    return {g: float(np.mean(v)) for g, v in groups.items()}


def part_a():
    tg = targets()
    C = {k: ck(k) for k in ("s2.5M", "s500M", "s600M", "x300_640", "ctrl640")}
    rng = np.random.default_rng(1)
    out = {}
    for t, (ids, ys) in tg.items():
        rows = [INDEX[t][i] for i in ids]
        D = emb(C["s500M"], t).shape[1]
        feats = {k: emb(c, t)[rows] for k, c in C.items()}
        feats["drift 2.5M->600M"] = (emb(C["s600M"], t) - emb(C["s2.5M"], t))[rows]
        feats["drift 600M->x300"] = (emb(C["x300_640"], t) - emb(C["s600M"], t))[rows]
        feats["drift 600M->ctrl640"] = (emb(C["ctrl640"], t) - emb(C["s600M"], t))[rows]
        res = {k: summarize(ys, v) for k, v in feats.items()}
        rnd = [summarize(ys, rng.standard_normal((len(rows), D))) for _ in range(10)]
        res["random N(0,1) (mean of 10)"] = {g: float(np.mean([r[g] for r in rnd])) for g in rnd[0]}
        res["random (95% hi)"] = {g: float(np.percentile([r[g] for r in rnd], 97.5)) for g in rnd[0]}
        out[t] = {"n_rows": len(rows), "probes": res}
        print(f"\n-- {t} ({len(rows)} rows, dim {D})")
        gs = list(res["random N(0,1) (mean of 10)"])
        print(f"{'features':28s} " + " ".join(f"{g[:10]:>10s}" for g in gs))
        for k, v in res.items():
            print(f"{k:28s} " + " ".join(f"{v.get(g, float('nan')):10.3f}" for g in gs))
    R["A"] = out


def part_b(d):
    s = split_obs(d["obs"])
    cats, nums, mvn = s["cats"], s["nums"], s["move_nums"]
    out = {}
    pres = cats[..., 0] > 1
    for label, cols in (("base stats+types", list(range(2, 8)) + list(range(15, 33))),
                        ("base stats+types+weight", list(range(2, 8)) + list(range(15, 33)) + [121]),
                        ("base stats+types+level+weight", list(range(2, 8)) + [14] + list(range(15, 33)) + [121])):
        sig = np.round(nums[pres][:, cols].astype(np.float32), 3)
        sp = cats[..., 0][pres]
        u, inv = np.unique(sig, axis=0, return_inverse=True)
        inv = inv.ravel()
        # for each signature, number of distinct species
        pairs = np.unique(np.stack([inv, sp], 1), axis=0)
        nsp = np.bincount(pairs[:, 0], minlength=len(u))
        uniq_tok = float((nsp[inv] == 1).mean())
        n_species = len(np.unique(sp))
        sp_unique = len(np.unique(pairs[nsp[pairs[:, 0]] == 1][:, 1]))
        out[f"species | {label}"] = {"tokens_uniquely_identified": uniq_tok, "species_seen": n_species,
                                     "species_with_unique_signature": sp_unique, "max_species_per_signature": int(nsp.max())}
    mp = cats[..., 3:] > 1
    sig = np.round(mvn[mp][:, MOVE_STATIC].astype(np.float32), 3)
    mid = cats[..., 3:][mp]
    u, inv = np.unique(sig, axis=0, return_inverse=True)
    inv = inv.ravel()
    pairs = np.unique(np.stack([inv, mid], 1), axis=0)
    nm = np.bincount(pairs[:, 0], minlength=len(u))
    out["moves | static move numerics"] = {"tokens_uniquely_identified": float((nm[inv] == 1).mean()),
                                           "moves_seen": int(len(np.unique(mid))),
                                           "moves_with_unique_signature": int(len(np.unique(pairs[nm[pairs[:, 0]] == 1][:, 1]))),
                                           "max_moves_per_signature": int(nm.max())}
    # what is known about items/abilities at all: fraction of mon tokens with a known (>1) id, own vs opp
    for t, col in (("items", 1), ("abilities", 2), ("species", 0)):
        for role, sl in (("own", slice(0, 6)), ("opp", slice(6, 12))):
            c = cats[:, sl, col]
            present = cats[:, sl, 0] > 0
            out[f"known {t} {role}"] = float((c[present] > 1).mean())
    km = cats[:, 6:, 3:]
    out["opp revealed move slots per present opp mon"] = float((km[cats[:, 6:, 0] > 0] > 1).sum(1).mean())
    R["B"] = out
    print("\n== B. redundancy")
    for k, v in out.items():
        print(" ", k, v)


def logreg(Xtr, ytr, Xte, yte, ncls, lam=1e-3, iters=100):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr, Xte = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(Xte, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr)
    W = torch.zeros(Xtr.shape[1], ncls, requires_grad=True)
    b = torch.zeros(ncls, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr_t) + lam * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = (Xte @ W + b).argmax(1).numpy()
    return float((pred == yte).mean())


def part_c(d):
    counts = token_counts(d)
    seen = {t: (counts[t]["all"] > 0) & (np.arange(len(counts[t]["all"])) >= 2) for t in TABLES}
    games = np.unique(d["game"])
    rng = np.random.default_rng(0)
    test_games = set(rng.choice(games, len(games) // 4, replace=False).tolist())
    rows = np.flatnonzero(d["mask"].sum(1) >= 1)[::4]
    X = d["obs"][rows].astype(np.float32)
    is_test = np.array([g in test_games for g in d["game"][rows]])
    s = split_obs(X)
    out = {}
    for netname, path in (("500M", "500M"), ("x300_640", f"{GLORIA}/runs/arm-x300-20261105-005736-371541/final_model.zip")):
        pol = load(path)
        reps = {}
        for label, p in [("token", pol)] + [(f"token|{t} mean", None) for t in ("species", "items", "abilities")]:
            if p is None:
                t = label.split("|")[1].split()[0]
                p = copy.deepcopy(pol)
                W = getattr(p.features_extractor, t).weight
                with torch.no_grad():
                    W[2:] = W[torch.as_tensor(np.flatnonzero(seen[t]))].mean(0)
            net = Net(p)
            tok, fin = [], []
            for i in range(0, len(X), 1024):
                c = {}
                with torch.no_grad():
                    net.forward(torch.as_tensor(X[i:i + 1024]), torch.as_tensor(d["mask"][rows[i:i + 1024]]), None, c)
                tok.append(c["tokens"][:, 1:13].numpy())
                if label == "token":
                    fin.append(c["final"][:, 1:13].numpy())
            reps[label] = np.concatenate(tok)
            if fin:
                reps["final"] = np.concatenate(fin)
        own_mvn = s["move_nums"].reshape(len(X), 12, -1)
        reps["numerics only"] = np.concatenate([s["nums"], own_mvn], -1)
        res = {}
        for t, col in (("species", 0), ("items", 1), ("abilities", 2)):
            for role, sl in (("own", slice(0, 6)), ("opp", slice(6, 12))):
                ids = s["cats"][:, sl, col]
                ok = ids > 1
                y_all = ids[ok]
                # classes: ids with >= 30 tokens in the train split, rest dropped
                tr_mask = np.repeat(~is_test[:, None], 6, 1)[ok]
                vals, cnt = np.unique(y_all[tr_mask], return_counts=True)
                keep_ids = vals[cnt >= 30]
                if len(keep_ids) < 2:
                    continue
                remap = {v: i for i, v in enumerate(keep_ids)}
                sel = np.isin(y_all, keep_ids)
                y = np.array([remap[v] for v in y_all[sel]])
                trm = tr_mask[sel]
                maj = float((y[~trm] == np.bincount(y[trm]).argmax()).mean())
                r = {"n_classes": int(len(keep_ids)), "n_train": int(trm.sum()), "n_test": int((~trm).sum()), "majority": maj}
                for lab, rep in reps.items():
                    Z = rep[:, sl][ok][sel]
                    # subsample train to 20k for speed
                    tri = np.flatnonzero(trm)
                    if len(tri) > 10000:
                        tri = rng.choice(tri, 10000, replace=False)
                    tei = np.flatnonzero(~trm)
                    if len(tei) > 5000:
                        tei = rng.choice(tei, 5000, replace=False)
                    r[lab] = logreg(Z[tri], y[tri], Z[tei], y[tei], len(keep_ids))
                res[f"{t} {role}"] = r
                print(netname, t, role, json.dumps(r), flush=True)
        out[netname] = res
        if netname == "500M":
            # opponent pool facts from the pool-mean input: does the pool contain move X (top-10 most common pool moves)
            pool = s["pool"][:, 6:]
            sp = s["cats"][:, 6:, 0]
            okp = sp > 1
            fe = pol.features_extractor
            with torch.no_grad():
                pe_ = fe.moves(torch.as_tensor(pool[okp]))
                pm = (pe_ * (torch.as_tensor(pool[okp]) > 0).unsqueeze(-1)).sum(1) / (torch.as_tensor(pool[okp]) > 0).sum(1, keepdim=True).clamp(min=1)
            pm = pm.numpy()
            trm = np.repeat(~is_test[:, None], 6, 1)[okp]
            flat = pool[okp]
            vals, cnt = np.unique(flat[flat > 1], return_counts=True)
            top = vals[np.argsort(-cnt)[:10]]
            accs = []
            for mvid in top:
                y = (flat == mvid).any(1).astype(int)
                tri, tei = np.flatnonzero(trm)[:20000], np.flatnonzero(~trm)[:8000]
                a = logreg(pm[tri], y[tri], pm[tei], y[tei], 2)
                base = max(y[tei].mean(), 1 - y[tei].mean())
                accs.append({"move": list(INDEX["moves"])[mvid - 2], "acc": a, "majority": float(base)})
            out["pool_mean_decodes_pool_membership"] = accs
            print("pool membership from pool mean:", accs)
    R["C"] = out


def main():
    import sys
    parts = sys.argv[1] if len(sys.argv) > 1 else "abc"
    if "a" in parts:
        part_a()
    d = load_data()
    if "b" in parts:
        part_b(d)
    if "c" in parts:
        part_c(d)
    json.dump(R, open(OUT / f"probe_{parts}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
