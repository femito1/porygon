"""Directional persistence of weight drift (supports analysis 7 of the emb brief).

For consecutive checkpoint segments a->b->c: cosine between the displacements (w_b - w_a) and (w_c - w_b),
per tensor (flattened; embeddings over seen rows) and per embedding row. A pure random walk gives cos ~ 0
(± 1/sqrt(n)); directional learning gives cos > 0. Also: the semantic content of the embedding drift per segment
(z-scored ridge R^2 of base stats / move category from the drift vectors) and the row-norm change.

    python scripts/wsp/emb_persist.py
"""
import json

import numpy as np

from emb_util import OUT, TABLES, ck, emb
from emb_probe import summarize, targets
from porygon.encoder import INDEX

TRIPLES = [("s2.5M", "s10M", "s21M"), ("s21M", "s66M", "s136M"), ("s136M", "s201M", "s298M"),
           ("s298M", "s400M", "s500M"), ("s400M", "s500M", "s600M"), ("s505M", "s545M", "s595M"),
           ("s505M", "s510M", "s515M")]
SEGS = [("s2.5M", "s21M"), ("s21M", "s136M"), ("s136M", "s298M"), ("s298M", "s400M"), ("s400M", "s600M"),
        ("s2.5M", "s600M"), ("s2.5M", "s136M")]


def main():
    cnt = np.load(OUT / "token_counts.npz")
    seen = {t: (cnt[f"{t}_all"] > 0) & (np.arange(len(cnt[f"{t}_all"])) >= 2) for t in TABLES}
    tags = sorted({t for tr in TRIPLES for t in tr} | {t for s in SEGS for t in s})
    C = {t: ck(t)["params"] for t in tags}
    R = {"tensor_cos": {}, "row_cos": {}, "probe_drift": {}, "norm_change": {}}
    names = [n for n in C["s2.5M"] if C["s2.5M"][n].numel() > 1000]
    print(f"{'tensor':55s} " + " ".join(f"{a[1:]}>{b[1:]}>{c[1:]}"[:16].rjust(16) for a, b, c in TRIPLES))
    for n in names:
        row = []
        for a, b, c in TRIPLES:
            wa, wb, wc = (C[x][n].numpy() for x in (a, b, c))
            t = n.split(".")[1] if n.endswith(".weight") and n.split(".")[1] in TABLES else None
            d1, d2 = (wb - wa), (wc - wb)
            if t:
                d1, d2 = d1[seen[t]], d2[seen[t]]
            d1, d2 = d1.ravel(), d2.ravel()
            row.append(float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-30)))
        R["tensor_cos"][n] = row
        print(f"{n:55s} " + " ".join(f"{x:16.3f}" for x in row))
    for t in TABLES:
        n = f"features_extractor.{t}.weight"
        out = []
        for a, b, c in TRIPLES:
            wa, wb, wc = (C[x][n].numpy()[seen[t]] for x in (a, b, c))
            d1, d2 = wb - wa, wc - wb
            cs = (d1 * d2).sum(1) / (np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) + 1e-30)
            out.append({"median": float(np.median(cs)), "mean": float(cs.mean()), "se": float(cs.std() / np.sqrt(len(cs))),
                        "null_sd_per_row": float(1 / np.sqrt(d1.shape[1]))})
        R["row_cos"][t] = out
        print(t, "row cos (mean ± se):", ["%.3f±%.3f" % (o["mean"], o["se"]) for o in out])
        e0, e6 = emb(ck("s2.5M"), t)[seen[t]], emb(ck("s600M"), t)[seen[t]]
        dn = (e6 ** 2).sum(1) - (e0 ** 2).sum(1)
        R["norm_change"][t] = {"mean_dsqnorm": float(dn.mean()), "se": float(dn.std() / np.sqrt(len(dn))),
                               "mean_sqnorm_2.5M": float((e0 ** 2).sum(1).mean())}
        print(t, "sq-norm change 2.5M->600M:", R["norm_change"][t])
    tg = targets()
    for t in ("species", "moves"):
        ids, ys = tg[t]
        rows = [INDEX[t][i] for i in ids]
        R["probe_drift"][t] = {}
        for a, b in SEGS:
            dE = (C[b][f"features_extractor.{t}.weight"].numpy() - C[a][f"features_extractor.{t}.weight"].numpy())[rows]
            R["probe_drift"][t][f"{a}->{b}"] = summarize(ys, dE)
            print(t, a, b, {k: round(v, 3) for k, v in R["probe_drift"][t][f"{a}->{b}"].items()})
    json.dump(R, open(OUT / "persist.json", "w"), indent=1)


if __name__ == "__main__":
    main()
