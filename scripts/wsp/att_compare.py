"""Head-role stability across checkpoints, on enrichment tables (attention mass minus the share expected from
token counts, so near-uniform heads score ~0 instead of looking similar).

    python scripts/wsp/att_compare.py
For each checkpoint and head: E[q,k] = mass[q,k] − n_k / 17 (n_k = mean number of keys of role k), over query roles
{fld, oA, oB, pA, pB, pU, mL}. Reports per head: ‖E‖ (how structured), corr(E_ckpt, E_500M) for the same head, and
whether the same head is the best match among all 24 heads of the other checkpoint (head identity preserved).
Output $WSP/out/att/compare.json + compare.md
"""
import json

import numpy as np

from att_common import OUT

Q = [0, 1, 2, 4, 5, 7, 8]


def enrich(name):
    z = np.load(OUT / f"patterns_{name}.npz")
    mass = z["mass_sum"] / np.maximum(z["mass_cnt"], 1)[None, None, :, None]
    share = z["mass_cnt"] / z["mass_cnt"].sum()
    return (mass - share[None, None, None, :])[:, :, Q].reshape(24, -1)


def main():
    names = ["2.5M", "400M", "500M", "600M"]
    E = {n: enrich(n) for n in names}
    ref = E["500M"]
    out = {"norm": {n: np.linalg.norm(E[n], axis=1).tolist() for n in names}}
    lines = ["| head | " + " | ".join(f"‖E‖ {n}" for n in names) + " | " +
             " | ".join(f"r vs 500M {n} (best-match?)" for n in names if n != "500M") + " |",
             "|---|" + "---|" * (2 * len(names) - 1)]
    out["corr"], out["best"] = {}, {}
    for n in names:
        if n == "500M":
            continue
        C = np.corrcoef(E[n], ref)[:24, 24:]               # rows: heads of n, cols: heads of 500M
        out["corr"][n] = np.diag(C).tolist()
        out["best"][n] = [int(np.argmax(C[:, h])) for h in range(24)]
    for h in range(24):
        cells = [f"{np.linalg.norm(E[n][h]):.2f}" for n in names]
        cells += [f"{out['corr'][n][h]:+.2f} ({'yes' if out['best'][n][h] == h else 'L%dH%d' % divmod(out['best'][n][h], 8)})"
                  for n in names if n != "500M"]
        lines.append(f"| L{h // 8}H{h % 8} | " + " | ".join(cells) + " |")
    off = np.corrcoef(ref)[np.triu_indices(24, 1)]
    lines.append(f"\nBaseline: correlation between different heads of 500M: mean {off.mean():+.2f}, "
                 f"95th pct {np.percentile(off, 95):+.2f}")
    out["between_head_500M"] = [float(off.mean()), float(np.percentile(off, 95))]
    (OUT / "compare.json").write_text(json.dumps(out))
    (OUT / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
