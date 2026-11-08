"""Format the res JSON outputs as markdown tables (prints to stdout).

    python scripts/wsp/res_tables.py out/res 500M [600M]
"""

import json
import sys
from pathlib import Path


def f(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, dict):
        ci = v.get("ci")
        s = f"{v['v']:.{nd}f}"
        return s + (f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]" if ci else "")
    return f"{v:.{nd}f}"


def probes(d, tag, short=False):
    p = json.loads((d / f"probes_{tag}.json").read_text())
    cols = ["shuffled@L2", "raw_linear", "raw_mlp", "L-1.out", "L0.out", "L1.out", "L2.out", "final", "mlp@L-1.out",
            "mlp@final"]
    print(f"\n### Probes {tag}\n")
    print("| token | target | metric | n | " + " | ".join(cols) + " |")
    print("|" + "---|" * (4 + len(cols)))
    for fam, tg in p.items():
        for name, out in tg.items():
            any_ = next(iter(out.values()))
            row = [fam, name, any_["metric"], str(any_["n"])]
            for c in cols:
                v = out.get(c)
                row.append("–" if v is None else (f"{v['v']:.3f}" if short or c not in ("L2.out", "raw_linear", "final")
                                                  else f(v)))
            print("| " + " | ".join(row) + " |")
    # KO AUC
    print("\nP(KO) > 0.5 AUC of the linear probe score:")
    for fam, tg in p.items():
        for name, out in tg.items():
            if name.endswith("pko"):
                print(f"- {fam} {name}: " + ", ".join(f"{c} {out[c]['auc_ko']:.3f}" for c in cols
                                                      if c in out and "auc_ko" in out[c]))


def main():
    d = Path(sys.argv[1])
    for tag in sys.argv[2:]:
        if (d / f"probes_{tag}.json").exists():
            probes(d, tag)


if __name__ == "__main__":
    main()
