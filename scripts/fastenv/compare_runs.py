"""Side-by-side TensorBoard scalars of two netv2-trainer runs at matched env steps (collector-level comparison of
the per-sample arms: episode length, opponent-target availability, aux accuracies, value EV, entropy, pool win rates).

    python scripts/fastenv/compare_runs.py RUN_A RUN_B [--at 0.5e6 1e6 2e6] [--window 0.25e6]

Each value is the mean over updates whose step lies within ±window of the target step."""
import argparse
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

TAGS = ["episode/decisions", "episode/turns", "perf/opp_target_frac", "loss/ent", "loss/kl", "loss/clip",
        "diag/explained_variance", "aux/item_acc_hidden", "aux/ability_acc_hidden", "aux/moves_acc_cand",
        "aux/opp_action_acc", "pool/win_rate_heuristic", "pool/win_rate_max_power", "pool/win_rate_snapshot",
        "perf/steps_per_s", "perf/collect_steps_per_s", "perf/update_s"]


def load(run: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for f in sorted(run.rglob("events.out.tfevents.*")):
        a = ea.EventAccumulator(str(f.parent), size_guidance={"scalars": 0})
        a.Reload()
        for t in a.Tags()["scalars"]:
            s = a.Scalars(t)
            st, v = out.get(t, (np.zeros(0), np.zeros(0)))
            out[t] = (np.concatenate([st, [e.step for e in s]]), np.concatenate([v, [e.value for e in s]]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--at", type=float, nargs="+", default=[0.5e6, 1e6, 2e6])
    p.add_argument("--window", type=float, default=0.25e6)
    args = p.parse_args()
    runs = [load(args.a), load(args.b)]
    print(f"A = {args.a}\nB = {args.b}")
    for at in args.at:
        print(f"--- at {at / 1e6:.2f}M (±{args.window / 1e6:.2f}M)")
        for t in TAGS:
            vals = []
            for r in runs:
                if t not in r:
                    vals.append("-")
                    continue
                st, v = r[t]
                sel = np.abs(st - at) <= args.window
                vals.append(f"{v[sel].mean():.4g} (n={sel.sum()})" if sel.any() else "-")
            print(f"  {t:28s} A {vals[0]:>16s}   B {vals[1]:>16s}")


if __name__ == "__main__":
    main()
