"""Learning-rate / Adam-step history of the gen4 PPO lineage from the runs' train.log tables.

For each run: (total_timesteps, learning_rate) per logged PPO update. Writes out/emb/lr_history.json and
prints per-segment sums Σlr and √Σlr² over Adam steps (Adam steps per update = n_epochs * rollout/batch).

    python scripts/wsp/emb_lr.py
"""
import json
import re
from pathlib import Path

G = Path("/vol/bitbucket/fr1226/gloria/runs")
RUNS = ["20261019-140413", "20261020-180025-299541", "20261020-181327-299545", "20261021-145709-300011",
        "20261022-120923-300097", "20261023-112550-300355", "20261023-125508-300389", "20261024-135245-300664",
        "20261025-171535-301033", "20261025-195826-301040", "20261026-082356-301120",
        "arm-ctrl-20261105-010231-371650", "arm-ctrl-20261105-115414-377837", "arm-x30-20261105-010656-371651",
        "arm-x300-20261105-005736-371541"]


def parse(path: Path):
    rows, ts = [], None
    for line in open(path, errors="replace"):
        m = re.search(r"total_timesteps\s*\|\s*(\d+)", line)
        if m:
            ts = int(m.group(1))
        m = re.search(r"learning_rate\s*\|\s*([0-9.eE+-]+)", line)
        if m and ts is not None:
            rows.append((ts, float(m.group(1))))
    return rows


def tb_rows(d: Path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows = []
    for ev in sorted(d.rglob("events.out.tfevents*")):
        ea = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        ea.Reload()
        if "train/learning_rate" in ea.Tags()["scalars"]:
            rows += [(int(e.step), float(e.value)) for e in ea.Scalars("train/learning_rate")]
    return sorted(rows)


def main():
    out = {}
    for r in RUNS:
        p = G / r / "train.log"
        rows = parse(p) if p.exists() else tb_rows(G / r / "tb")
        out[r] = rows
        if rows:
            print(f"{r}: {len(rows)} updates, steps {rows[0][0]:,}..{rows[-1][0]:,}, lr {rows[0][1]:.3g}..{rows[-1][1]:.3g}")
    Path("out/emb").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("out/emb/lr_history.json", "w"))


if __name__ == "__main__":
    main()
