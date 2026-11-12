"""dyn: training-log bookkeeping for the whole gen4 PPO lineage.

Reads every tensorboard event file of the lineage (all scalars) and the `[eval @ N]` lines of every train.log,
writes one long CSV (source, tag, step, value, wall) and the head of each train.log (args / first lr) so the
TOTAL each segment used can be reconstructed from lr(x) = 10^-4.23/(8x+1)^1.5, x = step/TOTAL.

    python scripts/wsp/dyn_logs.py OUT_DIR
"""
import csv
import glob
import os
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

G = "/vol/bitbucket/fr1226/gloria/runs"
RUNS = ["20261019-140413", "20261020-181327-299545", "20261021-145709-300011", "20261022-120923-300097",
        "20261023-112550-300355", "20261023-125508-300389", "20261024-135245-300664", "20261025-171535-301033",
        "20261025-195826-301040", "20261026-082356-301120"] + sorted(os.path.basename(p) for p in glob.glob(f"{G}/arm-*"))
EVAL = re.compile(r"\[eval @ ([\d,]+)\] (.*)")


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/scalars.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "tag", "step", "value", "wall"])
        for r in RUNS:
            for ev in sorted(glob.glob(f"{G}/{r}/tb/*/events.out.tfevents.*")):
                acc = EventAccumulator(ev, size_guidance={"scalars": 0})
                acc.Reload()
                src = f"{r}/{os.path.basename(ev)}"
                n = 0
                for tag in acc.Tags()["scalars"]:
                    for e in acc.Scalars(tag):
                        w.writerow([src, tag, e.step, e.value, e.wall_time])
                        n += 1
                print(src, n, "scalars", flush=True)
            log = f"{G}/{r}/train.log"
            if not os.path.exists(log):
                continue
            first_lr = None
            ts = None
            with open(log, errors="replace") as fh:
                for line in fh:
                    m = re.search(r"total_timesteps\s*\|\s*(\d+)", line)
                    if m:
                        ts = int(m.group(1))
                    m = re.search(r"learning_rate\s*\|\s*([\d.e+-]+)", line)
                    if m:
                        w.writerow([f"{r}/train.log", "log/learning_rate", ts, float(m.group(1)), ""])
                        if first_lr is None:
                            first_lr = (ts, float(m.group(1)))
                    for k in ("approx_kl", "clip_fraction", "entropy_loss", "explained_variance", "value_loss",
                              "policy_gradient_loss", "ep_len_mean", "ep_rew_mean", "n_updates", "fps"):
                        m = re.search(rf"\|\s*{k}\s*\|\s*([-\d.e+]+)", line)
                        if m:
                            w.writerow([f"{r}/train.log", f"log/{k}", ts, float(m.group(1)), ""])
                    m = EVAL.search(line)
                    if m:
                        step = int(m.group(1).replace(",", ""))
                        for kv in m.group(2).split():
                            k, v = kv.split("=")
                            w.writerow([f"{r}/train.log", f"eval/{k}", step, float(v.rstrip("%")) / 100, ""])
            print(r, "train.log first lr", first_lr, flush=True)


if __name__ == "__main__":
    main()
