"""Print the latest values (and a few history points) of a run's TensorBoard scalars.
    python scripts/fastenv/tb_scalars.py RUN_DIR [TAG ...]"""
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator as ea

run = Path(sys.argv[1])
tags = sys.argv[2:] or ["time/fps", "rollout/ep_len_mean", "train/entropy_loss", "train/explained_variance",
                        "train/approx_kl", "train/value_loss"]
for d in sorted({p.parent for p in run.rglob("events.out.tfevents.*")}):
    a = ea.EventAccumulator(str(d), size_guidance={"scalars": 0})
    a.Reload()
    have = a.Tags()["scalars"]
    print(d)
    for t in tags:
        if t in have:
            s = a.Scalars(t)
            pts = [s[0], s[len(s) // 2], s[-1]] if len(s) > 2 else s
            print(f"  {t}: " + "  ".join(f"{e.step / 1e6:.2f}M={e.value:.4g} ({(e.wall_time - s[0].wall_time) / 3600:.2f}h)"
                                       for e in pts))
