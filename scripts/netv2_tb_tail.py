"""Last N values of TensorBoard scalars matching a prefix (diagnostic): python scripts/netv2_tb_tail.py TB_DIR PREFIX [N]"""

import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tb, prefix = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
ea = EventAccumulator(tb, size_guidance={"scalars": 0})
ea.Reload()
for tag in sorted(t for t in ea.Tags()["scalars"] if t.startswith(prefix)):
    ev = ea.Scalars(tag)[-n:]
    print(f"{tag:32s} " + " ".join(f"{e.value:.3f}" for e in ev) + f"   (steps {ev[0].step}..{ev[-1].step})")
