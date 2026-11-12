"""dyn: weight averaging (SWA) over the late random walk. Late training random-walks in a basin (dyn_params:
R ≈ 1, efficiency 1.05, lag-1 cos −0.06), so a uniform average of the 5M checkpoints of a window should sit near
the basin centre. Builds the averaged weights for several windows, writes each as a loadable PPO zip (the 600M
zip with policy.pth replaced; the optimizer state is the 600M one) under OUT_DIR/swa/, and its policy/value on the
logged positions to OUT_DIR/func/swa_<name>.npz (read by dyn_func_an.py).

    python scripts/wsp/dyn_swa.py OUT_DIR DATA_DIR
"""
import io
import os
import sys
import time
import zipfile

import numpy as np
import torch

from dyn_ckpts import lineage, read
from wsp_common import Net, load, load_data, run_batched

WINDOWS = {"550-600": (550e6, 600.1e6), "500-600": (500e6, 600.1e6), "400-600": (400e6, 600.1e6),
           "300-600": (300e6, 600.1e6), "450-500": (450e6, 500.1e6)}


def main():
    out, data = sys.argv[1], sys.argv[2]
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(f"{out}/swa", exist_ok=True)
    lin = lineage()
    need = sorted({s for s, _ in lin for lo, hi in WINDOWS.values() if lo <= s <= hi})
    sums = {w: None for w in WINDOWS}
    cnt = {w: 0 for w in WINDOWS}
    t = time.time()
    for s, p in lin:
        if s not in need:
            continue
        sd, _, _ = read(p, optim=False)
        for w, (lo, hi) in WINDOWS.items():
            if lo <= s <= hi:
                if sums[w] is None:
                    sums[w] = {k: v.double().clone() for k, v in sd.items()}
                else:
                    for k in sums[w]:
                        sums[w][k] += sd[k].double()
                cnt[w] += 1
    print("read", len(need), "checkpoints", f"{time.time()-t:.0f}s", cnt, flush=True)
    d = load_data(data)
    pol = load("600M")
    ref = {k: v.clone() for k, v in pol.state_dict().items()}
    net = Net(pol)
    src = dict(lin)[600002347]
    for w, S in sums.items():
        avg = {k: (v / cnt[w]).to(ref[k].dtype) for k, v in S.items()}
        pol.load_state_dict(avg)
        pol.eval()
        r = run_batched(net, d["obs"], d["mask"], bs=4096)
        np.savez(f"{out}/func/swa_{w}.npz", probs=r["probs"].astype(np.float16), value=r["value"].astype(np.float32),
                 steps=cnt[w], path=w)
        buf = io.BytesIO()
        torch.save(avg, buf)
        with zipfile.ZipFile(src) as zi, zipfile.ZipFile(f"{out}/swa/swa_{w}.zip", "w") as zo:
            for item in zi.infolist():
                zo.writestr(item, buf.getvalue() if item.filename == "policy.pth" else zi.read(item.filename))
        dist = float(torch.sqrt(sum(((avg[k].double() - ref[k].double()) ** 2).sum() for k in avg
                                    if k.startswith(("features_extractor", "mlp_extractor", "action_net", "value_net")))))
        print(w, cnt[w], "checkpoints; |θ_swa − θ600| =", f"{dist:.3f}", flush=True)


if __name__ == "__main__":
    main()
