"""Forward passes of the 500M / 600M nets over all logged decisions -> $WSP/out/beh/fwd_<tag>.npz
(probs, logits (masked = -inf replaced by nan), value V). Checks 500M probs against the logged probs and derives
Φ = logged value - V.   python scripts/wsp/beh_fwd.py 500M 600M"""
import sys
import time

import numpy as np
import torch

from beh_common import OUT, load_local
from wsp_common import Net, load

torch.set_num_threads(2)
OUT.mkdir(parents=True, exist_ok=True)
d = load_local()
for tag in sys.argv[1:]:
    t0 = time.time()
    net = Net(load(tag))
    L, V = [], []
    with torch.no_grad():
        for s in range(0, len(d["obs"]), 1024):
            x = torch.as_tensor(d["obs"][s:s + 1024], dtype=torch.float32)
            m = torch.as_tensor(d["mask"][s:s + 1024])
            lg, v = net.forward(x, m)
            L.append(lg.numpy()); V.append(v.numpy())
    lg = np.concatenate(L); v = np.concatenate(V)
    p = torch.as_tensor(lg).softmax(-1).numpy()
    lg = np.where(d["mask"] > 0, lg, np.nan)
    np.savez(OUT / f"fwd_{tag}.npz", logits=lg.astype(np.float32), probs=p.astype(np.float32), value=v)
    print(tag, f"{time.time()-t0:.0f}s", "max|p - logged| =", float(np.abs(p - d["probs"]).max()),
          "mean|p - logged| =", float(np.abs(p - d["probs"]).mean()), flush=True)
    if tag == "500M":
        phi = d["value"] - v
        print("phi stats", np.nanpercentile(phi, [0, 5, 50, 95, 100]))
