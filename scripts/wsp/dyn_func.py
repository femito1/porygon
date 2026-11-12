"""dyn: each checkpoint's policy and value on the logged real positions ($WSP/data/decisions.npz).

Writes OUT_DIR/func/<steps>.npz with probs (float16 [N,10]) and value V (float32 [N]) per checkpoint; skips
checkpoints already done. Forward = wsp_common.Net (verified equal to the stock forward).

    python scripts/wsp/dyn_func.py OUT_DIR DATA_DIR [--device cpu] [--which lineage|arms|all]
"""
import argparse
import os
import time

import numpy as np
import torch

from dyn_ckpts import arms, lineage, read
from wsp_common import Net, load, load_data, run_batched


def pick():
    lin = lineage()
    coarse = {2499840, 9999360, 21261312, 65950032, 100949682, 135949332, 170948982, 195948732, 235948332,
              273474432, 300028848}
    finals = {300028848, 400029852, 403267524, 500036720, 600002347}
    # 300-600M: the 10M grid (checkpoints at ...5M) plus the run finals; set DYN_ALL5M=1 for every 5M
    every5 = os.environ.get("DYN_ALL5M") == "1"
    sel = [(s, p) for s, p in lin if s in coarse or s in finals
           or (s >= 300_000_000 and (every5 or s % 10_000_000 >= 5_000_000))]
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--which", default="all")
    ap.add_argument("--only", default="", help="comma-separated lineage steps to run (overrides --which)")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    os.makedirs(f"{a.out}/func", exist_ok=True)
    d = load_data(a.data)
    x, mask = d["obs"], d["mask"]
    pol = load("500M", device=a.device)
    net = Net(pol)
    todo = []
    if a.only:
        want = {int(v) for v in a.only.split(",")}
        todo = [("lin", s, p) for s, p in lineage() if s in want]
    elif a.which in ("lineage", "all"):
        todo += [("lin", s, p) for s, p in pick()]
    if not a.only and a.which in ("arms", "all"):
        for arm, ck in arms().items():
            todo += [(arm, s, p) for s, p in ck if s in (620002347,) or s > 640_000_000]
    first = {2499840, 21261312, 100949682, 195948732, 300028848, 350028848, 400029852, 450267524, 500036720,
             550036720, 600002347, 640025994, 640025414, 640032682}
    todo.sort(key=lambda t: (t[1] not in first, t[1] % 10_000_000 < 5_000_000, t[1]))  # key ones, then 10M grid
    for tag, s, p in todo:
        f = f"{a.out}/func/{s}.npz" if tag == "lin" else f"{a.out}/func/{tag}_{s}.npz"
        if os.path.exists(f):
            continue
        t = time.time()
        sd, _, n = read(p, optim=False)
        pol.load_state_dict(sd)
        pol.eval()
        r = run_batched(net, x, mask, bs=4096, device=a.device)
        np.savez(f, probs=r["probs"].astype(np.float16), value=r["value"].astype(np.float32), steps=n, path=p)
        print(tag, n, f"{time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
