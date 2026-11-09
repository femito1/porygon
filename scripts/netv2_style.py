"""Play-style masses of v2 nets on held-out supervised rows (diagnostic, like wsp's tracker columns): at voluntary
decisions (>= 2 legal actions, at least one move legal) the mean policy probability on switches (actions 0-5), on
attacking moves and on status moves (move actions 6-9 / tera 22-25 by the active mon's move category, porygon.v2.tables
move_raw: 3 = status), plus entropy and P(top >= 0.9).
    python scripts/netv2_style.py --gen 9 --set GLOB --net NAME=PATH ... [--max-rows 20000] [--temperature 1]"""

import argparse
import json

import numpy as np
import torch

from porygon.v2.model import batch_to_tensors, load
from porygon.v2.pretrain import StreamSource
from porygon.v2.tables import move_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--set", required=True, help="GLOB of v2 shards (held-out rows by the crc32 game split)")
    p.add_argument("--net", action="append", required=True, help="NAME=PATH")
    p.add_argument("--val-frac", type=float, default=0.06)
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = torch.device(a.device)
    d = StreamSource(a.set, [], a.gen, a.val_frac, 1e9, 0).vdata
    n = min(len(d["z"]), a.max_rows)
    d = {k: v[:n] for k, v in d.items()}
    cat = move_raw(a.gen)["category"]
    mask = d["action_mask"] > 0
    nA = mask.shape[1]
    active = np.argmax(d["mon_num"][:, :6, 2], axis=1)  # mon_num[..., 2] = active flag; slots 0-5 = our team
    mv = d["moves"][np.arange(n), active, :4].astype(np.int64)
    mcat = np.where((mv >= 0) & (mv < len(cat)), cat[np.clip(mv, 0, len(cat) - 1)], 0)  # [n, 4]
    is_status = np.zeros((n, nA), bool)
    is_attack = np.zeros((n, nA), bool)
    for j in range(4):
        for act in ([6 + j] + ([22 + j] if nA > 22 else [])):
            is_status[:, act] = mcat[:, j] == 3
            is_attack[:, act] = np.isin(mcat[:, j], (1, 2))
    is_switch = np.zeros((n, nA), bool)
    is_switch[:, :6] = True
    move_legal = (mask & (is_status | is_attack)).any(1)
    vol = (mask.sum(1) >= 2) & move_legal
    out = {"rows": int(n), "voluntary": int(vol.sum())}
    for name, path in (x.split("=", 1) for x in a.net):
        net = load(path, dev).eval()
        probs = []
        for i in range(0, n, 1024):
            mb = batch_to_tensors({k: v[i:i + 1024] for k, v in d.items()
                                   if k not in ("game", "src", "weight", "z", "target")}, dev)
            m = mb["action_mask"] > 0
            with torch.no_grad():
                lg = net(mb, aux=False)["logits"].float() / a.temperature
                lg = torch.where(m, lg, torch.finfo(torch.float32).min)
                probs.append(torch.softmax(lg, -1).cpu().numpy())
        q = np.concatenate(probs)[vol]
        ent = -(q * np.log(np.clip(q, 1e-12, 1))).sum(-1)
        out[name] = {"switch": float((q * is_switch[vol]).sum(1).mean()), "attack": float((q * is_attack[vol]).sum(1).mean()),
                     "status": float((q * is_status[vol]).sum(1).mean()), "entropy": float(ent.mean()),
                     "ptop_0.9": float((q.max(1) >= 0.9).mean())}
        r = out[name]
        print(f"{name:14s} switch {r['switch']:.3f} attack {r['attack']:.3f} status {r['status']:.3f} "
              f"entropy {r['entropy']:.3f} P(top>=0.9) {r['ptop_0.9']:.3f}", flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
