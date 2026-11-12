"""Policy statistics of v2 nets on held-out supervised rows (diagnostic, no training): per net and set, mean entropy
over rows with >= 2 legal actions, mean top probability, share of rows with top p >= 0.9, CE / top-1 vs the row's
target, and the target's own entropy; plus pairwise argmax agreement and KL between the nets.
    python scripts/netv2_policy_stats.py --gen 9 --set NAME=GLOB ... --net NAME=PATH ... [--max-rows 20000]"""

import argparse
import json

import numpy as np
import torch

from porygon.v2.model import batch_to_tensors, load
from porygon.v2.pretrain import StreamSource


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", type=int, required=True)
    p.add_argument("--set", action="append", required=True, help="NAME=GLOB (held-out rows by the crc32 game split)")
    p.add_argument("--net", action="append", required=True, help="NAME=PATH")
    p.add_argument("--val-frac", type=float, default=0.06)
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--temperature", type=float, default=1.0, help="report the policy at this sampling temperature")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = torch.device(a.device)
    nets = {k: load(v, dev).eval() for k, v in (x.split("=", 1) for x in a.net)}
    out = {}
    for spec in a.set:
        name, glob_ = spec.split("=", 1)
        src = StreamSource(glob_, [], a.gen, a.val_frac, 1e9, 0)  # held-out rows only (vdata)
        d = src.vdata
        rows = np.arange(len(d["z"]))[: a.max_rows]
        probs = {k: [] for k in nets}
        tgt, masks = [], []
        for i in range(0, len(rows), 1024):
            j = rows[i:i + 1024]
            mb = batch_to_tensors({k: v[j] for k, v in d.items() if k not in ("game", "src", "weight", "z", "target")}, dev)
            m = mb["action_mask"] > 0
            with torch.no_grad():
                for k, net in nets.items():
                    lg = net(mb, aux=False)["logits"].float() / a.temperature
                    lg = torch.where(m, lg, torch.finfo(torch.float32).min)
                    probs[k].append(torch.softmax(lg, -1).cpu().numpy())
            tgt.append(d["target"][j])
            masks.append(m.cpu().numpy())
        mask = np.concatenate(masks)
        t = np.concatenate(tgt) * mask
        has = t.sum(-1) > 0
        multi = mask.sum(-1) >= 2
        sel = has & multi
        t = t[sel] / t[sel].sum(-1, keepdims=True)
        res = {"rows": int(sel.sum()), "target_entropy": float(-(t * np.log(np.clip(t, 1e-12, 1))).sum(-1).mean()),
               "target_top_p": float(t.max(-1).mean())}
        P = {k: np.concatenate(v)[sel] for k, v in probs.items()}
        for k, q in P.items():
            ent = -(q * np.log(np.clip(q, 1e-12, 1))).sum(-1)
            res[k] = {"entropy": float(ent.mean()), "top_p": float(q.max(-1).mean()),
                      "ptop_0.9": float((q.max(-1) >= 0.9).mean()),
                      "ce_target": float(-(t * np.log(np.clip(q, 1e-12, 1))).sum(-1).mean()),
                      "top1_target": float((q.argmax(-1) == t.argmax(-1)).mean())}
        ks = list(P)
        for x in range(len(ks)):
            for y in range(x + 1, len(ks)):
                qa, qb = P[ks[x]], P[ks[y]]
                res[f"{ks[x]}~{ks[y]}"] = {
                    "argmax_agree": float((qa.argmax(-1) == qb.argmax(-1)).mean()),
                    "kl_a_b": float((qa * (np.log(np.clip(qa, 1e-12, 1)) - np.log(np.clip(qb, 1e-12, 1)))).sum(-1).mean())}
        out[name] = res
        print(name, json.dumps(res, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
