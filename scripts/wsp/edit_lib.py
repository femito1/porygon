"""Weight edits of the gen4 network (weight-space programming): each edit is a function policy -> None that writes
weights in place (no gradients), registered in EDITS with a one-line rationale. `make` loads the base checkpoint,
applies the edit, logs how many weights changed, checks the result on the logged positions (policy agreement /
KL vs the original, agreement with the search) and saves a pool-format .pt that every tool loads
(porygon.model.load_policy, h2h_dup, search --model).

    python scripts/wsp/edit_lib.py EDIT [EDIT ...] --base 500M --out ckpt/edits
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wsp_common import Net, load, load_data, run_batched

EDITS = {}


def edit(name, why):
    def deco(f):
        EDITS[name] = (f, why)
        return f
    return deco


@edit("temp05", "control: policy temperature 0.5 (score.2 weight and bias x2); sampled play closer to argmax")
def temp05(p):
    with torch.no_grad():
        p.action_net.score[2].weight.mul_(2.0)
        p.action_net.score[2].bias.mul_(2.0)


@edit("identity", "control: no change (A/A)")
def identity(p):
    pass


def changed(a: dict, b: dict) -> tuple[int, int]:
    n = tot = 0
    for k in a:
        if k.startswith(("pi_features_extractor", "vf_features_extractor")):
            continue
        tot += a[k].numel()
        n += int((a[k] != b[k]).sum())
    return n, tot


def summarize(base_net, new_net, d, idx):
    x, m = d["obs"][idx], d["mask"][idx]
    o = run_batched(base_net, x, m)
    e = run_batched(new_net, x, m)
    sh = d["shares"][idx]
    p0, p1 = o["probs"], e["probs"]
    kl = float(np.mean(np.sum(p0 * (np.log(p0 + 1e-9) - np.log(p1 + 1e-9)), 1)))
    a0, a1 = p0.argmax(1), p1.argmax(1)
    return {"KL(orig||edit)": kl, "top1_same": float((a0 == a1).mean()),
            "orig_top1=search": float((a0 == sh.argmax(1)).mean()),
            "edit_top1=search": float((a1 == sh.argmax(1)).mean()),
            "orig_E[share|argmax]": float(sh[np.arange(len(a0)), a0].mean()),
            "edit_E[share|argmax]": float(sh[np.arange(len(a1)), a1].mean()),
            "orig_E[share|sample]": float((sh * p0).sum(1).mean()),
            "edit_E[share|sample]": float((sh * p1).sum(1).mean()),
            "orig_entropy": float(-(p0 * np.log(p0 + 1e-9)).sum(1).mean()),
            "edit_entropy": float(-(p1 * np.log(p1 + 1e-9)).sum(1).mean()),
            "mean|dV|": float(np.abs(o["value"] - e["value"]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edits", nargs="+")
    ap.add_argument("--base", default="500M")
    ap.add_argument("--out", default="ckpt/edits")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-check", type=int, default=12000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_data(a.data)
    idx = np.random.default_rng(0).choice(len(d["obs"]), min(a.n_check, len(d["obs"])), replace=False)
    base = load(a.base)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    for name in a.edits:
        f, why = EDITS[name]
        p = load(a.base)
        f(p)
        n, tot = changed(base_sd, p.state_dict())
        s = summarize(Net(base), Net(p), d, idx)
        s.update({"edit": name, "why": why, "base": a.base, "weights_changed": n, "weights_total": tot,
                  "fraction_changed": n / tot})
        path = out / f"{a.base}_{name}.pt"
        p.save(str(path))
        (out / f"{a.base}_{name}.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s))


if __name__ == "__main__":
    main()
