"""Early-training diagnosis of network v2 checkpoints on the fixed position set (v2diag_record.py).
Per checkpoint: embedding drift vs init (dead tables?), dex-feature share of each id vector, attention selector
statistics per layer/head (runaway one-hot heads?), policy entropy / sharpness / switching, value AUC vs outcomes.
Flags alarming values. Writes one JSON per checkpoint and prints a compact table.

    python scripts/wsp/v2diag.py --init INIT.pt --pos v2diag/pos --out v2diag/out CKPT [CKPT ...]
"""

import argparse
import glob
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import porygon.v2.model as M

_REAL_SDPA = F.scaled_dot_product_attention
_ATT: list = []


def _sdpa_record(q, k, v, attn_mask=None, **kw):
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if attn_mask is not None:
        s = s + attn_mask
    w = torch.softmax(s, -1)
    finite = torch.isfinite(s)
    sf = torch.where(finite, s, torch.zeros_like(s))
    n = finite.sum(-1).clamp(min=1)
    mean = sf.sum(-1) / n
    std = torch.sqrt((torch.where(finite, (sf - mean[..., None]) ** 2, torch.zeros_like(s))).sum(-1) / n)
    ent = -(torch.where(w > 0, w * torch.log(w), torch.zeros_like(w))).sum(-1)
    _ATT.append({"maxw": w.amax(-1).mean((0, 2)), "p_onehot": (w.amax(-1) > 0.9).float().mean((0, 2)),
                 "ent": ent.mean((0, 2)), "score_std": std.mean((0, 2)), "uniform_ent": torch.log(n.float()).mean((0, 2))})
    return w @ v


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def load_npz_pos(pattern, searched_only=True):
    """Positions from the data stream's v2 shards (e.g. gen9 search labels): ref policy = the search target,
    ref value = the search's root value, outcome from z."""
    rows = []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        okeys = [k for k in z.files if k.startswith("obs/")]
        cols = {k[4:]: z[k] for k in okeys}
        sel = np.where(z["searched"] > 0)[0] if (searched_only and "searched" in z.files) else np.arange(len(z["z"]))
        for i in sel:
            rows.append({"gid": (f, int(z["game"][i])), "obs": {k: v[i] for k, v in cols.items()},
                         "v1_probs": z["target"][i].astype(np.float32), "v1_value": float(z["root_value"][i])
                         if "root_value" in z.files else 0.0, "won": bool(z["z"][i] > 0)})
    return rows


def load_pos(d):
    if d.endswith(".npz") or "*" in d:
        return load_npz_pos(d)
    rows = []
    for f in sorted(glob.glob(f"{d}/*/rows.pkl")) + sorted(glob.glob(f"{d}/rows.pkl")):
        with open(f, "rb") as fh:
            rows += pickle.load(fh)
    return rows


def run_net(net, rows, bs=512):
    out = {"logits": [], "value": []}
    _ATT.clear()
    M.F.scaled_dot_product_attention = _sdpa_record
    try:
        with torch.no_grad():
            for s in range(0, len(rows), bs):
                t = M.obs_to_tensors([r["obs"] for r in rows[s:s + bs]], "cpu")
                o = net(t, aux=False)
                out["logits"].append(o["logits"].numpy())
                out["value"].append(o["value"].numpy())
    finally:
        M.F.scaled_dot_product_attention = _REAL_SDPA
    L = net.cfg["n_layers"]
    att = {}
    for key in _ATT[0]:
        per = torch.stack([a[key] for a in _ATT]).view(-1, L, _ATT[0][key].shape[0]).mean(0)  # [L, H]
        att[key] = per.numpy().round(4).tolist()
    return {k: np.concatenate(v) for k, v in out.items()}, att


def emb_stats(net, init, rows):
    seen = {k: set() for k in ("species", "moves", "items", "abilities")}
    keymap = {"species": "species", "moves": "moves", "items": "item", "abilities": "ability"}
    for r in rows[:5000]:
        for kind, f in keymap.items():
            seen[kind].update(np.unique(r["obs"][f]).tolist())
    res = {}
    sd, sd0 = net.state_dict(), init.state_dict()
    for kind in seen:
        w, w0 = sd[f"ent_{kind}.emb.weight"], sd0[f"ent_{kind}.emb.weight"]
        ids = torch.tensor(sorted(i for i in seen[kind] if i > 1))
        d = (w - w0)
        ent = getattr(net, f"ent_{kind}")
        with torch.no_grad():
            feats = ent.feat(ent.table)[ids] if ent.table is not None else None
        res[kind] = {"n_seen": len(ids),
                     "rms_init_seen": float(w0[ids].pow(2).mean().sqrt()),
                     "rms_drift_seen": float(d[ids].pow(2).mean().sqrt()),
                     "rel_drift_seen": float(d[ids].norm() / w0[ids].norm().clamp(min=1e-12)),
                     "frac_rows_moved": float((d.abs().amax(1) > 1e-7).float().mean()),
                     "feat_over_id_norm": None if feats is None else float(feats.norm(dim=1).mean() / w[ids].norm(dim=1).mean())}
    groups = {}
    for n, p in sd.items():
        if not torch.is_floating_point(p) or n not in sd0 or p.numel() < 64:
            continue
        g = n.split(".")[0] if not n.startswith("blocks") else ".".join(n.split(".")[:2])
        a = groups.setdefault(g, [0.0, 0.0])
        a[0] += float((p - sd0[n]).pow(2).sum())
        a[1] += float(sd0[n].pow(2).sum())
    res["_groups_rel_drift"] = {g: round(math.sqrt(a / max(b, 1e-12)), 4) for g, (a, b) in groups.items()}
    return res


def policy_stats(o, rows):
    lg = o["logits"]
    mask = np.stack([r["obs"]["action_mask"] for r in rows]) > 0
    nA = mask.shape[1]
    lg = np.where(mask, lg, -1e30)
    p = np.exp(lg - lg.max(1, keepdims=True))
    p = np.where(mask, p, 0)
    p /= p.sum(1, keepdims=True)
    v1 = np.stack([np.pad(r["v1_probs"], (0, max(0, nA - len(r["v1_probs"]))))[:nA] for r in rows])
    ent = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-30)), 0)).sum(1)
    vol = mask[:, :6].any(1) & mask[:, 6:].any(1)
    won = np.array([r["won"] for r in rows])
    v = o["value"]
    return {"entropy": float(ent.mean()), "v1_entropy": float(-(np.where(v1 > 0, v1 * np.log(np.maximum(v1, 1e-30)), 0)).sum(1).mean()),
            "p_top_ge_0.9": float((p.max(1) >= 0.9).mean()),
            "argmax_eq_v1_argmax": float((p.argmax(1) == v1.argmax(1)).mean()),
            "voluntary_P_switch": float(p[vol, :6].sum(1).mean()), "v1_voluntary_P_switch": float(v1[vol, :6].sum(1).mean()),
            "value_auc": auc(v, won), "v1_value_auc": auc(np.array([r["v1_value"] for r in rows]), won),
            "mean_abs_value": float(np.abs(v).mean()), "value_std": float(v.std())}


def flags(st):
    f = []
    for kind, e in st["emb"].items():
        if kind.startswith("_"):
            continue
        if st["step"] >= 5e6 and e["rel_drift_seen"] < 0.01:
            f.append(f"DEAD? {kind} embeddings moved only {e['rel_drift_seen']:.4f} relative to init")
    for l, (mw, sstd) in enumerate(zip(st["att"]["maxw"], st["att"]["score_std"])):
        for h, (a, b) in enumerate(zip(mw, sstd)):
            if a > 0.85 or b > 8:
                f.append(f"SELECTOR? L{l}H{h} mean max weight {a:.2f}, score std {b:.1f}")
    p = st["policy"]
    if st["step"] >= 5e6 and p["entropy"] < 0.1:
        f.append(f"ENTROPY COLLAPSE? {p['entropy']:.3f} nats")
    if st["step"] >= 5e6 and p["value_auc"] < 0.6:
        f.append(f"VALUE WEAK? AUC {p['value_auc']:.3f} (v1 {p['v1_value_auc']:.3f})")
    if st["step"] >= 5e6 and p["value_std"] < 0.02:
        f.append(f"VALUE CONSTANT? std {p['value_std']:.4f}")
    return f


def step_of(path):
    name = Path(path).stem
    digits = "".join(c for c in name.split("_")[-1] if c.isdigit())
    return 0 if "init" in name else (int(digits) if digits else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--init", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12000)
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", 2)))
    rows = load_pos(a.pos)
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), min(a.n, len(rows)), replace=False))]
    init = M.load(a.init)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for c in a.ckpts:
        net = M.load(c)
        o, att = run_net(net, rows)
        st = {"ckpt": c, "step": step_of(c), "n_pos": len(rows), "emb": emb_stats(net, init, rows), "att": att,
              "policy": policy_stats(o, rows)}
        st["flags"] = flags(st)
        (out / f"{Path(c).parent.name}_{Path(c).stem}.json").write_text(json.dumps(st, indent=1))
        e = st["emb"]
        print(f"== {c} (step {st['step']:,})")
        print("  emb rel drift (seen rows):", {k: round(v["rel_drift_seen"], 4) for k, v in e.items() if not k.startswith("_")},
              " feat/id norm:", {k: (None if v["feat_over_id_norm"] is None else round(v["feat_over_id_norm"], 2)) for k, v in e.items() if not k.startswith("_")})
        print("  group rel drift:", e["_groups_rel_drift"])
        print("  att max weight per layer (max over heads):", [round(max(x), 3) for x in att["maxw"]],
              " score std (max):", [round(max(x), 2) for x in att["score_std"]],
              " entropy/uniform:", [round(float(np.mean(x) / np.mean(u)), 3) for x, u in zip(att["ent"], att["uniform_ent"])])
        print("  policy:", {k: round(v, 4) for k, v in st["policy"].items()})
        print("  FLAGS:", st["flags"] or "none")


if __name__ == "__main__":
    main()
