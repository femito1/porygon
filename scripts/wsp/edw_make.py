"""edw step 5: write the designed ActionHead weights into the checkpoint and save pool-format .pt files;
verify each with porygon.model.load_policy (stock forward): logits equal the cached-head evaluation, value
unchanged, weights changed counted.
    python scripts/wsp/edw_make.py 500M s1 s2 s3
"""
import json
import sys

import numpy as np
import torch

from beh_common import load_local
from edw_common import LOCAL, OUT, head_logits, split
from edw_lib import write_policy
from porygon.model import load_policy
from wsp_common import CKPTS, load

torch.set_num_threads(2)
tag, names = sys.argv[1], sys.argv[2:]
d = load_local()
te = np.flatnonzero(~split(d["game"]))[:2000]
x = torch.as_tensor(d["obs"][te].astype(np.float32)); m = torch.as_tensor(d["mask"][te])
base = load(CKPTS.get(tag, tag))
base_sd = {k: v.clone() for k, v in base.state_dict().items()}
obs = {"observation": x, "action_mask": m}
with torch.no_grad():
    v0 = base.predict_values(obs).squeeze(-1)
fin = np.load(LOCAL / f"{tag}_full.npz")["fin"][te]
out_dir = __import__("pathlib").Path("/vol/bitbucket/fr1226/porygon-ws-wsp/ckpt/edits"); out_dir.mkdir(parents=True, exist_ok=True)
for nm in names:
    W2 = torch.load(LOCAL / f"W_{tag}_{nm}.pt")
    p = load(CKPTS.get(tag, tag))
    write_policy(p, W2)
    path = out_dir / f"{tag}_switch_{nm}.pt"
    p.save(str(path))
    q = load_policy(path)
    with torch.no_grad():
        dist = q.get_distribution(obs).distribution.probs.double().numpy()
        v1 = q.predict_values(obs).squeeze(-1)
    lg = head_logits(fin, W2, d["mask"][te])
    pc = np.exp(lg - lg.max(1, keepdims=True)); pc /= pc.sum(1, keepdims=True)
    sd = q.state_dict()
    n_ch = sum(int((sd[k] != base_sd[k]).sum()) for k in sd if not k.startswith(("pi_features", "vf_features")))
    tot = sum(sd[k].numel() for k in sd if not k.startswith(("pi_features", "vf_features")))
    changed_keys = [k for k in sd if not k.startswith(("pi_features", "vf_features")) and (sd[k] != base_sd[k]).any()]
    rep = {"path": str(path), "weights_changed": n_ch, "weights_total": tot, "changed_tensors": changed_keys,
           "max|p_stock - p_cached_head|": float(np.abs(dist - pc).max()), "max|dV|": float((v1 - v0).abs().max())}
    (OUT / f"make_{tag}_{nm}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
