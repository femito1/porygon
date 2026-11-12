"""Sanity/timing check for att_common.XNet (equals Net / logged probs; Φ = logged value - V)."""
import time
import numpy as np, torch
from att_common import XNet, data, load, roles, ROLES, device
from wsp_common import Net
torch.set_grad_enabled(False)
d = data()
print({k: v.shape for k, v in d.items()})
dev = device(); print("device", dev)
pol = load("500M", dev); n = XNet(pol)
x = torch.as_tensor(d["obs"][:2048], dtype=torch.float32, device=dev); m = torch.as_tensor(d["mask"][:2048], device=dev)
t = time.time(); lg, v = n.fwd(x, m); print("fwd 2048: %.2fs" % (time.time() - t))
lg2, v2 = Net(pol).forward(x, m)
print("XNet vs Net", float((lg.softmax(-1) - lg2.softmax(-1)).abs().max()), float((v - v2).abs().max()))
print("vs logged probs", float(np.abs(lg.softmax(-1).cpu().numpy() - d["probs"][:2048]).max()))
phi = d["value"][:2048] - v.cpu().numpy()
print("phi stats", np.nanmin(phi), np.nanmax(phi), np.nanmean(phi))
r = roles(d["obs"], d["mask"])
print("role counts per decision (mean):", {ROLES[i]: float((r == i).sum(1).mean()) for i in range(10)})
print("own_act count dist", np.bincount((r == 1).sum(1)), "opp_act", np.bincount((r == 4).sum(1)))
print("won", np.unique(d["won"], return_counts=True), "games", len(np.unique(d["game"])))
print("value nan", np.isnan(d["value"]).sum())
