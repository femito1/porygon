"""Small extra numbers for the attention report: base rate of status moves among legal moves (for L2H6),
and the share of decisions with ≥2 legal moves."""
import numpy as np
from att_common import data, roles
from porygon.encoder import FIELD_DIM, MON_DIM, MOVE_DIM
from wsp_common import MOVENUM_OFF
d = data(); obs, mask = d["obs"], d["mask"]
R = roles(obs, mask); N = len(obs); ar = np.arange(N)
own_i = (R[:, 1:7] == 1).argmax(1)
mons = obs[:, FIELD_DIM:].reshape(N, 12, MON_DIM)
mv = mons[ar, own_i, MOVENUM_OFF:].reshape(N, 4, MOVE_DIM).astype(np.float32)
legal = mask[:, 6:10] > 0
multi = legal.sum(1) >= 2
st = mv[..., 7] > .5
print("status share among legal moves (multi):", float(st[multi][legal[multi]].mean()))
print("P(net top move is status | multi):", float(st[multi][np.arange(multi.sum()), d["probs"][multi, 6:10].argmax(1)].mean()))
print("decisions with >=2 legal moves:", float(multi.mean()))
