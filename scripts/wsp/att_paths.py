"""Which heads carry opponent-active information into our move scores? (path patching)

    python scripts/wsp/att_paths.py CKPT [--n 4096]

Corruption: the opponent-active mon block (all its inputs) is replaced by the opponent-active block of another
random decision; our own move numerics (incl. the type-effectiveness feature computed vs the true opponent) are
left unchanged, so any change of the move scores must come through attention (or the field token context).
Effect measures: centred legal-move logits (logit − mean over legal moves; the move *preference*),
Δ = ‖c_corrupt − c_clean‖² averaged over decisions with ≥2 legal moves; KL of the whole policy; |ΔV|.
Path patching: rerun the corrupted input but restore one head's clean z at (a) the move-token queries 13-16,
(b) the field query 0, (c) all queries; and whole layers at the move queries. Fraction of Δ removed =
1 − Δ_patched / Δ_corrupt. Also the reverse (noising): clean input, corrupted z of one head at move queries
(how much of the change a single head can carry by itself).
Output $WSP/out/att/paths_<CKPT>.json
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from att_common import CKPT, OUT, XNet, data, device, load, roles
from porygon.encoder import FIELD_DIM, MON_DIM

torch.set_grad_enabled(False)


class PNet(XNet):
    def layer_p(self, i, x, patch=None, cache=None):
        """patch: (z_src [B,T,H,dh], positions list, heads list) — overwrite z at those positions/heads."""
        L = self.layers[i]
        B, T, D = x.shape
        a = L.norm1(x)
        qkv = F.linear(a, L.self_attn.in_proj_weight, L.self_attn.in_proj_bias).view(B, T, 3, self.h, self.dh)
        q, k, v = qkv.unbind(2)
        pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.dh)).softmax(-1)
        z = torch.einsum("bhqk,bkhd->bqhd", pat, v)
        if patch is not None:
            zs, pos, hs = patch
            z = z.clone()
            for hh in hs:
                z[:, pos, hh] = zs[:, pos, hh]
        if cache is not None:
            cache[i] = z
        xm = x + L.self_attn.out_proj(z.reshape(B, T, D))
        return xm + L.linear2(F.gelu(L.linear1(L.norm2(xm))))

    def run(self, x, mask, patches=None, cache=None):
        h = self.embed_x(x, None)
        for i in range(3):
            h = self.layer_p(i, h, (patches or {}).get(i), cache)
        return self.head(h, mask)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    dev = device()
    d = data()
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(d["obs"]), a.n, replace=False))
    obs = d["obs"][sel].astype(np.float32); mask = d["mask"][sel]
    R = roles(obs, mask)
    opp_i = (R[:, 7:13] == 4).argmax(1)              # opp slot 0-5
    perm = rng.permutation(len(obs))
    xc = obs.copy()
    for b in range(len(obs)):
        src = perm[b]
        o_dst = FIELD_DIM + (6 + opp_i[b]) * MON_DIM
        o_src = FIELD_DIM + (6 + opp_i[src]) * MON_DIM
        xc[b, o_dst:o_dst + MON_DIM] = obs[src, o_src:o_src + MON_DIM]
    net = PNet(load(CKPT[a.ckpt], dev))
    legal_mv = torch.as_tensor(mask[:, 6:10] > 0, device=dev)
    nleg = legal_mv.sum(1)
    use = nleg >= 2
    bs = 1024

    def batches():
        for s in range(0, len(obs), bs):
            yield slice(s, s + bs)

    def centred(lg):
        m = lg[:, 6:10]
        return torch.where(legal_mv_b, m - (m * legal_mv_b).sum(1, keepdim=True) / nleg_b.clamp(min=1)[:, None], 0)

    # cache clean and corrupted z per batch, then patch runs
    names = []
    for i in range(3):
        for h in range(8):
            names += [f"L{i}H{h}@moves", f"L{i}H{h}@field", f"L{i}H{h}@all", f"L{i}H{h}@moves_noise"]
        names += [f"L{i}@moves", f"L{i}@field", f"L{i}@all"]
    acc = {n: 0.0 for n in names}
    tot = {"move_d2": 0.0, "kl": 0.0, "dv": 0.0, "switch_d2": 0.0, "n_use": 0, "n": 0, "clean_move_var": 0.0}
    for sl in batches():
        xb = torch.as_tensor(obs[sl], device=dev); xcb = torch.as_tensor(xc[sl], device=dev)
        mb = torch.as_tensor(mask[sl], device=dev)
        legal_mv_b, nleg_b, use_b = legal_mv[sl], nleg[sl], use[sl]
        cc, ck = {}, {}
        lg0, v0 = net.run(xb, mb, cache=cc)
        lg1, v1 = net.run(xcb, mb, cache=ck)
        c0, c1 = centred(lg0), centred(lg1)
        d2 = ((c1 - c0) ** 2).sum(1)
        tot["move_d2"] += float(d2[use_b].sum()); tot["n_use"] += int(use_b.sum()); tot["n"] += len(xb)
        tot["clean_move_var"] += float((c0 ** 2).sum(1)[use_b].sum())
        p0, p1 = lg0.softmax(-1), lg1.softmax(-1)
        tot["kl"] += float((p0 * ((p0 + 1e-12).log() - (p1 + 1e-12).log())).sum(1).sum())
        tot["dv"] += float((v1 - v0).abs().sum())
        for nm in names:
            if nm.count("H") and "@" in nm and nm.split("@")[0].count("H"):
                lh, where = nm.split("@")
                i, h = int(lh[1]), int(lh.split("H")[1]); hs = [h]
            else:
                lh, where = nm.split("@"); i = int(lh[1]); hs = list(range(8))
            pos = {"moves": [13, 14, 15, 16], "field": [0], "all": list(range(17)), "moves_noise": [13, 14, 15, 16]}[where]
            if where == "moves_noise":   # clean run, corrupted z of this head at the move queries
                lg, _ = net.run(xb, mb, patches={i: (ck[i], pos, hs)})
            else:                         # corrupted run, clean z restored
                lg, _ = net.run(xcb, mb, patches={i: (cc[i], pos, hs)})
            dd = ((centred(lg) - c0) ** 2).sum(1)
            acc[nm] += float(dd[use_b].sum())
        print(sl, flush=True)
    full = tot["move_d2"] / tot["n_use"]
    out = {"ckpt": a.ckpt, "n": tot["n"], "n_use": tot["n_use"],
           "move_pref_change_ms": full, "clean_move_pref_ms": tot["clean_move_var"] / tot["n_use"],
           "kl": tot["kl"] / tot["n"], "abs_dv": tot["dv"] / tot["n"], "frac_removed": {}, "frac_carried": {}}
    for nm in names:
        m = acc[nm] / tot["n_use"]
        if nm.endswith("_noise"):
            out["frac_carried"][nm.replace("@moves_noise", "")] = m / full
        else:
            out["frac_removed"][nm] = 1 - m / full
    (OUT / f"paths_{a.ckpt}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
    fr = out["frac_removed"]
    for i in range(3):
        print(f"L{i}", "moves", round(fr[f"L{i}@moves"], 3), "field", round(fr[f"L{i}@field"], 3), "all", round(fr[f"L{i}@all"], 3))
        print("  per head @moves", [round(fr[f"L{i}H{h}@moves"], 3) for h in range(8)])
        print("  per head @all  ", [round(fr[f"L{i}H{h}@all"], 3) for h in range(8)])
        print("  carried alone  ", [round(out["frac_carried"][f"L{i}H{h}"], 3) for h in range(8)])


if __name__ == "__main__":
    main()
