"""The near-one-hot head L2H6 (entropy 0.18 nats): which token does it select, and why?

    python scripts/wsp/att_l2h6.py [CKPT]
Per decision: the key every query attends to (argmax; agreement across queries), its role, and for selected move
tokens whether it is the net's argmax / the search's argmax / the highest bp×eff×STAB legal move / the lowest;
for selected opp tokens: which one. Also: the logged policy's probability of the selected move vs the others.
Output $WSP/out/att/l2h6_<CKPT>.json
"""
import json
import sys

import numpy as np
import torch

from att_common import CKPT, OUT, ROLES, XNet, boot_games, data, device, load, roles
from porygon.encoder import FIELD_DIM, MON_DIM, MOVE_DIM
from wsp_common import MOVENUM_OFF

torch.set_grad_enabled(False)


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else "500M"
    layer, head = 2, 6
    dev = device(); d = data()
    obs, mask = d["obs"], d["mask"]
    R = roles(obs, mask)
    net = XNet(load(CKPT[ck], dev))
    N = len(obs)
    sel_key = np.zeros(N, int); agree = np.zeros(N); maxw = np.zeros(N); probs = np.zeros((N, 10), np.float32)
    second = np.zeros(N, int)
    for s in range(0, N, 2048):
        xb = torch.as_tensor(obs[s:s + 2048], dtype=torch.float32, device=dev)
        mb = torch.as_tensor(mask[s:s + 2048], device=dev)
        cache = {}
        lg, _ = net.fwd(xb, mb, cache=cache)
        P = cache[f"L{layer}.pattern"][:, head]                     # B,Tq,Tk
        am = P.argmax(-1)                                           # B,Tq
        fq = am[:, 0]
        sel_key[s:s + 2048] = fq.cpu().numpy()
        agree[s:s + 2048] = (am == fq[:, None]).float().mean(1).cpu().numpy()
        maxw[s:s + 2048] = P.max(-1).values.mean(1).cpu().numpy()
        second[s:s + 2048] = P[:, 0].topk(2, -1).indices[:, 1].cpu().numpy()
        probs[s:s + 2048] = lg.softmax(-1).cpu().numpy()
    ar = np.arange(N)
    role_sel = R[ar, sel_key]
    out = {"ckpt": ck, "mean_max_weight": float(maxw.mean()), "query_agreement": float(agree.mean()),
           "role_of_selected": {ROLES[r]: float((role_sel == r).mean()) for r in range(10)},
           "slot_of_selected": np.bincount(sel_key, minlength=17).tolist()}
    # move selections
    mons = obs[:, FIELD_DIM:].reshape(N, 12, MON_DIM).astype(np.float32)
    own_i = (R[:, 1:7] == 1).argmax(1)
    mv = mons[ar, own_i, MOVENUM_OFF:].reshape(N, 4, MOVE_DIM)
    bp = mv[..., 0] * 150
    mult = np.where(mv[..., 28] > .5, 0, 2.0 ** (2 * mv[..., 27]))
    stab = np.where(mv[..., 26] > .5, 1.5, 1.0)
    dmg = bp * mult * stab
    legal = mask[:, 6:10] > 0
    dmg_l = np.where(legal, dmg, -1)
    ism = (sel_key >= 13)
    j = np.clip(sel_key - 13, 0, 3)
    nleg = legal.sum(1)
    multi = ism & (nleg >= 2)
    g = d["game"]
    net_arg = probs.argmax(1); s_arg = d["shares"].argmax(1)
    out["moves"] = {
        "n": int(multi.sum()),
        "is_net_argmax": boot_games((net_arg[multi] == 6 + j[multi]).astype(float), g[multi]),
        "is_net_top_move": boot_games((probs[multi, 6:10].argmax(1) == j[multi]).astype(float), g[multi]),
        "is_search_argmax": boot_games((s_arg[multi] == 6 + j[multi]).astype(float), g[multi]),
        "is_max_damage_legal": boot_games((dmg_l[multi].argmax(1) == j[multi]).astype(float), g[multi]),
        "is_status_move": boot_games((mv[multi, j[multi], 7] > .5).astype(float), g[multi]),
        "selected_legal": float(legal[ism, j[ism]].mean()),
        "chance_top_move": float((1 / nleg[multi]).mean()),
        "net_prob_selected": float(probs[multi, 6 + j[multi]].mean()),
        "net_prob_other_legal_mean": float(((probs[multi, 6:10] * legal[multi]).sum(1) - probs[multi, 6 + j[multi]]).mean()
                                           / max(1e-9, (nleg[multi] - 1).mean())),
    }
    # when does it select the opp active instead of a move?
    opp_sel = role_sel == 4
    has_mv = legal.any(1)
    out["select_opp_active_given_moves_legal"] = float(opp_sel[has_mv].mean())
    out["select_opp_active_given_no_moves"] = float(opp_sel[~has_mv].mean()) if (~has_mv).any() else None
    fs = (d["obs"][:, 85] > .5)
    out["force_switch_frac"] = float(fs.mean())
    out["role_given_force_switch"] = {ROLES[r]: float((role_sel[fs] == r).mean()) for r in range(10)}
    out["role_given_not_force_switch"] = {ROLES[r]: float((role_sel[~fs] == r).mean()) for r in range(10)}
    # does selecting the opp active coincide with the net preferring to switch?
    sw_mass = probs[:, :6].sum(1)
    out["net_switch_mass_when_opp_sel"] = float(sw_mass[opp_sel & has_mv].mean())
    out["net_switch_mass_when_move_sel"] = float(sw_mass[ism & has_mv].mean())
    out["search_switch_mass_when_opp_sel"] = float(d["shares"][opp_sel & has_mv, :6].sum(1).mean())
    out["search_switch_mass_when_move_sel"] = float(d["shares"][ism & has_mv, :6].sum(1).mean())
    (OUT / f"l2h6_{ck}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
