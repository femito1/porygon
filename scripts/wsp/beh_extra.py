"""beh extras: switch->switch breakdown (forced / voluntary / blind double switch), matchup of net vs search
switch-ins, and the key behaviour numbers recomputed with the 600M net's policy (fwd_600M.npz).
    python scripts/wsp/beh_extra.py  -> out/beh/extra.txt
"""
import numpy as np

from beh_common import OUT, FNT, decode, features, load_local, ratio_boot, action_type, coarse

d = load_local()
f = features(d)
der = np.load(OUT / "derived_r2-300k.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N)
mask = d["mask"] > 0
game = d["game"]
nums = decode(d["obs"])["nums"]
txt = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


def ci(e, lo, hi, p=3):
    return f"{e:.{p}f} [{lo:.{p}f},{hi:.{p}f}]"


cat, na, sa = der["cat"], der["net_arg"], der["srch_arg"]
gn, gs = der["gap_net"], der["gap_srch"]
opp_fnt = nums[r, f["opp_a"], FNT] > 0.5
blind = f["forced"] & opp_fnt
P("## switch->switch disagreements (tau 0.1), r2-300k Q gaps")
ss = cat == "switch->switch"
for name, s in (("forced (our active fainted / U-turn)", ss & f["forced"]), ("  of which blind (opp active also fainted)", ss & blind),
                ("  forced, opp active alive", ss & f["forced"] & ~opp_fnt), ("voluntary", ss & f["voluntary"])):
    e = ratio_boot(game[s], gn[s] - gs[s])
    P(f"{name}: n {s.sum()} ({s.sum()/ss.sum():.1%}), gap net-search {ci(*e)}")
P(f"base rates among all decisions: forced {f['forced'].mean():.3f}, blind {blind.mean():.3f}; disagreement rate on forced "
  f"{np.mean(der['dis'][f['forced']]):.3f}, on blind {np.mean(der['dis'][blind]):.3f}, on voluntary {np.mean(der['dis'][f['voluntary']]):.3f}")
s = ss & f["forced"] & ~opp_fnt
dn, ds = f["def_mult"][r, np.clip(na, 0, 5)], f["def_mult"][r, np.clip(sa, 0, 5)]
on, os_ = f["off"][r, np.clip(na, 0, 5)], f["off"][r, np.clip(sa, 0, 5)]
P(f"forced switch->switch, opp alive: def mult (opp STAB vs switch-in; lower=better) net {dn[s].mean():.3f} vs search {ds[s].mean():.3f}; "
  f"best offence net {on[s].mean():.1f} vs search {os_[s].mean():.1f}; net's switch-in has HP {nums[r, np.clip(na,0,5), 0][s].mean():.2f} "
  f"vs search's {nums[r, np.clip(sa,0,5), 0][s].mean():.2f}")
s = ss & f["voluntary"]
P(f"voluntary switch->switch: def mult net {dn[s].mean():.3f} vs search {ds[s].mean():.3f}; offence {on[s].mean():.1f} vs {os_[s].mean():.1f}; "
  f"HP {nums[r, np.clip(na,0,5), 0][s].mean():.2f} vs {nums[r, np.clip(sa,0,5), 0][s].mean():.2f}")
# switch->attack: what does the net switch to and why
s = cat == "switch->attack"
P(f"switch->attack: own active HP {f['hpA'][s].mean():.2f} (agree {f['hpA'][der['agr']].mean():.2f}); opp HP {f['hpB'][s].mean():.2f}; "
  f"search's attack is the max-dmg move {np.mean(sa[s] == f['maxdmg'][s]):.3f}; our STAB SE vs them {np.mean(f['our_se'][s]>=2):.3f}; "
  f"opp STAB SE vs us {np.mean(f['opp_se'][s]>=2):.3f}; net's switch-in def mult {dn[s].mean():.3f} vs staying active "
  f"{f['opp_se'][s].mean():.3f}")
# the search attack would KO? proxy: opp HP < 0.3
P(f"switch->attack with opp active HP<30%: {np.mean(f['hpB'][s] < .3):.3f} (all voluntary {np.mean(f['hpB'][f['voluntary']] < .3):.3f})")
s = cat == "status->attack"
tn = action_type(na, f)
P("status->attack: net's status move types: " + ", ".join(f"{k} {np.mean(tn[s]==k):.2f}" for k in ("status", "setup", "recover", "protect", "hazard"))
  + f"; opp HP<30% {np.mean(f['hpB'][s] < .3):.3f}")
names = f["move_names"][r, np.clip(na - 6, 0, 3)]
u, c = np.unique(names[s], return_counts=True)
P("status->attack: most common net status moves: " + ", ".join(f"{a} {b}" for a, b in sorted(zip(u, c), key=lambda t: -t[1])[:12]))
s = cat == "switch->status"
nm = f["move_names"][r, np.clip(sa - 6, 0, 3)]
u, c = np.unique(nm[s], return_counts=True)
P("switch->status: most common search status moves: " + ", ".join(f"{a} {b}" for a, b in sorted(zip(u, c), key=lambda t: -t[1])[:12]))

# ---- 600M check of the behaviour numbers
P("\n## 600M vs 500M (same positions; search shares as logged)")
for tag in ("500M", "600M"):
    fw = np.load(OUT / f"fwd_{tag}.npz")
    p = fw["probs"]
    arg = np.where(mask, p, -1).argmax(1)
    H = -(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1)), 0)).sum(1)
    vol = f["voluntary"]
    agree = arg == sa
    dis = (arg != sa) & (d["shares"][r, arg] < 0.1)
    gap = der["lab_q"][r, arg] - np.nanmax(der["lab_q"], 1)
    rec = (mask[:, 6:] & f["recover"]).any(1)
    ssw = (mask[:, 6:] & f["selfsw"]).any(1)
    P(f"{tag}: entropy {H.mean():.3f}; argmax agreement {ci(*ratio_boot(game, agree.astype(float)))}; disagreement(τ0.1) "
      f"{dis.mean():.3f}; mean Q gap of argmax {np.nanmean(gap):.4f}; vol P(switch) {p[vol,:6].sum(1).mean():.3f}; "
      f"P(recover|legal) {(p[:,6:]*f['recover']).sum(1)[rec].mean():.3f}; P(U-turn|legal) {(p[:,6:]*f['selfsw']).sum(1)[ssw].mean():.3f}; "
      f"oppSE vol P(switch) {p[vol & (f['opp_se']>=2), :6].sum(1).mean():.3f}; ourSE {p[vol & (f['our_se']>=2), :6].sum(1).mean():.3f}")
    if tag == "600M":
        v5 = np.load(OUT / "fwd_500M.npz")["value"]
        v6 = fw["value"]
        phi = d["value"] - v5
        won = d["won"]; ok = np.isin(won, (0, 1))
        def auc(s_, t):
            o = np.argsort(s_); rk = np.empty(len(s_)); rk[o] = np.arange(1, len(s_) + 1)
            n1 = t.sum(); n0 = len(t) - n1
            return (rk[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        P(f"value AUC for outcome: 500M V+Φ {auc((v5+phi)[ok], won[ok]):.3f}, 600M V+Φ {auc((v6+phi)[ok], won[ok]):.3f}; "
          f"corr(V500, V600) {np.corrcoef(v5, v6)[0,1]:.3f}")
(OUT / "extra.txt").write_text("\n".join(txt) + "\n")
