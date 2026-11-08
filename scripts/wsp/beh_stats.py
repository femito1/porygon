"""beh analyses 1 (policy behaviour stats), 2 (disagreement taxonomy + costs + worst examples), 4 (value calibration).

Inputs: $WSP/data (decisions.npz, games.json), out/beh/fwd_500M.npz (beh_fwd.py), oracle labels r1, replays.
Outputs: out/beh/stats.json (all numbers), out/beh/stats.txt (readable tables), out/beh/worst.txt.
    python scripts/wsp/beh_stats.py
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from beh_common import (GLORIA, OUT, action_type, coarse, decode, features, load_local, ratio_boot, vname, HP,
                        FNT, UNKNOWN, ACTIVE, OPPACT, TYPE_NAMES, TYP, STATUS)

d = load_local()
f = features(d)
fw = np.load(OUT / "fwd_500M.npz")
N = len(d["mask"])
probs, shares, mask = d["probs"], d["shares"], d["mask"] > 0
game, run, won, turn = d["game"], d["run"], d["won"], d["turn"]
games = d["games"]
R = {}  # results
txt = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    txt.append(s)


def ci(e, lo, hi, p=3):
    return f"{e:.{p}f} [{lo:.{p}f},{hi:.{p}f}]"


SW = np.zeros(10, bool); SW[:6] = True
net_arg = np.where(mask, probs, -1).argmax(1)
srch_arg = np.where(mask, shares, -1).argmax(1)
ok = mask.sum(1) >= 2  # real decisions
P(f"N={N} decisions, {len(np.unique(game))} games; >=2 legal: {ok.sum()}; run0 {np.sum(run==0)} run1 {np.sum(run==1)}")

# ---------------------------------------------------------------- 1. behaviour stats
P("\n## 1. Policy behaviour")
H = lambda p: -(np.where(p > 0, p * np.log(np.clip(p, 1e-12, 1)), 0)).sum(1)  # noqa
ent_n, ent_s = H(np.where(mask, probs, 0)), H(np.where(mask, shares, 0))
nleg = mask.sum(1)
R["entropy"] = {}
for name, e in (("net", ent_n), ("search", ent_s)):
    q = np.percentile(e[ok], [10, 25, 50, 75, 90])
    R["entropy"][name] = {"pct10_25_50_75_90": q.tolist(), "mean": ratio_boot(game[ok], e[ok])[0].item(),
                          "norm_mean": float(np.mean(e[ok] / np.log(nleg[ok])))}
    P(f"entropy {name}: mean {e[ok].mean():.3f} nats, pct10/25/50/75/90 {np.round(q,3)}, mean normalised (H/log n) "
      f"{np.mean(e[ok]/np.log(nleg[ok])):.3f}")
maxp = np.where(mask, probs, 0).max(1)
P(f"net max prob: pct10/50/90 {np.round(np.percentile(maxp[ok],[10,50,90]),3)}; P(max>0.9)={np.mean(maxp[ok]>0.9):.3f}")
agree = net_arg == srch_arg
for r_ in (0, 1):
    s = ok & (run == r_)
    e = ratio_boot(game[s], agree[s].astype(float))
    P(f"argmax agreement net vs search, run{r_}: {ci(*e)}")
    R[f"agree_run{r_}"] = [float(x) for x in e]
# KL(search || net)
kl = (np.where(mask & (shares > 0), shares * (np.log(np.clip(shares, 1e-9, 1)) - np.log(np.clip(probs, 1e-9, 1))), 0)).sum(1)
P(f"KL(search||net) mean {kl[ok].mean():.3f}, median {np.median(kl[ok]):.3f}")

vol = f["voluntary"]
psw_n, psw_s = probs[:, :6].sum(1), shares[:, :6].sum(1)
sit = {
    "voluntary (all)": vol,
    "vol & own HP<25%": vol & (f["hpA"] < .25), "vol & own HP 25-50%": vol & (f["hpA"] >= .25) & (f["hpA"] < .5),
    "vol & own HP 50-75%": vol & (f["hpA"] >= .5) & (f["hpA"] < .75), "vol & own HP>=75%": vol & (f["hpA"] >= .75),
    "vol & we are faster": vol & f["faster"], "vol & we are slower": vol & f["slower"],
    "vol & opp STAB-type SE vs us (>=2x)": vol & (f["opp_se"] >= 2), "vol & opp STAB-type not SE": vol & (f["opp_se"] < 2),
    "vol & opp revealed SE move": vol & (f["opp_rev_mult"] > 0.01),
    "vol & our STAB-type SE vs them": vol & (f["our_se"] >= 2),
    "vol & opp SE & we slower": vol & (f["opp_se"] >= 2) & f["slower"],
    "vol & opp SE & we slower & HP<50%": vol & (f["opp_se"] >= 2) & f["slower"] & (f["hpA"] < .5),
}
P("\nP(switch): situation | n | net P(switch) | search share(switch) | diff net-search | net argmax=switch | search argmax=switch")
R["switch"] = {}
for k, s in sit.items():
    y = np.stack([psw_n, psw_s, psw_n - psw_s, net_arg < 6, srch_arg < 6], 1)[s].astype(float)
    e, lo, hi = ratio_boot(game[s], y)
    R["switch"][k] = {"n": int(s.sum()), "est": e.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    P(f"{k} | {s.sum()} | " + " | ".join(ci(e[i], lo[i], hi[i]) for i in range(5)))

# move classes
P("\nMove classes (voluntary or move-only decisions where >=1 legal move of the class): class | n | net P | search share | diff | net argmax in class | search argmax in class")
hasmv = mask[:, 6:].any(1) & ok
R["classes"] = {}
for cls in ("status", "setup", "protect", "hazard", "selfsw", "recover", "attack"):
    leg = mask[:, 6:] & f[cls]
    s = hasmv & leg.any(1)
    pn, ps = (probs[:, 6:] * leg).sum(1), (shares[:, 6:] * leg).sum(1)
    an = (net_arg >= 6) & leg[np.arange(N), np.clip(net_arg - 6, 0, 3)]
    as_ = (srch_arg >= 6) & leg[np.arange(N), np.clip(srch_arg - 6, 0, 3)]
    y = np.stack([pn, ps, pn - ps, an, as_], 1)[s].astype(float)
    e, lo, hi = ratio_boot(game[s], y)
    R["classes"][cls] = {"n": int(s.sum()), "est": e.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    P(f"{cls} | {s.sum()} | " + " | ".join(ci(e[i], lo[i], hi[i]) for i in range(5)))

# max-damage heuristic
r = np.arange(N)
natk = (mask[:, 6:] & f["attack"]).sum(1)
dm = np.where(mask[:, 6:] & f["attack"], f["dmg"], -1)
srt = np.sort(dm, 1)
distinct = (natk >= 2) & (srt[:, -1] > srt[:, -2] * 1.05)
for who, a in (("net", net_arg), ("search", srch_arg)):
    is_atk = (a >= 6) & f["attack"][r, np.clip(a - 6, 0, 3)]
    s = distinct & is_atk
    e = ratio_boot(game[s], (a == f["maxdmg"])[s].astype(float))
    s2 = distinct
    e2 = ratio_boot(game[s2], (a == f["maxdmg"])[s2].astype(float))
    R[f"maxdmg_{who}"] = {"given_attack": [float(x) for x in e], "all": [float(x) for x in e2], "n": int(s.sum())}
    P(f"max-damage move (bp x mult x STAB x acc x hits) chosen by {who}: given its argmax is an attack {ci(*e)} (n={s.sum()}); "
      f"of all decisions with >=2 attacks of distinct damage {ci(*e2)}")
# prob mass on max-dmg among attack mass
pa = (probs[:, 6:] * (mask[:, 6:] & f["attack"])).sum(1)
pm = probs[r, f["maxdmg"]]
sa = (shares[:, 6:] * (mask[:, 6:] & f["attack"])).sum(1)
sm = shares[r, f["maxdmg"]]
P(f"share of attack mass on the max-damage move: net {np.mean(pm[distinct]/np.maximum(pa[distinct],1e-9)):.3f}, "
  f"search {np.mean(sm[distinct]/np.maximum(sa[distinct],1e-9)):.3f}")

# slot preferences (forced switches)
P("\nSwitch slot preferences: slot | n legal | forced: net P(slot|legal) | search | voluntary: net P(slot)/P(switch) | search")
R["slots"] = {}
forced = f["forced"] & ok
for s_ in range(6):
    lf = forced & mask[:, s_]
    lv = vol & mask[:, s_]
    a = ratio_boot(game[lf], np.stack([probs[lf, s_], shares[lf, s_]], 1))
    b = ratio_boot(game[lv], np.stack([probs[lv, s_] / np.maximum(psw_n[lv], 1e-9), shares[lv, s_] / np.maximum(psw_s[lv], 1e-9)], 1))
    R["slots"][s_] = {"forced": [x.tolist() for x in a], "vol": [x.tolist() for x in b]}
    P(f"{s_} | {lf.sum()} | {ci(a[0][0],a[1][0],a[2][0])} | {ci(a[0][1],a[1][1],a[2][1])} | {ci(b[0][0],b[1][0],b[2][0])} | {ci(b[0][1],b[1][1],b[2][1])}")
# expected under uniform-over-legal for forced switches
uni = np.mean([1 / mask[i, :6].sum() for i in np.flatnonzero(forced)])
P(f"(uniform over legal switches would give mean {uni:.3f})")
# matchup-based switch choice: does the net's forced switch-in have better defensive matchup than search's?
P("\nForced switches: defensive matchup (best opp-active STAB mult vs switch-in; lower = better) and offence of switch-in")
for who, a in (("net", net_arg), ("search", srch_arg)):
    s = forced & (a < 6) & f["has_opp"]
    dmul = f["def_mult"][r, np.clip(a, 0, 5)]
    off = f["off"][r, np.clip(a, 0, 5)]
    e1, e2 = ratio_boot(game[s], dmul[s]), ratio_boot(game[s], off[s])
    # random legal baseline
    rb = np.array([np.mean(f["def_mult"][i, mask[i, :6]]) for i in np.flatnonzero(s)])
    ro = np.array([np.mean(f["off"][i, mask[i, :6]]) for i in np.flatnonzero(s)])
    P(f"{who}: def mult {ci(*e1)}  (random legal {rb.mean():.3f}); best offence of switch-in {ci(*e2,1)} (random legal {ro.mean():.1f})")
    R[f"forced_matchup_{who}"] = {"def": [float(x) for x in e1], "off": [float(x) for x in e2], "rand_def": float(rb.mean()), "rand_off": float(ro.mean())}

# ---------------------------------------------------------------- oracle labels
import sys
LAB = sys.argv[1] if len(sys.argv) > 1 else "r1"
SUF = "" if LAB == "r1" else "_" + LAB
lab_q = np.full((N, 10), np.nan)
key = {(games[game[i]]["run"], int(d["src_line"][i])): i for i in range(N)}
nl = 0
for fn in glob.glob(f"{GLORIA}/runs/oracle-labels/{LAB}/labels-*.jsonl"):
    for line in open(fn):
        o = json.loads(line)
        rn = o["key"][0].split("/")[1]
        i = key.get((rn, int(o["key"][1])))
        if i is None or "q" not in o:
            continue
        lab_q[i] = [np.nan if q is None else q for q in o["q"]]
        nl += 1
has_lab = np.isfinite(lab_q).any(1)
P(f"\noracle labels {LAB} matched: {nl} rows, {has_lab.sum()} with q")
qmax = np.nanmax(np.where(has_lab[:, None], lab_q, -1), 1)


def qgap(a):
    q = lab_q[r, a]
    return q - qmax


gap_net, gap_srch = qgap(net_arg), qgap(srch_arg)
gap_rand = np.array([np.nanmean(lab_q[i][mask[i]] ) - qmax[i] if has_lab[i] else np.nan for i in range(N)])

# ---------------------------------------------------------------- 2. disagreements
P("\n## 2. Disagreement taxonomy (net argmax != search argmax AND search share of net argmax < tau)")
tn = action_type(net_arg, f)
ts = action_type(srch_arg, f)
cn, cs = coarse(tn), coarse(ts)
share_of_net = shares[r, net_arg]
# next decision of same game (value change)
order = np.lexsort((d["src_line"], game))
nxt = np.full(N, -1)
for a_, b_ in zip(order[:-1], order[1:]):
    if game[a_] == game[b_]:
        nxt[a_] = b_
val = d["value"]
dval = np.where(nxt >= 0, val[np.maximum(nxt, 0)] - val, np.nan)
# terminal: if last decision, use outcome as next value
dval_term = np.where(nxt >= 0, dval, (2 * won - 1) - val)
R["disagree"] = {}
for tau in (0.05, 0.1, 0.2, 1.01):
    dis = ok & (net_arg != srch_arg) & (share_of_net < tau)
    e = ratio_boot(game[ok], dis[ok].astype(float))
    P(f"tau={tau}: disagreement rate {ci(*e)}  run0 {dis[ok&(run==0)].mean():.3f} run1 {dis[ok&(run==1)].mean():.3f}")
    R["disagree"][str(tau)] = [float(x) for x in e]
tau = 0.1
dis = ok & (net_arg != srch_arg) & (share_of_net < tau)
agr = ok & (net_arg == srch_arg)

# fine category
def fine(i):
    a, b = tn[i], ts[i]
    if cn[i] == "attack" and cs[i] == "attack":
        dn, ds = f["dmg"][i, net_arg[i] - 6], f["dmg"][i, srch_arg[i] - 6]
        mn = f["opp_se"][i]
        return "attack->attack (lower dmg)" if dn < ds * 0.95 else "attack->attack (higher/equal dmg)"
    if cn[i] == "switch" and cs[i] == "switch":
        return "switch->switch"
    a = "status" if cn[i] == "status" else a
    b = "status" if cs[i] == "status" else b
    return f"{a}->{b}"


cat = np.array([fine(i) if dis[i] else "" for i in range(N)], dtype=object)
cat_fine = np.array([f"{tn[i]}->{ts[i]}" if dis[i] else "" for i in range(N)], dtype=object)
cats = [c for c, _ in sorted(defaultdict(int, {c: (cat == c).sum() for c in set(cat) if c}).items(), key=lambda kv: -kv[1])]
P(f"\ncategory (tau=0.1) | n | % of disagreements | per-decision rate | oracle Q gap net [CI] (n lab) | Q gap search | gap net-search | "
  f"Δ(V+Φ) next | Δ(V+Φ) minus agreement baseline")
base_d = ratio_boot(game[agr & np.isfinite(dval)], dval[agr & np.isfinite(dval)])
base_gn = ratio_boot(game[agr & has_lab], gap_net[agr & has_lab])
P(f"agreement baseline: n {agr.sum()}, Δ(V+Φ) {ci(*base_d)}, Q gap {ci(*base_gn)}; random legal action Q gap "
  f"{np.nanmean(gap_rand[ok & has_lab]):.3f}; search argmax Q gap all labelled {np.nanmean(gap_srch[ok&has_lab]):.3f}; "
  f"net argmax Q gap all labelled {np.nanmean(gap_net[ok&has_lab]):.3f}")
P("agreement Δ(V+Φ) baseline by the (shared) action type: " + ", ".join(
    f"{c}: {np.nanmean(dval[agr & (cs == c)]):+.3f} (n={np.sum(agr & (cs == c))})" for c in ("attack", "switch", "status")))
P("all decisions Δ(V+Φ) by the search's action type: " + ", ".join(
    f"{c}: {np.nanmean(dval[ok & (cs == c)]):+.3f}" for c in ("attack", "switch", "status")))
base_by = {c: np.nanmean(dval[agr & (cs == c)]) for c in ("attack", "switch", "status")}
R["cat"] = {}
for c in cats + ["ALL"]:
    s = dis & (cat == c) if c != "ALL" else dis
    sl = s & has_lab
    g = ratio_boot(game[sl], np.stack([gap_net[sl], gap_srch[sl], gap_net[sl] - gap_srch[sl]], 1)) if sl.sum() > 5 else None
    sv = s & np.isfinite(dval)
    dv = ratio_boot(game[sv], dval[sv]) if sv.sum() > 5 else (np.nan,) * 3
    R["cat"][c] = {"n": int(s.sum()), "n_lab": int(sl.sum()), "gap": None if g is None else [x.tolist() for x in g],
                   "dval": [float(x) for x in dv]}
    gs = " | ".join(ci(g[0][k], g[1][k], g[2][k]) for k in range(3)) if g is not None else "- | - | -"
    P(f"{c} | {s.sum()} | {s.sum()/dis.sum():.1%} | {s.sum()/ok.sum():.3f} | {gs} (n={sl.sum()}) | {ci(*dv)} | {dv[0]-base_d[0]:+.3f} | vs same-search-type agreement {dv[0]-base_by.get(c.split("->")[-1].split(" ")[0], np.nan) if c != "ALL" else np.nan:+.3f}")
P("\nfine categories (net type -> search type), tau=0.1: n, mean Q gap net-search (labelled n)")
for c in sorted(set(cat_fine) - {""}, key=lambda c: -(cat_fine == c).sum()):
    s = dis & (cat_fine == c)
    sl = s & has_lab
    P(f"  {c}: {s.sum()}  gapdiff {np.nanmean(gap_net[sl]-gap_srch[sl]) if sl.sum() else np.nan:.3f} (n={sl.sum()})  Δ(V+Φ) {np.nanmean(dval[s]):+.3f}")
# sensitivity of the category ranking to tau
P("\nsensitivity: category share of disagreements at tau 0.05 / 0.1 / 0.2")
for c in cats:
    row = []
    for t in (0.05, 0.1, 0.2):
        dd = ok & (net_arg != srch_arg) & (shares[r, net_arg] < t)
        cc = np.array([fine(i) if dd[i] else "" for i in np.flatnonzero(dd)], dtype=object)
        row.append(f"{np.mean(cc == c):.3f}")
    P(f"  {c}: " + " / ".join(row))
# attack->attack details: effectiveness
aa = dis & (cn == "attack") & (cs == "attack")
mult_n = 2.0 ** (2 * decode(d["obs"][aa])["mv"][np.arange(aa.sum()), f["own_a"][aa], net_arg[aa] - 6, 27])
mult_s = 2.0 ** (2 * decode(d["obs"][aa])["mv"][np.arange(aa.sum()), f["own_a"][aa], srch_arg[aa] - 6, 27])
P(f"attack->attack: net move less effective {np.mean(mult_n < mult_s):.3f}, more {np.mean(mult_n > mult_s):.3f}; "
  f"net move lower dmg-heuristic {np.mean(f['dmg'][aa, net_arg[aa]-6] < 0.95*f['dmg'][aa, srch_arg[aa]-6]):.3f}; "
  f"net move = max-dmg {np.mean(net_arg[aa] == f['maxdmg'][aa]):.3f}, search = max-dmg {np.mean(srch_arg[aa] == f['maxdmg'][aa]):.3f}")
# situation profile of the categories
P("\nsituation profile per category: own HP, opp HP, we slower, opp SE vs us, our SE vs them, turn, our alive, their alive")
for c in cats[:6] + ["AGREE"]:
    s = (dis & (cat == c)) if c != "AGREE" else agr
    P(f"  {c}: hpA {f['hpA'][s].mean():.2f} hpB {f['hpB'][s].mean():.2f} slower {f['slower'][s].mean():.2f} oppSE {np.mean(f['opp_se'][s]>=2):.2f} "
      f"ourSE {np.mean(f['our_se'][s]>=2):.2f} turn {turn[s].mean():.1f} alive {f['our_alive'][s].mean():.1f}/{f['their_alive'][s].mean():.1f}")

# outcome association: games with more costly-category disagreements lose more? (confounded; reported only)
ug = np.unique(game)
nd = np.bincount(game, weights=dis.astype(float), minlength=game.max() + 1)[ug]
nn = np.bincount(game, weights=ok.astype(float), minlength=game.max() + 1)[ug]
wg = np.array([won[game == g][0] for g in ug])
P(f"\ngame-level corr(disagreement rate, win) = {np.corrcoef(nd/np.maximum(nn,1), wg)[0,1]:.3f} (confounded by position quality)")

# ---------------------------------------------------------------- worst examples
P("\n## worst examples written to worst.txt")
cand = np.flatnonzero(dis & has_lab)
cand = cand[np.argsort(gap_net[cand] - gap_srch[cand])][:10]
ob = decode(d["obs"])
lines = []


def mon_str(i, s):
    n = ob["nums"][i, s]
    c = ob["cats"][i, s]
    if n[UNKNOWN] > .5:
        return "(unseen)"
    st = [x for k, x in enumerate(["brn", "frz", "par", "psn", "slp", "tox"]) if n[STATUS + k] > .5]
    ty = "/".join(TYPE_NAMES[k] for k in range(18) if n[TYP + k] > .5)
    return (f"{vname('species', c[0])} [{ty}] {int(round(100*n[HP]))}%{' FNT' if n[FNT]>.5 else ''}{' '+st[0] if st else ''} "
            f"item={vname('items', c[1])} abil={vname('abilities', c[2])} moves=" + ",".join(vname('moves', m) for m in c[3:]))


def act_str(i, a):
    if a < 6:
        return f"switch->{vname('species', ob['cats'][i, a, 0])}"
    return f"move {vname('moves', ob['cats'][i, f['own_a'][i], 3 + a - 6])}"


for i in cand:
    g = games[game[i]]
    lines.append(f"--- row {i}  {g['tag']}  run {g['run']}  turn {turn[i]}  won={won[i]}  V+Φ={val[i]:+.3f}  cat={cat[i]} ({cat_fine[i]})")
    lines.append(f"replay: {GLORIA}/runs/{g['run']}/replays/{g['tag']}.log  (player {g['player']})")
    lines.append(f"our active: {mon_str(i, f['own_a'][i])}")
    lines.append(f"opp active: {mon_str(i, f['opp_a'][i])}")
    lines.append("our bench: " + " | ".join(mon_str(i, s) for s in range(6) if s != f['own_a'][i]))
    lines.append("opp seen: " + " | ".join(mon_str(i, s) for s in range(6, 12) if s != f['opp_a'][i] and ob['nums'][i, s, UNKNOWN] < .5))
    lines.append("legal: " + ", ".join(f"{act_str(i,a)} net={probs[i,a]:.2f} srch={shares[i,a]:.2f} Q={lab_q[i,a]:.3f}"
                                       for a in range(10) if mask[i, a]))
    lines.append(f"net argmax: {act_str(i, net_arg[i])} (Q gap {gap_net[i]:.3f}); search argmax: {act_str(i, srch_arg[i])} (Q gap {gap_srch[i]:.3f})")
(OUT / f"worst{SUF}.txt").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- 4. value calibration
P("\n## 4. Value calibration")
V = fw["value"].astype(float)
phi = val - V
y = 2 * won - 1


def parse_end(path, me):
    hp, fnt = {}, {}
    side = None
    for line in open(path):
        p = line.rstrip("\n").split("|")
        if len(p) < 3:
            continue
        if p[1] == "player" and len(p) > 3 and p[3] == me:
            side = p[2]
        if p[1] in ("switch", "drag", "-damage", "-heal", "-sethp", "replace") and len(p) > 3:
            who = p[2].split(":")[0][:2]; nick = p[2].split(":", 1)[1].strip()
            hpf = p[4] if p[1] in ("switch", "drag", "replace") else p[3]
            m = re.match(r"(\d+)/(\d+)", hpf)
            if hpf.startswith("0"):
                hp[(who, nick)] = 0.0
            elif m:
                hp[(who, nick)] = int(m.group(1)) / int(m.group(2))
        if p[1] == "faint":
            who = p[2].split(":")[0][:2]; nick = p[2].split(":", 1)[1].strip()
            hp[(who, nick)] = 0.0
    if side is None:
        return np.nan
    v = 0.0
    for s_, sign in ((side, 1), ("p2" if side == "p1" else "p1", -1)):
        mons = [h for (w, _), h in hp.items() if w == s_]
        mons += [1.0] * (6 - len(mons))
        for h in mons:
            v += sign * (-0.05 if h <= 0 else 0.05 * h)
    return v


phi_end_g = {}
for gi, g in enumerate(games):
    pth = Path(f"{GLORIA}/runs/{g['run']}/replays/{g['tag']}.log")
    if pth.exists():
        phi_end_g[gi] = parse_end(pth, g["player"])
ys = np.array([phi_end_g.get(int(g), np.nan) for g in game])
phi_end = ys + y  # material at the end + the ±1 terminal
fin = np.isfinite(phi_end) & np.isfinite(val) & np.isfinite(y)
P(f"replays parsed {len(phi_end_g)}/{len(games)}; rows with Φ_end {fin.sum()}")


def auc(s, t):
    m = np.isin(t, (0, 1))
    s, t = s[m], t[m]
    o = np.argsort(s)
    rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    n1 = t.sum(); n0 = len(t) - n1
    return (rk[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logit_fit(X, t, l2=1e-3, it=50):
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - t) + l2 * w
        Hh = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        w -= np.linalg.solve(Hh, g)
    return w


def cv_logloss(X, t, grp, k=5):
    ug_ = np.unique(grp)
    rng = np.random.default_rng(0)
    fold = dict(zip(ug_, rng.integers(0, k, len(ug_))))
    fo = np.array([fold[g] for g in grp])
    pr = np.zeros(len(t))
    for j in range(k):
        w = logit_fit(X[fo != j], t[fo != j])
        pr[fo == j] = 1 / (1 + np.exp(-np.c_[np.ones((fo == j).sum()), X[fo == j]] @ w))
    pr = np.clip(pr, 1e-6, 1 - 1e-6)
    return float(-np.mean(t * np.log(pr) + (1 - t) * np.log(1 - pr))), float(np.mean((pr - t) ** 2)), pr


m = fin & np.isin(won, (0, 1)) & ok
t = won[m]
mat_mons = (f["our_alive"] - f["their_alive"])[m]
mat_hp = (f["hp_own"] - f["hp_opp"])[m]
tt = turn[m].astype(float)
feats = {
    "V+Φ (logged)": val[m][:, None], "Φ alone": phi[m][:, None], "V alone": V[m][:, None],
    "material: mons diff + HP diff": np.c_[mat_mons, mat_hp],
    "material + turn interactions": np.c_[mat_mons, mat_hp, mat_mons * tt / 30, mat_hp * tt / 30],
    "material + V+Φ": np.c_[mat_mons, mat_hp, val[m]],
    "material + V": np.c_[mat_mons, mat_hp, V[m]],
}
P("predictor | raw corr with outcome ±1 | AUC | CV log-loss (Platt/logistic, 5 game folds) | CV Brier")
base_ll = -np.mean(t * np.log(t.mean()) + (1 - t) * np.log(1 - t.mean()))
P(f"constant | - | 0.5 | {base_ll:.4f} | {np.mean((t-t.mean())**2):.4f}")
R["value_cal"] = {}
for k, X in feats.items():
    ll, br, pr = cv_logloss(X, t, game[m])
    a = auc(X[:, -1] if X.shape[1] == 1 else pr, t)
    c = np.corrcoef(X[:, -1] if X.shape[1] == 1 else pr, t)[0, 1]
    R["value_cal"][k] = {"auc": float(a), "ll": ll, "brier": br, "corr": float(c)}
    P(f"{k} | {c:.3f} | {a:.3f} | {ll:.4f} | {br:.4f}")
# bootstrap the log-loss improvement of material+V over material
_, _, p_mat = cv_logloss(feats["material: mons diff + HP diff"], t, game[m])
_, _, p_matv = cv_logloss(feats["material + V"], t, game[m])
_, _, p_vp = cv_logloss(feats["V+Φ (logged)"], t, game[m])
lldiff = (-(t * np.log(p_mat) + (1 - t) * np.log(1 - p_mat))) - (-(t * np.log(p_matv) + (1 - t) * np.log(1 - p_matv)))
e = ratio_boot(game[m], lldiff)
P(f"log-loss gain of material+V over material (per decision, game bootstrap): {ci(*e, 4)}")
lld2 = (-(t * np.log(p_mat) + (1 - t) * np.log(1 - p_mat))) - (-(t * np.log(p_vp) + (1 - t) * np.log(1 - p_vp)))
P(f"log-loss gain of V+Φ alone over material: {ci(*ratio_boot(game[m], lld2), 4)}")
R["ll_gain_matV_over_mat"] = [float(x) for x in e]
# raw Brier with p=(x+1)/2
pr_raw = np.clip((val[m] + 1) / 2, 0, 1)
P(f"raw Brier with p=clip((V+Φ+1)/2): {np.mean((pr_raw-t)**2):.4f}; Φ alone: {np.mean((np.clip((phi[m]+1)/2,0,1)-t)**2):.4f}")
# reliability diagram
P("\nreliability: V+Φ bin | n | mean V+Φ | P(win) [CI] | mean outcome ±1 | mean Φ_end")
bins = [-9, -1, -.75, -.5, -.25, 0, .25, .5, .75, 1, 9]
R["reliability"] = []
for lo_, hi_ in zip(bins[:-1], bins[1:]):
    s = fin & ok & (val >= lo_) & (val < hi_) & np.isin(won, (0, 1))
    if s.sum() < 20:
        continue
    e = ratio_boot(game[s], won[s])
    R["reliability"].append([lo_, hi_, int(s.sum()), float(val[s].mean()), *map(float, e), float(phi_end[s].mean())])
    P(f"[{lo_},{hi_}) | {s.sum()} | {val[s].mean():+.3f} | {ci(*e)} | {y[s].mean():+.3f} | {phi_end[s].mean():+.3f}")
# by phase
P("\nby phase: subset | n | AUC V+Φ | AUC Φ | AUC material(logistic in-sample) | corr(V+Φ, Φ_end)")
mons_left = f["our_alive"] + f["their_alive"]
phases = {"turn 1-5": turn <= 5, "turn 6-15": (turn > 5) & (turn <= 15), "turn 16-30": (turn > 15) & (turn <= 30),
          "turn >30": turn > 30, "mons left 10-12": mons_left >= 10, "mons left 6-9": (mons_left >= 6) & (mons_left < 10),
          "mons left 2-5": mons_left < 6}
R["phase"] = {}
for k, s0 in phases.items():
    s = fin & ok & np.isin(won, (0, 1)) & s0
    Xm = np.c_[(f["our_alive"] - f["their_alive"])[s], (f["hp_own"] - f["hp_opp"])[s]]
    w = logit_fit(Xm, won[s])
    am_ = auc(Xm @ w[1:], won[s])
    R["phase"][k] = [int(s.sum()), auc(val[s], won[s]), auc(phi[s], won[s]), am_]
    P(f"{k} | {s.sum()} | {auc(val[s], won[s]):.3f} | {auc(phi[s], won[s]):.3f} | {am_:.3f} | {np.corrcoef(val[s], phi_end[s])[0,1]:.3f}")
# V ≈ E[Φ_end] − Φ ?
s = fin & ok
tgt = phi_end[s] - phi[s]
A_ = np.c_[np.ones(s.sum()), V[s]]
coef = np.linalg.lstsq(A_, tgt, rcond=None)[0]
res = tgt - A_ @ coef
r2 = 1 - res.var() / tgt.var()
P(f"\nV vs (Φ_end − Φ): corr {np.corrcoef(V[s], tgt)[0,1]:.3f}, slope {coef[1]:.3f}, intercept {coef[0]:+.3f}, R² {r2:.3f}; "
  f"mean V {V[s].mean():+.3f} vs mean (Φ_end−Φ) {tgt.mean():+.3f}; corr(V+Φ, Φ_end) {np.corrcoef(val[s], phi_end[s])[0,1]:.3f}, "
  f"corr(Φ, Φ_end) {np.corrcoef(phi[s], phi_end[s])[0,1]:.3f}")
# do V and Φ partially cancel?
P(f"corr(V, Φ) {np.corrcoef(V[s], phi[s])[0,1]:.3f}; sd V {V[s].std():.3f}, sd Φ {phi[s].std():.3f}, sd V+Φ {val[s].std():.3f}")
R["V_vs_target"] = {"corr": float(np.corrcoef(V[s], tgt)[0, 1]), "slope": float(coef[1]), "icpt": float(coef[0]), "r2": float(r2)}
# value by opponent-run: bias (net was trained in self-play)
for r_ in (0, 1):
    s2 = s & (run == r_)
    gw = [won[game == g][0] for g in np.unique(game[s2])]
    P(f"run{r_}: games {len(gw)}, game win rate {np.mean(gw):.3f}, decision-weighted win rate {won[s2].mean():.3f}")
    P(f"run{r_}: mean V+Φ {val[s2].mean():+.3f}, mean outcome ±1 {y[s2].mean():+.3f}, mean Φ_end {phi_end[s2].mean():+.3f}")
# vs oracle Q at root
s = has_lab & fin & ok & np.isin(won, (0, 1))
P(f"\nvs oracle root max Q (n={s.sum()}): corr(V+Φ, Qmax) {np.corrcoef(val[s], qmax[s])[0,1]:.3f}; "
  f"Spearman {np.corrcoef(np.argsort(np.argsort(val[s])), np.argsort(np.argsort(qmax[s])))[0,1]:.3f}; "
  f"AUC outcome: V+Φ {auc(val[s], won[s]):.3f}, Qmax {auc(qmax[s], won[s]):.3f}, Φ {auc(phi[s], won[s]):.3f}, "
  f"V {auc(V[s], won[s]):.3f}")
Xq = np.c_[mat_mons, mat_hp]
# partial: does V+Φ add to Qmax?
mm = s
for k, X in {"Qmax": qmax[mm][:, None], "Qmax + V+Φ": np.c_[qmax[mm], val[mm]], "material": np.c_[(f['our_alive']-f['their_alive'])[mm], (f['hp_own']-f['hp_opp'])[mm]],
             "material + Qmax": np.c_[(f['our_alive']-f['their_alive'])[mm], (f['hp_own']-f['hp_opp'])[mm], qmax[mm]]}.items():
    ll, br, _ = cv_logloss(X, won[mm], game[mm])
    P(f"  labelled rows CV log-loss {k}: {ll:.4f} (Brier {br:.4f})")

np.savez(OUT / f"derived{SUF}.npz", net_arg=net_arg, srch_arg=srch_arg, dis=dis, agr=agr, cat=cat.astype(str),
         cat_fine=cat_fine.astype(str), gap_net=gap_net, gap_srch=gap_srch, lab_q=lab_q, phi=phi, phi_end=phi_end,
         dval=dval)
json.dump(R, open(OUT / f"stats{SUF}.json", "w"), indent=1, default=float)
(OUT / f"stats{SUF}.txt").write_text("\n".join(txt) + "\n")
