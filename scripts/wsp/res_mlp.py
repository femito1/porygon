"""MLP units: dead / always-on / saturated units, effective rank of activations and weights, top units and what
they correlate with. Covers the 3 encoder-layer MLPs (512 GELU), the first-layer encoders (move_proj 64,
mon_proj 256, field_proj 256, action_move_proj 256 GELU hidden), the ActionHead hidden (256 GELU) and the value
MLP (2 × 256 tanh).

Definitions: a unit is dead if its pre-activation is never > 0 on any token of its kind over all 48,518
decisions; always-on if pre > 0 on > 99.9% (it then acts linearly); tanh unit saturated on a sample if
|tanh| > 0.99. Effective rank = participation ratio (Σλ)²/Σλ² of the covariance eigenvalues (activations) or of
σ² (weights).

    python scripts/wsp/res_mlp.py 500M
"""

import json
import sys

import numpy as np
import torch

from porygon.encoder import BOOSTS, EFFECTS, FIELDS, SCREENS, HAZARDS, STATS, STATUSES, TYPES, WEATHERS
from res_lib import OUT, data
from wsp_common import Net, load

TT = ["field", "own_active", "own_bench", "opp_active", "opp_bench", "move"]


def mon_names():
    n = ["hp", "fainted"] + [f"base_{s}" for s in STATS] + [f"stat_{s}" for s in STATS] + ["level"]
    n += [f"type_{t.name}" for t in TYPES] + [f"tera_{t.name}" for t in TYPES] + ["is_tera"]
    n += [f"boost_{b}" for b in BOOSTS] + [f"status_{s.name}" for s in STATUSES] + ["status_counter"]
    n += [f"eff_{e.name}" for e in EFFECTS]
    n += ["active", "mine", "first_turn", "protect_counter", "recharge", "preparing", "log_weight", "trapped",
          "is_opp_active", "unknown"]
    return n


def move_names():
    return (["bp", "acc", "prio", "pp", "avail", "cat_phys", "cat_spec", "cat_status"]
            + [f"mtype_{t.name}" for t in TYPES]
            + ["stab", "log2mult", "immune", "heal", "drain", "recoil", "self_switch", "force_switch", "side_cond",
               "status_move", "volatile", "boosts", "self_boost", "protect", "breaks_protect", "hits",
               "weather", "self_destruct"])


def field_names():
    n = [f"w_{w.name}" for w in WEATHERS] + [f"wage_{w.name}" for w in WEATHERS] + ["no_weather"]
    n += [f"f_{f.name}" for f in FIELDS] + [f"fage_{f.name}" for f in FIELDS]
    for side in ("own", "opp"):
        n += [f"{side}_{s.name}" for s in SCREENS] + [f"{side}_age_{s.name}" for s in SCREENS]
        n += [f"{side}_{h.name}" for h in HAZARDS]
    n += ["turn", "opp_unseen", "own_alive", "opp_alive", "force_switch", "trapped", "maybe_trapped", "can_tera",
          "used_tera", "opp_used_tera", "wait", "no_active", "n_switches", "n_moves"]
    return n


def pr(ev):
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))


def ndim(ev, q):
    ev = np.sort(np.clip(ev, 0, None))[::-1]
    return int(np.searchsorted(np.cumsum(ev) / ev.sum(), q) + 1)


class Acc:
    """Streaming per-unit stats for one activation site and one token type."""

    def __init__(self, n):
        self.n = 0
        self.pos = np.zeros(n)
        self.s = np.zeros(n)
        self.sa = np.zeros(n)
        self.ss = np.zeros((n, n))
        self.sat = np.zeros(n)
        self.mx = np.full(n, -np.inf)

    def add(self, pre, post, tanh=False):
        pre = pre.double().numpy()
        post = post.double().numpy()
        self.n += len(post)
        self.pos += (pre > 0).sum(0)
        self.s += post.sum(0)
        self.sa += np.abs(post).sum(0)
        self.ss += post.T @ post
        self.mx = np.maximum(self.mx, pre.max(0))
        if tanh:
            self.sat += (np.abs(post) > 0.99).sum(0)

    def summary(self):
        if self.n == 0:
            return None
        mu = self.s / self.n
        cov = self.ss / self.n - np.outer(mu, mu)
        ev = np.linalg.eigvalsh(cov)
        fpos = self.pos / self.n
        out = {"n": int(self.n), "units": len(mu), "dead": int((self.mx <= 0).sum()),
               "near_dead_<0.1%": int((fpos < 1e-3).sum()), "always_on_>99.9%": int((fpos > 0.999).sum()),
               "mean_frac_active": float(fpos.mean()), "PR": pr(ev), "dims90": ndim(ev, 0.9),
               "dims99": ndim(ev, 0.99)}
        return out


def main():
    tag = sys.argv[1]
    out_name = tag.rsplit("/", 1)[-1].replace(".zip", "") if "/" in tag else tag
    d = data()
    T = d["truth"]
    N = len(d["obs"])
    pol = load(tag)
    net = Net(pol)
    fe = pol.features_extractor
    hooks, cur = [], {}

    def grab(name):
        def h(mod, inp, out):
            cur[name + ".pre"] = inp[0].detach()
            cur[name] = out.detach()
        return h
    sites = {"move_proj": fe.move_proj[1], "mon_proj": fe.mon_proj[1], "field_proj": fe.field_proj[1],
             "action_move_proj": fe.action_move_proj[1], "head": pol.action_net.score[1],
             "value1": pol.mlp_extractor.value_net[1], "value2": pol.mlp_extractor.value_net[3]}
    for k, m in sites.items():
        hooks.append(m.register_forward_hook(grab(k)))
    # token types per position
    nums = d["split"]["nums"]
    ttype = np.zeros((N, 17), int)
    ttype[:, 1:7] = 2
    ttype[np.arange(N)[d["own_act"] >= 0], 1 + d["own_act"][d["own_act"] >= 0]] = 1
    ttype[:, 7:13] = 4
    ttype[np.arange(N)[d["opp_act"] >= 0], 7 + d["opp_act"][d["opp_act"] >= 0]] = 3
    ttype[:, 13:] = 5
    valid_tok = np.ones((N, 17), bool)
    valid_tok[:, 13:] = d["mask"][:, 6:10] > 0              # move tokens only when the move is legal
    valid_tok[:, 7:13] = nums[:, 6:, -1] < 0.5              # skip never-seen opponent slots
    valid_tok[:, 1:13] &= nums[:, :, 1] < 0.5               # skip fainted mons
    accs = {}
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(N, 12000, replace=False))
    subpos = {int(i): k for k, i in enumerate(sub)}
    keep_sub = {f"L{i}": np.zeros((len(sub), 17, 512), np.float16) for i in range(3)}

    def acc(name, t, pre, post, tanh=False):
        k = (name, t)
        if k not in accs:
            accs[k] = Acc(post.shape[-1])
        accs[k].add(pre, post, tanh)

    bs = 1024
    with torch.no_grad():
        for s in range(0, N, bs):
            b = np.arange(s, min(N, s + bs))
            x = torch.as_tensor(d["obs"][b], dtype=torch.float32)
            m = torch.as_tensor(d["mask"][b])
            cache = {}
            net.forward(x, m, None, cache)
            # instrumented forward does not call action_net/value through hooks? it does (modules are called)
            tt, vt = ttype[b], valid_tok[b]
            for i in range(3):
                pre, hid = cache[f"L{i}.mlp_pre"], cache[f"L{i}.mlp_hid"]
                for t in range(6):
                    sel = torch.as_tensor((tt == t) & vt)
                    if sel.any():
                        acc(f"L{i}", TT[t], pre[sel], hid[sel])
                rows = [subpos[int(r)] for r in b if int(r) in subpos]
                if rows:
                    loc = [k for k, r in enumerate(b) if int(r) in subpos]
                    keep_sub[f"L{i}"][rows] = hid[loc].half().numpy()
            # encoders
            mp = cur["mon_proj.pre"], cur["mon_proj"]
            for t in range(1, 5):
                sel = torch.as_tensor((tt[:, 1:13] == t) & vt[:, 1:13])
                if sel.any():
                    acc("mon_proj", TT[t], mp[0][sel], mp[1][sel])
            acc("field_proj", "field", cur["field_proj.pre"], cur["field_proj"])
            sel = torch.as_tensor(vt[:, 13:])
            acc("action_move_proj", "move", cur["action_move_proj.pre"][sel], cur["action_move_proj"][sel])
            mvsel = torch.as_tensor(d["split"]["cats"][b, :, 3:] > 0)       # real move slots of every mon
            acc("move_proj", "all_moves", cur["move_proj.pre"][mvsel], cur["move_proj"][mvsel])
            hsel = torch.as_tensor(d["mask"][b] > 0)
            acc("head", "legal_actions", cur["head.pre"][hsel], cur["head"][hsel])
            acc("value1", "all", cur["value1.pre"], cur["value1"], tanh=True)
            acc("value2", "all", cur["value2.pre"], cur["value2"], tanh=True)
    for h in hooks:
        h.remove()
    res = {"sites": {}}
    for (name, t), a in accs.items():
        sm = a.summary()
        if name.startswith("value"):
            fs = a.sat / a.n
            sm.update(sat_frac_mean=float(fs.mean()), units_sat_gt50pct=int((fs > 0.5).sum()),
                      units_sat_gt90pct=int((fs > 0.9).sum()), dead=None, mean_abs=float((a.sa / a.n).mean()))
        res["sites"][f"{name}|{t}"] = sm
    # union-over-token-types dead count for the encoder layers
    for i in range(3):
        mx = np.max([accs[(f"L{i}", t)].mx for t in TT if (f"L{i}", t) in accs], 0)
        pos = np.sum([accs[(f"L{i}", t)].pos for t in TT if (f"L{i}", t) in accs], 0)
        n = sum(accs[(f"L{i}", t)].n for t in TT if (f"L{i}", t) in accs)
        res["sites"][f"L{i}|union"] = {"dead": int((mx <= 0).sum()), "near_dead_<0.1%": int((pos / n < 1e-3).sum()),
                                       "always_on_>99.9%": int((pos / n > 0.999).sum()), "units": 512}
    mx = np.max([accs[("mon_proj", t)].mx for t in TT[1:5]], 0)
    res["sites"]["mon_proj|union"] = {"dead": int((mx <= 0).sum()), "units": len(mx)}
    # weights
    W = {}
    for i, L in enumerate(net.layers):
        W[f"L{i}.linear1"] = L.linear1.weight
        W[f"L{i}.linear2"] = L.linear2.weight
        W[f"L{i}.in_proj"] = L.self_attn.in_proj_weight
        W[f"L{i}.out_proj"] = L.self_attn.out_proj.weight
    for k in ("move_proj", "mon_proj", "field_proj", "action_move_proj"):
        W[f"{k}.0"] = getattr(fe, k)[0].weight
        W[f"{k}.2"] = getattr(fe, k)[2].weight
    W["head.0"] = pol.action_net.score[0].weight
    W["value.0"] = pol.mlp_extractor.value_net[0].weight
    W["value.2"] = pol.mlp_extractor.value_net[2].weight
    res["weights"] = {}
    for k, w in W.items():
        sv = torch.linalg.svdvals(w.detach().double()).numpy()
        res["weights"][k] = {"shape": list(w.shape), "PR_sigma2": pr(sv ** 2), "dims90": ndim(sv ** 2, 0.9),
                             "dims99": ndim(sv ** 2, 0.99), "top_sv": float(sv[0])}
    # top units per layer: importance = ||W2[:,u]|| × mean|hid_u| (per token type); correlate on the subsample
    mn, mvn, fdn = mon_names(), move_names(), field_names()
    s = d["split"]
    tsub = ttype[sub]
    vsub = valid_tok[sub]
    feats_state = {k: T[k][sub] for k in ("thr_frac", "thr_pko", "first", "hp_own", "hp_opp", "left_own",
                                           "left_opp", "spe_ratio")}
    feats_state["won"] = d["won"][sub]
    feats_state["value_logged"] = d["value"][sub]
    res["top_units"] = {}
    for i in range(3):
        w2n = net.layers[i].linear2.weight.detach().norm(dim=0).numpy()
        H = keep_sub[f"L{i}"].astype(np.float32)
        lst = []
        for t in range(6):
            a = accs.get((f"L{i}", TT[t]))
            if a is None:
                continue
            imp = w2n * a.sa / a.n
            for u in np.argsort(-imp)[:6]:
                lst.append((float(imp[u]), t, int(u)))
        lst.sort(reverse=True)
        out = []
        for imp, t, u in lst[:12]:
            rows, pos = np.nonzero((tsub == t) & vsub)
            h = H[rows, pos, u]
            cands = {}
            r_ = sub[rows]
            if t == 5:
                j = pos - 13
                oa = np.maximum(d["own_act"][r_], 0)
                mvx = s["move_nums"][r_, oa, j].astype(np.float32)
                for c, nme in enumerate(mvn):
                    cands["move." + nme] = mvx[:, c]
                for k in ("our_frac", "our_pko"):
                    cands["truth." + k] = T[k][r_, j]
            elif t == 0:
                fx = s["field"][r_].astype(np.float32)
                for c, nme in enumerate(fdn):
                    cands["field." + nme] = fx[:, c]
            else:
                mon = pos - 1
                mx_ = s["nums"][r_, mon].astype(np.float32)
                for c, nme in enumerate(mn):
                    cands["mon." + nme] = mx_[:, c]
                if t == 2:
                    cands["truth.bench_pko"] = T["bench_pko"][r_, mon]
            for k, v in feats_state.items():
                cands["truth." + k] = v[rows]
            cors = {}
            for k, v in cands.items():
                ok = np.isfinite(v) & np.isfinite(h)
                if ok.sum() > 100 and v[ok].std() > 0 and h[ok].std() > 0:
                    cors[k] = float(np.corrcoef(h[ok], v[ok])[0, 1])
            top = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:4]
            out.append({"unit": u, "token": TT[t], "importance": imp,
                        "frac_active": float(accs[(f"L{i}", TT[t])].pos[u] / accs[(f"L{i}", TT[t])].n),
                        "top_corr": top})
        res["top_units"][f"L{i}"] = out
    (OUT / f"mlp_{out_name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res["sites"].items()}, indent=0)[:6000])


if __name__ == "__main__":
    main()
