"""beh analysis 3: occlusion attribution over named feature groups.

For each (token role, feature group) the group is replaced by its dataset mean for that role (numerics: column
means over real tokens of the role; ids: the mean embedding vector over real tokens of the role) on real tokens
only (never on unseen-opponent / padding slots), and the effect on
  (i)  log π(net argmax)            (and total-variation distance of the policy),
  (ii) switch-vs-move margin  logsumexp(switch logits) - logsumexp(move logits)  (voluntary positions only),
  (iii) value V
is recorded per position. Roles: own active, own bench, opp active, opp bench, field.
    python scripts/wsp/beh_attr.py 500M [N_POS]   -> out/beh/attr_<tag>.npz, attr_<tag>.txt
"""
import sys
import time

import numpy as np
import torch

from beh_common import OUT, FIELD_DIM, MON_DIM, NUM_OFF, MOVENUM_OFF, ACTIVE, OPPACT, UNKNOWN, features, load_local, ratio_boot
from porygon.encoder import ACTIVE_FLAG, MOVE_DIM, N_CAT, N_MOVES
from wsp_common import Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
NPOS = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
d = load_local()
f = features(d)
rng = np.random.default_rng(0)
rows = np.sort(rng.choice(len(d["mask"]), NPOS, replace=False))
X = d["obs"][rows].astype(np.float32)
MK = d["mask"][rows]
game = d["game"][rows]
vol = f["voluntary"][rows]
N = len(rows)
mons = X[:, FIELD_DIM:].reshape(N, 12, MON_DIM)
nums = mons[..., NUM_OFF:MOVENUM_OFF]
own_a = f["own_a"][rows]
opp_a = f["opp_a"][rows]
real = (nums[..., UNKNOWN] < 0.5) & (mons[..., 0] > 0)  # seen, non-padding tokens
role = np.zeros((N, 12), dtype=object)
role[:, :6] = "own_bench"
role[:, 6:] = "opp_bench"
role[np.arange(N), own_a] = np.where(f["has_act"][rows], "own_active", "own_bench")
role[np.arange(N), opp_a] = np.where(f["has_opp"][rows], "opp_active", "opp_bench")
ROLES = ["own_active", "own_bench", "opp_active", "opp_bench"]
rmask = {r: torch.as_tensor((role == r) & real) for r in ROLES}

NUM_GROUPS = {"hp+fainted": [0, 1], "base stats": list(range(2, 8)), "stats+level": list(range(8, 15)),
              "types": list(range(15, 33)), "boosts": list(range(52, 59)), "status": list(range(59, 66)),
              "volatile effects": list(range(66, 115)), "flags (first turn, protect, recharge, trapped..)": list(range(117, 123))}
MOVE_GROUPS = {"move power": [0], "move acc/prio/pp/avail": [1, 2, 3, 4], "move category": [5, 6, 7],
               "move type": list(range(8, 26)), "move STAB+effectiveness": [26, 27, 28], "move effect flags": list(range(29, 44))}
ID_GROUPS = ["species id", "item id", "ability id", "move ids", "movepool ids"]
FIELD_GROUPS = {"weather": list(range(0, 17)), "field effects (trick room..)": list(range(17, 45)),
                "our side (screens, hazards)": list(range(45, 63)), "opp side (screens, hazards)": list(range(63, 81)),
                "turn": [81], "counts (unseen, alive)": [82, 83, 84], "flags (force switch, trapped, n legal)": list(range(85, 95))}

net = Net(load(tag))
fe = net.fe


def col_means():
    """per role: mean of mon numerics block and move numerics over real tokens (whole dataset sample)"""
    out = {}
    for r in ROLES:
        m = (role == r) & real
        out[r] = mons[m].mean(0)
    return out


MEANS = col_means()
FIELD_MEAN = X[:, :FIELD_DIM].mean(0)


def emb_means():
    with torch.no_grad():
        cats = torch.as_tensor(mons[..., :N_CAT]).long()
        pool = torch.as_tensor(mons[..., N_CAT:NUM_OFF]).long()
        out = {}
        for r in ROLES:
            m = rmask[r]
            mv = cats[..., 3:][m]
            mvf = mv.flatten()
            out[r] = {"species": fe.species(cats[..., 0][m]).mean(0), "items": fe.items(cats[..., 1][m]).mean(0),
                      "abilities": fe.abilities(cats[..., 2][m]).mean(0),
                      "moves": fe.moves(mvf[mvf > 0]).mean(0)}
            pm = (pool[m] > 0).unsqueeze(-1).float()
            pmean = (fe.moves(pool[m]) * pm).sum(-2) / pm.sum(-2).clamp(min=1)
            out[r]["pool"] = pmean.mean(0)
        return out


EMEANS = emb_means()


class BNet(Net):
    rep = None  # (kind, role) or ("token", role)

    def embed(self, x, iv=None, cache=None):
        fe = self.fe
        B = x.shape[0]
        mons_ = x[:, FIELD_DIM:].reshape(B, 12, MON_DIM)
        cats = mons_[..., :N_CAT].long()
        pool = mons_[..., N_CAT:NUM_OFF].long()
        nums_ = mons_[..., NUM_OFF:MOVENUM_OFF]
        move_nums = mons_[..., MOVENUM_OFF:].reshape(B, 12, N_MOVES, MOVE_DIM)
        sp, it, ab = fe.species(cats[..., 0]), fe.items(cats[..., 1]), fe.abilities(cats[..., 2])
        mv_e = fe.moves(cats[..., 3:])
        pm = (pool > 0).unsqueeze(-1).float()
        pool_mean = (fe.moves(pool) * pm).sum(-2) / pm.sum(-2).clamp(min=1)
        if self.rep is not None:
            kind, r, m = self.rep  # m: [B,12] bool
            mm = m.unsqueeze(-1)
            E = EMEANS[r]
            if kind == "species id":
                sp = torch.where(mm, E["species"], sp)
            elif kind == "item id":
                it = torch.where(mm, E["items"], it)
            elif kind == "ability id":
                ab = torch.where(mm, E["abilities"], ab)
            elif kind == "move ids":
                mv_e = torch.where(mm.unsqueeze(-1) & (cats[..., 3:] > 0).unsqueeze(-1), E["moves"], mv_e)
            elif kind == "movepool ids":
                pool_mean = torch.where(mm, E["pool"], pool_mean)
        moves = fe.move_proj(torch.cat([mv_e, move_nums], -1))
        mon = torch.cat([sp, it, ab, moves.flatten(-2), pool_mean, nums_], -1)
        if self.rep is not None and self.rep[0] == "token":
            mon = torch.where(self.rep[2].unsqueeze(-1), self.tokmean[self.rep[1]], mon)
        active = nums_[:, :6, ACTIVE_FLAG]
        active_moves = torch.einsum("bs,bsmd->bmd", active, moves[:, :6])
        tokens = torch.cat([fe.field_proj(x[:, :FIELD_DIM]).unsqueeze(1), fe.mon_proj(mon),
                            fe.action_move_proj(active_moves)], 1)
        if cache is not None:
            cache.update(mon_in=mon, tokens=tokens)
        return tokens + fe.pos


bnet = BNet(net.p)
# mean mon_in vector per role (for whole-token occlusion)
with torch.no_grad():
    cache = {}
    bnet.forward(torch.as_tensor(X[:2000]), torch.as_tensor(MK[:2000]), cache=cache)
    mi = cache["mon_in"]
    bnet.tokmean = {r: mi[rmask[r][:2000]].mean(0) for r in ROLES}


def run(xm, rep=None, bs=1000):
    L, V = [], []
    with torch.no_grad():
        for s in range(0, N, bs):
            bnet.rep = None if rep is None else (rep[0], rep[1], rep[2][s:s + bs])
            lg, v = bnet.forward(torch.as_tensor(xm[s:s + bs]), torch.as_tensor(MK[s:s + bs]))
            L.append(lg); V.append(v)
    bnet.rep = None
    return torch.cat(L).numpy().astype(np.float64), torch.cat(V).numpy()


def summarise(lg):
    lg = np.where(MK > 0, lg, -np.inf)
    lse = np.logaddexp.reduce(lg, 1)
    lp = lg - lse[:, None]
    sw = np.logaddexp.reduce(lg[:, :6], 1)
    mv = np.logaddexp.reduce(lg[:, 6:], 1)
    return lp, sw - mv


t0 = time.time()
lg0, v0 = run(X)
lp0, marg0 = summarise(lg0)
arg0 = lp0.argmax(1)
p0 = np.exp(lp0)
results = {}


def record(name, lg, v):
    lp, marg = summarise(lg)
    p = np.exp(lp)
    dlp = lp[np.arange(N), arg0] - lp0[np.arange(N), arg0]
    tv = 0.5 * np.abs(p - p0).sum(1)
    flip = lp.argmax(1) != arg0
    # move-only redistribution: TV among move actions renormalised (decisions with >=2 legal moves)
    pm, pm0 = p[:, 6:] / np.maximum(p[:, 6:].sum(1, keepdims=True), 1e-12), p0[:, 6:] / np.maximum(p0[:, 6:].sum(1, keepdims=True), 1e-12)
    tvm = 0.5 * np.abs(pm - pm0).sum(1)
    dm = np.where(vol, np.abs(marg - marg0), np.nan)
    results[name] = dict(dlp=dlp, tv=tv, flip=flip, tvm=tvm, dmarg=dm, dv=np.abs(v - v0))


for r in ROLES:
    m = rmask[r].numpy()
    for g, cols in NUM_GROUPS.items():
        xm = X.copy()
        mm = xm[:, FIELD_DIM:].reshape(N, 12, MON_DIM)
        for c in cols:
            mm[..., NUM_OFF + c] = np.where(m, MEANS[r][NUM_OFF + c], mm[..., NUM_OFF + c])
        record(f"{r}|{g}", *run(xm))
    for g, cols in MOVE_GROUPS.items():
        xm = X.copy()
        mm = xm[:, FIELD_DIM:].reshape(N, 12, MON_DIM)
        for j in range(4):
            for c in cols:
                k = MOVENUM_OFF + j * MOVE_DIM + c
                mm[..., k] = np.where(m, MEANS[r][k], mm[..., k])
        record(f"{r}|{g}", *run(xm))
    for g in ID_GROUPS:
        record(f"{r}|{g}", *run(X, (g, r, torch.as_tensor(m))))
    record(f"{r}|WHOLE TOKEN", *run(X, ("token", r, torch.as_tensor(m))))
    print(r, f"{time.time()-t0:.0f}s", flush=True)
for g, cols in FIELD_GROUPS.items():
    xm = X.copy()
    xm[:, cols] = FIELD_MEAN[cols]
    record(f"field|{g}", *run(xm))
xm = X.copy(); xm[:, :FIELD_DIM] = FIELD_MEAN
record("field|WHOLE TOKEN", *run(xm))
# all opponent information (both opp roles, whole tokens) at once
mopp = torch.as_tensor(((role == "opp_active") | (role == "opp_bench")) & real)
# opp active and bench replaced by their own role means
def run_opp_all():
    L, V = [], []
    with torch.no_grad():
        for s in range(0, N, 1000):
            xb = torch.as_tensor(X[s:s + 1000]); mb = torch.as_tensor(MK[s:s + 1000])
            # two-step: compute tokens with opp_active replaced, then replace opp_bench tokens by mean too
            bnet.rep = ("token", "opp_active", rmask["opp_active"][s:s + 1000])
            c = {}
            h = bnet.embed(xb, None, c)
            mon = c["mon_in"]
            mon = torch.where(rmask["opp_bench"][s:s + 1000].unsqueeze(-1), bnet.tokmean["opp_bench"], mon)
            tok = torch.cat([fe.field_proj(xb[:, :FIELD_DIM]).unsqueeze(1), fe.mon_proj(mon), c["tokens"][:, 13:]], 1) + fe.pos
            bnet.rep = None
            for i in range(3):
                tok = bnet.layer(i, tok, None, None)
            lat = fe.norm(tok).flatten(1)
            lg = bnet.p.action_net(lat).masked_fill(mb == 0, torch.finfo(torch.float32).min)
            v = bnet.p.value_net(bnet.p.mlp_extractor.forward_critic(lat)).squeeze(-1)
            L.append(lg); V.append(v)
    return torch.cat(L).numpy().astype(np.float64), torch.cat(V).numpy()
record("opp ALL TOKENS|WHOLE", *run_opp_all())
print("done", f"{time.time()-t0:.0f}s", flush=True)

names = list(results)
np.savez(OUT / f"attr_{tag}.npz", names=np.array(names), rows=rows,
         **{f"{k}__{n}": results[n][k] for n in names for k in results[n]})
lines = [f"occlusion attribution, {tag}, {N} positions ({vol.sum()} voluntary); mean over positions, 95% CI by game bootstrap",
         "group | |Δ log π(argmax)| | TV(policy) | argmax flips | TV among moves | |Δ switch-move margin| (vol) | |ΔV|"]
order = sorted(names, key=lambda n: -np.nanmean(results[n]["tv"]))
for n in order:
    R = results[n]
    cells = []
    for k in ("dlp", "tv", "flip", "tvm", "dmarg", "dv"):
        y = np.abs(R[k]).astype(float) if k == "dlp" else R[k].astype(float)
        e, lo, hi = ratio_boot(game, np.nan_to_num(y), w=np.isfinite(y).astype(float), n_boot=300)
        cells.append(f"{e:.3f} [{lo:.3f},{hi:.3f}]")
    lines.append(f"{n} | " + " | ".join(cells))
(OUT / f"attr_{tag}.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
