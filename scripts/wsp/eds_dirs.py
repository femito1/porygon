"""eds step 1: where "this move is Sleep Talk", "our active is asleep" and the status counter live in the move-token
residual stream, read three ways:
  probe   difference of means / least squares on the logged rows (cached by eds_cache.py)
  causal  counterfactual inputs on the 707 asleep-ST rows (counter set to c, sleep removed, ST id -> UNK): the
          change of the stream at the Sleep Talk token, per layer and per attention head (reading passes, logged)
  weights the embedding row / numeric columns pushed through move_proj -> action_move_proj and mon_proj
Then: readouts of the gate directions on f2 = norm2(L2.mid) for every token group of every row, and an exact
mean-ablation screen of the L2 MLP units with the smallest variance·|W2 col|² (heads-only passes on the cache).
    python scripts/wsp/eds_dirs.py 500M  -> out/eds/dirs_<tag>.txt, dirs_<tag>.npz
"""
import sys
from pathlib import Path

import numpy as np
import torch

from beh_common import NUM_OFF, load_local
from porygon.encoder import FIELD_DIM, MON_DIM, VOCAB
from wsp_common import Net, load

torch.set_num_threads(2)
tag = sys.argv[1] if len(sys.argv) > 1 else "500M"
OUT = Path("/vol/bitbucket/fr1226/porygon-ws-wsp/out/eds"); LOC = Path("/tmp/wsp-674320/eds")
d = load_local(); ph = np.load(OUT / "phase.npz", allow_pickle=True)
N = len(d["mask"]); r = np.arange(N)
mons = d["obs"][:, FIELD_DIM:].reshape(N, 12, MON_DIM)
own_a = mons[:, :6, NUM_OFF + 115].astype(np.float32).argmax(1)
MOVES = np.array(["<pad>", "<unk>"] + VOCAB["moves"])
ST_ID = int(np.flatnonzero(MOVES == "sleeptalk")[0]); REST_ID = int(np.flatnonzero(MOVES == "rest")[0])
ids = mons[r, own_a, 3:7].astype(np.int64)
is_st = ids == ST_ID                         # [N,4] move token j is Sleep Talk
slp = ph["slp"]; cnt = ph["cnt"]
H = slp & ph["st_legal"].any(1)
jst = is_st.argmax(1)
pol = load(tag); net = Net(pol); fe = net.fe
txt = []
checks = []  # forward-pass log


def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); txt.append(s)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------- probe directions (cached streams)
S = np.load(OUT / f"{tag}_sub.npz")
rows = S["rows"]; layers = ["L-1_out", "L0_out", "L1_out", "L2_mid"]
f2 = np.load(LOC / f"{tag}_f2.npy", mmap_mode="r")
stk = is_st[rows]; has_st = stk.any(1)
P(f"== {tag}: probe directions at the move tokens (rows with Sleep Talk in the active's moveset: {has_st.sum()})")
dirs = {}
for L in layers + ["f2"]:
    X = S[L][:, 1:] if L != "f2" else np.asarray(f2[rows][:, 13:17])          # [n,4,256] move tokens
    st_tok = X[has_st, jst[rows][has_st]]
    oth = X[has_st][~stk[has_st]]
    dST = st_tok.mean(0) - oth.mean(0)
    # asleep vs awake at the ST token
    a = slp[rows][has_st]
    dSLP = st_tok[a].mean(0) - st_tok[~a].mean(0)
    # counter: LS slope (per counter step) on asleep ST tokens with counter 0..7, and the 3 -> 4 step
    c = cnt[rows][has_st][a]; Y = st_tok[a]
    Xc = np.stack([np.ones(len(c)), c], 1)
    beta = np.linalg.lstsq(Xc, Y, rcond=None)[0][1]
    d34 = Y[c >= 4].mean(0) - Y[(c == 3)].mean(0)
    sd = st_tok.std(0).mean()
    dirs[L] = dict(st=dST, slp=dSLP, cnt=beta, d34=d34)
    P(f"{L:7s} |ΔST| {np.linalg.norm(dST):6.2f} |Δasleep| {np.linalg.norm(dSLP):6.2f} |counter slope| "
      f"{np.linalg.norm(beta):5.2f}/step |Δ(c≥4 - c=3)| {np.linalg.norm(d34):5.2f}  token sd/dim {sd:.2f} | "
      f"cos(ST,asleep) {cos(dST, dSLP):+.2f} cos(asleep,counter) {cos(dSLP, beta):+.2f} cos(slope, c34) {cos(beta, d34):+.2f}")
for k in ("st", "slp", "cnt"):
    P(f"  persistence of the {k} direction: cos with the L-1 version: " +
      ", ".join(f"{L} {cos(dirs[L][k], dirs['L-1_out'][k]):+.2f}" for L in layers[1:]) +
      f"; cos(L2.mid, f2) {cos(dirs['L2_mid'][k], dirs['f2'][k]):+.2f}")

# ---------------------------------------------------------------- causal: counterfactual inputs at the ST token
idx = np.flatnonzero(H); n = len(idx); ar = np.arange(n)
X0 = d["obs"][idx].astype(np.float32); MK = d["mask"][idx]; oa = own_a[idx]; jj = jst[idx]


def setm(X, fn):
    X = X.copy(); m = X[:, FIELD_DIM:].reshape(n, 12, MON_DIM); fn(m); return X


def run(X, label):
    c = {}
    with torch.no_grad():
        lg, v = net.forward(torch.as_tensor(X), torch.as_tensor(MK), None, c)
    checks.append(f"reading pass ({n} asleep-ST rows): {label}")
    out = {k: c[k][ar, 13 + jj].numpy() for k in ("L-1.out", "L0.out", "L1.out", "L2.mid")}
    out["f2"] = net.layers[2].norm2(c["L2.mid"])[ar, 13 + jj].detach().numpy()
    out["head0"] = c["L0.head_out"][ar, 13 + jj].numpy(); out["head1"] = c["L1.head_out"][ar, 13 + jj].numpy()
    out["head2"] = c["L2.head_out"][ar, 13 + jj].numpy()
    out["pst"] = lg.softmax(-1)[ar, 6 + jj].numpy(); out["v"] = v.numpy()
    return out


base = run(X0, "stock")
sweep = {}
for cc in range(0, 8):
    sweep[cc] = run(setm(X0, lambda m: m.__setitem__((ar, oa, NUM_OFF + 65), cc / 10)), f"counter := {cc}")
awake = run(setm(X0, lambda m: (m.__setitem__((ar, oa, NUM_OFF + 63), 0.0), m.__setitem__((ar, oa, NUM_OFF + 65), 0.0))),
            "sleep removed (status none, counter 0)")


def unk(m):
    m[ar, oa, 3 + jj] = 1.0  # ST id -> UNK in the active's moveset (feeds the mon token and the move token)


noST = run(setm(X0, unk), "Sleep Talk id -> UNK")
P(f"\n== causal directions at the Sleep Talk token ({n} rows): mean Δ stream; cos with the probe direction")
P("counter sweep, P(ST): " + " ".join(f"{cc}:{sweep[cc]['pst'].mean():.2f}" for cc in sweep))
cdirs = {}
for L in ("L-1.out", "L0.out", "L1.out", "L2.mid", "f2"):
    Lp = L.replace(".", "_")
    dc = (sweep[4][L] - sweep[3][L]).mean(0); dc_all = (sweep[6][L] - sweep[0][L]).mean(0) / 6
    ds = (base[L] - awake[L]).mean(0)       # asleep (with its counter) vs not asleep
    ds0 = (sweep[0][L] - awake[L]).mean(0)  # asleep at counter 0 vs not asleep: the status flag alone
    dst = (base[L] - noST[L]).mean(0)
    cdirs[L] = dict(c34=dc, cslope=dc_all, slp0=ds0, st=dst)
    P(f"{L:7s} counter 3→4 |Δ| {np.linalg.norm(dc):5.2f} (cos probe c≥4−c3 {cos(dc, dirs[Lp]['d34']):+.2f}, probe slope "
      f"{cos(dc, dirs[Lp]['cnt']):+.2f}) | asleep flag |Δ| {np.linalg.norm(ds0):5.2f} (cos probe {cos(ds0, dirs[Lp]['slp']):+.2f}) | "
      f"ST id |Δ| {np.linalg.norm(dst):5.2f} (cos probe {cos(dst, dirs[Lp]['st']):+.2f})"
      + ("" if L == "L-1.out" else ""))
    # linearity of the counter: consecutive steps projected on the 3→4 direction
    u = dc / np.linalg.norm(dc)
    P("        projection of mean stream on u(3→4) by counter: " +
      " ".join(f"{cc}:{(sweep[cc][L].mean(0) @ u):+.2f}" for cc in sweep) + f" (row sd at c=4 {np.std(sweep[4][L] @ u):.2f})")
P("\nper-head share of the counter 0→6 change at the ST token (Δ head_out · Δ total / |Δ total|²):")
for li in (0, 1, 2):
    dh = (sweep[6][f"head{li}"] - sweep[0][f"head{li}"]).mean(0)       # [H,256]
    tot = (sweep[6]["L2.mid" if li == 2 else ("L1.out" if li == 1 else "L0.out")] - sweep[0]["L2.mid" if li == 2 else ("L1.out" if li == 1 else "L0.out")]).mean(0)
    sh = dh @ tot / (tot @ tot)
    P(f"  L{li}: " + " ".join(f"h{h} {sh[h]:+.2f}" for h in range(8)))

# ---------------------------------------------------------------- weights: push embedding row / numeric columns
with torch.no_grad():
    E = fe.moves.weight.detach()
    st_num = torch.as_tensor(d["obs"][idx[0]].astype(np.float32)[FIELD_DIM:].reshape(12, MON_DIM)[oa[0], NUM_OFF + 125 + 44 * jj[0]: NUM_OFF + 125 + 44 * (jj[0] + 1)])
    ebar = E[2:].mean(0)
    tok = lambda e: fe.action_move_proj(fe.move_proj(torch.cat([e, st_num]))).numpy()  # noqa: E731
    w_emb = tok(E[ST_ID]) - tok(ebar)
    P(f"\nweights: ST token input from the embedding row alone (row vs table mean, same numerics): |Δ| {np.linalg.norm(w_emb):.2f}, "
      f"cos with the probe ST direction at L-1 {cos(w_emb, dirs['L-1_out']['st']):+.2f}, at f2 {cos(w_emb, dirs['f2']['st']):+.2f}; "
      f"|e_ST| {E[ST_ID].norm():.2f} vs mean row norm {E[2:].norm(dim=1).mean():.2f}; |e_ST - e_REST| {(E[ST_ID]-E[REST_ID]).norm():.2f}")
    # numeric columns of the own-active token: slp flag (63) and counter (65) through mon_proj (first layer)
    W1 = fe.mon_proj[0].weight.detach().numpy()   # [256, 573]
    num0 = 64 + 32 + 32 + 4 * 64 + 64
    P(f"mon_proj layer-1 input weights: |col slp| {np.linalg.norm(W1[:, num0 + 63]):.2f}, |col counter| {np.linalg.norm(W1[:, num0 + 65]):.2f}, "
      f"|col tox| {np.linalg.norm(W1[:, num0 + 64]):.2f}, median |numeric col| {np.median(np.linalg.norm(W1[:, num0:num0 + 125], axis=0)):.2f}; "
      f"cos(slp col, counter col) {cos(W1[:, num0 + 63], W1[:, num0 + 65]):+.2f}")
    own_in = (sweep[6]["L-1.out"] - sweep[0]["L-1.out"])  # at the ST token: zero (counter enters only the own-active token)
    P(f"counter change at the ST token's own input (L-1): |Δ| {np.abs(own_in).max():.1e} (enters only via attention)")

np.savez(OUT / f"dirs_{tag}.npz", **{f"probe_{L}_{k}": v for L in dirs for k, v in dirs[L].items()},
         **{f"causal_{L.replace('.', '_')}_{k}": v for L in cdirs for k, v in cdirs[L].items()})

# ---------------------------------------------------------------- unit screen: exact mean-ablation, heads only
U = np.load(OUT / f"{tag}_units.npz"); order = np.argsort(U["proxy"])[:12]
o2 = np.load(LOC / f"{tag}_out2.npy", mmap_mode="r")
L2 = net.layers[2]
rng = np.random.default_rng(0); samp = np.sort(rng.choice(N, 8000, replace=False))
P(f"\n== L2 MLP unit screen (8,000 random rows; heads-only passes from the cached L2.out): mean-ablating unit u")
P("unit | var·|W2col|² | mean hid | frac(pre > 0) | mean TV | p99 TV | mean |ΔV| | argmax flips")
fs = torch.as_tensor(np.asarray(f2[samp])); os_ = torch.as_tensor(np.asarray(o2[samp])); mk = torch.as_tensor(d["mask"][samp])


def heads(o):
    lat = fe.norm(o); flat = lat.flatten(1)
    lg = pol.action_net(flat).masked_fill(mk == 0, torch.finfo(torch.float32).min)
    return lg.softmax(-1), pol.value_net(pol.mlp_extractor.forward_critic(flat)).squeeze(-1)


with torch.no_grad():
    p0, v0 = heads(os_)
    res = []
    for u in order:
        pre = fs @ L2.linear1.weight[u] + L2.linear1.bias[u]
        hid = torch.nn.functional.gelu(pre)
        delta = (U["mean"][u] - hid).unsqueeze(-1) * L2.linear2.weight[:, u]
        p1, v1 = heads(os_ + delta)
        tv = 0.5 * (p1 - p0).abs().sum(1).numpy()
        res.append((u, tv.mean()))
        P(f"{u} | {U['proxy'][u]:.2e} | {U['mean'][u]:+.3f} | {(pre > 0).float().mean():.4f} | {tv.mean():.2e} | "
          f"{np.percentile(tv, 99):.2e} | {(v1 - v0).abs().mean():.2e} | {(p1.argmax(1) != p0.argmax(1)).float().mean():.4f}")
        checks.append(f"heads-only reading pass (8000 rows): mean-ablate L2 unit {u}")
best = min(res, key=lambda t: t[1])[0]
P(f"least-used unit (smallest mean TV): {best}")
P("\nforward-pass log (reading passes, not edit checks):"); [P("  " + c) for c in checks]
(OUT / f"dirs_{tag}.txt").write_text("\n".join(txt) + "\n")
