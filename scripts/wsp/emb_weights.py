"""Weight-space evidence on why the categorical embeddings did not learn (analyses 1, 2, 3-SVD, 7 of the emb brief).

Reads out/emb/ckpt/*.pt (emb_extract.py), out/emb/lr_history.json (emb_lr.py) and the logged decisions (token
frequencies). Writes out/emb/weights.json (+ printed tables).

1. rows bit-identical between checkpoints (Adam: a row with zero gradient forever never moves), vs seen tokens;
   0->2.5M drift from row statistics (unseen rows are pristine N(0,1) samples).
2. Adam state per parameter tensor: gradient RMS sqrt(v), eps-damping, rho^2 = E[m^2]/E[v] (pure noise: 1/19),
   effective step |m|/(sqrt(v)+eps) in units of lr, relative step lr*u/rms(w); per-row vs token frequency.
3. SVD of the first-layer column blocks (mon_proj.0, move_proj.0) vs random matrices and vs 2.5M.
7. drift scaling D(dt) within the 500-600M run (random walk: D ~ sqrt(sum lr^2), directional: D ~ sum lr);
   measured drift vs both predictions per tensor.
"""
import json

import numpy as np
import torch

from emb_util import ADAM_PER_ENV_STEP, OUT, TABLES, ck, emb, load_data, token_counts
from porygon.encoder import INDEX
from porygon.inference import library

EPS = 1e-5
NOISE_RHO2 = (1 - 0.9) / (1 + 0.9)  # E[m^2]/E[v] for zero-mean iid gradients, beta1 = 0.9
R = {}
STEPS = {"s2.5M": 2_499_840, "s10M": 9_999_360, "s21M": 21_261_312, "s66M": 65_950_032, "s136M": 135_949_332,
         "s201M": 200_948_682, "s298M": 297_811_982, "s400M": 400_029_852, "s498M": 498_267_524,
         "s500M": 500_036_720, "s505M": 505_036_720, "s510M": 510_036_720, "s515M": 515_036_720,
         "s525M": 525_036_720, "s545M": 545_036_720, "s575M": 575_036_720, "s595M": 595_036_720,
         "s600M": 600_002_347, "ctrl640": 640_032_682, "x30_640": 640_025_994, "x300_640": 640_025_414,
         "x300_605": 605_002_347, "x30_605": 605_002_347}


# ---------------------------------------------------------------- lr history -> sum lr, sum lr^2 over Adam steps
def lr_fn():
    h = json.load(open(OUT / "lr_history.json"))
    pts = []
    for r in ["20261020-181327-299545", "20261021-145709-300011", "20261022-120923-300097",
              "20261023-112550-300355", "20261023-125508-300389", "20261024-135245-300664",
              "20261025-171535-301033", "20261025-195826-301040", "20261026-082356-301120"]:
        pts += [tuple(p) for p in h.get(r, [])]
    # 0-60M has no log: the schedule 10^-4.23/(8x+1)^1.5 started at 5.89e-5; the optimizer lr stored in the
    # 2.5M / 10M / 21M checkpoints is 3.57e-5 / 1.41e-5 / 7.84e-6; the 60M resume restarted at 1.80e-5.
    early = [(0, 10 ** -4.23), (2_499_840, 3.569e-5), (9_999_360, 1.4066e-5), (21_261_312, 7.843e-6),
             (60_000_000, 7.0e-6), (60_088_632, 1.80e-5)]
    # (21M->60M: unknown; assume the 50M-total schedule continued to ~7e-6, i.e. lr ~ 1e-5 -> 7e-6)
    pts = sorted(early + [p for p in pts if p[0] > 60_088_632])
    t = np.array([p[0] for p in pts], float)
    lr = np.array([p[1] for p in pts], float)

    def sums(a, b, mult=1.0):
        grid = np.linspace(a, b, 4001)
        l = np.interp(grid, t, lr) * mult
        n_adam = (b - a) * ADAM_PER_ENV_STEP
        return float(l.mean() * n_adam), float(np.sqrt((l ** 2).mean() * n_adam))
    return sums


def arm_sums(arm, a, b, mult):
    h = json.load(open(OUT / "lr_history.json"))[arm]
    t = np.array([p[0] for p in h], float)
    lr = np.array([p[1] for p in h], float)
    grid = np.linspace(a, b, 2001)
    l = np.interp(grid, t, lr) * mult
    n = (b - a) * ADAM_PER_ENV_STEP
    return float(l.mean() * n), float(np.sqrt((l ** 2).mean() * n))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else float("nan")


def seen_rows(name, counts):
    for t in TABLES:
        if name == f"features_extractor.{t}.weight":
            return counts[t]["all"] > 0
    return None


def main():
    d = load_data()
    counts = token_counts(d)
    np.savez(OUT / "token_counts.npz", **{f"{t}_{r}": v for t in counts for r, v in counts[t].items()})
    del d
    lib = library()
    lib_tokens = {"species": {s for s in lib}, "moves": set(), "items": set(), "abilities": set()}
    for s, (_, sets) in lib.items():
        for st in sets:
            lib_tokens["moves"].update(st.moves)
            lib_tokens["items"].add(st.item)
            lib_tokens["abilities"].add(st.ability)
    sums = lr_fn()
    C = {k: ck(k) for k in STEPS}
    s0, s5, s6 = C["s2.5M"], C["s500M"], C["s600M"]

    # ---------------- 1. identical rows / seen tokens / drift vs frequency
    R["rows"] = {}
    print("\n== 1. rows bit-identical between checkpoints, vs tokens seen in the logged data / gen4 library")
    for t in TABLES:
        e0, e5, e6 = emb(s0, t), emb(s5, t), emb(s6, t)
        V = len(e0)
        real = np.arange(V) >= 2
        same = np.all(e0 == e6, 1) & real
        seen = (counts[t]["all"] > 0) & real
        inlib = np.zeros(V, bool)
        for tok in lib_tokens[t]:
            i = INDEX[t].get(tok)
            if i is not None:
                inlib[i] = True
        r = {"V_real": int(real.sum()), "identical_2.5M_600M": int(same.sum()), "seen_in_logs": int(seen.sum()),
             "in_library": int(inlib.sum()), "identical&seen": int((same & seen).sum()),
             "moved&unseen": int((~same & ~seen & real).sum()), "moved&unseen&notlib": int((~same & ~seen & ~inlib & real).sum()),
             "identical&inlib": int((same & inlib).sum()),
             "unk_row_moved_rel": float(np.linalg.norm(e6[1] - e0[1]) / np.linalg.norm(e0[1]))}
        for a, b in (("s2.5M", "s10M"), ("s10M", "s21M"), ("s21M", "s66M"), ("s400M", "s600M")):
            r[f"identical_{a}_{b}"] = int((np.all(emb(C[a], t) == emb(C[b], t), 1) & real).sum())
        # drift of moved rows 2.5M->600M, relative to row norm; vs frequency
        mv = ~same & real
        dr = np.linalg.norm(e6 - e0, axis=1)
        nr = np.linalg.norm(e0, axis=1)
        f = counts[t]["all"]
        r["moved_rows"] = int(mv.sum())
        r["drift_rel_moved_median"] = float(np.median(dr[mv] / nr[mv]))
        r["drift_rel_table_frob"] = float(np.linalg.norm((e6 - e0)[real]) / np.linalg.norm(e0[real]))
        sel = mv & seen
        if sel.sum() > 5:
            r["spearman_drift_vs_freq"] = float(spearman(dr[sel], f[sel]))
            qs = np.quantile(f[sel], [0, .25, .5, .75, 1])
            r["drift_by_freq_quartile"] = []
            for lo, hi in zip(qs[:-1], qs[1:]):
                s_ = sel & (f >= lo) & (f <= hi)
                r["drift_by_freq_quartile"].append({"freq_per_decision": [float(lo), float(hi)], "n": int(s_.sum()),
                                                     "drift_median": float(np.median(dr[s_])),
                                                     "drift_rel_median": float(np.median(dr[s_] / nr[s_]))})
        # 0 -> 2.5M from row statistics: identical rows are pristine N(0,1) init samples
        D = e0.shape[1]
        pr = e0[same]
        sq = (e0 ** 2).sum(1)
        r["pristine_sqnorm_mean"] = float(sq[same].mean()) if same.any() else None
        r["pristine_sqnorm_var"] = float(sq[same].var()) if same.any() else None
        r["pristine_elem_mean"] = float(pr.mean()) if same.any() else None
        r["pristine_elem_var"] = float(pr.var()) if same.any() else None
        r["moved_sqnorm_mean_2.5M"] = float(sq[mv].mean()) if mv.any() else None
        r["dof"] = D
        if mv.sum() > 2:
            # E|e_2.5|^2 = D + E|delta|^2 (+2 E e0.delta): excess over the pristine mean (SE from chi2: sqrt(2D/n))
            r["excess_sqnorm_moved_vs_D"] = float(sq[mv].mean() - D)
            r["excess_sqnorm_se"] = float(np.sqrt(2 * D / mv.sum()))
            m_ = e0[mv].mean(0)
            r["mean_vec_sq_moved"] = float((m_ ** 2).sum())
            r["mean_vec_sq_expected_iid"] = float(D / mv.sum())
            ev = np.linalg.eigvalsh(np.cov(e0[mv].T))[::-1]
            gam = D / mv.sum()
            r["cov_top_eig"] = [float(x) for x in ev[:5]]
            r["mp_edge"] = float((1 + np.sqrt(gam)) ** 2)
            if same.sum() > D:
                evp = np.linalg.eigvalsh(np.cov(e0[same].T))[::-1]
                r["cov_top_eig_pristine"] = [float(x) for x in evp[:5]]
                r["mp_edge_pristine"] = float((1 + np.sqrt(D / same.sum())) ** 2)
        R["rows"][t] = r
        print(t, json.dumps({k: v for k, v in r.items() if k != "drift_by_freq_quartile"}))
        for q in r.get("drift_by_freq_quartile", []):
            print("   ", q)

    # ---------------- lr sums
    seg = {"0-2.5M": (0, 2_499_840), "2.5-10M": (2_499_840, 9_999_360), "10-21M": (9_999_360, 21_261_312),
           "2.5M-600M": (2_499_840, 600_002_347), "400-600M": (400_029_852, 600_002_347),
           "500-600M": (500_036_720, 600_002_347), "0-600M": (0, 600_002_347)}
    R["lr_sums"] = {k: dict(zip(("sum_lr", "sqrt_sum_lr2"), sums(*v))) for k, v in seg.items()}
    print("\n== lr sums over Adam steps (per-element max directional drift = sum lr; random walk = sqrt(sum lr^2))")
    for k, v in R["lr_sums"].items():
        print(f"  {k:10s} sum lr {v['sum_lr']:.4g}   sqrt(sum lr^2) {v['sqrt_sum_lr2']:.4g}")

    # ---------------- 2. Adam state per tensor
    print("\n== 2. Adam state per tensor (500M / 600M / x300_640); rho2 = E[m^2]/E[v], noise = 1/19 = 0.0526")
    R["adam"] = {}
    for tag in ("s400M", "s500M", "s600M", "s2.5M", "s21M", "ctrl640", "x30_640", "x300_640"):
        c = C[tag]
        lr = c["hyper"]["lr"]
        rows = {}
        for n, w in c["params"].items():
            if n not in c["m"]:
                continue
            w, m, v = w.numpy(), c["m"][n].numpy(), c["v"][n].numpy()
            sr = seen_rows(n, counts)
            if sr is not None:  # embedding: rows with v>0 (received gradient in the last ~1e5 Adam steps)
                act = (v.sum(1) > 0)
                w, m, v = w[act], m[act], v[act]
            w, m, v = w.ravel(), m.ravel(), v.ravel()
            sv = np.sqrt(v)
            nz = v > 0
            u = m / (sv + EPS)
            mult = 1.0
            if sr is not None and tag.startswith("x30_"):
                mult = 30.0
            if sr is not None and tag.startswith("x300_"):
                mult = 300.0
            rho2 = float((m[nz] ** 2).sum() / v[nz].sum()) if nz.any() else float("nan")
            rows[n] = {"n": int(w.size), "rms_w": rms(w), "grad_rms": float(np.sqrt(v.mean())),
                       "sqrtv_median": float(np.median(sv)), "frac_sqrtv_lt_eps": float((sv < EPS).mean()),
                       "damp": float(np.mean(sv / (sv + EPS))),
                       "rho2": rho2, "mean_abs_m_over_sqrtv": float(np.mean(np.abs(m[nz]) / sv[nz])) if nz.any() else None,
                       "signal_frac": (rho2 - NOISE_RHO2) / (1 - NOISE_RHO2),
                       "u_rms": rms(u), "rel_step": lr * mult * rms(u) / rms(w)}
        R["adam"][tag] = rows
    hdr = f"{'tensor':48s} {'n':>8s} {'rms w':>8s} {'grad rms':>9s} {'sqrtv<eps':>9s} {'damp':>6s} {'rho2':>7s} {'E|m|/sv':>7s} {'u_rms':>7s} {'rel step':>9s}"
    for tag in ("s500M", "s600M", "x300_640"):
        print(f"\n-- {tag} (lr {C[tag]['hyper']['lr']:.3g})\n{hdr}")
        for n, r in R["adam"][tag].items():
            print(f"{n:48s} {r['n']:8d} {r['rms_w']:8.4f} {r['grad_rms']:9.2e} {r['frac_sqrtv_lt_eps']:9.3f} {r['damp']:6.3f} "
                  f"{r['rho2']:7.4f} {r['mean_abs_m_over_sqrtv'] or 0:7.3f} {r['u_rms']:7.3f} {r['rel_step']:9.2e}")

    # per-row Adam state vs frequency (500M)
    print("\n== 2b. per-row Adam state vs token frequency (500M)")
    R["adam_rows"] = {}
    for t in TABLES:
        n = f"features_extractor.{t}.weight"
        out = {}
        for tag in ("s500M", "s600M", "x300_640"):
            m, v = C[tag]["m"][n].numpy(), C[tag]["v"][n].numpy()
            f = counts[t]["all"]
            act = (v.sum(1) > 0)
            act[:2] = False
            gr = np.sqrt(v.mean(1))
            rho = (m ** 2).sum(1) / np.maximum(v.sum(1), 1e-300)
            sel = act & (f > 0)
            qs = np.quantile(f[sel], [0, .25, .5, .75, 1])
            bins = []
            for lo, hi in zip(qs[:-1], qs[1:]):
                s_ = sel & (f >= lo) & (f <= hi)
                bins.append({"freq": [float(lo), float(hi)], "n": int(s_.sum()), "grad_rms_median": float(np.median(gr[s_])),
                             "rho2_median": float(np.median(rho[s_])), "damp_median": float(np.median((np.sqrt(v[s_]) / (np.sqrt(v[s_]) + EPS)).mean(1)))})
            out[tag] = {"active_rows": int(act.sum()), "active_unseen_rows": int((act & (f == 0)).sum()),
                        "spearman_gradrms_freq": float(spearman(gr[sel], f[sel])),
                        "spearman_rho2_freq": float(spearman(rho[sel], f[sel])), "bins": bins}
        R["adam_rows"][t] = out
        print(t, json.dumps(out["s500M"]))

    # ---------------- 7. drift scaling within the 500-600M run and vs predictions
    print("\n== 7. drift D(dt) from 505M (per-element RMS of the change); alpha = d log D / d log dt")
    ends = ["s510M", "s515M", "s525M", "s545M", "s575M", "s595M"]
    R["drift_scaling"] = {}
    for n in C["s505M"]["params"]:
        w0 = C["s505M"]["params"][n].numpy()
        sr = seen_rows(n, counts)
        Ds, dts, slr, slr2 = [], [], [], []
        for e in ends:
            dw = C[e]["params"][n].numpy() - w0
            if sr is not None:
                dw = dw[sr]
            Ds.append(rms(dw))
            dts.append(STEPS[e] - STEPS["s505M"])
            a, b = sums(STEPS["s505M"], STEPS[e])
            slr.append(a)
            slr2.append(b)
        Ds = np.array(Ds)
        ok = Ds > 0
        alpha = float(np.polyfit(np.log(np.array(dts)[ok]), np.log(Ds[ok]), 1)[0]) if ok.sum() > 2 else float("nan")
        a6 = R["adam"]["s500M"].get(n)
        damp = a6["damp"] if a6 else 1.0
        R["drift_scaling"][n] = {"D": Ds.tolist(), "dt": dts, "alpha": alpha,
                                 "D_over_rw": float(Ds[-1] / (slr2[-1] * damp)), "D_over_dir": float(Ds[-1] / slr[-1]),
                                 "rms_w": rms(w0 if sr is None else w0[sr])}
    print(f"{'tensor':48s} {'alpha':>6s} {'D(90M)':>9s} {'D/rw':>7s} {'D/sumlr':>8s} {'D/rms w':>8s}")
    for n, r in R["drift_scaling"].items():
        print(f"{n:48s} {r['alpha']:6.2f} {r['D'][-1]:9.2e} {r['D_over_rw']:7.2f} {r['D_over_dir']:8.4f} {r['D'][-1] / max(r['rms_w'], 1e-30):8.2e}")

    # long segments: measured drift vs predictions (2.5M -> 600M; 400 -> 600M; 0 -> 2.5M estimate)
    print("\n== 7b. measured per-element drift vs random-walk / directional predictions over long segments")
    R["drift_long"] = {}
    for (a, b) in (("s2.5M", "s10M"), ("s10M", "s21M"), ("s2.5M", "s600M"), ("s400M", "s600M")):
        sl, sl2 = sums(STEPS[a], STEPS[b])
        out = {}
        for n in C[a]["params"]:
            dw = C[b]["params"][n].numpy() - C[a]["params"][n].numpy()
            sr = seen_rows(n, counts)
            w0 = C[a]["params"][n].numpy()
            if sr is not None:
                dw, w0 = dw[sr], w0[sr]
            out[n] = {"D": rms(dw), "rel": rms(dw) / max(rms(w0), 1e-30), "D_over_rw": rms(dw) / sl2, "D_over_dir": rms(dw) / sl}
        R["drift_long"][f"{a}->{b}"] = {"sum_lr": sl, "sqrt_sum_lr2": sl2, "tensors": out}
        print(f"-- {a}->{b}: sum lr {sl:.3g}, sqrt sum lr^2 {sl2:.3g}")
        for n, r in out.items():
            if any(k in n for k in ("species", "items", "abilities", "moves.", "mon_proj.0", "move_proj.0", "layers.0.linear1",
                                    "layers.1.self_attn.in_proj_weight", "value_net", "action_net.score.0", "mlp_extractor.value_net.0")):
                print(f"   {n:48s} D {r['D']:.3e} rel {r['rel']:.3%}  D/rw {r['D_over_rw']:.2f}  D/sumlr {r['D_over_dir']:.4f}")

    # arms: drift 600M -> 640M of the embeddings, vs predictions with the boosted step
    print("\n== 7c. arms 600M->640M: embedding drift (seen rows) vs random-walk prediction with the boost")
    R["arms"] = {}
    arm_run = {"ctrl640": ("arm-ctrl-20261105-115414-377837", 1), "x30_640": ("arm-x30-20261105-010656-371651", 30),
               "x300_640": ("arm-x300-20261105-005736-371541", 300)}
    for tag, (run, mult) in arm_run.items():
        out = {}
        for t in TABLES:
            n = f"features_extractor.{t}.weight"
            sr = seen_rows(n, counts)
            dw = (C[tag]["params"][n].numpy() - s6["params"][n].numpy())[sr]
            try:
                sl, sl2 = arm_sums(run, STEPS["s600M"], STEPS[tag], mult)
            except Exception:
                sl, sl2 = sums(STEPS["s600M"], STEPS[tag], mult)
            damp = R["adam"][tag][n]["damp"]
            out[t] = {"D": rms(dw), "rel": rms(dw) / rms(s6["params"][n].numpy()[sr]),
                      "D_over_rw": rms(dw) / (sl2 * damp), "D_over_dir": rms(dw) / sl}
        for n in ("features_extractor.mon_proj.0.weight", "features_extractor.encoder.layers.0.linear1.weight"):
            dw = C[tag]["params"][n].numpy() - s6["params"][n].numpy()
            out[n] = {"D": rms(dw), "rel": rms(dw) / rms(s6["params"][n].numpy())}
        R["arms"][tag] = out
        print(tag, json.dumps(out))

    # ---------------- 3. SVD of first-layer column blocks
    print("\n== 3. first-layer column blocks: Frobenius, mean column norm, singular values vs random")
    blocks = {"features_extractor.mon_proj.0.weight": {"species": (0, 64), "item": (64, 96), "ability": (96, 128),
                                                       "move_proj_out": (128, 384), "pool_mean": (384, 448),
                                                       "numerics": (448, 573)},
              "features_extractor.move_proj.0.weight": {"move_emb": (0, 64), "move_numerics": (64, 108)}}
    R["svd"] = {}
    rng = np.random.default_rng(0)
    for n, bl in blocks.items():
        for tag in ("s2.5M", "s500M", "s600M", "x300_640"):
            W = C[tag]["params"][n].numpy()
            fan_in = W.shape[1]
            init_std = 1 / np.sqrt(3 * fan_in)
            for b, (lo, hi) in bl.items():
                Wb = W[:, lo:hi]
                s = np.linalg.svd(Wb, compute_uv=False)
                fro = float(np.linalg.norm(Wb))
                rnd = np.mean([np.linalg.svd(rng.standard_normal(Wb.shape), compute_uv=False) for _ in range(20)], 0)
                rnd *= fro / np.linalg.norm(rnd)
                key = f"{n.split('.')[1]}:{b}"
                R["svd"].setdefault(key, {})[tag] = {
                    "fro": fro, "fro_init_expected": float(init_std * np.sqrt(Wb.size)),
                    "col_norm_mean": float(np.linalg.norm(Wb, axis=0).mean()),
                    "sv_top5": s[:5].tolist(), "sv_rand_top5": rnd[:5].tolist(),
                    "sv_ratio_top1": float(s[0] / rnd[0]), "eff_rank": float(np.exp(-(p := s ** 2 / (s ** 2).sum()) @ np.log(p))),
                    "eff_rank_rand": float(np.exp(-(q := rnd ** 2 / (rnd ** 2).sum()) @ np.log(q))),
                    "top8_energy": float((s[:8] ** 2).sum() / (s ** 2).sum()), "top8_energy_rand": float((rnd[:8] ** 2).sum() / (rnd ** 2).sum())}
            if tag != "s2.5M":
                W0 = C["s2.5M"]["params"][n].numpy()
                for b, (lo, hi) in bl.items():
                    key = f"{n.split('.')[1]}:{b}"
                    R["svd"][key][tag]["rel_change_vs_2.5M"] = float(np.linalg.norm(W[:, lo:hi] - W0[:, lo:hi]) / np.linalg.norm(W0[:, lo:hi]))
    for key, v in R["svd"].items():
        for tag, r in v.items():
            print(f"{key:24s} {tag:9s} fro {r['fro']:.3f} (init {r['fro_init_expected']:.3f}) colnorm {r['col_norm_mean']:.3f} "
                  f"sv1/rand {r['sv_ratio_top1']:.2f} effrank {r['eff_rank']:.1f}/{r['eff_rank_rand']:.1f} top8E {r['top8_energy']:.3f}/{r['top8_energy_rand']:.3f}"
                  f" chg {r.get('rel_change_vs_2.5M', 0):.3f}")

    # does the embedding drift live in what the first layer reads? energy of (e600 - e2.5) in top-k right singular vectors
    print("\n== 3b. embedding drift 2.5M->600M projected on the right singular vectors of its first-layer block (500M)")
    R["drift_alignment"] = {}
    for t, (n, lo, hi) in {"species": ("features_extractor.mon_proj.0.weight", 0, 64),
                           "items": ("features_extractor.mon_proj.0.weight", 64, 96),
                           "abilities": ("features_extractor.mon_proj.0.weight", 96, 128),
                           "moves": ("features_extractor.move_proj.0.weight", 0, 64)}.items():
        W = C["s500M"]["params"][n].numpy()[:, lo:hi]
        _, s, Vt = np.linalg.svd(W, full_matrices=False)
        sr = seen_rows(f"features_extractor.{t}.weight", counts)
        dE = (emb(s6, t) - emb(s0, t))[sr]
        proj = (dE @ Vt.T) ** 2
        frac = proj.sum(0) / proj.sum()
        D = hi - lo
        k = max(1, D // 8)
        # gradient direction prediction: dL/de = W^T delta, so energy should be ~ s_i^2 weighted
        R["drift_alignment"][t] = {"top_k": k, "energy_top_k": float(frac[:k].sum()), "uniform": k / D,
                                   "sv2_weighted_top_k": float((s[:k] ** 2).sum() / (s ** 2).sum()),
                                   "corr_energy_vs_sv2": float(np.corrcoef(frac, s ** 2)[0, 1])}
        print(t, R["drift_alignment"][t])
    json.dump(R, open(OUT / "weights.json", "w"), indent=1)


if __name__ == "__main__":
    main()
