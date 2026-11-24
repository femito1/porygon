"""Council test P2 (docs/netsearch/P2-EXIT-SOFT-PREREG.md = $C/council/p2-exit-soft/PREREG.txt): ExIt with a soft
completed-Q target, offline on ExIt-LA2's labelled positions (ns-27, $C/runs/exit-la2/labels). The one change from
ns-27's arm E is the policy target:

    t(a) = pi_imp(a) ∝ p(a)·exp(beta·(Q̂(a) − V))   over our legal actions

p = the parent's T-1 policy at the position (label part `probs`), Q̂ = lookahead2's Q(a) for actions with a row and V
for a legal action without one (completed Q), V = x·Q (lookahead2's root value, meta `value`). Subcommands, in the
order scripts/fastenv/p2_exit_soft.sbatch runs them:

  calibrate  the data check (fields and row alignment of every label part) and beta by bisection: mean KL(pi_imp || p)
             = 0.05 nats on the calibration split (training rows whose sha1("p2-exit-soft-cal:" + record id) is 0 mod
             10; no outcomes read) -> JSON. --expect B fails unless the result reproduces the pre-registered B.
  shards     ns-27's E shards with `target` replaced by pi_imp at --beta (obs, z, game and order unchanged; each shard
             is checked against its label part).
  check      information only, on the held-out games (pretrain.is_val(game, 0.05)): KL(pi_imp || p_X) at T 1, r' = 1 −
             KL_X / KL_S, top-1 agreement with argmax pi_imp where it differs from S's argmax, value log-loss / AUC.
  verdict    the PREREG rule on the two eval JSONs (exit_la2.py eval) -> VERDICT.txt text.

Pure numpy except `check` (torch, the frozen netv2 tree via netsearch_study.setup, as exit_la2.py check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exit_la2 as X  # noqa: E402  (ns-27's helpers: label_parts, sqrt_target, kl_rows, is_val, auc)

KL_TARGET = 0.05  # nats: mean KL(pi_imp || p) on the calibration split
CAL_SALT = "p2-exit-soft-cal"
CAL_MOD = 10  # one training row in ten
VAL_FRAC = 0.05  # pretrain's held-out split, as ns-27 (5,776 of 119,967 rows)
GO_VS_C, GO_VS_S = 50.5, 50.0  # lower 95% bounds (percent) the rule needs


# ---------------------------------------------------------------- pure functions (tests/test_p2_exit_soft.py)

def completed_adv(n: int, legal, rows, q, value: float) -> np.ndarray:
    """A(a) = Q(a) − V for a legal action with a lookahead row; 0 for a legal action without one (its Q completed with
    V, as Gumbel MuZero completes the Q of unvisited actions) and off the legal set."""
    legal = {int(a) for a in legal}
    adv = np.zeros(n)
    for r, v in zip(rows, q):
        if int(r) not in legal:
            raise ValueError(f"row {r} is not a legal action")
        adv[int(r)] = float(v) - float(value)
    return adv


def _scores(p, adv, beta: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    s = np.where(p > 0, beta * np.asarray(adv, dtype=np.float64), -np.inf)
    return p, s


def pi_imp(p, adv, beta: float) -> np.ndarray:
    """pi_imp(a) ∝ p(a)·exp(beta·A(a)) over p > 0 (one vector or the rows of a 2-D array); supp(pi_imp) = supp(p)."""
    p, s = _scores(p, adv, beta)
    w = np.where(p > 0, p * np.exp(s - s.max(-1, keepdims=True)), 0.0)
    return w / w.sum(-1, keepdims=True)


def kl_imp(p, adv, beta: float) -> np.ndarray:
    """KL(pi_imp || p) per row, exactly: beta·E_pi_imp[A] − log sum_a p(a)·exp(beta·A(a)) (no floor on p)."""
    p, s = _scores(p, adv, beta)
    m = s.max(-1, keepdims=True)
    w = np.where(p > 0, p * np.exp(s - m), 0.0)
    z = w.sum(-1, keepdims=True)
    e = ((w / z) * np.where(p > 0, s, 0.0)).sum(-1)
    return np.maximum(e - (m[..., 0] + np.log(z[..., 0])), 0.0)


def bisect_beta(p, adv, target: float = KL_TARGET, iters: int = 100) -> float:
    """The beta >= 0 with mean KL(pi_imp || p) = target. The mean is increasing in beta (d/dbeta KL = beta·Var_pi(A))."""
    def f(b):
        return float(kl_imp(p, adv, b).mean())
    lo, hi = 0.0, 1.0
    while f(hi) < target:
        lo, hi = hi, 2.0 * hi
        if hi > 1e9:
            raise ValueError(f"mean KL {f(hi)} < {target} at beta {hi}: target unreachable")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def in_cal(record_id: str) -> bool:
    """The fixed calibration split: one record in CAL_MOD by sha1 of the record id (applied to training rows)."""
    return int(hashlib.sha1(f"{CAL_SALT}:{record_id}".encode()).hexdigest()[:12], 16) % CAL_MOD == 0


def lower(r: dict) -> float:
    """Lower 95% bound in percent of an exit_la2.py eval JSON (pair-mean interval)."""
    return 100.0 * (r["win_rate"] - r["ci95"])


def verdict_p2(t1: dict, t2: dict) -> str:
    """PREREG rule: GO iff E' vs C (t1) lower 95% > 50.5 AND E' vs S (t2) lower 95% > 50.0; else NO-GO."""
    return "GO" if lower(t1) > GO_VS_C and lower(t2) > GO_VS_S else "NO-GO"


# ---------------------------------------------------------------- label parts

def read_part(part: Path) -> tuple[dict, list[dict]]:
    """(arrays probs, mask, game, pi; the meta rows) of one label part; row j of the npz is line j of the meta."""
    z = np.load(part)
    arr = {"probs": z["probs"].astype(np.float64), "mask": z["obs/action_mask"] > 0, "game": z["game"],
           "pi": z["pi"].astype(np.float64)}
    meta = [json.loads(s) for s in part.with_name(part.name[:-4] + ".meta.jsonl").read_text().splitlines()
            if s.strip()]
    return arr, meta


def part_adv(arr: dict, meta: list[dict]) -> tuple[np.ndarray, dict]:
    """Completed advantages [n, A] of one part, and its data check (raises on a missing field or a misaligned row)."""
    probs, mask = arr["probs"], arr["mask"]
    n, na = probs.shape
    if len(meta) != n:
        raise ValueError(f"{len(meta)} meta rows vs {n} npz rows")
    p05 = probs ** 2 / (probs ** 2).sum(-1, keepdims=True)  # the T 0.5 draw of a T-1 softmax
    moved = np.abs(arr["pi"] - p05).sum(-1) / 2
    adv = np.zeros((n, na))
    st = {"rows": n, "legal": 0, "with_q": 0, "rows_missing_q": 0, "max_v_err": 0.0, "max_moved_err": 0.0,
          "max_p_off_mask": 0.0, "max_p_sum_err": 0.0}
    for j, m in enumerate(meta):
        legal = np.flatnonzero(mask[j])
        if len(legal) != m["n_legal"] or len(legal) < 2:
            raise ValueError(f"row {j} ({m['id']}): {len(legal)} legal in the mask vs n_legal {m['n_legal']}")
        rows, q, x = m["rows"], np.asarray(m["q"], dtype=np.float64), np.asarray(m["x"], dtype=np.float64)
        if not (len(rows) == len(q) == len(x) >= 1) or not np.isfinite(q).all() or int(m["a_l"]) not in rows:
            raise ValueError(f"row {j} ({m['id']}): bad rows / q / x / a_l")
        adv[j] = completed_adv(na, legal, rows, q, m["value"])
        st["legal"] += len(legal)
        st["with_q"] += len(rows)
        st["rows_missing_q"] += int(len(rows) < len(legal))
        st["max_v_err"] = max(st["max_v_err"], abs(float(x @ q) - float(m["value"])))
        st["max_moved_err"] = max(st["max_moved_err"], abs(float(moved[j]) - float(m["moved"])))
    st["max_p_off_mask"] = float(np.abs(np.where(mask, 0.0, probs)).max())
    st["max_p_sum_err"] = float(np.abs(probs.sum(-1) - 1).max())
    return adv, st


def merge_stats(acc: dict, st: dict) -> None:
    for k, v in st.items():
        acc[k] = max(acc.get(k, 0.0), v) if k.startswith("max_") else acc.get(k, 0) + v


def data_ok(st: dict) -> list[str]:
    """Tolerances of the data check: V = x·Q up to the meta's 6-decimal rounding, the stored pi_LA and p reproduce the
    meta's moved mass (row alignment), p is a distribution on the legal mask."""
    bad = []
    if st["max_v_err"] > 2e-5:
        bad.append(f"|x·Q − V| up to {st['max_v_err']:.2e}")
    if st["max_moved_err"] > 1e-4:
        bad.append(f"moved mass mismatch up to {st['max_moved_err']:.2e} (npz / meta rows misaligned?)")
    if st["max_p_off_mask"] > 0 or st["max_p_sum_err"] > 1e-4:
        bad.append(f"p off the mask {st['max_p_off_mask']:.2e}, |sum p − 1| {st['max_p_sum_err']:.2e}")
    return bad


# ---------------------------------------------------------------- calibrate

def cmd_calibrate(a) -> int:
    parts = X.label_parts(Path(a.labels))
    cal_p, cal_adv, cal_pi = [], [], []
    acc: dict = {}
    n_train = n_val = 0
    for part in parts:
        arr, meta = read_part(part)
        adv, st = part_adv(arr, meta)
        merge_stats(acc, st)
        val = X.is_val(arr["game"], VAL_FRAC)
        n_val += int(val.sum())
        n_train += int((~val).sum())
        sel = ~val & np.array([in_cal(m["id"]) for m in meta])
        cal_p.append(arr["probs"][sel])
        cal_adv.append(adv[sel])
        cal_pi.append(arr["pi"][sel])
    p, adv, pi = np.concatenate(cal_p), np.concatenate(cal_adv), np.concatenate(cal_pi)
    bad = data_ok(acc)
    beta = bisect_beta(p, adv, a.kl)
    beta_rec = float(f"{beta:.6g}")  # the value written in the PREREG
    imp = pi_imp(p, adv, beta_rec)
    qhat_gain = ((imp - p) * adv).sum(-1)  # E_pi_imp[Q̂] − E_p[Q̂] (V cancels)
    hard = X.sqrt_target(pi).astype(np.float64)  # ns-27's T-1 target t_E on the same rows (information)
    out = {"parts": len(parts), "rows": acc.get("rows", 0), "train_rows": n_train, "held_out": n_val,
           "cal_rows": int(len(p)), "cal_rule": f"training rows with int(sha1('{CAL_SALT}:' + id)[:12], 16) % "
           f"{CAL_MOD} == 0", "kl_target": a.kl, "beta": beta, "beta_recorded": beta_rec,
           "mean_kl_cal_at_beta_recorded": float(kl_imp(p, adv, beta_rec).mean()),
           "mean_qhat_gain_cal": float(qhat_gain.mean()), "min_qhat_gain_cal": float(qhat_gain.min()),
           "mean_kl_hard_target_tE_vs_p_cal": float(X.kl_rows(hard, p).mean()),
           "legal_actions": acc.get("legal", 0), "legal_with_q": acc.get("with_q", 0),
           "rows_missing_q": acc.get("rows_missing_q", 0),
           **{k: acc[k] for k in acc if k.startswith("max_")}, "data_check": "OK" if not bad else "; ".join(bad)}
    if a.expect is not None:
        out["expect"] = a.expect
        if not abs(beta_rec - a.expect) <= 1e-4 * a.expect:
            bad.append(f"beta {beta_rec} does not reproduce the pre-registered {a.expect}")
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    if bad:
        print("FAIL: " + "; ".join(bad))
        return 1
    print(f"beta {beta_rec} (mean KL {out['mean_kl_cal_at_beta_recorded']:.6f} on {out['cal_rows']} calibration rows)")
    return 0


# ---------------------------------------------------------------- shards

def cmd_shards(a) -> int:
    parts = X.label_parts(Path(a.labels))[:a.parts] if a.parts else X.label_parts(Path(a.labels))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n, acc = 0, {}
    for i, part in enumerate(parts):
        arr, meta = read_part(part)
        adv, st = part_adv(arr, meta)
        merge_stats(acc, st)
        e = np.load(Path(a.e_shards) / f"shard-{i:05d}.npz")  # ns-27's arm E, built from this part
        if not (np.array_equal(e["game"], arr["game"]) and np.array_equal(e["target"], X.sqrt_target(arr["pi"]))
                and np.array_equal(e["obs/action_mask"] > 0, arr["mask"]) and len(e["z"]) == len(meta)):
            raise SystemExit(f"E shard {i} does not match label part {part.name}")
        t = pi_imp(arr["probs"], adv, a.beta).astype(np.float32)
        np.savez(out / f"shard-{i:05d}.npz", **{k: e[k] for k in e.files if k != "target"}, target=t)
        n += len(t)
    bad = data_ok(acc)
    if bad:
        raise SystemExit("data check: " + "; ".join(bad))
    (out / "SHARDS.json").write_text(json.dumps({"rows": n, "parts": len(parts), "beta": a.beta}) + "\n")
    print(f"shards: {n} rows, {len(parts)} parts, beta {a.beta}")
    return 0 if n else 1


# ---------------------------------------------------------------- check (information)

def cmd_check(a) -> int:
    import netsearch_study as ST
    ST.setup(a.format)
    import torch
    from porygon.v2.model import batch_to_tensors, load, model_keys
    data = defaultdict(list)
    for p in sorted(Path(a.shards).glob("shard-*.npz")):
        z = np.load(p)
        v = X.is_val(z["game"], VAL_FRAC)
        for k in z.files:
            data[k].append(z[k][v])
    data = {k: np.concatenate(v) for k, v in data.items()}
    n = len(data["z"])
    tgt = data["target"].astype(np.float64)
    nets = {"S": a.start, **dict(x.split("=", 1) for x in a.net)}
    probs, win = {}, {}
    for name, path in nets.items():
        net = load(path, a.device).eval()
        keys = model_keys(net.gen)
        ps, ws = [], []
        for s in range(0, n, 1024):
            obs = {k[4:]: v[s:s + 1024] for k, v in data.items() if k.startswith("obs/") and k[4:] in keys}
            with torch.no_grad():
                o = net(batch_to_tensors(obs, a.device), aux=False)
            m = torch.as_tensor(data["obs/action_mask"][s:s + 1024] > 0, device=a.device)
            ps.append(torch.softmax(torch.where(m, o["logits"].double(), -math.inf), -1).cpu().numpy())
            ws.append(torch.sigmoid(o["win_logit"].double()).cpu().numpy())
        probs[name], win[name] = np.concatenate(ps), np.concatenate(ws)
    kl_s = float(X.kl_rows(tgt, probs["S"]).mean())
    moved = tgt.argmax(-1) != probs["S"].argmax(-1)
    y = (data["z"] > 0).astype(np.float64)
    out = {"held_out": n, "nets": nets, "argmax_changed_rows": int(moved.sum())}
    for name in nets:
        kl = float(X.kl_rows(tgt, probs[name]).mean())
        w = np.clip(win[name], 1e-7, 1 - 1e-7)
        out[name] = {"kl_to_pi_imp": kl, "r": 1 - kl / kl_s if kl_s > 0 else math.nan,
                     "top1_argmax_changed": float((probs[name][moved].argmax(-1) == tgt[moved].argmax(-1)).mean())
                     if moved.any() else math.nan,
                     "value_logloss": float(-(y * np.log(w) + (1 - y) * np.log(1 - w)).mean()),
                     "value_auc": X.auc(win[name], y)}
    Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0


# ---------------------------------------------------------------- verdict

def cmd_verdict(a) -> int:
    t1, t2 = json.loads(Path(a.t1).read_text()), json.loads(Path(a.t2).read_text())
    cal = json.loads(Path(a.calibration).read_text())
    line = verdict_p2(t1, t2)

    def f(r, need):
        return (f"{100 * r['win_rate']:.2f}% ± {100 * r['ci95']:.2f} [{lower(r):.2f}, "
                f"{100 * (r['win_rate'] + r['ci95']):.2f}] ({r['pairs']} pairs, seed {r['seed']}; GO needs lower 95% "
                f"> {need})")
    txt = [line,
           f"E′ vs C (ns-29's control): {f(t1, GO_VS_C)}",
           f"E′ vs S (G9q-cd@805.03M): {f(t2, GO_VS_S)}",
           f"  target pi_imp ∝ p·exp(β(Q̂ − V)), β {cal['beta_recorded']} (mean KL(pi_imp ‖ p) "
           f"{cal['mean_kl_cal_at_beta_recorded']:.4f} on {cal['cal_rows']} calibration rows; ns-27's hard t_E "
           f"{cal['mean_kl_hard_target_tE_vs_p_cal']:.4f}); {cal['rows']} positions, {cal['rows_missing_q']} with a "
           f"legal action completed with V"]
    ck = Path(a.check)
    if ck.exists():
        c = json.loads(ck.read_text())
        names = [k for k in ("S", "Ep", "E", "C") if k in c]
        txt.append(f"  information (held out {c['held_out']}): KL(pi_imp ‖ p_X) "
                   + ", ".join(f"{k} {c[k]['kl_to_pi_imp']:.4f}" for k in names) + "; r′ "
                   + ", ".join(f"{k} {c[k]['r']:.3f}" for k in names if k != "S") + "; value log-loss / AUC "
                   + ", ".join(f"{k} {c[k]['value_logloss']:.4f} / {c[k]['value_auc']:.4f}" for k in names))
    else:
        txt.append("  information check: not available")
    if line == "NO-GO":
        txt.append("  NO-GO closes the ExIt-LA2 line in both forms (hard and soft target).")
    Path(a.out).write_text("\n".join(txt) + "\n")
    print("\n".join(txt))
    return 0


# ---------------------------------------------------------------- main

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    cb = sub.add_parser("calibrate")
    cb.add_argument("--labels", required=True)
    cb.add_argument("--json", required=True)
    cb.add_argument("--kl", type=float, default=KL_TARGET)
    cb.add_argument("--expect", type=float, help="the pre-registered beta; fail unless reproduced (1e-4 relative)")
    sh = sub.add_parser("shards")
    sh.add_argument("--labels", required=True)
    sh.add_argument("--e-shards", required=True, help="ns-27's arm-E shard dir")
    sh.add_argument("--beta", type=float, required=True)
    sh.add_argument("--out", required=True)
    sh.add_argument("--parts", type=int, default=0, help="first N parts only (dry runs)")
    ck = sub.add_parser("check")
    ck.add_argument("--shards", required=True)
    ck.add_argument("--start", required=True)
    ck.add_argument("--net", action="append", default=[], help="NAME=PATH")
    ck.add_argument("--json", required=True)
    ck.add_argument("--device", default="cuda")
    ck.add_argument("--format", default="gen9randombattle")
    vd = sub.add_parser("verdict")
    vd.add_argument("--t1", required=True, help="E' vs C eval JSON")
    vd.add_argument("--t2", required=True, help="E' vs S eval JSON")
    vd.add_argument("--calibration", required=True)
    vd.add_argument("--check", required=True)
    vd.add_argument("--out", required=True)
    a = p.parse_args()
    return {"calibrate": cmd_calibrate, "shards": cmd_shards, "check": cmd_check, "verdict": cmd_verdict}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
