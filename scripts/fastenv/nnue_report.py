"""NNUE gate 1 report and verdict (docs/netsearch/NNUE-GATE1-PREREG.md, "Metrics" / "Decision rule").

  report  set S (self-play held-out, split 2 of nnue_gen.py's shards, a fixed seeded subsample of <= 200k) and set H
          (lookahead2_study.py outcome --student on Step 2's 3,000 human positions): AUC and calibrated log-loss
          (2-fold, game-cross-fitted affine logit recalibration, as lookahead2_study.analyze_outcome; evaluate() on
          eval / 100) of the student, the teacher and evaluate(); criteria 1 / 2 per set -> PASS / PARTIAL / FAIL;
          game-clustered bootstrap intervals of the differences, by-turn tables, decile calibration, timing,
          parameters, throughput, parity rates. Writes REPORT.txt / REPORT.json; line 1 = the verdict line.
  timing  the per-leaf CPU cost (run it pinned to one core): numpy forward of one State, median over 10,000
          held-out States (a) from precomputed features, (b) features from the State object + forward.

    $PY scripts/fastenv/nnue_report.py timing --data D --student S.npz --out timing.json
    $PY scripts/fastenv/nnue_report.py report --data D --student S.npz --seth H.jsonl --timing timing.json --out REPORT.txt
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lookahead2_study import _logit_fit, _logloss, _predict  # noqa: E402

AUC_TEACHER, LL_TEACHER, AUC_EVAL, LL_EVAL = 0.01, 0.01, 0.015, 0.017
TURN_BINS = ((0, 5), (6, 15), (16, 30), (31, 100000))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC, ties 1/2 (= lookahead2_study._auc, vectorised)."""
    s = np.asarray(score, float)
    o = np.argsort(s, kind="mergesort")
    ss = s[o]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    ends = np.r_[starts[1:], len(ss)]
    r = np.empty(len(s))
    r[o] = np.repeat((starts + ends - 1) / 2 + 1, ends - starts)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrated(x: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """2-fold cross-fitted affine logit recalibration of the score x (a logit or eval / 100)."""
    out = np.zeros(len(x))
    for f in (0, 1):
        b = _logit_fit(x[fold != f, None], y[fold != f])
        out[fold == f] = _predict(b, x[fold == f, None])
    return out


class Scored:
    """One set: predictors {name: raw probability or eval}, outcome y, game ids, turns."""

    def __init__(self, preds: dict, y: np.ndarray, games: np.ndarray, turn: np.ndarray):
        self.y, self.games, self.turn = y.astype(float), games, turn
        fold = (np.abs(games) % 2).astype(int)
        self.raw, self.score, self.cal = {}, {}, {}
        for k, v in preds.items():
            v = np.asarray(v, float)
            x = v / 100 if k == "eval" else logit(v)
            self.raw[k] = 1 / (1 + np.exp(-0.0125 * v)) if k == "eval" else v  # eval at the search's scale
            self.score[k] = v
            self.cal[k] = calibrated(x, self.y, fold)

    def metrics(self, idx=None) -> dict:
        idx = np.arange(len(self.y)) if idx is None else idx
        y = self.y[idx]
        out = {"n": int(len(idx)), "result_mean": float(y.mean()),
               "ll_const": _logloss(np.full(len(idx), y.mean()), y)}
        for k in self.score:
            out[k] = {"auc": auc(self.score[k][idx], y), "ll_cal": _logloss(self.cal[k][idx], y),
                      "ll_raw": _logloss(self.raw[k][idx], y), "brier_cal": float(((self.cal[k][idx] - y) ** 2).mean())}
        return out

    def boot(self, n: int, seed: int) -> dict:
        """Game-clustered bootstrap of the rule's differences."""
        rng = np.random.default_rng(seed)
        o = np.argsort(self.games, kind="mergesort")
        g = self.games[o]
        starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
        ends = np.r_[starts[1:], len(g)]
        diffs = {"auc_student_minus_teacher": [], "ll_student_minus_teacher": [], "auc_student_minus_eval": [],
                 "ll_eval_minus_student": []}
        for _ in range(n):
            pick = rng.integers(0, len(starts), len(starts))
            size = (ends - starts)[pick]
            off = np.repeat(np.cumsum(size) - size, size)
            idx = o[np.repeat(starts[pick], size) + np.arange(size.sum()) - off]
            m = self.metrics(idx)
            diffs["auc_student_minus_teacher"].append(m["student"]["auc"] - m["teacher"]["auc"])
            diffs["ll_student_minus_teacher"].append(m["student"]["ll_cal"] - m["teacher"]["ll_cal"])
            diffs["auc_student_minus_eval"].append(m["student"]["auc"] - m["eval"]["auc"])
            diffs["ll_eval_minus_student"].append(m["eval"]["ll_cal"] - m["student"]["ll_cal"])
        return {k: [float(np.quantile(v, q)) for q in (0.025, 0.5, 0.975)] for k, v in diffs.items()} | {"n": n}

    def by_turn(self) -> dict:
        out = {}
        for lo, hi in TURN_BINS:
            idx = np.flatnonzero((self.turn >= lo) & (self.turn <= hi))
            if len(idx) >= 50 and 0 < self.y[idx].mean() < 1:
                out[f"{lo}-{hi if hi < 100000 else 'end'}"] = self.metrics(idx)
        return out

    def calibration(self) -> dict:
        out = {}
        for k, p in self.raw.items():
            qs = np.quantile(p, np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, 9)
            out[k] = [{"mean_pred": float(p[b == j].mean()), "freq": float(self.y[b == j].mean()), "n": int((b == j).sum())}
                      for j in range(10) if (b == j).any()]
        return out


def criteria(m: dict) -> dict:
    s, t, e = m["student"], m["teacher"], m["eval"]
    c1 = s["auc"] >= t["auc"] - AUC_TEACHER and s["ll_cal"] <= t["ll_cal"] + LL_TEACHER
    c2 = s["auc"] - e["auc"] >= AUC_EVAL and e["ll_cal"] - s["ll_cal"] >= LL_EVAL
    return {"c1": bool(c1), "c2": bool(c2), "auc_s_minus_t": s["auc"] - t["auc"], "ll_s_minus_t": s["ll_cal"] - t["ll_cal"],
            "auc_s_minus_e": s["auc"] - e["auc"], "ll_e_minus_s": e["ll_cal"] - s["ll_cal"]}


def level(cs: dict, ch: dict) -> str:
    if cs["c1"] and cs["c2"] and ch["c1"] and ch["c2"]:
        return "PASS"
    if cs["c2"] and ch["c2"]:
        return "PARTIAL"
    return "FAIL"


def shards(data: Path) -> list[Path]:
    return sorted(p.with_suffix(".npz") for p in data.glob("shard-*.json") if p.with_suffix(".npz").exists())


def load_set_s(data: Path, cap: int, seed: int) -> dict:
    keep = {k: [] for k in ("ids", "mon", "side", "field", "teacher", "outcome", "eval", "turn", "game")}
    for s in shards(data):
        with np.load(s) as z:
            sel = z["split"] == 2
            for k in keep:
                keep[k].append(z[k][sel])
    d = {k: np.concatenate(v) for k, v in keep.items()}
    n = len(d["teacher"])
    if n > cap:
        idx = np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))
        d = {k: v[idx] for k, v in d.items()}
    d["n_heldout"] = n
    return d


def student_probs(ns, d: dict, bs: int = 8192) -> np.ndarray:
    out = []
    for i in range(0, len(d["ids"]), bs):
        lg = ns.logits(d["ids"][i:i + bs].astype(np.int64), d["mon"][i:i + bs], d["side"][i:i + bs],
                       d["field"][i:i + bs])
        out.append(1 / (1 + np.exp(-lg.astype(np.float64))))
    return np.concatenate(out)


def gen_stats(data: Path) -> dict:
    last = {}
    for p in data.glob("shard-*.json"):
        m = json.loads(p.read_text())
        key = (m["attempt"], m["worker"])
        if key not in last or m["rows"] > last[key]["rows"]:
            last[key] = m
    rows = sum(m["rows"] for m in last.values())
    games = sum(m["games"] for m in last.values())
    mans = [json.loads(p.read_text()) for p in sorted(data.glob("manifest-a*.json"))]
    secs = sum(m.get("seconds", 0) for m in mans)
    decs = sum(m.get("decisions", 0) for m in mans)
    return {"positions": rows, "games": games, "positions_per_game": rows / max(games, 1),
            "dropped_games": sum(m["dropped_games"] for m in last.values()), "attempts": len(mans),
            "hours": secs / 3600, "decisions_per_s": decs / max(secs, 1), "kept_positions_per_s": rows / max(secs, 1)}


def cmd_timing(a) -> int:
    import poke_engine as pe
    from porygon.nnue.features import features
    from porygon.nnue.student import NumpyStudent
    ns = NumpyStudent.load(a.student)
    strs = []
    for s in shards(Path(a.data)):
        sp = s.with_name(s.stem + ".states.jsonl.gz")
        if not sp.exists():
            continue
        with gzip.open(sp, "rt") as fh:
            for line in fh:
                x = json.loads(line)
                if x["split"] == 2:
                    strs.append(x["state"])
        if len(strs) >= a.n:
            break
    strs = strs[:a.n]
    states = [pe.State.from_string(s) for s in strs]
    feats = [features(st) for st in states]
    for f in feats[:200]:  # warm-up
        ns.p_win(f)
    ta, tb = [], []
    for f in feats:
        t0 = time.perf_counter_ns()
        ns.p_win(f)
        ta.append(time.perf_counter_ns() - t0)
    for st in states:
        t0 = time.perf_counter_ns()
        ns.p_win(features(st))
        tb.append(time.perf_counter_ns() - t0)
    q = lambda v, p: float(np.quantile(np.array(v) / 1000, p))  # noqa: E731  (microseconds)
    out = {"n": len(states), "forward_us_median": q(ta, 0.5), "forward_us_p90": q(ta, 0.9),
           "features_forward_us_median": q(tb, 0.5), "features_forward_us_p90": q(tb, 0.9),
           "host": __import__("os").uname().nodename, "affinity": sorted(__import__("os").sched_getaffinity(0))}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
    return 0 if len(states) >= min(a.n, 1000) else 1


def fmt_metrics(name: str, m: dict) -> list[str]:
    return [f"  {name} (n {m['n']}, result mean {m['result_mean']:.3f}, const log-loss {m['ll_const']:.4f}):"] + [
        f"    {k:8s} AUC {m[k]['auc']:.4f}  LL_cal {m[k]['ll_cal']:.4f}  LL_raw {m[k]['ll_raw']:.4f}  Brier_cal "
        f"{m[k]['brier_cal']:.4f}" for k in ("student", "teacher", "eval")]


def cmd_report(a) -> int:
    from porygon.nnue.student import NumpyStudent, n_params
    ns = NumpyStudent.load(a.student)
    # set S
    d = load_set_s(Path(a.data), a.cap, a.seed)
    S = Scored({"student": student_probs(ns, d), "teacher": d["teacher"].astype(float), "eval": d["eval"].astype(float)},
               d["outcome"], d["game"], d["turn"].astype(int))
    # set H
    rows = [json.loads(x) for x in open(a.seth) if x.strip()]
    rows = [r for r in rows if r.get("result") in (1, -1)]
    g = np.array([int(hashlib.sha1(str(r["rec"]).encode()).hexdigest()[:8], 16) for r in rows])
    yh = np.array([(r["result"] + 1) / 2 for r in rows])
    col = lambda k: np.concatenate([[r["a"][k] for r in rows], [r["b"][k] for r in rows]])  # noqa: E731
    H = Scored({"student": col("student"), "teacher": col("win"), "eval": col("eval")}, np.concatenate([yh, yh]),
               np.concatenate([g, g]), col("turn").astype(int))
    overlap = None
    if a.step2 and Path(a.step2).exists():
        ref = {(r["rec"], r["t"], r["t1"]) for r in map(json.loads, open(a.step2)) if r.get("result") in (1, -1)}
        mine = {(r["rec"], r["t"], r["t1"]) for r in rows}
        overlap = {"step2_pairs": len(ref), "pairs": len(mine), "common": len(ref & mine)}
    ms, mh = S.metrics(), H.metrics()
    cs, ch = criteria(ms), criteria(mh)
    lev = level(cs, ch)
    timing = json.loads(Path(a.timing).read_text()) if a.timing and Path(a.timing).exists() else {}
    fit = json.loads(Path(a.fit).read_text()) if a.fit and Path(a.fit).exists() else {}
    gs = gen_stats(Path(a.data))
    par = {}
    for name, p in (("state", a.parity), ("feature", a.feature_parity)):
        if p and Path(p).exists():
            par[name] = json.loads(Path(p).read_text())
    yes = lambda b: "met" if b else "NOT met"  # noqa: E731
    head = (f"{lev}: set S c1 {yes(cs['c1'])} / c2 {yes(cs['c2'])}; set H c1 {yes(ch['c1'])} / c2 {yes(ch['c2'])} "
            f"(S: AUC s-t {cs['auc_s_minus_t']:+.4f}, LL s-t {cs['ll_s_minus_t']:+.4f}, AUC s-e {cs['auc_s_minus_e']:+.4f}, "
            f"LL e-s {cs['ll_e_minus_s']:+.4f}; H: {ch['auc_s_minus_t']:+.4f}, {ch['ll_s_minus_t']:+.4f}, "
            f"{ch['auc_s_minus_e']:+.4f}, {ch['ll_e_minus_s']:+.4f})")
    bs, bh = S.boot(a.boot, 5), H.boot(a.boot, 5)
    lines = [head, "",
             "NNUE gate 1 (docs/netsearch/NNUE-GATE1-PREREG.md v2, md5 43f045da): student on poke-engine State features vs "
             "the teacher G9q-cd@805.03M's value and stock evaluate()",
             f"criterion 1: AUC(student) >= AUC(teacher) - {AUC_TEACHER} and LL_cal(student) <= LL_cal(teacher) + {LL_TEACHER}",
             f"criterion 2: AUC(student) - AUC(evaluate) >= {AUC_EVAL} and LL_cal(evaluate) - LL_cal(student) >= {LL_EVAL}",
             "PASS = 1 and 2 on both sets; PARTIAL = 2 on both, 1 not on at least one; FAIL = otherwise", "",
             f"Set S (self-play held-out, true-position State; {len(S.y)} of {d['n_heldout']} held-out positions, seed {a.seed}):"]
    lines += fmt_metrics("all", ms)
    lines += [f"  bootstrap 95% (games, {a.boot}): " + ", ".join(f"{k} {v}" for k, v in bs.items() if k != "n")]
    lines += [f"Set H (human, Step 2's positions, K 4 determinizations; {mh['n']} positions" +
              (f"; pairs in common with Step 2: {overlap['common']} of {overlap['step2_pairs']}" if overlap else "") + "):"]
    lines += fmt_metrics("all", mh)
    lines += [f"  bootstrap 95% (games, {a.boot}): " + ", ".join(f"{k} {v}" for k, v in bh.items() if k != "n")]
    for name, sc in (("S", S), ("H", H)):
        lines += ["", f"By turn, set {name} (AUC / LL_cal: student, teacher, eval):"]
        for k, m in sc.by_turn().items():
            lines.append(f"  turns {k:7s} n {m['n']:7d}: AUC {m['student']['auc']:.4f} {m['teacher']['auc']:.4f} "
                         f"{m['eval']['auc']:.4f} | LL {m['student']['ll_cal']:.4f} {m['teacher']['ll_cal']:.4f} "
                         f"{m['eval']['ll_cal']:.4f} (const {m['ll_const']:.4f})")
        lines.append(f"Decile calibration, set {name} (raw prediction -> observed; eval at sigmoid(0.0125 eval)):")
        for k, c in sc.calibration().items():
            lines.append(f"  {k:8s} " + ", ".join(f"{x['mean_pred']:.2f}->{x['freq']:.2f}" for x in c))
    spec = ns.spec
    lines += ["", "Also reported (information):",
              f"  parameters {n_params(spec)} (spec {spec['hash']}); fit: best epoch {fit.get('best_epoch')}, val BCE "
              f"{fit.get('best_val_bce')}, train rows {fit.get('rows_train')}, val rows {fit.get('rows_val')}",
              f"  per-leaf CPU (numpy, one core {timing.get('affinity')} on {timing.get('host')}, median of {timing.get('n')} "
              f"held-out States): forward {timing.get('forward_us_median', float('nan')):.1f} us (p90 "
              f"{timing.get('forward_us_p90', float('nan')):.1f}); features from the State + forward "
              f"{timing.get('features_forward_us_median', float('nan')):.1f} us (p90 {timing.get('features_forward_us_p90', float('nan')):.1f})",
              f"  generation: {gs['positions']} positions, {gs['games']} games ({gs['positions_per_game']:.1f} positions "
              f"per game), {gs['dropped_games']} games dropped (ties / turn cap / crashes), {gs['attempts']} attempt(s), "
              f"{gs['hours']:.2f} h, {gs['decisions_per_s']:.0f} decisions/s"]
    if "state" in par:
        lines.append(f"  State parity (step 3): {par['state']['verdict']}, rates > 0.5%: " + ", ".join(
            f"{k} {v:.2%}" for k, v in par["state"]["rates"].items() if v > 0.005))
    if "feature" in par:
        f = par["feature"]
        worst = max(f["column_mismatch"].values(), default=0.0)
        lines.append(f"  FEATURE parity: {f['verdict']}, whole vector equal {f['vector_equal']:.2%}, worst column "
                     f"mismatch {worst:.2%} (spec {f['spec']})")
    text = "\n".join(lines) + "\n"
    Path(a.out).write_text(text)
    Path(a.out).with_suffix(".json").write_text(json.dumps(
        {"level": lev, "set_S": {"metrics": ms, "criteria": cs, "boot": bs, "by_turn": S.by_turn()},
         "set_H": {"metrics": mh, "criteria": ch, "boot": bh, "by_turn": H.by_turn(), "overlap_step2": overlap},
         "timing": timing, "fit": {k: v for k, v in fit.items() if k != "epochs"}, "generation": gs, "parity": par},
        indent=1, default=float))
    print(text)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("timing")
    t.add_argument("--data", required=True)
    t.add_argument("--student", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--n", type=int, default=10000)
    r = sub.add_parser("report")
    r.add_argument("--data", required=True)
    r.add_argument("--student", required=True)
    r.add_argument("--seth", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--timing")
    r.add_argument("--fit")
    r.add_argument("--parity")
    r.add_argument("--feature-parity")
    r.add_argument("--step2", help="Step 2's outcome-human.jsonl (the same positions check)")
    r.add_argument("--cap", type=int, default=200000)
    r.add_argument("--seed", type=int, default=5)
    r.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    return cmd_timing(a) if a.cmd == "timing" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
