"""LA2-gen4, stage 1 (docs/netsearch/LA2-GEN4-PREREG.md = council/la2-gen4/PREREG.txt): the mechanical parts of the
prep chain (la2_gen4_prep.sbatch: self-play records -> split -> leaf parity + gen9 identity -> the δ* study) and of
the slot chain (la2_gen4.sbatch: smoke -> 1,000 duplicate pairs -> VERDICT.txt).

  split     the self-play record parts (exit_la2.py gen --all-decisions: per game one record per side, each with one
            decision drawn uniformly among the side's decisions with >= 2 legal actions) -> games ordered by
            sha1("<seed>:<game>"); per game ONE side (sha1("<seed>:<game>:side") % 2; the other side if that one has
            no record); the first --parity games -> parity-records.jsonl.gz (the leaf-parity battles), the next
            --study games -> study-records.jsonl.gz + study-rows.jsonl (one position per game: that side's drawn
            decision). split.json: counts and md5s.
  ident     gen9 byte-identity of porygon.v3leaf against the benchmark bot's frozen copy (--old): per gen9 position of a
            study rows file, K determinizations; lookahead2.expand_world (min_prob 0, as the player) once with each
            module: rows, cols, cells, counts and every leaf observation array (dtype, shape, bytes) must be equal.
  parity    the reading of v3_gates.py p1's p1.json at the gen9 standard (ns-10): PASS iff test A element agreement
            >= 0.99 and A mean |dP| <= 0.01 and B mean |dP| <= 0.03; VOID if > 10% of the battles fail to replay.
  rows-diff two p1-rows.jsonl byte-identical (gen9 P1 with the old and the new code)
  delta     δ* from analyze-bench's JSON (Bet 1's rule, pooled, >= 98% of the positions usable)
  probs     informational: the bench's replayed T-1 probabilities vs the self-play forward's at the same positions
  smoke     the slot smoke's mechanical checks (the lookahead2 arm vs the raw net, which has no search statistics)
  verdict   VERDICT.txt: line 1 GO | STOP (GO iff the pair-mean lower 95% > 50.0%), then the pair-mean with its 95%
            CI and the override rate; VOID (HARNESS_FAIL, > 2% fallbacks, < 900 pairs) instead of a reading

split / parity / rows-diff / delta / probs / smoke / verdict are pure (standard library); ident needs the gen9
environment (PORYGON_GEN=9, gloria's .engine-gen9 first on PYTHONPATH, NETV2_SRC = the frozen netv2 tree's src).
"""
import argparse
import asyncio
import glob
import gzip
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRECISION_STANDARD = {"A_agree": 0.99, "A_dp": 0.01, "B_dp": 0.03}  # ns-10 (V3-GATES-PREREG P1)
GO_LO = 0.50  # main's rule: GO iff the pair-mean lower 95% > 50.0% (strict)
MIN_PAIRS, VOID_FALLBACKS = 900, 0.02  # the slot void rule (LA2-H's 450 of 500, scaled to 1,000 pairs)
USABLE_MIN = 0.98  # Bet 1's δ rule: >= 98% of the study positions usable


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_gz(path: Path, lines: list[str]) -> None:
    """A deterministic gzip file (mtime 0, no file name), written atomically."""
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
        for line in lines:
            gz.write(line.encode())
    tmp.replace(path)


# ---------------------------------------------------------------- split

def game_key(seed: int, game: str) -> str:
    return hashlib.sha1(f"{seed}:{game}".encode()).hexdigest()


def side_of(seed: int, game: str) -> int:
    return int(hashlib.sha1(f"{seed}:{game}:side".encode()).hexdigest()[:12], 16) % 2


def choose(recs: list[dict], seed: int) -> list[dict]:
    """One record per game, games in sha1("<seed>:<game>") order (the module docstring's rule)."""
    games: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in recs:
        games[r["game"]][int(r["side"])] = r
    out = []
    for g in sorted(games, key=lambda x: game_key(seed, x)):
        s = side_of(seed, g)
        rec = games[g].get(s) or games[g].get(1 - s)
        if rec is not None:
            out.append(rec)
    return out


def study_row(rec: dict) -> dict:
    d = rec["decisions"][0]
    return {"rec": rec["id"], "side": rec["side"], "idx": d["idx"], "turn": d["turn"], "n_legal": d["n_legal"],
            "game": rec["game"]}


def cmd_split(a) -> int:
    parts = sorted(glob.glob(str(Path(a.gen) / "rec-*.jsonl.gz")))
    assert parts, f"no record parts in {a.gen}"
    recs = []
    for f in parts:
        with gzip.open(f, "rt") as fh:
            recs += [json.loads(x) for x in fh if x.strip()]
    chosen = choose(recs, a.seed)
    need = a.parity + a.study
    if len(chosen) < need:
        print(f"FAIL: {len(chosen)} games with a record, need {need}")
        return 1
    if a.parity and not all("all_decisions" in r for r in chosen[:a.parity]):
        print("FAIL: parity records without all_decisions (gen needs --all-decisions)")
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    par, stu = chosen[:a.parity], chosen[a.parity:need]
    write_gz(out / "parity-records.jsonl.gz", [json.dumps(r) + "\n" for r in par])
    write_gz(out / "study-records.jsonl.gz", [json.dumps(r) + "\n" for r in stu])
    rows = [study_row(r) for r in stu]
    (out / "study-rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    info = {"seed": a.seed, "parts": len(parts), "records": len(recs), "games_with_record": len(chosen),
            "parity_battles": len(par), "study_positions": len(rows),
            "study_sides": {s: sum(r["side"] == s for r in rows) for s in (0, 1)},
            "study_turn_median": sorted(r["turn"] for r in rows)[len(rows) // 2] if rows else None,
            "md5": {n: md5_file(out / n) for n in ("parity-records.jsonl.gz", "study-records.jsonl.gz",
                                                    "study-rows.jsonl")}}
    (out / "split.json").write_text(json.dumps(info, indent=1))
    print(f"OK: {len(par)} parity battles, {len(rows)} study positions (one per game) from {len(chosen)} games; "
          f"study-rows.jsonl md5 {info['md5']['study-rows.jsonl']}")
    return 0


# ---------------------------------------------------------------- ident (gen9 environment)

def world_diffs(x: dict, y: dict) -> list[str]:
    """Fields in which two expand_world results differ (empty = identical, the leaves byte for byte)."""
    diffs = [k for k in ("rows", "cols", "cells", "n_leaves", "branches", "terminal", "switch_expansions", "errors")
             if x.get(k) != y.get(k)]
    ox, oy = x.get("obs"), y.get("obs")
    if (ox is None) != (oy is None):
        diffs.append("obs None")
    elif ox is not None:
        if set(ox) != set(oy):
            diffs.append("obs keys")
        else:
            for k in sorted(ox):
                a, b = ox[k], oy[k]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    diffs.append(f"obs.{k}")
    return diffs


def obs_diffs(x: dict, y: dict) -> list[str]:
    if set(x) != set(y):
        return ["keys"]
    return [k for k in sorted(x) if x[k].dtype != y[k].dtype or x[k].shape != y[k].shape
            or x[k].tobytes() != y[k].tobytes()]


async def ident(a) -> int:
    import importlib.util
    sys.path.insert(0, str(HERE))
    import netsearch_study as ST
    NS, S, R = ST.setup("gen9randombattle")
    import porygon
    from porygon import lookahead2 as LA
    from porygon import v3leaf as NL
    from porygon.v2.encoder import encode
    spec = importlib.util.spec_from_file_location("porygon._v3leaf_old", a.old)
    OL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OL)
    assert NL.__file__ != OL.__file__

    def expand(mod, root, conv, legal):
        saved = sys.modules["porygon.v3leaf"], porygon.v3leaf
        sys.modules["porygon.v3leaf"], porygon.v3leaf = mod, mod
        try:
            return LA.expand_world(root, conv, legal, 0.0)
        finally:
            sys.modules["porygon.v3leaf"], porygon.v3leaf = saved

    rows = [json.loads(x) for x in open(a.rows) if x.strip()][:a.positions]
    want: dict[tuple, set] = defaultdict(set)
    for r in rows:
        want[(r["rec"], r.get("side"))].add(r["idx"])
    recs = [r for r in ST.records(a.records, a.source, a.seed) if (r["id"], r.get("side")) in want]
    res = {"positions": 0, "worlds": 0, "leaves": 0, "worlds_with_errors": 0, "differ": 0, "control_differ": 0,
           "replay_failed": 0, "diff_fields": {}, "examples": [], "old": a.old, "new": NL.__file__}
    for rec in recs:
        key = (rec["id"], rec.get("side"))
        try:
            async for d, battle, first, _ in ST.replay_targets(rec, R, NS, 9, want[key]):
                legal = encode(battle)["action_mask"] > 0
                if legal.sum() < 2:
                    continue
                res["positions"] += 1
                root_n, root_o = NL.LeafRoot(battle, first), OL.LeafRoot(battle, first)
                bad = obs_diffs(root_n.obs, root_o.obs)
                hist = S.History(battle)
                for k in range(a.dets):
                    conv = S.convert(battle, random.Random(f"ident:{rec['id']}:{d.msg}:{k}"), first, hist)
                    try:
                        wn = expand(NL, root_n, conv, legal)
                    except Exception as e:  # a world the engine cannot expand: both modules must fail alike
                        wn = {"error": f"{type(e).__name__}: {e}"}
                    try:
                        wo = expand(OL, root_o, conv, legal)
                    except Exception as e:
                        wo = {"error": f"{type(e).__name__}: {e}"}
                    res["worlds"] += 1
                    if "error" in wn or "error" in wo:
                        res["worlds_with_errors"] += 1
                        diffs = [] if wn.get("error") == wo.get("error") else ["error"]
                    else:
                        res["leaves"] += wn["n_leaves"]
                        diffs = bad + world_diffs(wn, wo)
                        if res["positions"] <= a.control:  # determinism control: the new module twice
                            res["control_differ"] += bool(world_diffs(wn, expand(NL, NL.LeafRoot(battle, first), conv,
                                                                                legal)))
                    if diffs:
                        res["differ"] += 1
                        for f in diffs:
                            res["diff_fields"][f] = res["diff_fields"].get(f, 0) + 1
                        if len(res["examples"]) < 5:
                            res["examples"].append({"rec": rec["id"], "idx": d.msg, "det": k, "fields": diffs})
        except Exception as e:  # noqa: BLE001
            res["replay_failed"] += 1
            print(f"record {rec['id']}: {type(e).__name__}: {e}", file=sys.stderr)
    ok = res["differ"] == 0 and res["control_differ"] == 0 and res["positions"] >= a.min_positions and res["leaves"] > 0
    res["pass"] = ok
    res["line"] = (("PASS" if ok else "FAIL") + f": gen9 v3leaf new vs old ({Path(a.old).parent.parent.parent.name}"
                   f" code): {res['positions']} positions x {a.dets} worlds = {res['worlds']} lookahead2 expansions, "
                   f"{res['leaves']} leaves; differing expansions {res['differ']} {res['diff_fields'] or ''}; "
                   f"determinism control {res['control_differ']} of {min(a.control, res['positions'])} positions; "
                   f"engine-failed worlds {res['worlds_with_errors']}; replay failures {res['replay_failed']}")
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(res["line"])
    return 0 if ok else 1


# ---------------------------------------------------------------- readings (pure)

def parity_line(p1: dict) -> tuple[str, str]:
    """(PASS | FAIL | VOID, the line) for a p1.json at the gen9 standard."""
    A, B = p1["A"], p1["B"]
    std = PRECISION_STANDARD
    ok = A["agree"] >= std["A_agree"] and A["mean_abs_dp"] <= std["A_dp"] and B["mean_abs_dp"] <= std["B_dp"]
    word = "VOID" if p1.get("void") else ("PASS" if ok else "FAIL")
    fmt = p1.get("format", "gen9randombattle")
    bad_a = ", ".join(f"{k} {v}" for k, v in list(A.get("bad", {}).items())[:6]) or "none"
    bad_b = ", ".join(f"{k} {v}" for k, v in list(B.get("bad", {}).items())[:8]) or "none"
    line = (f"{word}: {fmt} leaf parity (ns-10 method; gen9 standard A agree >= {std['A_agree']}, A mean|dP| <= "
            f"{std['A_dp']}, B mean|dP| <= {std['B_dp']}): {p1.get('records', '?')} battles "
            f"({p1.get('records_with_pairs', '?')} with pairs, {p1['failed_records']} failed), {p1['pairs']} pairs x "
            f"{p1['dets']} determinizations; A agree {A['agree']:.5f} ({A['elements']} elements), A mean|dP| "
            f"{A['mean_abs_dp']:.2e} (q99 {A.get('q99', math.nan):.2e}); B agree {B['agree']:.4f}, B mean|dP| "
            f"{B['mean_abs_dp']:.4f} (q90 {B.get('q90', math.nan):.3f}; revealed {B.get('mean_abs_dp_revealed') or 0:.4f} "
            f"n {B.get('n_revealed')}, not revealed {B.get('mean_abs_dp_not_revealed') or 0:.4f}); A disagreements: "
            f"{bad_a}; B disagreements: {bad_b}")
    return word, line


def delta_line(bench: dict, n_positions: int) -> tuple[float | None, str]:
    """(δ*, line) by Bet 1's pooled rule from analyze-bench's JSON; δ* None if fewer than 98% usable or no margin."""
    m = bench["margin_rule"]
    usable = bench.get("usable", 0)
    if usable < USABLE_MIN * n_positions or m["delta"] is None:
        return None, f"NONE usable {usable} of {n_positions} delta {m['delta']}"
    cur = next(c for c in bench["curves"] if c["delta"] == m["delta"])
    gain = m.get("judgeR_gain_vs_raw")
    return m["delta"], (f"{m['delta']} override {m['override']:.4f} precision_R {cur['precision_R']:.4f} "
                        f"judgeR_gain_vs_raw {gain:+.5f} usable {usable} of {n_positions}")


def probs_parity(bench_rows: list[dict], recs: dict[str, dict]) -> dict:
    """max / mean |p_replay - p_selfplay| over the legal actions of the study positions (informational)."""
    worst, tot, n, legal_mismatch = 0.0, 0.0, 0, 0
    for row in bench_rows:
        rec = recs.get(row["rec"])
        if rec is None:
            continue
        d = rec["decisions"][0]
        if d["idx"] != row["idx"]:
            continue
        legal = [i for i, m in enumerate(row["mask"]) if m]
        if legal != d["legal"]:
            legal_mismatch += 1
            continue
        for j, a in enumerate(legal):
            dp = abs(row["probs"][a] - d["probs"][j])
            worst, tot, n = max(worst, dp), tot + dp, n + 1
    return {"positions": len(bench_rows), "compared_actions": n, "max_abs_dp": worst,
            "mean_abs_dp": tot / n if n else math.nan, "legal_mismatch": legal_mismatch}


def smoke(run: Path, games: int) -> tuple[bool, str]:
    r = json.loads((run / "result.json").read_text())
    checks = {"one arm": len(r["arms"]) == 1, "no HARNESS_FAIL": not (run / "HARNESS_FAIL").exists()}
    for a in r["arms"]:
        k, s = a["arm"], a["stats_a"]
        ev = "\n".join(p.read_text(errors="replace") for p in run.glob(f"arm-{k}/block-*/evaluate.log"))
        checks[f"arm {k} {games} games"] = a["games"] == games
        checks[f"arm {k} no Traceback"] = "Traceback" not in ev
        checks[f"arm {k} searched/leaves > 0"] = s.get("searched", 0) > 0 and s.get("leaves", 0) > 0
        checks[f"arm {k} gate rows"] = a["gate"]["decisions"] > 0
        checks[f"arm {k} failed worlds 0"] = s.get("failed_worlds", 0) == 0
        checks[f"arm {k} fallbacks <= max(2, 2%)"] = s.get("fallbacks", 0) <= max(2, 0.02 * s.get("decisions", 0))
    ok = all(checks.values())
    return ok, ("PASS " if ok else "FAIL ") + "; ".join(f"{k} {'ok' if v else 'NO'}" for k, v in checks.items())


def pct(x, digits: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}%}"


def verdict(run: Path, delta: str, label: str, seed: str) -> str:
    r = json.loads((run / "result.json").read_text())
    arms = r["arms"]
    if len(arms) != 1:
        raise SystemExit(f"{run}: {len(arms)} arms (want 1)")
    a = arms[0]
    hf = (run / "HARNESS_FAIL").exists()
    void = hf or bool(a["void"]) or a["pairs"] < MIN_PAIRS
    ci = f"{pct(a['mean'])} [{pct(a['lo'])}, {pct(a['hi'])}]"
    if void:
        head = [f"VOID: harness failure, > {VOID_FALLBACKS:.0%} fallbacks or < {MIN_PAIRS} pairs ({a['pairs']} pairs, "
                f"HARNESS_FAIL {'yes' if hf else 'no'}, fallback void {bool(a['void'])}); no reading; replay with the "
                "same seeds after the fix"]
    else:
        go = a["lo"] > GO_LO
        head = ["GO" if go else "STOP",
                f"pair-mean {ci} = {pct(a['mean'])} ± {pct(a['hw'])} (95%, mean ± 1.96 sd / √n over {a['pairs']} "
                f"duplicate pairs, {a['games']} games): lookahead2({label}, δ* {delta}, K 8, T 0.5) vs raw {label} "
                f"(T 0.5), gen4; lower 95% {pct(a['lo'], 3)} {'>' if go else '<='} 50.0%",
                f"override rate {pct(a.get('override_rate'), 1)} of {a['gate']['decisions']} searched decisions",
                ("  => GO: main decides the follow-up (LA2-gen4 vs the incumbent on fresh identical Foul Play @200k pairs)"
                 if go else "  => STOP: the gen4 bot stays the incumbent search bot (search + fitted eval + v1@500M "
                 "prior); result recorded")]
    s = a["stats_a"]
    return "\n".join(head + [
        f"  arm {a['arm']}: {ci} ({a['pairs']} pairs, {a['games']} games; won both {a['won_both']}, split {a['split']}, "
        f"lost both {a['lost_both']}; incomplete {a.get('incomplete_pairs')}); {a['seconds'] / 3600:.2f} h",
        f"  lookahead2: searched {int(s.get('searched', 0))} of {int(s.get('decisions', 0))} decisions, overrides "
        f"{int(s.get('overrides', 0))}, fallbacks {pct(a.get('fallback_rate_a'), 3)}, failed worlds "
        f"{int(s.get('failed_worlds', 0))}, leaves {int(s.get('leaves', 0))}, lookahead s {s.get('la_seconds', 0):.0f}",
        f"  A {a['spec']}",
        f"  B {r['opponent']}",
        f"  seed tag {seed} (block j: {seed}b<j>)",
        "  per block: " + ", ".join(f"{b['block']} {b['pairs']} pairs {(b.get('seconds') or 0) / 3600:.2f} h"
                                    for b in a.get("per_block", [])),
        f"  rule (main, verbatim): GO iff the pair-mean lower 95% > 50.0%; else STOP (the gen4 bot stays the incumbent "
        f"search bot; record the result); run {run}"])


# ---------------------------------------------------------------- CLI

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--gen", required=True, help="exit_la2.py gen output dir")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--parity", type=int, default=300)
    s.add_argument("--study", type=int, default=1403)
    i = sub.add_parser("ident")
    i.add_argument("--rows", required=True)
    i.add_argument("--records", nargs="+", required=True)
    i.add_argument("--source")
    i.add_argument("--seed", type=int, default=11)
    i.add_argument("--old", required=True, help="the frozen v3leaf.py to compare with")
    i.add_argument("--positions", type=int, default=300)
    i.add_argument("--dets", type=int, default=2)
    i.add_argument("--control", type=int, default=10, help="positions also run twice with the new module")
    i.add_argument("--min-positions", type=int, default=200)
    i.add_argument("--json", required=True)
    pa = sub.add_parser("parity")
    pa.add_argument("p1json", type=Path)
    rd = sub.add_parser("rows-diff")
    rd.add_argument("a", type=Path)
    rd.add_argument("b", type=Path)
    de = sub.add_parser("delta")
    de.add_argument("bench_json", type=Path)
    de.add_argument("--positions", type=int, default=1403)
    pr = sub.add_parser("probs")
    pr.add_argument("bench", type=Path)
    pr.add_argument("records", type=Path)
    sm = sub.add_parser("smoke")
    sm.add_argument("run", type=Path)
    sm.add_argument("--games", type=int, default=4)
    v = sub.add_parser("verdict")
    v.add_argument("run", type=Path)
    v.add_argument("--delta", required=True)
    v.add_argument("--label", default="R1b-F-cd@640M")
    v.add_argument("--seed", required=True)
    a = p.parse_args()
    if a.cmd == "split":
        return cmd_split(a)
    if a.cmd == "ident":
        return asyncio.run(ident(a))
    if a.cmd == "parity":
        word, line = parity_line(json.loads(a.p1json.read_text()))
        print(line)
        return 0 if word == "PASS" else 1
    if a.cmd == "rows-diff":
        x, y = a.a.read_bytes(), a.b.read_bytes()
        nx, ny = x.count(b"\n"), y.count(b"\n")
        same = x == y and nx > 0
        print(("IDENTICAL" if same else "DIFFERENT") + f": {a.a} ({nx} rows, md5 {hashlib.md5(x).hexdigest()}) vs {a.b} "
              f"({ny} rows, md5 {hashlib.md5(y).hexdigest()})")
        return 0 if same else 1
    if a.cmd == "delta":
        d, line = delta_line(json.loads(a.bench_json.read_text()), a.positions)
        print(line)
        return 0 if d is not None else 1
    if a.cmd == "probs":
        rows = [json.loads(x) for x in a.bench.open() if x.strip()]
        with gzip.open(a.records, "rt") as fh:
            recs = {r["id"]: r for r in map(json.loads, fh)}
        print(json.dumps(probs_parity(rows, recs)))
        return 0
    if a.cmd == "smoke":
        print(smoke(a.run, a.games)[1])
        return 0
    print(verdict(a.run, a.delta, a.label, a.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
