"""Sequential monitor for a running duplicate-pair head-to-head: current statistic, bounds, decision.

DRAFT (wsp seq plan, .orchestra/wsp/seqtest-plan.md) -- not in use until the orchestrator approves.

Input (files or run directories; a directory is searched recursively for --glob, default *.jsonl):
  * per-game records = our battles.jsonl: {"pair", "game", "winner", "p1", "p2", ...}; A is the player whose name
    starts with --a-prefix (default wspa; netv2 Foul Play runs: pol, gloria Foul Play benchmarks: sea). Empty winner = tie.
    Each file is one shard, pair ids from 0.
  * per-pair records, one JSON line per completed pair: spd_h2h.py's {"pair", "shard", "games": [{"winner": "A" |
    "B" | "tie", "void", ...}, {...}], "score"} (live shard-KKKK.part.jsonl and final shard-KKKK.jsonl of a shard
    merge; the shard's first pair id comes from shard-KKKK.meta.json when it exists, else shard * --shard-pairs,
    else --first-id), or generic {"pair", "shard" (optional), "winners": [w0, w1]} with w = "A" | "B" | null/"tie"
    (or a username, resolved with --a-prefix), "scores": [s0, s1], "a_score" (pair score) or "a_wins" (0, 1, 2).
Ordering: complete pairs are taken in pair-id order within each shard, up to the first id that is not complete yet
(a pair still being played holds back every later id; --stale N drops one that stays incomplete while N later ids
complete, reported; --final drops all incomplete ones), and shards are interleaved round-robin. Look j uses the
first j*k pairs of that sequence, so the data at a look never depend on which games happened to finish first and a
later re-run reproduces the same decision. The decision is the first scheduled look at which a bound was crossed;
pairs played after it (overrun) are reported but not used.

    python scripts/wsp/seq_monitor.py OUT_DIR --design OUT_DIR/seq_design.json
    python scripts/wsp/seq_monitor.py out/h2h/688986/edits_500M_temp05_vs_m500 --preset gain [--final]
    python scripts/wsp/seq_monitor.py out/h2h/*/* --mix          # pair-outcome mix (planning input)
Exit status: 0 continue, 10 success, 20 futility, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seq_design as sd  # noqa: E402


# --------------------------------------------------------------------------------------------------- reading
def _game_score(w, a_re) -> float | None:
    """A's score for one game from a winner field: 1 / 0 / .5 (tie); None if not decodable."""
    if w is None or w == "" or (isinstance(w, str) and w.lower() in ("tie", "draw", "none")):
        return 0.5
    if isinstance(w, bool):
        return float(w)
    if isinstance(w, (int, float)):
        return float(w)
    if isinstance(w, str):
        if w in ("A", "a"):
            return 1.0
        if w in ("B", "b"):
            return 0.0
        return 1.0 if a_re.match(w) else 0.0
    return None


def read_records(paths: list[str], a_prefix: str = "wspa", glob: str = "*.jsonl") -> dict:
    """{shard key: {"pairs": {pair_id: [score_game0 or None, score_game1 or None]}, "files": [...], "shard": s}}.
    Records with a "shard" field are keyed by it (so a shard's part and final files merge); others by file."""
    a_re = re.compile(re.escape(a_prefix) if not a_prefix.startswith("^") else a_prefix)
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.rglob(glob)) if p.is_dir() else [p]
    files = [f for f in files if f.name not in ("results.jsonl", "decisions.jsonl")] or files
    shards: dict = {}
    for fi, f in enumerate(files):
        for ln, line in enumerate(open(f)):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if ln == 0:
                    raise
                continue  # a line being written right now
            if "pair" not in r:
                continue
            key = ("shard", r["shard"]) if "shard" in r else ("file", fi)
            sh = shards.setdefault(key, {"pairs": {}, "files": set(), "shard": r.get("shard")})
            sh["files"].add(f)
            pair = sh["pairs"].setdefault(int(r["pair"]), [None, None])
            if "game" in r and "winner" in r:                  # per-game record
                if a_prefix and not (a_re.match(str(r.get("p1", ""))) or a_re.match(str(r.get("p2", "")))):
                    raise SystemExit(f"{f}: neither player matches --a-prefix {a_prefix!r}: {r.get('p1')} / {r.get('p2')}")
                pair[int(r["game"])] = _game_score(r["winner"], a_re)
            elif "games" in r:                                  # spd_h2h.py: void games score as ties
                pair[0], pair[1] = (0.5 if g.get("void") else _game_score(g.get("winner"), a_re) for g in r["games"])
            elif "winners" in r or "scores" in r:
                v = r.get("winners", r.get("scores"))
                pair[0], pair[1] = (_game_score(x, a_re) for x in v)
            elif "a_score" in r or "score" in r:
                sc = float(r.get("a_score", r.get("score")))
                pair[0] = pair[1] = sc                            # two halves with the pair's mean
            elif "a_wins" in r:
                pair[0] = pair[1] = float(r["a_wins"]) / 2
    return shards


def first_ids(shards: dict, first_id: int = 0, shard_pairs: int | None = None) -> dict:
    """First pair id of each shard: its spd meta file's "first", else shard * shard_pairs, else first_id."""
    out = {}
    for key, sh in shards.items():
        fid = None
        for f in sh["files"]:
            m = Path(str(f).replace(".part.jsonl", ".jsonl").replace(".jsonl", ".meta.json"))
            if m.exists():
                try:
                    fid = int(json.loads(m.read_text())["first"])
                except (ValueError, KeyError, json.JSONDecodeError):
                    pass
        if fid is None and shard_pairs and sh["shard"] is not None:
            fid = int(sh["shard"]) * shard_pairs
        if fid is None:
            if sh["shard"] not in (None, 0) and first_id == 0 and min(sh["pairs"]) >= 1:
                raise SystemExit(f"shard {sh['shard']}: first pair id unknown (no meta file yet); give "
                                 "--shard-pairs N (spd_h2h --pairs) or --first-id")
            fid = first_id
        out[key] = fid
    return out


def ordered_scores(shards: dict, stale: int = 0, final: bool = False, first_id: int = 0,
                   shard_pairs: int | None = None):
    """Pair scores in the canonical order (id order within shards, round-robin across shards), with bookkeeping.
    Ids run first, first + 1, ... in each shard; a missing or incomplete id holds back all later ones (stale > 0:
    unless that many later ids are complete; final: never)."""
    prefixes, dropped = [], []
    firsts = first_ids(shards, first_id, shard_pairs)
    for key in sorted(shards, key=lambda k: (str(k[0]), k[1])):
        prs = shards[key]["pairs"]
        ids = sorted(prs)
        complete = [i for i in ids if None not in prs[i]]
        cset = set(complete)
        out = []
        n_later = len(complete)
        for i in range(min([firsts[key]] + ids), (ids[-1] + 1) if ids else firsts[key]):
            if i in cset:
                out.append((prs[i][0] + prs[i][1]) / 2)
                n_later -= 1
                continue
            if final or (stale and n_later >= stale):  # incomplete and long overtaken: drop, report
                dropped.append((key, i))
                continue
            break
        prefixes.append(out)
    seq, r = [], 0
    while True:
        live = [p for p in prefixes if r < len(p)] if final else prefixes
        if not live or (not final and any(r >= len(p) for p in prefixes)):
            break
        seq += [p[r] for p in live]
        r += 1
    if not final:  # partial round: shards in order until the first one that has no pair at rank r
        for p in prefixes:
            if r < len(p):
                seq.append(p[r])
            else:
                break
    n_all = sum(1 for sh in shards.values() for v in sh["pairs"].values() if None not in v)
    return np.array(seq), {"held_back": n_all - len(seq), "dropped": dropped, "shards": len(prefixes),
                           "complete_pairs": n_all}


# --------------------------------------------------------------------------------------------------- deciding
def evaluate(d: sd.Design, x: np.ndarray, final: bool = False) -> dict:
    """Evaluate every scheduled look the data reach; decision at the first crossing."""
    looks, decision, at = [], "continue", None
    for j, n in enumerate(d.n):
        if len(x) < n:
            break
        xs = x[:n]
        m, s = float(xs.mean()), float(xs.std(ddof=1))
        se = max(s, 1e-9) / math.sqrt(n)
        z0, z1 = (m - d.theta0) / se, (m - d.theta1) / se
        hi, lo = d.theta0 + d.b[j] * se, d.theta1 - d.c[j] * se
        succ = z0 >= d.b[j]
        fut = (z1 <= -d.c[j]) or (j == len(d.n) - 1 and not succ)
        state = "SUCCESS" if succ else ("FUTILITY" if fut else "continue")
        looks.append(dict(look=j + 1, pairs=n, mean=m, sd=s, se=se, z0=z0, z1=z1, succ_bound=hi, fut_bound=lo,
                          state=state))
        if state != "continue":
            decision, at = state.lower(), j
            break
    return {"decision": decision, "look": at, "looks": looks}


def labels(d: sd.Design) -> dict:
    if d.kind == "threshold":
        c = (d.theta0 + d.theta1) / 2
        return {"success": f"win rate >= {100 * c:.1f}% (errors <= {d.alpha:.0%} outside {100 * d.theta0:.1f}-"
                           f"{100 * d.theta1:.1f}%)",
                "futility": f"win rate < {100 * c:.1f}% (errors <= {d.beta:.0%} outside {100 * d.theta0:.1f}-"
                            f"{100 * d.theta1:.1f}%)"}
    return {"success": f"claim: win rate > {100 * d.theta0:.1f}% (one-sided alpha {d.alpha})",
            "futility": f"no claim: win rate < {100 * d.theta1:.1f}% indicated (power {1 - d.beta:.0%} design)"}


def report(d: sd.Design, x: np.ndarray, meta: dict, final: bool = False) -> tuple[str, int]:
    ev = evaluate(d, x, final)
    L = [f"seq monitor -- design {d.name or d.kind}: theta0 {100 * d.theta0:.2f}%, theta1 {100 * d.theta1:.2f}%, "
         f"alpha {d.alpha}, power {1 - d.beta:.2f}, look every {d.k} pairs, n_max {d.n_max} pairs",
         f"  data: {len(x)} pairs usable in id order ({meta['complete_pairs']} complete, {meta['held_back']} held "
         f"back behind running pairs, {len(meta['dropped'])} dropped as stale/incomplete, {meta['shards']} shard(s))",
         "  look  pairs   win%    SE  |   Z0  succ if win% >= |   Z1  fut if win% <= | state"]
    for r in ev["looks"]:
        L.append(f"  {r['look']:4d} {r['pairs']:6d} {100 * r['mean']:6.2f} {100 * r['se']:5.2f} | {r['z0']:+5.2f}"
                 f" {100 * r['succ_bound']:13.2f}   | {r['z1']:+5.2f} {100 * r['fut_bound']:12.2f}   | {r['state']}")
    lab = labels(d)
    if ev["decision"] == "continue":
        nxt = next((n for n in d.n if n > len(x)), None)
        L.append(f"DECISION continue: {len(x)} pairs usable, next look at {nxt} pairs" if nxt else
                 "DECISION continue (no scheduled look left?)")
        if ev["looks"]:
            r = ev["looks"][-1]
            j = r["look"] - 1
            L.append(f"  at look {r['look']} ({r['pairs']} pairs): {100 * r['mean']:.2f}% (naive), 95% repeated CI "
                     f"[{100 * (r['mean'] - d.r[j] * r['se']):.2f}, {100 * (r['mean'] + d.r[j] * r['se']):.2f}] "
                     "(valid at any look; not a decision)")
        return "\n".join(L), 0
    j = ev["look"]
    r = ev["looks"][-1]
    fi = sd.final_inference(d, j, r["mean"], r["sd"])
    L.append(f"DECISION {ev['decision'].upper()} at look {j + 1} ({fi['pairs']} pairs, {2 * fi['pairs']} games): "
             f"{lab[ev['decision']]}")
    L.append(f"  estimate: median-unbiased {100 * fi['mue']:.2f}% [95% CI {100 * fi['ci_lo']:.2f}, {100 * fi['ci_hi']:.2f}]"
             f" (stage-wise ordering); naive {100 * fi['naive']:.2f}%, 95% repeated CI [{100 * fi['rci'][0]:.2f}, "
             f"{100 * fi['rci'][1]:.2f}]")
    lo, hi = fi["rci"]
    if hi < 0.5 or lo > 0.5:
        L.append(f"  the 95% repeated CI excludes 50%: A is {'stronger' if lo > 0.5 else 'weaker'} than B "
                 "(two-sided 0.05, valid at any look)")
    extra = len(x) - fi["pairs"]
    if extra > 0:
        m, s = x.mean(), x.std(ddof=1)
        L.append(f"  overrun: {extra} more pairs already played (not used for the decision); all {len(x)} pairs: "
                 f"{100 * m:.2f}% (naive, informational)")
    return "\n".join(L), 10 if ev["decision"] == "success" else 20


def mix_report(paths: list[str], a_prefix: str, glob: str) -> str:
    """Pair-outcome mix per run (each path = one run) and pooled: the planning input of seq_design.py."""
    rows, tot = [], np.zeros(5)
    L = ["  run                                          pairs  both-won  split  both-lost  one-tie  split%   mean    SD  var/indep"]
    for p in paths:
        x, _ = ordered_scores(read_records([p], a_prefix, glob), final=True)
        c = np.array([(np.abs(x - v) < 1e-9).sum() for v in sd.VALUES])
        tot += c
        n, m, s = len(x), x.mean(), x.std(ddof=1)
        rows.append((n, c[2]))
        L.append(f"  {'/'.join(Path(p).parts[-2:])[-44:]:44s} {n:6d} {c[4]:8d} {c[2]:6d} {c[0]:9d} {c[1] + c[3]:8d}"
                 f"  {c[2] / n:.3f}  {100 * m:5.1f}  {s:.3f}  {s * s / (m * (1 - m) / 2):.3f}")
    N = tot.sum()
    ps = tot[2] / N
    chi = sum((sp - n * ps) ** 2 / (n * ps * (1 - ps)) for n, sp in rows)
    L.append(f"  pooled: {int(N)} pairs, split {ps:.4f} ± {1.96 * math.sqrt(ps * (1 - ps) / N):.4f}, heterogeneity "
             f"chi2 {chi:.1f} on {len(rows) - 1} df; counts for seq_design.py --mix-counts (X = 0 .25 .5 .75 1): "
             + " ".join(str(int(v)) for v in tot))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--design", help="design JSON written by seq_design.py --save (the pre-registered design)")
    ap.add_argument("--preset", choices=["gain", "margin10", "noninf", "threshold"])
    ap.add_argument("--c", type=float, help="threshold preset: decision point")
    ap.add_argument("--a-prefix", default="wspa", help="A's username prefix (regex if it starts with ^)")
    ap.add_argument("--stale", type=int, default=0, help="drop an incomplete pair once this many later ids of its "
                    "shard are complete (default 0 = never: a lost pair stalls the looks until --final)")
    ap.add_argument("--final", action="store_true", help="the run has ended: drop incomplete pairs, use all shards")
    ap.add_argument("--first-id", type=int, default=0, help="first pair id of a shard without meta/--shard-pairs")
    ap.add_argument("--shard-pairs", type=int, help="spd_h2h.py --pairs N: shard K's first pair id is K*N")
    ap.add_argument("--glob", default="*.jsonl", help="files to read inside a run directory (e.g. 'pairs*.jsonl' "
                                                      "if a run writes both per-game and per-pair records)")
    ap.add_argument("--json", action="store_true", help="also print a machine-readable line")
    ap.add_argument("--stop-file", help="write the decision line to this file when a bound is crossed")
    ap.add_argument("--mix", action="store_true", help="print the pair-outcome mix of each path (= one run) instead")
    a = ap.parse_args()
    if a.mix:
        print(mix_report(a.paths, a.a_prefix, a.glob))
        return 0
    if bool(a.design) == bool(a.preset):
        ap.error("give exactly one of --design / --preset")
    d = sd.load(a.design) if a.design else sd.solve(sd.preset(a.preset, c=a.c))
    try:
        x, meta = ordered_scores(read_records(a.paths, a.a_prefix, a.glob), a.stale, a.final, a.first_id,
                                 a.shard_pairs)
    except (OSError, ValueError, KeyError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    text, code = report(d, x, meta, a.final)
    print(text)
    if a.json:
        ev = evaluate(d, x, a.final)
        print("JSON " + json.dumps({"decision": ev["decision"], "look": None if ev["look"] is None else ev["look"] + 1,
                                    "pairs_used": len(x), "design": d.name}))
    if a.stop_file and code in (10, 20):
        Path(a.stop_file).write_text(next(ln for ln in text.splitlines() if ln.startswith("DECISION")) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
