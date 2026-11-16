"""NNUE gate 1 FEATURE-level parity (main's condition of 2026-11-16 on the step-3 State parity; notes.log): on the
same parity games as step 3, the student's inputs from the training path vs the serving path, per column:
  A = features(from_string(convert_full(dumped state, seen_hp=True)))      (what nnue_gen.py writes)
  B = features(from_string(search.convert(battle, ..., truth, oracle_all=True)))   (a serving-time converter)
Rule: every column equal on >= 99.5% of decisions and the whole vector on >= 98% (and >= 50 games / 2,000
decisions, conversion errors <= 0.5%). Line 1 of --out = PASS / FAIL. Also printed: per-volatile presence counts
in the raw States (which side holds a volatile the other lacks), to find fields poke-env does not track.
The training view needs the dumps' seen_hp (host.js since 2026-11-16), which step 3's dumps lack, so the games are
re-played: `--play 60 --workers 6 --seed 916` = step 3's arguments (nnue_parity.sbatch, 8 CPUs), and `--same-as`
checks that every re-played game has the same Showdown input log as step 3's (the same games).

    cd $C && PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src $PY scripts/fastenv/nnue_feature_parity.py \\
        --play 60 --workers 6 --seed 916 --dump-dir nnue/gate1/fparity \\
        --same-as 'nnue/gate1/parity/dumps/dump-*.jsonl.gz' --out nnue/gate1/FEATURE-PARITY.txt
"""
import argparse
import asyncio
import glob
import gzip
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import netsearch_study as study  # noqa: E402
import nnue_parity as NP  # noqa: E402

COL_MIN, VEC_MIN = 0.995, 0.98


def load(paths: list[str]):
    games, states = {}, {}
    for p in paths:
        with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as fh:
            for line in fh:
                x = json.loads(line)
                if x["t"] == "G":
                    if not x["crashed"] and not x["truncated"] and "streams" in x:
                        games[x["game"]] = x
                else:
                    states[(x["game"], x["side"], x["e"])] = x
    return games, states


def mon_context(col: str, fa, fb, names, inv) -> str:
    """For a mon column: that slot's species / ability on both sides (e.g. Regenerator behind an HP difference)."""
    m = re.match(r"(s[12]\.mon\d)\.", col)
    if not m:
        return ""
    j = names.index(f"{m.group(1)}.species")
    sp = lambda f: inv["species"].get(int(f[j]), int(f[j]))  # noqa: E731
    ab = lambda f: inv["abilities"].get(int(f[j + 2]), int(f[j + 2]))  # noqa: E731
    return f"  [train {sp(fa)}/{ab(fa)}, serve {sp(fb)}/{ab(fb)}]"


def group(col: str) -> str:
    """s2.mon3.hp -> s2.mon*.hp (summary by kind)."""
    return re.sub(r"mon\d", "mon*", col)


async def run(a) -> int:
    netsearch, S, R = study.setup(a.format)
    import poke_engine as pe
    from porygon.nnue import features as F
    from porygon.nnue.convert_full import convert_full
    gen = int(re.match(r"gen(\d+)", a.format).group(1))
    t0 = time.time()
    paths = NP.play(a.play, a.format, a.workers, Path(a.dump_dir), a.seed) if a.play else sorted(
        f for p in a.dump for f in glob.glob(p))
    games, states = load(paths)
    same = ""
    if a.same_as:
        ref, _ = load(sorted(f for p in a.same_as for f in glob.glob(p)))
        diff = [g for g in set(ref) | set(games) if g not in ref or g not in games
                or ref[g]["inputlog"] != games[g]["inputlog"]]
        same = (f"same games as {a.same_as[0]}: {'yes' if not diff else 'NO'} ({len(games)} vs {len(ref)} finished "
                f"games, {len(diff)} differ)")
        if diff:
            print(same, file=sys.stderr)
    names = F.column_names()
    miss = np.zeros(len(names), np.int64)
    n_dec = n_err = n_vec = n_unreached = 0
    errs = Counter()
    examples = defaultdict(list)
    inv = {c: {i: n for n, i in v.items()} for c, v in F.vocab().items()}
    vol = Counter()  # (side, name, "full only" / "convert only" / "both")
    for gid in sorted(games):
        for side, rec in NP.records(games[gid], states, a.format):
            chosen = {d["idx"] for d in rec["decisions"]}
            got = set()
            async for d, battle, first, _ in study.replay_targets(rec, R, netsearch, gen, chosen):
                got.add(d.msg)
                x = states[(gid, side, d.msg)]
                try:
                    sa = pe.State.from_string(convert_full(x["state"], seen_hp=True).to_string())
                    conv = S.convert(battle, random.Random(0), first, None, rec["true_team"], True)
                    sb = pe.State.from_string(conv.state_str)
                    fa, fb = F.flat(F.features(sa)), F.flat(F.features(sb))
                except BaseException as e:  # engine panics are pyo3 PanicExceptions
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    n_err += 1
                    errs[repr(e)[:160]] += 1
                    continue
                n_dec += 1
                bad = fa != fb
                miss += bad
                n_vec += int(not bad.any())
                for j in np.flatnonzero(bad):
                    if len(examples[names[j]]) < a.examples:
                        examples[names[j]].append(f"{gid} p{side + 1} e{d.msg} turn {x['turn']}: train {fa[j]:.6g} vs "
                                                  f"serve {fb[j]:.6g}{mon_context(names[j], fa, fb, names, inv)}")
                for s_name, s_a, s_b in (("side_one", sa.side_one, sb.side_one), ("side_two", sa.side_two, sb.side_two)):
                    if s_a.pokemon[int(s_a.active_index)].hp <= 0:
                        continue
                    va = {str(v).upper() for v in s_a.volatile_statuses} - {"NONE"}
                    vb = {str(v).upper() for v in s_b.volatile_statuses} - {"NONE"}
                    for v in va | vb:
                        vol[(s_name, v, "both" if v in va and v in vb else "full only" if v in va else "convert only")] += 1
            n_unreached += len(chosen - got)
    n = max(n_dec, 1)
    rate = miss / n
    bad_cols = [names[j] for j in np.flatnonzero(rate > 1 - COL_MIN)]
    vec_rate = n_vec / n
    enough = len(games) >= a.min_games and n_dec >= a.min_decisions
    ok = enough and not bad_cols and vec_rate >= VEC_MIN and n_err <= 0.005 * n and "NO" not in same
    why = ([] if enough else [f"too few games / decisions ({len(games)} / {n_dec})"]) \
        + ([f"columns below {COL_MIN:.1%}: {', '.join(bad_cols)}"] if bad_cols else []) \
        + ([f"whole vector {vec_rate:.4%} < {VEC_MIN:.0%}"] if vec_rate < VEC_MIN else []) \
        + ([f"{n_err} conversion errors"] if n_err > 0.005 * n else []) + (["not the same games"] if "NO" in same else [])
    lines = [("PASS" if ok else "FAIL") + (f": {'; '.join(why)}" if why else ""),
             f"NNUE gate 1 FEATURE parity: features(convert_full(seen_hp)) vs features(search.convert(truth=, "
             f"oracle_all=True)); format {a.format}; feature spec {F.spec()['hash']} ({len(names)} columns)",
             f"{len(games)} games, {n_dec} decisions, {n_unreached} dumped rows the replay did not reach, {n_err} "
             f"conversion errors; {time.time() - t0:.0f} s",
             f"whole feature vector equal: {vec_rate:.4%} (rule >= {VEC_MIN:.0%}); worst column equal: "
             f"{1 - rate.max():.4%} (rule >= {COL_MIN:.1%})",
             f"inputs: {len(paths)} dump files ({', '.join(paths[:2])}{' ...' if len(paths) > 2 else ''})"]
    lines += [same] if same else []
    lines += ["",
             "Columns with any mismatch (rate, count):"]
    for j in np.argsort(-miss):
        if miss[j] == 0:
            break
        lines.append(f"  {names[j]:40s} {rate[j]:8.4%}  ({miss[j]})")
    by_group = Counter()
    for j in np.flatnonzero(miss):
        by_group[group(names[j])] += int(miss[j])
    lines += ["", "By kind (summed over mon slots; counts):"] + [f"  {k:40s} {c}" for k, c in by_group.most_common()]
    lines += ["", "Raw-State volatiles of live actives (side, name: both / full only / convert only):"]
    for s_name, v in sorted({(k[0], k[1]) for k in vol}):
        b, fo, co = vol[(s_name, v, "both")], vol[(s_name, v, "full only")], vol[(s_name, v, "convert only")]
        tag = "" if v in F.volatiles() else "  [not an input]"
        lines.append(f"  {s_name} {v:24s} {b:6d} / {fo:6d} / {co:6d}{tag}")
    lines += ["", "Examples:"]
    for k in sorted(examples):
        lines.append(f"  {k}:")
        lines += [f"    {e}" for e in examples[k]]
    if errs:
        lines += ["", "Conversion errors:"] + [f"  {c} x {e}" for e, c in errs.most_common(10)]
    text = "\n".join(lines) + "\n"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    (out.parent / (out.stem + ".json")).write_text(json.dumps(
        {"verdict": "PASS" if ok else "FAIL", "games": len(games), "decisions": n_dec, "errors": n_err,
         "vector_equal": vec_rate, "spec": F.spec()["hash"],
         "column_mismatch": {names[j]: float(rate[j]) for j in np.flatnonzero(miss)}}, indent=1))
    print(text)
    return 0 if a.no_verdict or ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--dump", nargs="*", default=[])
    p.add_argument("--play", type=int, default=0, help="re-play this many games first (nnue_parity.play), else --dump")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=916)
    p.add_argument("--dump-dir", default="nnue/gate1/fparity")
    p.add_argument("--same-as", nargs="*", default=[], help="dumps whose games the re-played ones must equal")
    p.add_argument("--out", required=True)
    p.add_argument("--min-games", type=int, default=50)
    p.add_argument("--min-decisions", type=int, default=2000)
    p.add_argument("--examples", type=int, default=4)
    p.add_argument("--no-verdict", action="store_true", help="exit 0 whatever the verdict (tests)")
    a = p.parse_args()
    os.environ.setdefault("PORYGON_GEN", re.match(r"gen(\d+)", a.format).group(1))
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
