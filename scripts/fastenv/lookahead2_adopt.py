"""The mechanical parts of an LA2-H-style adoption test (lookahead2_adopt.sbatch; LA2-1350 = docs/netsearch/
LA2-1350-PREREG.md; la2_2030.sbatch = LA2-2030-PREREG.md): the gate on netv2's confirmation file, the smoke's checks,
the pre-registered reading and the VERDICT text, and LA2-2030's FP200 measurement text. Pure functions of files,
standard library only (the cluster venv or a laptop python3).

    python lookahead2_adopt.py gate RESULTS.txt [--md5 6cf40c0c] [--name F1350]  # exit 0 = PLAY, 3 = WITHDRAWN
    python lookahead2_adopt.py smoke RUN [--games 4] [--fp]          # one line: PASS ... / FAIL ...
    python lookahead2_adopt.py reading LO                            # the reading for a lower 95% bound (fraction)
    python lookahead2_adopt.py verdict RUN --delta D [--label ...]    # VERDICT.txt on stdout (line 1 = the reading)
    python lookahead2_adopt.py fp200 RUN --delta D --iters N --seed S [--voids V]   # FP200.txt (line 1 = the number)
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

GATE_OK = "CONFIRMED (lower 95% bound > 50%)"  # netv2 conf1350.sbatch reading(): the only confirmed wording
WITHDRAWN_EXIT = 3
BETTER = "ADOPTED (BETTER)"
NON_INFERIOR = "ADOPTED (NON-INFERIOR, margin 3 points)"
NOT_ADOPTED = "NOT ADOPTED"
LO_BETTER, LO_NON_INFERIOR = 0.50, 0.47  # lower 95% bound of the pair means, strict >
MIN_PAIRS, VOID_FALLBACKS = 450, 0.02  # the LA2-H void rule


def gate(path: Path, md5: str | None = None, name: str = "F1350") -> tuple[bool, str]:
    """PLAY iff line 2 of netv2's RESULTS.txt contains GATE_OK (and not "NOT CONFIRMED"), and line 1 names the net's
    md5 when one is given. Anything else (NOT CONFIRMED, FAILED, missing, unreadable) withdraws, with the reason.
    `name` is the confirmation's name in the messages (F1350, F2030)."""
    if not path.is_file():
        return False, f"{path} missing ({name} wrote no RESULTS.txt)"
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        return False, f"{path} has no line 2"
    l1, l2 = lines[0].strip(), lines[1].strip()
    if "NOT CONFIRMED" in l2:
        return False, f"{name} NOT CONFIRMED: {l2}"
    if GATE_OK not in l2:
        why = "FAILED" if "FAILED" in l2 else "no CONFIRMED reading"
        return False, f"{name} {why}: {l2}"
    if md5 and f"md5 {md5}" not in l1:
        return False, f"{name} line 1 does not name the net md5 {md5}: {l1}"
    return True, f"{name} CONFIRMED: {l2}"


def reading(lo: float) -> str:
    """The pre-registered reading of the candidate's lower 95% bound (a fraction): > 50% BETTER, > 47% NON-INFERIOR,
    else NOT ADOPTED. NaN (no pairs) reads NOT ADOPTED; the void rule catches it first."""
    if lo > LO_BETTER:
        return BETTER
    if lo > LO_NON_INFERIOR:
        return NON_INFERIOR
    return NOT_ADOPTED


ITERS_COUNTS = re.compile(r"\(([^()]*)\)\s*$")


def iters_counts(text: str) -> Counter:
    """The counts of one ITERS_OK line (fp_iters_check.py: "all iteration counts OK (1000 single-option 3, N 130,
    N/2 25, decisions 70)")."""
    c: Counter = Counter()
    m = ITERS_COUNTS.search(text.strip())
    for part in (m.group(1).split(", ") if m and m.group(1) else []):
        key, _, v = part.rpartition(" ")
        if key and v.isdigit():
            c[key] += int(v)
    return c


def iters_status(run: Path) -> tuple[bool, Counter, list[str]]:
    """Foul Play's fixed-iteration check over a run's blocks: OK iff every block has ITERS_OK and none (nor the run)
    has ITERS_MISMATCH, and full-budget states ("N") were seen. Returns (ok, summed counts, problems)."""
    blocks = sorted(run.glob("arm-*/block-*"))
    total: Counter = Counter()
    bad = [] if blocks else ["no blocks"]
    if (run / "ITERS_MISMATCH").exists():
        bad.append("ITERS_MISMATCH: " + (run / "ITERS_MISMATCH").read_text(errors="replace").strip()[:300])
    for b in blocks:
        if (b / "ITERS_MISMATCH").exists():
            bad.append(f"{b.parent.name}/{b.name} ITERS_MISMATCH")
        elif not (b / "ITERS_OK").exists():
            bad.append(f"{b.parent.name}/{b.name} no ITERS_OK")
        else:
            total += iters_counts((b / "ITERS_OK").read_text(errors="replace"))
    if not bad and total["N"] == 0:
        bad.append("no full-budget state checked")
    return not bad, total, bad


def smoke(run: Path, games: int, fp: bool = False) -> tuple[bool, str]:
    """The smoke's mechanical checks (LA2-H's, plus the lookahead opponent's as in the scale gate's smoke). fp: the
    opponent is Foul Play at a fixed iteration count (no opponent search stats; its iteration check must pass)."""
    r = json.loads((run / "result.json").read_text())
    checks = {}
    for a in r["arms"]:
        k = a["arm"]
        ev = "\n".join(p.read_text(errors="replace") for p in run.glob(f"arm-{k}/block-*/evaluate.log"))
        checks[f"arm {k} {games} games"] = a["games"] == games
        checks[f"arm {k} no Traceback"] = "Traceback" not in ev
        sides = [(f"arm {k}", a["stats_a"], a["gate"])]
        if not fp:
            sides.append((f"arm {k} opponent", a["stats_b"], a["gate_opp"]))
        for tag, s, g in sides:
            checks[f"{tag} searched/leaves > 0"] = s.get("searched", 0) > 0 and s.get("leaves", 0) > 0
            checks[f"{tag} gate rows"] = g["decisions"] > 0
            checks[f"{tag} failed worlds 0"] = s.get("failed_worlds", 0) == 0
            checks[f"{tag} fallbacks <= max(2, 2%)"] = s.get("fallbacks", 0) <= max(2, 0.02 * s.get("decisions", 0))
    checks["one arm"] = len(r["arms"]) == 1
    checks["no HARNESS_FAIL"] = not (run / "HARNESS_FAIL").exists()
    if fp:
        checks["Foul Play iteration check"] = iters_status(run)[0]
    ok = all(checks.values())
    return ok, ("PASS " if ok else "FAIL ") + "; ".join(f"{k} {'ok' if v else 'NO'}" for k, v in checks.items())


def pct(x: float | None, digits: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}%}"


def verdict(run: Path, delta: str, label: str, bot_label: str, bot_delta: str = "0.005") -> str:
    """VERDICT.txt: line 1 = the reading (or VOID), then the arm's numbers."""
    r = json.loads((run / "result.json").read_text())
    arms = r["arms"]
    if len(arms) != 1:
        raise SystemExit(f"{run}: {len(arms)} arms (want 1)")
    a = arms[0]
    void = (run / "HARNESS_FAIL").exists() or bool(a["void"]) or a["pairs"] < MIN_PAIRS
    if void:
        head = (f"VOID: harness failure, > {VOID_FALLBACKS:.0%} fallbacks or < {MIN_PAIRS} pairs ({a['pairs']} pairs, "
                f"HARNESS_FAIL {'yes' if (run / 'HARNESS_FAIL').exists() else 'no'}, fallback void {bool(a['void'])}); "
                "replay with the same seeds after the fix")
        then = "no reading"
    else:
        rd = reading(a["lo"])
        rule = {BETTER: "> 50%", NON_INFERIOR: "> 47%, <= 50%", NOT_ADOPTED: "<= 47%"}[rd]
        then = (f"the benchmark bot becomes lookahead2 on {label} (δ* {delta})" if rd != NOT_ADOPTED
                else f"the benchmark bot stays lookahead2 on {bot_label} (δ {bot_delta})")
        head = (f"{rd}: lower 95% {pct(a['lo'], 3)} {rule}; lookahead2({label}, δ* {delta}) vs lookahead2({bot_label}, "
                f"δ {bot_delta}) {pct(a['mean'])} ± {pct(a['hw'])} [{pct(a['lo'])}, {pct(a['hi'])}], {a['pairs']} pairs")
    out = [head,
           f"  => {then}",
           f"  arm {a['arm']}: {pct(a['mean'])} ± {pct(a['hw'])} [{pct(a['lo'])}, {pct(a['hi'])}] ({a['pairs']} pairs, "
           f"{a['games']} games; won both {a['won_both']}, split {a['split']}, lost both {a['lost_both']}; incomplete "
           f"{a.get('incomplete_pairs')}); {a['seconds'] / 3600:.2f} h",
           f"  overrides: candidate {pct(a.get('override_rate'), 1)} of {a['gate']['decisions']} searched decisions, "
           f"benchmark bot {pct(a.get('override_rate_opp'), 1)} of {a['gate_opp']['decisions']}; fallbacks "
           f"{pct(a.get('fallback_rate_a'), 3)} / {pct(a.get('fallback_rate_b'), 3)}",
           f"  candidate {a['spec']}",
           f"  benchmark bot {r['opponent']}",
           f"  per block: " + ", ".join(f"{b['block']} {b['pairs']} pairs {(b.get('seconds') or 0) / 3600:.2f} h"
                                        for b in a.get("per_block", [])),
           f"  rule: lower 95% of the pair means > 50% {BETTER}; else > 47% {NON_INFERIOR}; else {NOT_ADOPTED}; "
           f"run {run}"]
    return "\n".join(out)


def fp200(run: Path, delta: str, label: str, iters: int, seed: str, voids: int = 0, voids_text: str = "") -> str:
    """FP200.txt (LA2-2030 stage 2, a measurement): line 1 = the new benchmark bot's pair-mean 95% interval vs Foul
    Play at a fixed iteration count, or VOID (HARNESS_FAIL, > 2% fallbacks, < 450 pairs, or Foul Play's iteration check
    not passed on every block)."""
    r = json.loads((run / "result.json").read_text())
    arms = r["arms"]
    if len(arms) != 1:
        raise SystemExit(f"{run}: {len(arms)} arms (want 1)")
    a = arms[0]
    iok, counts, bad = iters_status(run)
    hf = (run / "HARNESS_FAIL").exists()
    if hf or bool(a["void"]) or a["pairs"] < MIN_PAIRS or not iok:
        head = (f"VOID: harness failure, > {VOID_FALLBACKS:.0%} fallbacks, < {MIN_PAIRS} pairs or Foul Play's iteration "
                f"check not passed ({a['pairs']} pairs, HARNESS_FAIL {'yes' if hf else 'no'}, fallback void "
                f"{bool(a['void'])}, iteration check {'OK' if iok else '; '.join(bad)[:300]}); replay with the same seed "
                "after the fix")
    else:
        head = (f"LA2-FP200: lookahead2({label}, δ* {delta}, K 8, T 0.5) vs Foul Play gen9 at a FIXED {iters} iterations "
                f"per state: {pct(a['mean'])} ± {pct(a['hw'])} [{pct(a['lo'])}, {pct(a['hi'])}] ({a['pairs']} duplicate "
                f"pairs, {a['games']} games); measurement, no decision rule")
    return "\n".join([
        head,
        f"  arm {a['arm']}: {pct(a['mean'])} ± {pct(a['hw'])} [{pct(a['lo'])}, {pct(a['hi'])}] (pair means; {a['pairs']} "
        f"pairs, {a['games']} games; won both {a['won_both']}, split {a['split']}, lost both {a['lost_both']}; "
        f"incomplete {a.get('incomplete_pairs')}); {a['seconds'] / 3600:.2f} h",
        f"  Foul Play iteration check (wsp fp_iters_check.py, every block): {'OK' if iok else 'NOT OK'} ("
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")" + ("" if iok else f"; {'; '.join(bad)[:300]}"),
        f"  overrides: {pct(a.get('override_rate'), 1)} of {a['gate']['decisions']} searched decisions; fallbacks "
        f"{pct(a.get('fallback_rate_a'), 3)}",
        f"  bot {a['spec']}",
        f"  opponent {r['opponent']}; seed tag {seed} (block j: {seed}b<j>)",
        f"  infrastructure voids replayed: {voids}{voids_text}",
        "  per block: " + ", ".join(f"{b['block']} {b['pairs']} pairs {(b.get('seconds') or 0) / 3600:.2f} h"
                                    for b in a.get("per_block", [])),
        f"  run {run}",
    ])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("results", type=Path)
    g.add_argument("--md5")
    g.add_argument("--name", default="F1350")
    s = sub.add_parser("smoke")
    s.add_argument("run", type=Path)
    s.add_argument("--games", type=int, default=4)
    s.add_argument("--fp", action="store_true", help="the opponent is Foul Play at a fixed iteration count")
    rd = sub.add_parser("reading")
    rd.add_argument("lo", type=float)
    v = sub.add_parser("verdict")
    v.add_argument("run", type=Path)
    v.add_argument("--delta", required=True)
    v.add_argument("--label", default="G9q-cd@1355.04M")
    v.add_argument("--bot-label", default="G9q-cd@805.03M")
    v.add_argument("--bot-delta", default="0.005")
    f = sub.add_parser("fp200")
    f.add_argument("run", type=Path)
    f.add_argument("--delta", required=True)
    f.add_argument("--label", default="G9q-cd@2035.04M")
    f.add_argument("--iters", type=int, required=True)
    f.add_argument("--seed", required=True)
    f.add_argument("--voids", type=int, default=0)
    f.add_argument("--voids-text", default="")
    a = p.parse_args()
    if a.cmd == "gate":
        ok, why = gate(a.results, a.md5, a.name)
        print(("PLAY: " if ok else "WITHDRAWN: ") + why.replace("|", "/"))
        return 0 if ok else WITHDRAWN_EXIT
    if a.cmd == "smoke":
        print(smoke(a.run, a.games, a.fp)[1])
        return 0
    if a.cmd == "reading":
        print(reading(a.lo))
        return 0
    if a.cmd == "fp200":
        print(fp200(a.run, a.delta, a.label, a.iters, a.seed, a.voids, a.voids_text))
        return 0
    print(verdict(a.run, a.delta, a.label, a.bot_label, a.bot_delta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
