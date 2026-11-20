"""The mechanical parts of an LA2-H-style adoption test (lookahead2_adopt.sbatch; LA2-1350 = docs/netsearch/
LA2-1350-PREREG.md): the gate on netv2's confirmation file, the smoke's checks, the pre-registered reading and the
VERDICT text. Pure functions of files, standard library only (the cluster venv or a laptop python3).

    python lookahead2_adopt.py gate RESULTS.txt [--md5 6cf40c0c]   # exit 0 = PLAY, 3 = WITHDRAWN (one line out)
    python lookahead2_adopt.py smoke RUN [--games 4]                # one line: PASS ... / FAIL ...
    python lookahead2_adopt.py reading LO                           # the reading for a lower 95% bound (fraction)
    python lookahead2_adopt.py verdict RUN --delta D [--label ...]   # VERDICT.txt on stdout (line 1 = the reading)
"""
import argparse
import json
import math
import sys
from pathlib import Path

GATE_OK = "CONFIRMED (lower 95% bound > 50%)"  # netv2 conf1350.sbatch reading(): the only confirmed wording
WITHDRAWN_EXIT = 3
BETTER = "ADOPTED (BETTER)"
NON_INFERIOR = "ADOPTED (NON-INFERIOR, margin 3 points)"
NOT_ADOPTED = "NOT ADOPTED"
LO_BETTER, LO_NON_INFERIOR = 0.50, 0.47  # lower 95% bound of the pair means, strict >
MIN_PAIRS, VOID_FALLBACKS = 450, 0.02  # the LA2-H void rule


def gate(path: Path, md5: str | None = None) -> tuple[bool, str]:
    """PLAY iff line 2 of netv2's RESULTS.txt contains GATE_OK (and not "NOT CONFIRMED"), and line 1 names the net's
    md5 when one is given. Anything else (NOT CONFIRMED, FAILED, missing, unreadable) withdraws, with the reason."""
    if not path.is_file():
        return False, f"{path} missing (F1350 wrote no RESULTS.txt)"
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        return False, f"{path} has no line 2"
    l1, l2 = lines[0].strip(), lines[1].strip()
    if "NOT CONFIRMED" in l2:
        return False, f"F1350 NOT CONFIRMED: {l2}"
    if GATE_OK not in l2:
        why = "FAILED" if "FAILED" in l2 else "no CONFIRMED reading"
        return False, f"F1350 {why}: {l2}"
    if md5 and f"md5 {md5}" not in l1:
        return False, f"F1350 line 1 does not name the net md5 {md5}: {l1}"
    return True, f"F1350 CONFIRMED: {l2}"


def reading(lo: float) -> str:
    """The pre-registered reading of the candidate's lower 95% bound (a fraction): > 50% BETTER, > 47% NON-INFERIOR,
    else NOT ADOPTED. NaN (no pairs) reads NOT ADOPTED; the void rule catches it first."""
    if lo > LO_BETTER:
        return BETTER
    if lo > LO_NON_INFERIOR:
        return NON_INFERIOR
    return NOT_ADOPTED


def smoke(run: Path, games: int) -> tuple[bool, str]:
    """The smoke's mechanical checks (LA2-H's, plus the lookahead opponent's as in the scale gate's smoke)."""
    r = json.loads((run / "result.json").read_text())
    checks = {}
    for a in r["arms"]:
        k = a["arm"]
        ev = "\n".join(p.read_text(errors="replace") for p in run.glob(f"arm-{k}/block-*/evaluate.log"))
        checks[f"arm {k} {games} games"] = a["games"] == games
        checks[f"arm {k} no Traceback"] = "Traceback" not in ev
        for tag, s, g in ((f"arm {k}", a["stats_a"], a["gate"]), (f"arm {k} opponent", a["stats_b"], a["gate_opp"])):
            checks[f"{tag} searched/leaves > 0"] = s.get("searched", 0) > 0 and s.get("leaves", 0) > 0
            checks[f"{tag} gate rows"] = g["decisions"] > 0
            checks[f"{tag} failed worlds 0"] = s.get("failed_worlds", 0) == 0
            checks[f"{tag} fallbacks <= max(2, 2%)"] = s.get("fallbacks", 0) <= max(2, 0.02 * s.get("decisions", 0))
    checks["one arm"] = len(r["arms"]) == 1
    checks["no HARNESS_FAIL"] = not (run / "HARNESS_FAIL").exists()
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("results", type=Path)
    g.add_argument("--md5")
    s = sub.add_parser("smoke")
    s.add_argument("run", type=Path)
    s.add_argument("--games", type=int, default=4)
    rd = sub.add_parser("reading")
    rd.add_argument("lo", type=float)
    v = sub.add_parser("verdict")
    v.add_argument("run", type=Path)
    v.add_argument("--delta", required=True)
    v.add_argument("--label", default="G9q-cd@1355.04M")
    v.add_argument("--bot-label", default="G9q-cd@805.03M")
    v.add_argument("--bot-delta", default="0.005")
    a = p.parse_args()
    if a.cmd == "gate":
        ok, why = gate(a.results, a.md5)
        print(("PLAY: " if ok else "WITHDRAWN: ") + why.replace("|", "/"))
        return 0 if ok else WITHDRAWN_EXIT
    if a.cmd == "smoke":
        print(smoke(a.run, a.games)[1])
        return 0
    if a.cmd == "reading":
        print(reading(a.lo))
        return 0
    print(verdict(a.run, a.delta, a.label, a.bot_label, a.bot_delta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
