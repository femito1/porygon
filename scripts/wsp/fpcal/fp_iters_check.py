"""fpcal iteration check: every per-state MCTS iteration count Foul Play logs must be the fixed budget.

    python fp_iters_check.py N LOG [LOG ...]      exit 0 = all counts OK, 1 = a mismatch (printed), 2 = usage

Reads Foul Play's INFO logs (one instance per file, decisions in order). A decision is the block from
"Sampling K battles at X ms each" to "Choice: ...". Each "Iterations i: V" line in it must be
  - N (1000 ms states) or N/2 rounded up to 1000 (500 ms early states), or
  - exactly 1000 on a state whose side one has a single option, recognised by that state's "Policy i: <move>
    visited 100.0%" line (with >= 2 options every option gets visited, so 100.0% cannot occur at 1000 visits):
    poke-engine's binding replaces the iteration budget by 100 when side one has <= 1 option ("force a quick exit")
    and its loop runs iterations in batches of 1000, so such a state always reports 1000 visits, in time-based runs
    as well (job 1173380's fp0.log, time-based: 168 "Iterations i: 1000" lines = 84 states, each logged by the worker
    and again by the main process; every one of them is a single-option state).
Why this exists: the first fpcal job (1173956) voided both arms because its check rejected those forced states;
the fixed-iteration launcher itself had taken effect (its fp0.logs show only N, N/2 and these 1000s).
An unfinished trailing decision (the instance is still searching) is not checked. Prints one summary line.
"""
import re
import sys
from collections import Counter

SAMPLING = re.compile(r"Sampling (\d+) battles at (\d+)ms each")
ITERS = re.compile(r"Iterations (\d+): (\d+)\s*$")
POLICY = re.compile(r"Policy (\d+): .* visited ([0-9.]+)% ")
CHOICE = re.compile(r"\bChoice: ")
FORCED = 1000


def check(n: int, paths: list[str]) -> tuple[list[str], Counter]:
    half = (n // 2 + 999) // 1000 * 1000
    bad: list[str] = []
    seen: Counter = Counter()
    for path in paths:
        try:
            lines = open(path, errors="replace").read().splitlines()
        except FileNotFoundError:
            continue
        block = None  # [(state index, iterations)] of the current decision; None outside a decision
        visited: dict[int, float] = {}  # state index -> % of root visits on side one's most visited option
        for line in lines:
            if SAMPLING.search(line):
                block, visited = [], {}
                continue
            m = ITERS.search(line)
            if m:
                if block is None:  # an iteration line outside a decision block: report it rather than skip it
                    bad.append(f"{path}: outside a decision: {line.strip()}")
                else:
                    block.append((int(m.group(1)), int(m.group(2))))
                continue
            if block is None:
                continue
            m = POLICY.search(line)
            if m:
                visited[int(m.group(1))] = float(m.group(2))
                continue
            if CHOICE.search(line):
                for i, v in block:
                    if v == n or v == half:
                        seen["N" if v == n else "N/2"] += 1
                    elif v == FORCED and visited.get(i) == 100.0:
                        seen["1000 single-option"] += 1
                    else:
                        bad.append(f"{path}: Iterations {i}: {v} (state visited {visited.get(i)}%)")
                seen["decisions"] += 1
                block = None
    return bad, seen


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(2)
    n = int(sys.argv[1])
    bad, seen = check(n, sys.argv[2:])
    summary = ", ".join(f"{k} {v}" for k, v in sorted(seen.items()))
    if bad:
        print(f"expected {n} or {(n // 2 + 999) // 1000 * 1000} (or 1000 on a single-option state), saw: "
              + " | ".join(bad[:3]) + f" [{len(bad)} bad; {summary}]")
        sys.exit(1)
    print(f"all iteration counts OK ({summary})")


if __name__ == "__main__":
    main()
