"""Citation check for council outputs: every result-row id a member cites must exist in results.csv, and a number
written right after an id (within ~60 characters) should match that row's estimate / CI bounds (±0.15). Prints
UNKNOWN ids and NUMBER mismatches per file; the orchestrator adjudicates (a mismatch can be a legitimate derived
number). Usage: python3 verify_citations.py results.csv FILE.md...
"""
import csv, re, sys

rows = {r["id"]: r for r in csv.DictReader(open(sys.argv[1], newline=""))}
idre = re.compile(r"\b([a-z][a-z0-9]{0,6}-[a-z0-9]{1,8}[a-z]?)\b")
numre = re.compile(r"(?<![\d.])(\d{1,3}\.\d{1,3})(?![\d.])")
looks = re.compile(r"^(?:v1|v2g4|v2g9|g9q|g9s|ns|fp4|fpr|fpc|s4|bc|dg|bn|cal|la2|fpc)-")
bad = 0
for path in sys.argv[2:]:
    text = open(path, encoding="utf-8", errors="replace").read()
    seen, unknown, mism = set(), set(), []
    for m in idre.finditer(text):
        i = m.group(1)
        if i in rows:
            seen.add(i)
            r = rows[i]
            vals = []
            for k in ("estimate_pct", "ci_low", "ci_high"):
                try:
                    vals.append(float(r[k]))
                except (TypeError, ValueError):
                    pass
            window = text[m.end():m.end() + 60]
            nxt = re.search(r"\b[a-z][a-z0-9]{0,6}-[a-z0-9]{1,8}\b", window)
            if nxt:
                window = window[:nxt.start()]
            for n in numre.findall(window)[:3]:
                if vals and not any(abs(float(n) - v) <= 0.15 for v in vals):
                    mism.append(f"{i}: text {n} vs row {r['estimate_pct']} [{r['ci_low']}, {r['ci_high']}]")
        elif looks.match(i):
            unknown.add(i)
    print(f"== {path}: {len(seen)} known ids cited")
    for u in sorted(unknown):
        print(f"  UNKNOWN id {u}")
    for x in mism:
        print(f"  NUMBER? {x}")
    bad += len(unknown) + len(mism)
print(f"total flags: {bad}")
