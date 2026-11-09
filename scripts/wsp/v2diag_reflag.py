"""Recompute the flags column of an existing tracker table (and its flags.txt) from the per-step outputs with the
current v2diag_track.compute_flags, e.g. after a flag-rule change. Every other column is kept; the old table.md and
flags.txt are kept as *.pre-reflag-<UTC stamp>. Pass the run's conf options, so the rows get exactly the tracker's rules:

    bash -c 'source v2diag/conf/g9s.conf; python scripts/wsp/v2diag_reflag.py --dir $D --ref $REFNAME "${TRACK_ARGS[@]}"'
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from v2diag_track import add_flag_args, compute_flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="tracker dir: table.md, flags.txt, out/<step>/decay.json")
    ap.add_argument("--boundary", default="", help="accepted and ignored (the conf's TRACK_ARGS may carry it)")
    add_flag_args(ap)
    a = ap.parse_args()
    D = Path(a.dir)
    steps = {f"{int(o.name) / 1e6:.2f}M": int(o.name) for o in (D / "out").iterdir()
             if o.name.isdigit() and (o / "done").exists()}
    stamp = time.strftime("%Y%m%d-%H%M", time.gmtime())
    out, flagged, changed, rows = [], [], [], 0
    for ln in (D / "table.md").read_text().splitlines(keepends=True):
        parts = ln.split("|")
        key = parts[1].strip() if len(parts) > 13 else ""
        if key in steps:
            st = steps[key]
            dec = json.load(open(D / "out" / str(st) / "decay.json"))
            fl = compute_flags(dec[f"s_{st}.pt"], dec.get(a.ref) if a.ref else None, a, st)
            new = "; ".join(fl) or "—"
            if parts[12].strip() != new:
                changed.append(f"{key}: {parts[12].strip()} -> {new}")
            parts[12] = f" {new} "
            ln = "|".join(parts)
            rows += 1
            if fl and not fl[0].startswith("(suppressed"):
                w = parts[13].strip()  # MM-DD HH:MM, when the row was written
                flagged.append(f"2026-{w[:5]}T{w[6:]}Z step {st:,}: {'; '.join(fl)}\n")
        out.append(ln)
    for f in ("table.md", "flags.txt"):
        if (D / f).exists():
            shutil.copy2(D / f, D / f"{f}.pre-reflag-{stamp}")
    tmp = D / "table.md.tmp"
    tmp.write_text("".join(out))
    tmp.replace(D / "table.md")
    tmp = D / "flags.txt.tmp"
    tmp.write_text(f"# re-flagged {stamp} UTC by v2diag_reflag.py (old flags: flags.txt.pre-reflag-{stamp})\n" + "".join(flagged))
    tmp.replace(D / "flags.txt")
    print(f"{D}: {rows} rows re-flagged, {len(changed)} changed, {len(flagged)} flagged now")
    for c in changed:
        print("  " + c)


if __name__ == "__main__":
    main()
