#!/bin/bash
# argsdiff sweep for netv2.env v23 (council P1, g9 MODE p1tera): OLD (v22) -> NEW (v23) over every g9 mode and every r1
# mode (R1b-F's g4r1bf jobs are in every case), both trees (current, now), the state-marker cases of v22's sweep, the
# committed-swap tree and today's cluster (r1 g9q, g9 g9p4, both started). Every one must print "identical": v23 changes
# nothing outside MODE p1tera (which v22 does not know; not part of the comparison).
#   bash scripts/netv2_p1tera/argsdiff-sweep.sh OLD NEW OUTDIR
set -u
WT=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")
O=$(readlink -f "${1:?OLD}"); N=$(readlink -f "${2:?NEW}"); OUT=${3:?OUTDIR}; R=${ARGSDIFF_ROOT:-$OUT.tree}
export NV2_PARK_HOLD_ROUNDS=1 NV2_G9T_HOLD_ROUNDS=1
rm -rf "$OUT"; mkdir -p "$OUT"; n=0; same=0
one() {
  local name=$1 what=$2; shift 2
  env ARGSDIFF_NO_AS=1 ARGSDIFF_ROOT="$R" "$@" bash "$WT/scripts/netv2_swap/argsdiff.sh" "$O" "$N" "$what" > "$OUT/$name.txt" 2>&1
  n=$((n + 1)); if grep -q "^identical" "$OUT/$name.txt"; then same=$((same + 1)); else echo "DIFF $name: $(head -n 3 "$OUT/$name.txt" | tr '\n' ' ' | cut -c1-300)"; fi
}
for what in current now; do
  for m in g9 g9p g9s g9q g9p2 g9r g9t g9q2 g9qbf g9qent g9p4 g9e; do one "$what-g9-$m" "$what" ARGSDIFF_G9_MODE=$m; done
  one "$what-g9-g9p4-started" "$what" ARGSDIFF_G9_MODE=g9p4 ARGSDIFF_TOUCH=runs/g9p4/G9P4_STARTED
  for m in r1b g9 g9s g9p g9q g9t; do one "$what-r1-$m" "$what" ARGSDIFF_R1_MODE=$m; done
  one "$what-r1-g9q-started" "$what" ARGSDIFF_R1_MODE=g9q ARGSDIFF_TOUCH=runs/r1/G9Q_A30_STARTED
done
one swapped swapped
one now-today now ARGSDIFF_R1_MODE=g9q ARGSDIFF_G9_MODE=g9p4 "ARGSDIFF_TOUCH=runs/g9p4/G9P4_STARTED runs/r1/G9Q_A30_STARTED"
rm -rf "$R" "$R".*
echo "argsdiff sweep v22 -> v23: $same / $n cases identical"
[ "$same" = "$n" ]
