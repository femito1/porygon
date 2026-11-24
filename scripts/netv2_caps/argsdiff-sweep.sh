#!/bin/bash
# argsdiff sweep for netv2.env v22 (the step caps): OLD -> NEW over every g9 mode and every r1 mode (R1b-F's g4r1bf jobs
# are in every case: the run has no MODE), both trees (current, now), plus the cases with the state markers that make
# MODE g9p4 / r1 MODE g9q reach their launch args (ARGSDIFF_TOUCH), the committed-swap tree, and today's cluster (r1
# g9q, g9 g9p4, both started). The full old / new outputs are kept per case (ARGSDIFF_KEEP) for caps-check.sh.
#   bash scripts/netv2_caps/argsdiff-sweep.sh OLD NEW OUTDIR && bash scripts/netv2_caps/caps-check.sh OUTDIR
set -u
WT=$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")
O=$(readlink -f "${1:?OLD}"); N=$(readlink -f "${2:?NEW}"); OUT=${3:?OUTDIR}; R=${ARGSDIFF_ROOT:-$OUT.tree}
export NV2_PARK_HOLD_ROUNDS=1 NV2_G9T_HOLD_ROUNDS=1  # refused first starts in the fake tree exit 3 instead of spinning
rm -rf "$OUT"; mkdir -p "$OUT"
one() {  # one NAME WHAT [VAR=VALUE...]: one argsdiff run, its summary + kept outputs in OUT/NAME/
  local name=$1 what=$2; shift 2
  mkdir -p "$OUT/$name"
  env ARGSDIFF_NO_AS=1 ARGSDIFF_ROOT="$R" ARGSDIFF_KEEP="$OUT/$name" "$@" bash "$WT/scripts/netv2_swap/argsdiff.sh" "$O" "$N" "$what" > "$OUT/$name/summary.txt" 2>&1
  echo "$name: $(head -n 1 "$OUT/$name/summary.txt")"
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
