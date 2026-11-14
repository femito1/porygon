#!/bin/bash
# netv2 2026-11-14, G9t chain stage S1 (orchestrator-approved chain plan 15:2x; runs/g9t/PREREG.txt md5 e57b8ab4), in the
# data host (h2h-only step, outside refresh rounds):
#   rule 1 (25M harm guard, seed 9825; its h2h already runs as its own step -> runs/g9t/cmp/25M): wait for RESULT.txt,
#     apply: "G9p > G9s" (= "G9q > G9t") SUCCESS at the decision of record => STOP;
#   rule 3 (50M drift, seed 9875): raw G9t@50M (first pool snapshot >= 50M) vs raw G9t@25M (rule 1's snapshot) ->
#     runs/g9t/cmp/drift50: "G9p > G9s" (= "G9t@25M > G9t@50M") SUCCESS => STOP;
#   rule 4 (info, seed 9876): raw G9t@50M vs raw G9q runs/g9/20261109-021316-741849/G9q/pool/00050038810.pt -> info50.
# Each rule writes OUT/READING.txt (r--): "VERDICT STOP|CONTINUE|INFO". Rules 3 / 4 are skipped after a rule-1 STOP.
# The stop itself (park G9t) is S2's (it runs in G9t's job). Status lines: runs/g9t/chain/STATUS; a failure appends
# runs/g9t/chain/FAILED and ends the chain (S2 sees it).
#   setsid nohup srun --jobid=<data host> --overlap --ntasks=1 -c 8 --gres=gpu:1 --mem=8G -J nv2-g9t-s1 \
#     bash $C/scripts/netv2_g9t/chain-s1.sh > $C/logs/g9t-chain-s1-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; G=$C/runs/g9t; K=$G/chain; mkdir -p "$K"
TPOOL=$C/runs/g9/20261112-015035-810367/G9t/pool; Q50=$C/runs/g9/20261109-021316-741849/G9q/pool/00050038810.pt
st() { echo "$(date -u +%FT%T) S1 $*" | tee -a "$K/STATUS"; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) S1 $*" >> "$K/FAILED"; exit 1; }
[ ! -e "$K/FAILED" ] || { st "runs/g9t/chain/FAILED exists: not running"; exit 1; }
[ "$(md5sum < "$G/PREREG.txt" | cut -c1-8)" = e57b8ab4 ] || fail "PREREG md5 is not e57b8ab4"
dor() { sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$1/RESULT.txt"; }
waitres() {  # waitres DIR MINUTES: RESULT.txt complete (its last block written)
  local i; for i in $(seq 1 "$2"); do
    if grep -q "^--- all pairs played" "$1/RESULT.txt" 2>/dev/null; then sleep 30; return 0; fi; sleep 60; done; return 1; }
reading() {  # reading DIR RULE VERDICT TEXT
  { echo "G9t PREREG rule $2 (runs/g9t/PREREG.txt e57b8ab4), read mechanically by scripts/netv2_g9t/chain-s1.sh"
    echo "VERDICT $3"; echo "$4"; echo "RESULT.txt md5 $(md5sum < "$1/RESULT.txt" | cut -c1-8)"; grep -A1 '^DECISION OF RECORD' "$1/RESULT.txt"
    grep -A3 '^--- all pairs played' "$1/RESULT.txt"; } > "$1/READING.txt.tmp" && mv -f "$1/READING.txt.tmp" "$1/READING.txt" && chmod a-w "$1/READING.txt"; }
cmp_run() {  # cmp_run STEP SEED OUT S_DIRS P_DIRS
  cp "$C/scripts/netv2_cmp_step.sh" "/tmp/nv2-g9t-cmp-$$.sh" || return 1
  [ "$(md5sum < "/tmp/nv2-g9t-cmp-$$.sh" | cut -c1-8)" = a80c7da2 ] || { echo "netv2_cmp_step.sh md5 is not a80c7da2"; return 1; }
  CMP_WORKERS=7 timeout 7200 bash "/tmp/nv2-g9t-cmp-$$.sh" "$@"; local rc=$?; rm -f "/tmp/nv2-g9t-cmp-$$.sh"; return $rc; }

# --- rule 1 ---
D=$G/cmp/25M
st "rule 1: waiting for $D/RESULT.txt"
waitres "$D" 120 || fail "rule 1: no complete RESULT.txt in 2 h"
read -r x y <<< "$(dor "$D")"; [ -n "${y:-}" ] || fail "rule 1: no DECISION OF RECORD line"
grep -q "00025034161.pt" "$D/NETS.txt" && grep -q "00025038951.pt" "$D/NETS.txt" || fail "rule 1: NETS.txt does not name the PREREG snapshots"
if [ "$y" = 10 ]; then reading "$D" 1 STOP "\"G9q > G9t\" SUCCESS (exit 10) => stop G9t (reject lambda 0.5)"; st "rule 1 VERDICT STOP (G9q > G9t SUCCESS): G9t to be parked by S2"; st "S1 done (rules 3 / 4 skipped)"; exit 0; fi
reading "$D" 1 CONTINUE "\"G9q > G9t\" exit $y, \"G9t > G9q\" exit $x (10 SUCCESS, 20 FUTILITY, 0 undecided) => no stop; a G9t win at 25M is not evidence"
st "rule 1 VERDICT CONTINUE (G9t>G9q exit $x, G9q>G9t exit $y)"

# --- rule 3 ---
D=$G/cmp/drift50; mkdir -p "$D"
if [ ! -f "$D/g9s.pt" ] || [ ! -f "$D/g9p.pt" ]; then
  s=""; for f in "$TPOOL"/*.pt; do n=$(basename "$f" .pt); case $n in *[!0-9]*|"") continue ;; esac
    if [ "$((10#$n))" -ge 50000000 ]; then s=$f; break; fi; done  # glob order = numeric (zero-padded names)
  [ -n "$s" ] || fail "rule 3: no G9t pool snapshot >= 50M in $TPOOL"
  b=$G/cmp/25M/g9s.pt
  cp -p "$s" "$D/g9s.pt.tmp" && mv "$D/g9s.pt.tmp" "$D/g9s.pt" && cp -p "$b" "$D/g9p.pt.tmp" && mv "$D/g9p.pt.tmp" "$D/g9p.pt" || fail "rule 3: copy"
  printf 'G9s %s\nG9p %s (= rule 1 G9t@25M %s)\nseed %s\nmd5 g9s %s g9p %s\n' "$s" "$b" "$(sed -n 's/^G9s //p' "$G/cmp/25M/NETS.txt" | head -n 1)" 9875 \
    "$(md5sum < "$D/g9s.pt" | cut -c1-32)" "$(md5sum < "$D/g9p.pt" | cut -c1-32)" > "$D/NETS.txt"
fi
st "rule 3: h2h G9t@50M vs G9t@25M, seed 9875 ($(head -n 1 "$D/NETS.txt"))"
cmp_run 50000000 9875 "$D" "$TPOOL" "$G/cmp/25M" > "$D/run.log" 2>&1 || fail "rule 3: h2h exited $? (log $D/run.log)"
read -r x y <<< "$(dor "$D")"; [ -n "${y:-}" ] || fail "rule 3: no DECISION OF RECORD line"
if [ "$y" = 10 ]; then reading "$D" 3 STOP "\"G9t@25M > G9t@50M\" SUCCESS (exit 10) => stop G9t"; st "rule 3 VERDICT STOP (G9t@25M > G9t@50M SUCCESS): G9t to be parked by S2"
else reading "$D" 3 CONTINUE "\"G9t@25M > G9t@50M\" exit $y, \"G9t@50M > G9t@25M\" exit $x => no stop (informational otherwise)"; st "rule 3 VERDICT CONTINUE (50M>25M exit $x, 25M>50M exit $y)"; fi

# --- rule 4 (info) ---
D=$G/cmp/info50; mkdir -p "$D"
[ -s "$Q50" ] || fail "rule 4: no $Q50"
if [ ! -f "$D/g9s.pt" ] || [ ! -f "$D/g9p.pt" ]; then
  cp -p "$G/cmp/drift50/g9s.pt" "$D/g9s.pt.tmp" && mv "$D/g9s.pt.tmp" "$D/g9s.pt" && cp -p "$Q50" "$D/g9p.pt.tmp" && mv "$D/g9p.pt.tmp" "$D/g9p.pt" || fail "rule 4: copy"
  printf 'G9s %s (= rule 3 G9t@50M)\nG9p %s\nseed %s\nmd5 g9s %s g9p %s\n' "$(sed -n 's/^G9s //p' "$G/cmp/drift50/NETS.txt" | head -n 1)" "$Q50" 9876 \
    "$(md5sum < "$D/g9s.pt" | cut -c1-32)" "$(md5sum < "$D/g9p.pt" | cut -c1-32)" > "$D/NETS.txt"
fi
st "rule 4 (info): h2h G9t@50M vs G9q@50M, seed 9876"
cmp_run 50000000 9876 "$D" "$G/cmp/drift50" "$D" > "$D/run.log" 2>&1 || fail "rule 4: h2h exited $? (log $D/run.log)"
read -r x y <<< "$(dor "$D")"; [ -n "${y:-}" ] || fail "rule 4: no DECISION OF RECORD line"
reading "$D" 4 INFO "informational (no rule): \"G9t@50M > G9q@50M\" exit $x, \"G9q@50M > G9t@50M\" exit $y"
st "rule 4 INFO (G9t>G9q exit $x, G9q>G9t exit $y)"
st "S1 done"
