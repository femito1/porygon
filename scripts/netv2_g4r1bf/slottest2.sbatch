#!/bin/bash
# netv2 2026-11-14: R1b-F prior slot test #2 as ONE cluster-side chain job (runs/g4r1bf/SLOTTEST2-PREREG.txt):
# smoke (8 games, read mechanically) -> main run (1000 duplicate pairs, fixed n) -> VERDICT.txt (r--) -> one ledger
# RELEASE line in $C/LEDGER-OUTBOX.md. Any failure: STATUS + FAILED + a RELEASE line naming the failed stage.
# Submit HELD from $C (the harness uses SLURM_SUBMIT_DIR); the orchestrator releases it:
#   cd $C && sbatch --hold scripts/netv2_g4r1bf/slottest2.sbatch
#SBATCH --job-name=ws-netv2-slottest2
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=31:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; cd "$C" || exit 1
D=$C/runs/g4r1bf/slottest2; mkdir -p "$D"
PRE=$C/runs/g4r1bf/SLOTTEST2-PREREG.txt; PRE_MD5=${PRE_MD5:-5ee05392}
P=$C/runs/g4r1bf/prior-R1bF-300M.pt; P_MD5=${P_MD5:-c3ec9433}
V=/vol/bitbucket/fr1226/gloria/runs/20261025-195826-301040/final_model.zip; V_MD5=51488bda
H=$C/scripts/netv2_vs_search.sbatch; H_MD5=7d74e0ad; SEED=9625
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
J=${SLURM_JOB_ID:?}
st() { echo "$(date -u +%FT%T) $*" | tee -a "$D/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$D/FAILED"
  outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | R1b-F prior slot test #2 FAILED ($*); no reading; records runs/g4r1bf/slottest2/; slot FREE"; exit 1; }
md5of() { md5sum < "$1" | cut -c1-8; }
[ ! -e "$D/FAILED" ] || { st "FAILED exists: not running"; exit 1; }
st "job $J on $(hostname -s) ($SLURM_CPUS_ON_NODE CPUs), start"
[ -f "$PRE" ] && [ ! -w "$PRE" ] && [ "$(md5of "$PRE")" = "$PRE_MD5" ] || fail "PREREG missing / writable / md5 not $PRE_MD5"
[ "$(md5of "$P")" = "$P_MD5" ] && [ ! -w "$P" ] || fail "prior $P md5 not $P_MD5 or writable"
[ "$(md5of "$V")" = "$V_MD5" ] || fail "v1@500M md5"
[ "$(md5of "$H")" = "$H_MD5" ] || fail "harness md5"
[ "${SLURM_CPUS_ON_NODE:-0}" -ge 24 ] || fail "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 24)"
newest() { ls -td "$C"/runs/vs-search-*-"$J" 2>/dev/null | head -n 1; }
# --- smoke ---
st "smoke: 8 games, seed smoke$SEED"
bash "$H" "search:$P,workers=10" 8 --gen 4 --search "search:$V,workers=10" --shards 1 --concurrency 2 --seed "smoke$SEED" > "$D/smoke.log" 2>&1 || fail "smoke harness exited $?"
S=$(newest); [ -n "$S" ] && [ -f "$S/result.txt" ] || fail "smoke: no result.txt"
sm=$("$PY" -I -c "
import json, sys
s = json.load(open(sys.argv[1] + '/shard-0/summary.json')); a, b = s['search_stats_a'], s['search_stats_b']
ok = s['games'] == 8 and a['searched'] > 0 and a['fallbacks'] <= b['fallbacks'] + 2 and a['failed_samples'] <= b['failed_samples'] + 2
print(('PASS' if ok else 'FAIL'), 'games', s['games'], 'A searched %d/%d fallbacks %d failed %d' % (a['searched'], a['decisions'], a['fallbacks'], a['failed_samples']),
      '| B searched %d/%d fallbacks %d failed %d' % (b['searched'], b['decisions'], b['fallbacks'], b['failed_samples']))" "$S" 2>&1)
grep -q Traceback "$S"/shard-*/evaluate.log && sm="FAIL (Traceback in evaluate.log) $sm"
{ echo "smoke $S"; echo "$sm"; head -n 1 "$S/result.txt"; } > "$D/SMOKE.txt"
st "smoke: $sm"
case $sm in PASS*) ;; *) fail "smoke FAILED: $sm" ;; esac
# --- main run ---
st "main: 2000 games (1000 duplicate pairs), seed $SEED"
bash "$H" "search:$P,workers=10" 2000 --gen 4 --search "search:$V,workers=10" --shards 1 --concurrency 2 --seed "$SEED" > "$D/main.log" 2>&1 || fail "main harness exited $?"
M=$(newest); [ -n "$M" ] && [ "$M" != "$S" ] && [ -f "$M/result.txt" ] || fail "main: no result.txt"
l1=$(head -n 1 "$M/result.txt")
grep -q -- "--seed $SEED " "$M/manifest.txt" || fail "main: manifest lacks --seed $SEED"
pairs=$(grep -oE '\(([0-9]+) duplicate pairs' <<< "$l1" | tr -dc 0-9)
[ "${pairs:-0}" -ge 1000 ] || fail "main VOID: ${pairs:-?} pairs (< 1000)"
p=$(grep -oE ': [0-9.]+%' <<< "$l1" | head -n 1 | tr -dc '0-9.'); h=$(grep -oE '± [0-9.]+%' <<< "$l1" | head -n 1 | tr -dc '0-9.')
[ -n "$p" ] && [ -n "$h" ] || fail "main: cannot parse '$l1'"
lo=$(awk -v p="$p" -v h="$h" 'BEGIN { printf "%.1f", p - h }'); hi=$(awk -v p="$p" -v h="$h" 'BEGIN { printf "%.1f", p + h }')
if awk -v x="$lo" 'BEGIN { exit !(x > 50) }'; then v="BETTER (lo $lo > 50) => propose the separately pre-registered gen4 benchmark of A vs Foul Play (fixed n, vs the 59.2% standing); the headline gen4 bot changes only after that run"
elif awk -v x="$hi" 'BEGIN { exit !(x < 50) }'; then v="WORSE (hi $hi < 50) => keep the v1@500M prior"
else v="INCONCLUSIVE ([$lo, $hi] contains 50) => keep the v1@500M prior; no extension of n"; fi
info=$("$PY" -I -c "
import json, glob, sys
for side in 'ab':
    t = {}
    for f in glob.glob(sys.argv[1] + '/shard-*/summary.json'):
        for k, x in json.load(open(f))['search_stats_' + side].items():
            if isinstance(x, (int, float)): t[k] = t.get(k, 0) + x
    d = max(t.get('decisions', 0), 1)
    print(side.upper(), 'decisions %d searched %d agree-with-net %.1f%% fallbacks %d failed samples %d' % (t.get('decisions', 0), t.get('searched', 0), 100 * t.get('agree', 0) / d, t.get('fallbacks', 0), t.get('failed_samples', 0)))" "$M" 2>&1)
{ echo "R1b-F prior slot test #2 (runs/g4r1bf/SLOTTEST2-PREREG.txt md5 $PRE_MD5), read mechanically by scripts/netv2_g4r1bf/slottest2.sbatch"
  echo "A = search with the R1b-F@300M prior ($P, md5 $P_MD5) vs B = search with the v1@500M prior; seed $SEED; job $J"
  echo "VERDICT: $v"; echo "p $p%, 95% [$lo, $hi]"; echo "--- $M/result.txt (md5 $(md5of "$M/result.txt"))"; cat "$M/result.txt"
  echo "--- informational"; echo "$info"; echo "--- smoke"; cat "$D/SMOKE.txt"; } > "$D/VERDICT.txt.tmp" && mv -f "$D/VERDICT.txt.tmp" "$D/VERDICT.txt" && chmod a-w "$D/VERDICT.txt"
st "VERDICT: $v (p $p, [$lo, $hi])"
outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | R1b-F prior slot test #2 done: search with R1b-F@300M prior vs v1@500M prior $p% ± $h% [$lo, $hi] ($pairs pairs, seed $SEED) => $(cut -d' ' -f1 <<< "$v"); verdict runs/g4r1bf/slottest2/VERDICT.txt; result $M/result.txt; slot FREE"
st "chain done (RELEASE line in $C/LEDGER-OUTBOX.md)"
