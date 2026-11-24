#!/bin/bash
# netv2 2026-11-24: council P1 VERDICT chain (council/p1-tera/PREREG.txt): waits for council/p1-tera/ARMS_DONE (both cooled
# finals, published r-- by netv2.env v23's MODE p1tera on the A40), then plays E-cd vs C-cd in-process (spd_h2h_tera.py =
# wsp's spd_h2h ee76994 + per-side encoder routing, md5 pinned; raw T 0.5 both; gen9randombattle; November sets; 4,000
# duplicate pairs = 10 shards of 400; pair seed 9241103; 5 workers, cuda) as an srun --overlap step in the RUNNING
# ws-data-host job with the most time left (>= 1.5 h; R2030 pattern: shards already complete are kept, so a host that ends
# mid-h2h costs one rerun in the next host), merges (pair-mean 95% interval) and reads the pre-registered rule mechanically
# => council/p1-tera/VERDICT.txt (r--): line 1 GO | REPLICATE | PARK, line 2 the pair-mean and its 95% CI. Any failure
# writes council/p1-tera/FAILED (one line) and stops. This chain is a 1-CPU step in data's host job:
#   setsid nohup srun --jobid=<data host> --overlap --ntasks=1 -c 1 --mem=1G -J nv2-p1t-verdict \
#     bash -c 'cp $C/scripts/netv2_p1tera/verdict-chain.sh /tmp/nv2-p1tv-$$.sh && exec bash /tmp/nv2-p1tv-$$.sh' \
#     > $C/logs/p1t-verdict-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=${MOCK_C:-/vol/bitbucket/fr1226/porygon-ws-netv2}; P=$C/council/p1-tera; K=$P/h2h; mkdir -p "$K"
PRE_MD5=088a5650; H2H_MD5=669386f3; SEED=9241103; NSH=10; SP=400; WORKERS=5; T=$C/tree-2e03cc6
G=/vol/bitbucket/fr1226/gloria; PY=$G/.venv/bin/python; NODE=$G/.node/bin/node; PSB=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
WAIT_S=172800; POLL_S=60; HOST_WAIT_S=86400; MIN_LEFT=5400; H2H_LIMIT=14400; U=fr1226; SRUN=srun
if [ -n "${MOCK_C:-}" ]; then  # laptop mock only
  PRE_MD5=$MOCK_PRE_MD5; H2H_MD5=$MOCK_H2H_MD5; WAIT_S=$MOCK_WAIT_S; POLL_S=1; HOST_WAIT_S=$MOCK_WAIT_S; U=$(id -un); PY=python3; SRUN=$MOCK_SRUN; T=$MOCK_T
fi
st() { echo "$(date -u +%FT%T) $*" | tee -a "$K/STATUS" >&2; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) verdict chain: $*" >> "$P/FAILED"; exit 1; }
pick_host() {  # the RUNNING ws-data-host job with the most time left, if >= MIN_LEFT s
  local now best="" bl=0 j e l
  now=$(date +%s)
  while read -r j e; do
    [ -n "$j" ] || continue
    l=$(( $(date -d "$e" +%s 2>/dev/null || echo 0) - now ))
    if [ "$l" -ge "$MIN_LEFT" ] && [ "$l" -gt "$bl" ]; then best=$j; bl=$l; fi
  done < <(squeue -h -u "$U" -n ws-data-host -t RUNNING -o "%i %e" 2>/dev/null)
  echo "$best"
}
running() { [ "$(squeue -h -j "$1" -o %T 2>/dev/null | head -n 1)" = RUNNING ]; }
if [ -e "$P/VERDICT.txt" ]; then st "council/p1-tera/VERDICT.txt exists: nothing to do"; exit 0; fi
[ ! -e "$P/FAILED" ] || { st "council/p1-tera/FAILED exists ($(tail -n 1 "$P/FAILED")): not running"; exit 1; }
if [ ! -s "$P/PREREG.txt" ] || [ -w "$P/PREREG.txt" ] || [ "$(md5sum < "$P/PREREG.txt" | cut -c1-8)" != "$PRE_MD5" ]; then fail "no read-only council/p1-tera/PREREG.txt with md5 $PRE_MD5"; fi
trap 'fail "killed by SIGTERM (host job ended or the chain step was cancelled): rerun the same pre-registration"' TERM
T0=$(date +%s); st "chain start on $(hostname -s) (job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}); waiting <= $((WAIT_S / 3600)) h for council/p1-tera/ARMS_DONE"
while [ ! -f "$P/ARMS_DONE" ]; do
  if [ -e "$P/FAILED" ]; then st "arms FAILED ($(tail -n 1 "$P/FAILED")): no verdict"; exit 1; fi
  if [ $(( $(date +%s) - T0 )) -ge "$WAIT_S" ]; then fail "no ARMS_DONE within $((WAIT_S / 3600)) h"; fi
  sleep "$POLL_S"
done
declare -A NET MD
for arm in E C; do
  read -r f m s < "$P/$arm/FINAL" || fail "no council/p1-tera/$arm/FINAL"
  [ -f "$f" ] && [ ! -w "$f" ] || fail "arm $arm final $f missing or writable"
  [ "$(md5sum < "$f" | cut -c1-8)" = "$m" ] || fail "arm $arm final $f md5 is not $m"
  case $f in "$P/$arm/final/"*.pt) ;; *) fail "arm $arm final $f is not in council/p1-tera/$arm/final/" ;; esac
  NET[$arm]=$f; MD[$arm]=$m
done
grep -q "^${NET[E]} " "$P/ARMS_DONE" && grep -q "^${NET[C]} " "$P/ARMS_DONE" || fail "ARMS_DONE does not name the two finals"
H=$K/spd_h2h_tera.py
if [ ! -f "$H" ]; then cp "$C/scripts/netv2_p1tera/spd_h2h_tera.py" "$H.tmp" && mv -f "$H.tmp" "$H" && chmod a-w "$H" || fail "could not copy spd_h2h_tera.py"; fi
[ "$(md5sum < "$H" | cut -c1-8)" = "$H2H_MD5" ] || fail "spd_h2h_tera.py md5 is not $H2H_MD5"
printf 'A E-cd %s %s\nB C-cd %s %s\nseed %s pairs %s shards %sx%s T 0.5 gen9randombattle PS %s\n' "${NET[E]}" "${MD[E]}" "${NET[C]}" "${MD[C]}" \
  "$SEED" "$((NSH * SP))" "$NSH" "$SP" "$PSB" > "$K/NETS.txt"
st "A = E-cd ${NET[E]} (${MD[E]}), B = C-cd ${NET[C]} (${MD[C]}); seed $SEED, $NSH x $SP pairs, $WORKERS workers"
cat > "$K/play.sh" <<PLAY
#!/bin/bash
# one pass over the shards not yet complete (run inside the data host)
set -o pipefail
export SPD_FASTENV_JS=$T/fastenv/js PORYGON_PS=$PSB PORYGON_NODE=$NODE OMP_NUM_THREADS=1
for j in \$(seq 0 $((NSH - 1))); do
  [ -f "$K/shards/shard-\$(printf %04d "\$j").jsonl" ] && continue
  $PY "$H" run --a "v2:${NET[E]},temperature=0.5" --b "v2:${NET[C]},temperature=0.5" --pairs $SP --shard "\$j" --seed $SEED \\
    --out "$K/shards" --format gen9randombattle --workers $WORKERS --v2-src "$T/src" --device cuda --progress 0 --overwrite 2>&1 | tail -n 2 || exit 1
done
PLAY
chmod a-w "$K/play.sh"
T1=$(date +%s); rc=1
for att in 1 2; do
  HJ=""; T2=$(date +%s)
  while [ -z "$HJ" ]; do
    HJ=$(pick_host); [ -n "$HJ" ] && break
    if [ $(( $(date +%s) - T2 )) -ge "$HOST_WAIT_S" ]; then fail "no RUNNING ws-data-host job with >= $((MIN_LEFT / 60)) min left within $((HOST_WAIT_S / 3600)) h"; fi
    sleep "$POLL_S"
  done
  left=$(( H2H_LIMIT - ($(date +%s) - T1) )); [ "$left" -gt 60 ] || fail "the $((H2H_LIMIT / 3600)) h h2h limit is used up"
  st "h2h attempt $att in data host job $HJ (ends $(squeue -h -j "$HJ" -o %e 2>/dev/null | head -n 1)); limit ${left}s; $(ls "$K"/shards/shard-*.jsonl 2>/dev/null | wc -l) of $NSH shards already complete"
  ( for v in $(compgen -e | grep -E '^SLURM'); do unset "$v"; done  # a clean Slurm environment (no inherited CPU binding)
    exec $SRUN --jobid="$HJ" --overlap --ntasks=1 -c 6 --gres=gpu:1 --mem=14G -J nv2-p1t-h2h timeout "$left" bash "$K/play.sh" ) >> "$K/run.log" 2>&1 < /dev/null; rc=$?
  n=$(ls "$K"/shards/shard-*.jsonl 2>/dev/null | wc -l)
  [ "$rc" = 0 ] && [ "$n" = "$NSH" ] && break
  if [ "$att" = 1 ] && ! running "$HJ"; then st "h2h attempt 1: rc $rc, $n shards, host job $HJ is no longer RUNNING: once more in the next host"; continue; fi
  fail "h2h exited $rc with $n of $NSH shards in host $HJ (attempt $att; log $K/run.log)"
done
[ "$(ls "$K"/shards/shard-*.jsonl 2>/dev/null | wc -l)" = "$NSH" ] || fail "not all $NSH shards complete"
$PY "$H" merge "$K/shards" --out "$K/MERGE.json" > "$K/merge.txt" 2>&1 || fail "merge failed ($K/merge.txt)"
read -r pairs games ties voids mean half lo hi miss <<< "$($PY -c "
import json; s = json.load(open('$K/MERGE.json'))
m, h = 100 * s['win_rate'], 100 * s['ci95']
print(s['pairs'], s['games'], s['ties'], s['void_games'], f'{m:.2f}', f'{h:.2f}', f'{m - h:.2f}', f'{m + h:.2f}', s['missing_in_range'])
")" || fail "could not read MERGE.json"
[ "${pairs:-0}" = "$((NSH * SP))" ] && [ "${miss:-1}" = 0 ] || fail "MERGE.json has ${pairs:-?} pairs (${miss:-?} missing), want $((NSH * SP))"
v=$($PY -c "
lo = $lo
print('GO' if lo > 50.5 else 'REPLICATE' if lo > 50.0 else 'PARK')
") || fail "rule evaluation failed"
klines() { local f; f=$(ls -t "$C"/runs/g9/*/"$1"/train.log 2>/dev/null | head -n 1); [ -n "$f" ] && grep "kl-monitor @" "$f" | sed -n '1p;$p' | cut -c1-200; }
{ echo "$v"
  echo "E-cd vs C-cd pair-mean $mean% [$lo, $hi] (95%, +- $half; $pairs duplicate pairs = $games games, $ties ties, $voids void; seed $SEED; raw T 0.5 both, in-process, per-side encoder routing)"
  echo "A = E-cd (spec v2.2a learner) ${NET[E]} md5 ${MD[E]}; B = C-cd (v2.1) ${NET[C]} md5 ${MD[C]}; PREREG council/p1-tera/PREREG.txt $PRE_MD5; harness $H $H2H_MD5; MERGE.json md5 $(md5sum < "$K/MERGE.json" | cut -c1-8)"
  echo "rule (verbatim): GO iff E-cd vs C-cd pair-mean lower 95% > 50.0%; lower bound in (50.0, 50.5] => REPLICATE (a second training seed before any promotion); else PARK (a 30M fork can be too short: not a kill of the idea). Read: lower > 50.5 GO, (50.0, 50.5] REPLICATE, <= 50.0 PARK; lower = $lo"
  echo "info (no rule): KL-monitor first / last, E: $(klines P1E | tr '\n' '|')"
  echo "info (no rule): KL-monitor first / last, C: $(klines P1C | tr '\n' '|')"
  echo "info (no rule): guard trips: $( [ -f "$P/TRIPPED" ] && wc -l < "$P/TRIPPED" || echo 0)"; } > "$P/VERDICT.txt.tmp" && mv -f "$P/VERDICT.txt.tmp" "$P/VERDICT.txt" && chmod a-w "$P/VERDICT.txt" || fail "could not write VERDICT.txt"
st "VERDICT: $(head -n 2 "$P/VERDICT.txt" | tr '\n' ' ')"
st "chain done"
