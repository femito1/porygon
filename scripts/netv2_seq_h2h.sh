#!/bin/bash
# A pre-registered sequential head to head of two raw v2 nets (wsp's approved plan): 100-pair shards of the in-process
# h2h (wsp's spd_h2h.py, vendored ws/wsp ee76994; duplicate pairs, A = side one) and after each shard both one-sided
# gain tests of the design (default runs/cmp/seq-gain.json) on the same pairs: "A > B" and "B > A" (winners swapped,
# scripts/netv2_cmp_flip.py); each test's first decision is kept; stops when both have decided (n_max 2400 pairs).
#   nohup srun --jobid=J --overlap --ntasks=1 -c 3 --mem=8G bash $C/scripts/netv2_seq_h2h.sh OUT_DIR SEED \
#     "v2:A.pt,temperature=T" "v2:B.pt,temperature=T" [FORMAT] > $C/logs/<name>.out 2>&1 &
# Summary: OUT_DIR/RESULT.txt.
set -uo pipefail
C=${NV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}; V=$C/vendor/wsp-ee76994; PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
D=$1 SEED=$2 A=$3 B=$4 FMT=${5:-gen9randombattle}
DESIGN=${SEQ_DESIGN:-$C/runs/cmp/seq-gain.json} WORKERS=${SEQ_WORKERS:-2}
say() { echo "$(date -u +%T) $*"; }
mkdir -p "$D"
printf 'A %s\nB %s\nseed %s\nformat %s\ndesign %s\n' "$A" "$B" "$SEED" "$FMT" "$DESIGN" > "$D/SPEC.txt"
d1=0; d2=0; j1=-1; j2=-1
for j in $(seq 0 23); do
  if [ ! -f "$D/h2h/shard-$(printf %04d "$j").jsonl" ]; then
    bash "$C/scripts/netv2_spd.sh" run --a "$A" --b "$B" --pairs 100 --shard "$j" --seed "$SEED" --out "$D/h2h" \
      --format "$FMT" --workers "$WORKERS" --v2-src "$C/src-p2" --device cuda --progress 0 2>&1 | tail -n 1
  fi
  $PY "$C/scripts/netv2_cmp_flip.py" "$D/h2h" "$D/h2h-rev"
  $PY "$V/seq_monitor.py" "$D/h2h" --design "$DESIGN" --shard-pairs 100 > "$D/look-$j.txt" 2>&1; r1=$?
  $PY "$V/seq_monitor.py" "$D/h2h-rev" --design "$DESIGN" --shard-pairs 100 > "$D/look-$j-rev.txt" 2>&1; r2=$?
  if [ "$d1" = 0 ] && [ "$r1" != 0 ]; then d1=$r1; j1=$j; fi
  if [ "$d2" = 0 ] && [ "$r2" != 0 ]; then d2=$r2; j2=$j; fi
  say "look $((j + 1)) ($(( (j + 1) * 100 )) pairs): A>B exit $r1, B>A exit $r2 | $(grep -m1 DECISION "$D/look-$j.txt") | $(grep -m1 DECISION "$D/look-$j-rev.txt")"
  [ "$d1" != 0 ] && [ "$d2" != 0 ] && break
done
{
  cat "$D/SPEC.txt"
  echo "test A > B: exit $d1 at look $((j1 + 1)) (10 = SUCCESS, 20 = FUTILITY, 0 = undecided)"
  echo "test B > A: exit $d2 at look $((j2 + 1))"
  echo "--- A > B, look $((j1 + 1)):"; tail -n 8 "$D/look-$j1.txt" 2>/dev/null
  echo "--- B > A, look $((j2 + 1)):"; tail -n 8 "$D/look-$j2-rev.txt" 2>/dev/null
  echo "--- all pairs played ($(cat "$D"/h2h/shard-*.jsonl | wc -l)):"
  $PY "$V/spd_h2h.py" merge "$D/h2h" 2>&1 | tail -n 3
} > "$D/RESULT.txt"
cat "$D/RESULT.txt"
