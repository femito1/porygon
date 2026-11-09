#!/bin/bash
# Pre-registered in-process head to head of two given raw v2 nets, the loop of netv2_cmp.sh without its wait-for-step
# logic: OUT/a.pt (A) vs OUT/b.pt (B), both T 0.5, duplicate pairs, gen9randombattle, --seed SEED, 100-pair shards of
# wsp's spd_h2h.py (vendored ws/wsp ee76994); after every shard (= one look) both one-sided tests of the sequential
# design (default runs/cmp/seq-gain.json: θ0 50 → θ1 52 %, α 0.025 each, power 0.8, looks every 100 pairs, n_max 2400):
# "A > B" on the records and "B > A" on a copy with the winners swapped (netv2_cmp_flip.py), on the same pairs; the
# first decision of each test is kept and it stops when both have decided. Refuses to start unless OUT/PREREG.txt,
# OUT/a.pt and OUT/b.pt exist (write the pre-registration before game 1). Summary: OUT/RESULT.txt. Run inside a job,
# from a node-local copy (a re-sync must not hit a running bash):
#   nohup srun --jobid=718701 --overlap --ntasks=1 -c 4 --mem=8G bash -c \
#     "cp $C/scripts/netv2_cmp_pair.sh /tmp/cmp-pair-\$\$.sh && bash /tmp/cmp-pair-\$\$.sh OUT SEED G9s G9p" > LOG 2>&1 &
set -uo pipefail
C=${NV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}; V=$C/vendor/wsp-ee76994; PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
D=${1:?usage: netv2_cmp_pair.sh OUT SEED [LABEL_A LABEL_B]}; SEED=${2:?seed}; LA=${3:-A}; LB=${4:-B}
WORKERS=${CMP_WORKERS:-3}; DESIGN=${CMP_DESIGN:-$C/runs/cmp/seq-gain.json}; SHARDS=${CMP_SHARDS:-24}
say() { echo "$(date -u +%T) $*"; }
for f in "$D/PREREG.txt" "$D/a.pt" "$D/b.pt" "$DESIGN"; do
  [ -s "$f" ] || { say "missing $f: not playing" >&2; exit 2; }
done
say "A = $LA ($D/a.pt), B = $LB ($D/b.pt), seed $SEED, design $DESIGN, $(md5sum < "$D/PREREG.txt" | cut -c1-8) = PREREG md5"
d1=0; d2=0; j1=-1; j2=-1
for j in $(seq 0 $((SHARDS - 1))); do
  if [ ! -f "$D/h2h/shard-$(printf %04d "$j").jsonl" ]; then
    bash "$C/scripts/netv2_spd.sh" run --a "v2:$D/a.pt,temperature=0.5" --b "v2:$D/b.pt,temperature=0.5" \
      --pairs 100 --shard "$j" --seed "$SEED" --out "$D/h2h" --format gen9randombattle --workers "$WORKERS" \
      --v2-src "$C/src-p2" --device cuda --progress 0 2>&1 | tail -n 2
  fi
  $PY "$C/scripts/netv2_cmp_flip.py" "$D/h2h" "$D/h2h-rev"
  $PY "$V/seq_monitor.py" "$D/h2h" --design "$DESIGN" --shard-pairs 100 > "$D/look-$j.txt" 2>&1; r1=$?
  $PY "$V/seq_monitor.py" "$D/h2h-rev" --design "$DESIGN" --shard-pairs 100 > "$D/look-$j-rev.txt" 2>&1; r2=$?
  if [ "$d1" = 0 ] && [ "$r1" != 0 ]; then d1=$r1; j1=$j; fi
  if [ "$d2" = 0 ] && [ "$r2" != 0 ]; then d2=$r2; j2=$j; fi
  say "look $((j + 1)) ($(( (j + 1) * 100 )) pairs): $LA>$LB exit $r1, $LB>$LA exit $r2 | $(grep -m1 DECISION "$D/look-$j.txt") | $(grep -m1 DECISION "$D/look-$j-rev.txt")"
  if [ "$d1" != 0 ] && [ "$d2" != 0 ]; then break; fi
done
{
  echo "$LA vs $LB (A = $LA), both T 0.5, seed $SEED, fastenv in-process; pre-registration $D/PREREG.txt"
  echo "test $LA > $LB: exit $d1 at look $((j1 + 1)) (10 = SUCCESS, 20 = FUTILITY, 0 = undecided)"
  echo "test $LB > $LA: exit $d2 at look $((j2 + 1))"
  echo "--- $LA > $LB, look $((j1 + 1)):"; tail -n 8 "$D/look-$j1.txt" 2>/dev/null
  echo "--- $LB > $LA, look $((j2 + 1)):"; tail -n 8 "$D/look-$j2-rev.txt" 2>/dev/null
  echo "--- all pairs played ($(cat "$D"/h2h/shard-*.jsonl | wc -l)):"
  $PY "$V/spd_h2h.py" merge "$D/h2h" 2>&1 | tail -n 3
} > "$D/RESULT.txt"
cat "$D/RESULT.txt"
