#!/bin/bash
# G9s vs G9p at a matched step after the 2026-11-09 12:40 GPU swap (netv2_cmp.sh's check, harness relocated; deviation
# record runs/cmp/50M-DEVIATION.txt): waits for each lineage's FIRST pool snapshot >= STEP in its dirs, copies both to
# OUT/g9s.pt / OUT/g9p.pt (+ NETS.txt), then plays 100-pair shards of wsp's spd_h2h.py exactly as netv2_cmp.sh does
# (vendored ws/wsp ee76994; duplicate pairs, gen9randombattle, A = G9s, B = G9p, both raw at T 0.5, --seed SEED, 3 workers
# x 48 battles, src-p2 nets, cuda). After every shard both one-sided tests of runs/cmp/seq-gain.json ("G9s > G9p" on the
# records, "G9p > G9s" on a winner-swapped copy) are evaluated in TWO pair orders: nested = all shards merged in pair-id
# order (scripts/netv2_cmp_nested.py; look j = pairs 0 .. 100j-1) = the DECISION OF RECORD, and round-robin across shards
# (seq_monitor's default, what netv2_cmp.sh reported) alongside. Play stops when both orders have decided both tests, or at
# n_max 2400 pairs. Summary OUT/RESULT.txt (with an AGREE / DISAGREE line). Run inside a job:
#   nohup srun --jobid=718701 --overlap --ntasks=1 -c 4 --mem=8G -J nv2-cmp50 bash -c "cp $C/scripts/netv2_cmp_step.sh \
#     /tmp/cmp-step-\$\$.sh && bash /tmp/cmp-step-\$\$.sh 50000000 9050 $C/runs/cmp/50M 'S_DIRS' 'P_DIRS'" > LOG 2>&1 &
set -uo pipefail
C=${NV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}; V=$C/vendor/wsp-ee76994; PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
STEP=${1:?usage: netv2_cmp_step.sh STEP SEED OUT S_DIRS P_DIRS}; SEED=${2:?}; D=${3:?}; S_DIRS=${4:?}; P_DIRS=${5:?}
WORKERS=${CMP_WORKERS:-3}; DESIGN=$C/runs/cmp/seq-gain.json; SHARDS=24
say() { echo "$(date -u +%T) $*"; }
first_snap() {  # first snapshot (numeric name) >= STEP in the given pool dirs
  local d f n
  for d in "$@"; do
    for f in "$d"/*.pt; do
      [ -e "$f" ] || continue
      n=$(basename "$f" .pt); case $n in *[!0-9]*|"") continue ;; esac
      if [ "$((10#$n))" -ge "$STEP" ]; then echo "$((10#$n)) $f"; fi
    done
  done | sort -n | head -n 1 | cut -d' ' -f2
}
mkdir -p "$D"
if [ ! -f "$D/g9s.pt" ] || [ ! -f "$D/g9p.pt" ]; then
  say "waiting for G9s and G9p snapshots >= $STEP (G9s in: $S_DIRS | G9p in: $P_DIRS)"
  while true; do
    s=$(first_snap $S_DIRS); p=$(first_snap $P_DIRS)
    if [ -n "$s" ] && [ -n "$p" ]; then break; fi
    sleep 60
  done
  sleep 30  # a mirror copy may still be in flight: re-pick
  s=$(first_snap $S_DIRS); p=$(first_snap $P_DIRS)
  cp -p "$s" "$D/g9s.pt.tmp" && mv "$D/g9s.pt.tmp" "$D/g9s.pt"
  cp -p "$p" "$D/g9p.pt.tmp" && mv "$D/g9p.pt.tmp" "$D/g9p.pt"
  printf 'G9s %s\nG9p %s\nseed %s\nmd5 g9s %s g9p %s\n' "$s" "$p" "$SEED" "$(md5sum < "$D/g9s.pt" | cut -c1-32)" \
    "$(md5sum < "$D/g9p.pt" | cut -c1-32)" > "$D/NETS.txt"
fi
say "nets: $(tr '\n' ' ' < "$D/NETS.txt")"
mkdir -p "$D/nested"
dec() { grep -m1 DECISION "$1" 2>/dev/null || echo "?"; }
r1=0; r2=0; n1=0; n2=0; jr1=-1; jr2=-1; jn1=-1; jn2=-1
for j in $(seq 0 $((SHARDS - 1))); do
  if [ ! -f "$D/h2h/shard-$(printf %04d "$j").jsonl" ]; then
    bash "$C/scripts/netv2_spd.sh" run --a "v2:$D/g9s.pt,temperature=0.5" --b "v2:$D/g9p.pt,temperature=0.5" \
      --pairs 100 --shard "$j" --seed "$SEED" --out "$D/h2h" --format gen9randombattle --workers "$WORKERS" \
      --v2-src "$C/src-p2" --device cuda --progress 0 2>&1 | tail -n 2
  fi
  $PY "$C/scripts/netv2_cmp_flip.py" "$D/h2h" "$D/h2h-rev"
  $PY "$V/seq_monitor.py" "$D/h2h" --design "$DESIGN" --shard-pairs 100 > "$D/look-$j.txt" 2>&1; a=$?
  $PY "$V/seq_monitor.py" "$D/h2h-rev" --design "$DESIGN" --shard-pairs 100 > "$D/look-$j-rev.txt" 2>&1; b=$?
  for x in h2h h2h-rev; do
    $PY "$C/scripts/netv2_cmp_nested.py" "$D/$x" "$D/nested/$x" > /dev/null || say "NESTED MERGE FAILED ($x): nested looks below are stale"
  done
  $PY "$V/seq_monitor.py" "$D/nested/h2h" --design "$DESIGN" --shard-pairs 100 > "$D/nested/look-$j.txt" 2>&1; c=$?
  $PY "$V/seq_monitor.py" "$D/nested/h2h-rev" --design "$DESIGN" --shard-pairs 100 > "$D/nested/look-$j-rev.txt" 2>&1; e=$?
  if [ "$r1" = 0 ] && [ "$a" != 0 ]; then r1=$a; jr1=$j; fi
  if [ "$r2" = 0 ] && [ "$b" != 0 ]; then r2=$b; jr2=$j; fi
  if [ "$n1" = 0 ] && [ "$c" != 0 ]; then n1=$c; jn1=$j; fi
  if [ "$n2" = 0 ] && [ "$e" != 0 ]; then n2=$e; jn2=$j; fi
  say "look $((j + 1)) ($(( (j + 1) * 100 )) pairs): nested G9s>G9p $c G9p>G9s $e | round-robin G9s>G9p $a G9p>G9s $b | nested: $(dec "$D/nested/look-$j.txt") | $(dec "$D/nested/look-$j-rev.txt")"
  if [ "$r1" != 0 ] && [ "$r2" != 0 ] && [ "$n1" != 0 ] && [ "$n2" != 0 ]; then break; fi
done
agree=AGREE; { [ "$n1" = "$r1" ] && [ "$n2" = "$r2" ]; } || agree=DISAGREE
{
  echo "G9s vs G9p at $((STEP / 1000000))M (A = G9s), both T 0.5, seed $SEED, fastenv in-process; harness scripts/netv2_cmp_step.sh (post-swap dirs; runs/cmp/50M-DEVIATION.txt)"
  tr '\n' ' ' < "$D/NETS.txt"; echo
  echo "DECISION OF RECORD (nested pair-id order): test G9s > G9p exit $n1 at look $((jn1 + 1)); test G9p > G9s exit $n2 at look $((jn2 + 1)) (10 = SUCCESS, 20 = FUTILITY, 0 = undecided)"
  echo "alongside (round-robin, netv2_cmp.sh's order): test G9s > G9p exit $r1 at look $((jr1 + 1)); test G9p > G9s exit $r2 at look $((jr2 + 1))"
  echo "orders: $agree"
  echo "--- nested, G9s > G9p:"; tail -n 8 "$D/nested/look-$jn1.txt" 2>/dev/null
  echo "--- nested, G9p > G9s:"; tail -n 8 "$D/nested/look-$jn2-rev.txt" 2>/dev/null
  echo "--- round-robin, G9s > G9p:"; tail -n 8 "$D/look-$jr1.txt" 2>/dev/null
  echo "--- round-robin, G9p > G9s:"; tail -n 8 "$D/look-$jr2-rev.txt" 2>/dev/null
  echo "--- all pairs played ($(cat "$D"/h2h/shard-*.jsonl | wc -l)):"
  $PY "$V/spd_h2h.py" merge "$D/h2h" 2>&1 | tail -n 3
} > "$D/RESULT.txt"
cat "$D/RESULT.txt"
