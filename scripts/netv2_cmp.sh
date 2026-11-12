#!/bin/bash
# G9s vs G9p at a matched step (pre-registered in .orchestra/netv2/STATUS.md 05:15): waits for each lineage's first
# pool snapshot at >= STEP, copies both to runs/cmp/<N>M/, then plays 100-pair shards of the in-process head to head
# (wsp's spd_h2h.py, vendored copy ws/wsp ee76994; duplicate pairs, gen9randombattle, A = G9s, B = G9p, both raw at
# T 0.5, --seed SEED). After every shard (= one look) it runs both one-sided gain tests of wsp's approved sequential
# plan (design runs/cmp/seq-gain.json: θ0 50 → θ1 52 %, α 0.025 each, power 0.8, looks every 100 pairs, n_max 2400):
# "G9s > G9p" on the records and "G9p > G9s" on a copy with the winners swapped; the first decision of each test is
# kept and it stops when both have decided. Summary: runs/cmp/<N>M/RESULT.txt. Launch inside a job:
#   nohup srun --jobid=718701 --overlap --ntasks=1 -c 4 --mem=8G bash $C/scripts/netv2_cmp.sh 10000000 9010 \
#     > $C/logs/cmp-10M.out 2>&1 &
set -uo pipefail
C=${NV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}; V=$C/vendor/wsp-ee76994; PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
STEP=$1 SEED=$2 WORKERS=${CMP_WORKERS:-3}
D=$C/runs/cmp/$((STEP / 1000000))M
# pool dirs of each lineage wherever it trained (node-local ones are only visible inside that job; the mirrors lag
# <= 10 min); after the MAIN-RUN RULE swap (STATUS.md, agent 4) G9s trains in 741849 (run/G9s) and G9p in 718701
# (run/G9p). Override with CMP_S_DIRS / CMP_P_DIRS.
S_DIRS=${CMP_S_DIRS:-"/tmp/ws-netv2-718701/run/G9s/pool $C/runs/r1/20261108-220401-718701/G9s/pool /tmp/ws-netv2-741849/run/G9s/pool $C/runs/g9/20261109-021316-741849/G9s/pool"}
P_DIRS=${CMP_P_DIRS:-"$C/runs/r1/handover-g9p/pool /tmp/ws-netv2-741849/run/G9/pool $C/runs/g9/20261109-021316-741849/G9/pool /tmp/ws-netv2-718701/run/G9p/pool $C/runs/r1/20261108-220401-718701/G9p/pool"}
say() { echo "$(date -u +%T) $*"; }
first_snap() {  # first snapshot (numeric name) >= STEP in the given pool dirs
  local d f n
  for d in "$@"; do
    for f in "$d"/*.pt; do
      [ -e "$f" ] || continue
      n=$(basename "$f" .pt); case $n in *[!0-9]*|"") continue ;; esac
      [ "$((10#$n))" -ge "$STEP" ] && echo "$((10#$n)) $f"
    done
  done | sort -n | head -n 1 | cut -d' ' -f2
}
mkdir -p "$D"
if [ ! -f "$D/g9s.pt" ] || [ ! -f "$D/g9p.pt" ]; then
  say "waiting for G9s and G9p snapshots >= $STEP"
  while true; do
    s=$(first_snap $S_DIRS); p=$(first_snap $P_DIRS)
    [ -n "$s" ] && [ -n "$p" ] && break
    sleep 60
  done
  sleep 30  # a snapshot is written atomically by the trainer; the mirror copy may still be in flight: re-pick
  s=$(first_snap $S_DIRS); p=$(first_snap $P_DIRS)
  cp -p "$s" "$D/g9s.pt.tmp" && mv "$D/g9s.pt.tmp" "$D/g9s.pt"
  cp -p "$p" "$D/g9p.pt.tmp" && mv "$D/g9p.pt.tmp" "$D/g9p.pt"
  printf 'G9s %s\nG9p %s\nseed %s\n' "$s" "$p" "$SEED" > "$D/NETS.txt"
fi
say "nets: $(tr '\n' ' ' < "$D/NETS.txt")"
d1=0; d2=0; j1=-1; j2=-1
for j in $(seq 0 23); do
  if [ ! -f "$D/h2h/shard-$(printf %04d "$j").jsonl" ]; then
    bash "$C/scripts/netv2_spd.sh" run --a "v2:$D/g9s.pt,temperature=0.5" --b "v2:$D/g9p.pt,temperature=0.5" \
      --pairs 100 --shard "$j" --seed "$SEED" --out "$D/h2h" --format gen9randombattle --workers "$WORKERS" \
      --v2-src "$C/src-p2" --device cuda --progress 0 2>&1 | tail -n 2
  fi
  $PY "$C/scripts/netv2_cmp_flip.py" "$D/h2h" "$D/h2h-rev"
  $PY "$V/seq_monitor.py" "$D/h2h" --design "$C/runs/cmp/seq-gain.json" --shard-pairs 100 > "$D/look-$j.txt" 2>&1; r1=$?
  $PY "$V/seq_monitor.py" "$D/h2h-rev" --design "$C/runs/cmp/seq-gain.json" --shard-pairs 100 > "$D/look-$j-rev.txt" 2>&1; r2=$?
  if [ "$d1" = 0 ] && [ "$r1" != 0 ]; then d1=$r1; j1=$j; fi
  if [ "$d2" = 0 ] && [ "$r2" != 0 ]; then d2=$r2; j2=$j; fi
  say "look $((j + 1)) ($(( (j + 1) * 100 )) pairs): G9s>G9p exit $r1, G9p>G9s exit $r2 | $(grep -m1 DECISION "$D/look-$j.txt") | $(grep -m1 DECISION "$D/look-$j-rev.txt")"
  [ "$d1" != 0 ] && [ "$d2" != 0 ] && break
done
{
  echo "G9s vs G9p at $((STEP / 1000000))M (A = G9s), both T 0.5, seed $SEED, fastenv in-process, $(tr '\n' ' ' < "$D/NETS.txt")"
  echo "test G9s > G9p: exit $d1 at look $((j1 + 1)) (10 = SUCCESS, 20 = FUTILITY, 0 = undecided)"
  echo "test G9p > G9s: exit $d2 at look $((j2 + 1))"
  echo "--- G9s > G9p, look $((j1 + 1)):"; cat "$D/look-$j1.txt" 2>/dev/null | tail -n 8
  echo "--- G9p > G9s, look $((j2 + 1)):"; cat "$D/look-$j2-rev.txt" 2>/dev/null | tail -n 8
  echo "--- all pairs played ($(cat "$D"/h2h/shard-*.jsonl | wc -l)):"
  $PY "$V/spd_h2h.py" merge "$D/h2h" 2>&1 | tail -n 3
} > "$D/RESULT.txt"
cat "$D/RESULT.txt"
