#!/bin/bash
# netv2 2026-11-15: R1b-F guard-trip HOLD response (runs/g4r1bf/HOLD-PREREG.txt; orchestrator decision ~20:40), a step of
# R1b-F's job: waits for a NEW runs/g4r1bf/HOLD + a new GUARD_TRIPPED.second-* (the disarmed follow-up's trip path);
# T1 = TRIP vs R1b-F@300M in-process, fixed 1000 pairs (decides); T2 = TRIP vs v1@500M on Showdown (information only);
# T1 upper 95% < 50% => ROLLBACK (walk back to the newest checkpoint not clearly worse; resume it at --lr 3.5e-5 through
# the netv2.env hook runs/g4r1bf/ROLLBACK) else CONTINUE (rm HOLD). Up to 3 HOLDs while every response is CONTINUE; a
# ROLLBACK or a failure ends the chain. Outputs runs/g4r1bf-hold<k>/{STATUS, VERDICT.txt (r--), FAILED}.
#   setsid nohup srun --jobid=1137378 --overlap --ntasks=1 -c 8 --gres=gpu:1 --mem=16G -J nv2-hold \
#     bash -c 'cp $C/scripts/netv2_g4r1bf/hold-chain.sh /tmp/nv2-hold-$$.sh && exec bash /tmp/nv2-hold-$$.sh' \
#     > $C/logs/hold-<t>.out 2>&1 < /dev/null &
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; G=$C/runs/g4r1bf; J=${SLURM_JOB_ID:?}; LD=/tmp/ws-netv2-$J/run/R1bF; LOG=$LD/train.log
PRE_MD5=af0441c6; B300=$G/check-R1bF-ckpt_00300027150.pt; B300_MD5=c3ec9433
V500=/vol/bitbucket/fr1226/gloria/runs/20261025-195826-301040/final_model.zip; S300=3600301; S500=3600501
PY=/vol/bitbucket/fr1226/gloria/.venv/bin/python; V=$C/vendor/wsp-ee76994
PAT='^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train( |$)'
K=$C/runs/g4r1bf-hold-chain; mkdir -p "$K"
st() { echo "$(date -u +%FT%T) $*" | tee -a "$K/STATUS" >&2; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$K/FAILED"; exit 1; }
cnt() { local c; c=$(grep -c "$1" "$LOG" 2>/dev/null); echo "${c:-0}"; }
if [ ! -s "$G/HOLD-PREREG.txt" ] || [ -w "$G/HOLD-PREREG.txt" ] || [ "$(md5sum < "$G/HOLD-PREREG.txt" | cut -c1-8)" != "$PRE_MD5" ]; then fail "no read-only runs/g4r1bf/HOLD-PREREG.txt with md5 $PRE_MD5"; fi
[ "$(md5sum < "$B300" | cut -c1-8)" = "$B300_MD5" ] || fail "R1b-F@300M md5 is not $B300_MD5"
[ -f "$V500" ] || fail "no v1@500M at $V500"
trap 'st "chain step got SIGTERM (job end / cancel): a pending HOLD is left to the orchestrator"; exit 1' TERM
# h2h_fixed A B SEED OUT: in-process gen4, raw T 0.5, 10 shards x 100 duplicate pairs; echoes the merge line ("" on failure)
h2h_fixed() {
  local a=$1 b=$2 s=$3 o=$4 j
  mkdir -p "$o"
  for j in $(seq 0 9); do
    if [ -f "$o/shard-$(printf %04d "$j").jsonl" ]; then continue; fi
    timeout 1800 bash "$C/scripts/netv2_spd.sh" run --a "v2:$a,temperature=0.5" --b "v2:$b,temperature=0.5" --pairs 100 --shard "$j" \
      --seed "$s" --out "$o" --format gen4randombattle --workers 7 --v2-src "$C/src-24c633f" --device cuda --progress 0 >> "$o.log" 2>&1 || return 1
  done
  $PY "$V/spd_h2h.py" merge "$o" 2>&1 | grep "A win rate" | tail -n 1
}
# upper LINE: "point lower upper" of the pair-mean 95% interval (needs pairs 0..999 with 0 missing and <= 10 void)
upper() {
  local p h
  local v; v=$(sed -n 's/.*pairs 0\.\.999 (0 missing), void \([0-9]*\).*/\1/p' <<< "$1")
  [ -n "$v" ] && [ "$v" -le 10 ] || { echo ""; return 0; }
  p=$(grep -oE 'A win rate [0-9.]+% ± [0-9.]+%' <<< "$1" | sed -E 's/A win rate ([0-9.]+)% ± ([0-9.]+)%/\1/')
  h=$(grep -oE 'A win rate [0-9.]+% ± [0-9.]+%' <<< "$1" | sed -E 's/A win rate ([0-9.]+)% ± ([0-9.]+)%/\2/')
  awk -v p="$p" -v h="$h" 'BEGIN { printf "%.1f %.1f %.1f", p, p - h, p + h }'
}
step_of() { local n; n=$(basename "$1" .pt); n=${n#ckpt_}; n=${n#check-R1bF-ckpt_}; echo "$((10#$n))"; }
no_async_eval() {  # true when the trainer's last 'started (async' eval has its result line (or its pid is gone)
  local s ln epid estep
  s=$(grep -n "started (async, pid" "$LOG" | tail -n 1); [ -z "$s" ] && return 0
  ln=${s%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$s"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$s")
  tail -n +"$ln" "$LOG" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null
}
seen=" $(ls "$LD"/GUARD_TRIPPED.second-* 2>/dev/null | tr '\n' ' ')"
st "chain start on $(hostname -s) (job $J step ${SLURM_STEP_ID:-?}, PREREG $PRE_MD5); trips already handled before: '${seen# }'; polling every 60 s"
k=0
while [ "$k" -lt 3 ]; do
  tf=""
  while [ -z "$tf" ]; do
    sleep 60
    [ -e "$G/HOLD" ] || continue
    for f in "$LD"/GUARD_TRIPPED.second-*; do [ -e "$f" ] || continue; case $seen in *" $f "*) ;; *) tf=$f ;; esac; done
    if [ -z "$tf" ] && [ ! -e "$K/.hold-noted" ]; then st "runs/g4r1bf/HOLD without a new GUARD_TRIPPED.second-* (not this chain's trigger)"; touch "$K/.hold-noted"; fi
  done
  rm -f "$K/.hold-noted"; seen="$seen$tf "; k=$((k + 1)); H=$C/runs/g4r1bf-hold$k; mkdir -p "$H"
  hst() { echo "$(date -u +%FT%T) $*" | tee -a "$H/STATUS" "$K/STATUS" >&2; }
  hfail() { hst "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$H/FAILED"; exit 1; }
  starts=$(cnt "v2 trainer:")
  hst "HOLD #$k: trip $(head -c 160 "$tf"); trainer processes: '$(pgrep -u fr1226 -f "$PAT" | tr '\n' ' ')'"
  ts=$(grep "saved final_model.pt at" "$LOG" | tail -n 1 | grep -oE "at [0-9,]+ steps" | tr -dc 0-9)
  [ -n "$ts" ] && [ -f "$LD/final_model.pt" ] || hfail "no saved final_model.pt / step for the trip"
  TRIP=$H/trip-$(printf %011d "$ts").pt; cp -p "$LD/final_model.pt" "$TRIP.tmp" && mv -f "$TRIP.tmp" "$TRIP" || hfail "could not copy the trip net"
  hst "TRIP = $TRIP (steps $ts, md5 $(md5sum < "$TRIP" | cut -c1-8)); T1 vs R1b-F@300M in-process 1000 pairs seed $S300"
  l1=$(h2h_fixed "$TRIP" "$B300" "$S300" "$H/T1") || hfail "T1 h2h failed (log $H/T1.log)"
  u1=$(upper "$l1"); [ -n "$u1" ] || hfail "T1 merge line unreadable / not 1000 complete pairs: '$l1'"
  read -r p1 lo1 hi1 <<< "$u1"; hst "T1: $l1"
  # T2 (information only): Showdown, hang guard 12 min with 0 games, cap 60 min
  hst "T2 (info): TRIP vs v1@500M on Showdown, 1000 games, seed $S500"; touch "$H/.t2"
  ( cd "$C" && exec bash scripts/netv2_h2h.sbatch "$TRIP,temperature=0.5" "$V500,temperature=0.5" 1000 --gen 4 --seed "$S500" ) > "$H/T2.log" 2>&1 &
  t2=$!; t0=$(date +%s); r2="void"
  while kill -0 "$t2" 2>/dev/null; do
    sleep 30; hd=""; for h in "$C"/runs/h2h-*-"$J"; do if [ -d "$h" ] && [ "$h" -nt "$H/.t2" ]; then hd=$h; fi; done
    n=0; if [ -n "$hd" ]; then n=$(cat "$hd"/shard-*/results.jsonl 2>/dev/null | wc -l); fi
    el=$(( $(date +%s) - t0 ))
    if { [ "$el" -ge 720 ] && [ "$n" = 0 ]; } || [ "$el" -ge 3600 ]; then
      hst "T2 stopped after $((el / 60)) min with $n games (hang guard / cap): void"; kill -TERM "$t2" 2>/dev/null
      for p in $(pgrep -u fr1226 -f "porygon.v2.evaluate .*$(basename "$TRIP")" || true); do kill -TERM "$p" 2>/dev/null; done
      sleep 20; break
    fi
  done
  wait "$t2" 2>/dev/null
  if [ -n "${hd:-}" ] && [ -f "$hd/result.txt" ]; then r2=$(head -n 1 "$hd/result.txt"); fi
  hst "T2 (info): $r2"
  if awk -v h="$hi1" 'BEGIN { exit !(h < 50) }'; then
    hst "T1 upper 95% $hi1 < 50 => CLEARLY WORSE than R1b-F@300M => ROLLBACK: walk back from below $ts"
    R=""; Rl=""
    for x in $( { ls "$G"/*/R1bF/checkpoints/ckpt_*.pt "$LD"/checkpoints/ckpt_*.pt 2>/dev/null; } | awk -F'ckpt_' '{print $NF" "$0}' | sort -rn | awk '!s[$1]++ {print $2}'); do
      s=$(step_of "$x"); if [ "$s" -ge "$ts" ] || [ "$s" -le 300027150 ]; then continue; fi
      lw=$(h2h_fixed "$x" "$B300" "$S300" "$H/walk-$(printf %011d "$s")") || hfail "walk-back h2h failed at $x"
      uw=$(upper "$lw"); [ -n "$uw" ] || hfail "walk-back merge line unreadable at $x: '$lw'"
      read -r pw low hiw <<< "$uw"; hst "walk $s: $pw% [$low, $hiw] ($x)"
      if awk -v h="$hiw" 'BEGIN { exit !(h >= 50) }'; then R=$x; Rl="$pw% [$low, $hiw]"; break; fi
    done
    if [ -z "$R" ]; then R=$B300; Rl="(none of the checkpoints 305M+ was not clearly worse: R = 300M)"; fi
    rs=$(step_of "$R"); RB=$H/rollback-$(printf %011d "$rs").pt
    cp -p "$R" "$RB.tmp" && mv -f "$RB.tmp" "$RB" || hfail "could not copy R"
    rm5=$(md5sum < "$RB" | cut -c1-8)
    dec="ROLLBACK to $(basename "$R") (steps $rs, md5 $rm5; vs 300M $Rl) at --lr 3.5e-5"
    if ! grep -q 'runs/g4r1bf/ROLLBACK' "$C/netv2.env"; then
      { echo "R1b-F HOLD #$k response (runs/g4r1bf/HOLD-PREREG.txt $PRE_MD5): $dec — NOT APPLIED: the live netv2.env has no rollback hook"
        echo "T1 TRIP@$ts vs R1b-F@300M: $l1"; echo "T2 (info) vs v1@500M: $r2"; } > "$H/VERDICT.txt"; chmod a-w "$H/VERDICT.txt"
      hfail "rollback hook not deployed (netv2.env has no runs/g4r1bf/ROLLBACK): HOLD left for the orchestrator; R = $RB"
    fi
    echo "$RB $rs $rm5" > "$G/ROLLBACK.tmp" && mv -f "$G/ROLLBACK.tmp" "$G/ROLLBACK" || hfail "could not write runs/g4r1bf/ROLLBACK"
  else
    dec="CONTINUE (T1 upper 95% $hi1 >= 50: not clearly worse than R1b-F@300M); the same config continues"
  fi
  { echo "R1b-F HOLD #$k response (runs/g4r1bf/HOLD-PREREG.txt $PRE_MD5), read mechanically by scripts/netv2_g4r1bf/hold-chain.sh"
    echo "DECISION: $dec"
    echo "T1 TRIP@$ts vs R1b-F@300M (in-process, 1000 pairs, seed $S300): $l1"
    echo "T2 (info) TRIP vs v1@500M (Showdown, seed $S500): $r2"
    ls "$H"/walk-*.log > /dev/null 2>&1 && grep -h "walk " "$H/STATUS"
    true; } > "$H/VERDICT.txt.tmp" && mv -f "$H/VERDICT.txt.tmp" "$H/VERDICT.txt" && chmod a-w "$H/VERDICT.txt"
  hst "VERDICT: $dec"
  # apply: rm HOLD; if the HOLD already expired (the job relaunched the same config), a ROLLBACK needs a restart in place
  relaunched_before=""; if [ "$(cnt "v2 trainer:")" -gt "$starts" ]; then relaunched_before=1; fi
  touch "$G/restart"; rm -f "$G/HOLD"; hst "runs/g4r1bf/HOLD removed; restart flag armed"
  if [ -n "$relaunched_before" ] && [ -f "$G/ROLLBACK" ] && [ ! -f "$G/ROLLBACK.started" ]; then
    for i in $(seq 1 60); do no_async_eval && break; sleep 30; done
    ps=$(pgrep -u fr1226 -f "$PAT" || true); set -- $ps
    [ $# = 1 ] || hfail "HOLD had expired; expected 1 trainer to SIGINT for the rollback, got '$ps'"
    starts=$(cnt "v2 trainer:"); kill -INT "$1" && hst "the HOLD had expired: SIGINT $1 so the next relaunch applies the ROLLBACK"
  fi
  for i in $(seq 1 360); do if [ "$(cnt "v2 trainer:")" -gt "$starts" ]; then break; fi; sleep 10; done
  touch "$G/restart"
  npid=""; for i in $(seq 1 30); do ps=$(pgrep -u fr1226 -f "$PAT" || true); set -- $ps; if [ $# = 1 ]; then npid=$1; break; fi; sleep 10; done
  [ -n "$npid" ] || hfail "no single trainer after the response (found '${ps:-}')"
  tr '\0' ' ' < "/proc/$npid/cmdline" > "$H/relaunch-argv.txt"
  if [ -f "$G/ROLLBACK" ]; then
    grep -q -- "--lr 3.5e-5" "$H/relaunch-argv.txt" && grep -q -- "--resume $RB" "$H/relaunch-argv.txt" && [ -f "$G/ROLLBACK.started" ] || hfail "relaunch does not show the rollback (--lr 3.5e-5 / --resume $RB / ROLLBACK.started): $H/relaunch-argv.txt"
    hst "ROLLBACK applied: trainer $npid resumes $RB at --lr 3.5e-5; restart flag re-armed; chain ends (a further HOLD = orchestrator)"
    exit 0
  fi
  grep -q -- "--lr 7e-5" "$H/relaunch-argv.txt" || hfail "relaunch argv lacks --lr 7e-5: $H/relaunch-argv.txt"
  hst "CONTINUE applied: trainer $npid relaunched (same config); restart flag re-armed; waiting for a further HOLD"
done
st "3 HOLDs handled: chain ends"
