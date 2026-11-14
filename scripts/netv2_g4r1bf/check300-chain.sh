#!/bin/bash
# netv2 2026-11-14: R1b-F 300M strength watch as a cluster-side chain (orchestrator-approved; pre-registered in
# runs/g4r1bf/NOTES.txt "300M STRENGTH WATCH" before any game). A step in R1b-F's job (1136569):
#   setsid nohup srun --jobid=1136569 --overlap --ntasks=1 -c 1 --mem=1G -J nv2-g4-check300 \
#     bash $C/scripts/netv2_g4r1bf/check300-chain.sh > $C/logs/g4-check300-<t>.out 2>&1 < /dev/null &
# 1 waits until no async eval is running; 2 runs scripts/netv2_check_at.sh (88bff2c4: copies R1b-F's first ckpt >= 300M,
# queues the two h2h as runs/g4r1bf/INLINE_CMD, SIGINT, waits for the relaunch, re-arms restart); 3 reads both
# result.txt files and applies the rule mechanically -> runs/g4r1bf/check300/VERDICT.txt (r--). FAILED on any failure.
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; D=$C/runs/g4r1bf/check300; mkdir -p "$D"
J=${SLURM_JOB_ID:?}; LOG=/tmp/ws-netv2-$J/run/R1bF/train.log
V46=/vol/bitbucket/fr1226/gloria/runs/resume/pool/ppo_46259712_steps.pt
V500=/vol/bitbucket/fr1226/gloria/runs/20261025-195826-301040/final_model.zip
st() { echo "$(date -u +%FT%T) $*" | tee -a "$D/STATUS"; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$D/FAILED"; exit 1; }
[ ! -e "$D/FAILED" ] || { st "FAILED exists: not running"; exit 1; }
grep -q "^300M STRENGTH WATCH, PRE-REGISTERED" "$C/runs/g4r1bf/NOTES.txt" || fail "no 300M pre-registration in NOTES.txt"
[ "$(md5sum < "$C/scripts/netv2_check_at.sh" | cut -c1-8)" = 88bff2c4 ] || fail "netv2_check_at.sh md5"
for f in INLINE_CMD INLINE_CMD.running HOLD WAIT_FOR CHECK_STOP; do [ ! -e "$C/runs/g4r1bf/$f" ] || fail "runs/g4r1bf/$f present"; done
st "start in job $J: waiting for no async eval"
for i in $(seq 1 60); do
  ln0=$(grep -n "started (async, pid" "$LOG" | tail -n 1)
  [ -z "$ln0" ] && break
  ln=${ln0%%:*}; epid=$(sed -n 's/.*started (async, pid \([0-9]*\)).*/\1/p' <<< "$ln0"); estep=$(sed -n 's/.*eval @ \([0-9,]*\) started.*/\1/p' <<< "$ln0")
  if tail -n +"$ln" "$LOG" | grep -qF "[eval @ $estep]" || ! kill -0 "$epid" 2>/dev/null; then break; fi
  sleep 30
done
touch "$D/.t0"; st "no eval running: netv2_check_at.sh g4r1bf R1bF 300000000 (v1@46M | v1@500M) 1000 --gen 4 --seed 9600"
cp "$C/scripts/netv2_check_at.sh" "/tmp/nv2-check300-$$.sh"
# hang guard (attempt 1, 15:22-15:44: all 8 shards timed out in the websocket handshake and their evaluate processes
# hung with 0 games, the trainer paused): an h2h of this attempt with 0 results after 12 min => kill this INLINE (its
# bash, then its evaluate processes, by PID; the trainer relaunches), attempt void
( while sleep 60; do
    [ -e "$D/.guard-stop" ] && exit 0
    for h in "$C"/runs/h2h-*-"$J"; do
      [ -d "$h" ] && [ "$h" -nt "$D/.t0" ] && [ ! -f "$h/result.txt" ] || continue
      age=$(( $(date +%s) - $(stat -c %Y "$h/manifest.txt" 2>/dev/null || date +%s) ))
      n=$(cat "$h"/shard-*/results.jsonl 2>/dev/null | wc -l)
      if [ "$age" -ge 720 ] && [ "$n" = 0 ]; then
        echo "$(date -u +%FT%T) HANG GUARD: $h has 0 games after $((age / 60)) min ($(grep -l handshake "$h"/shard-*/evaluate.log 2>/dev/null | wc -l) shards with handshake errors): killing the INLINE" | tee -a "$D/STATUS"
        for p in $(pgrep -u fr1226 -f "INLINE_CMD.running" || true); do tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "runs/g4r1bf/INLINE_CMD.running" && kill -TERM "$p"; done
        for p in $(pgrep -u fr1226 -f "porygon.v2.evaluate .*check-R1bF-ckpt_00300027150" || true); do kill -TERM "$p"; done
        touch "$D/.hung"; exit 0
      fi
    done
  done ) & guard=$!
NV2_POLL=60 bash "/tmp/nv2-check300-$$.sh" g4r1bf R1bF 300000000 "$V46,temperature=0.5|$V500,temperature=0.5" 1000 --gen 4 --seed 9600 > "$D/check_at.log" 2>&1
rc=$?; rm -f "/tmp/nv2-check300-$$.sh"; touch "$D/.guard-stop"; kill "$guard" 2>/dev/null
[ ! -e "$D/.hung" ] || fail "attempt void: an h2h hung with 0 games (hang guard; R1b-F relaunched)"
st "check_at exit $rc: $(grep -E 'checkpoint|relaunched' "$D/check_at.log" | tr '\n' ' ' | cut -c1-300)"
[ "$rc" = 0 ] || fail "netv2_check_at.sh exited $rc (log $D/check_at.log)"
r46=""; r500=""
for r in $(ls -tr "$C"/runs/h2h-*-"$J"/result.txt 2>/dev/null); do
  [ "$r" -nt "$D/.t0" ] || continue
  grep -q "check-R1bF-ckpt_00300027150.pt" "$r" || continue
  if grep -q "$V46" "$r"; then r46=$r; elif grep -q "$V500" "$r"; then r500=$r; fi
done
[ -n "$r46" ] && [ -n "$r500" ] || fail "result files missing (v1@46M '${r46}', v1@500M '${r500}')"
grep -q -- "--seed 9600" "$r46" && grep -q -- "--seed 9600" "$r500" || fail "a result file lacks --seed 9600"
p46=$(head -n 1 "$r46" | grep -oE ': [0-9.]+%' | tr -dc '0-9.'); p500=$(head -n 1 "$r500" | grep -oE ': [0-9.]+%' | tr -dc '0-9.')
[ -n "$p46" ] && [ -n "$p500" ] || fail "cannot parse the win rates"
le() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a <= b) }'; }; ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }
if le "$p46" 46.9; then v1="(i) DETERIORATED (vs v1@46M $p46% <= 46.9%) => pause + follow-up #1 per the user rule: REPORT TO MAIN FIRST (nothing started)"
else v1="(i) continue (vs v1@46M $p46% > 46.9%)"; fi
if ge "$p500" 60; then v2="(ii) vs v1@500M $p500% >= 60% => candidate prior + NEW TRIGGER: propose a second slot test (R1b-F@300M prior vs v1@500M prior; its own pre-registration) = main"
elif ge "$p500" 55; then v2="(ii) vs v1@500M $p500% in [55, 60) => R1b-F stays the candidate gen4 prior; below the 60% trigger: no slot test"
elif le "$p500" 45; then v2="(ii) vs v1@500M $p500% <= 45% => REPORT (with (i))"
else v2="(ii) vs v1@500M $p500% in (45, 55) => continue"; fi
{ echo "R1b-F 300M strength watch (runs/g4r1bf/NOTES.txt '300M STRENGTH WATCH'; NOTES md5 at launch $(md5sum < "$C/runs/g4r1bf/NOTES.txt" | cut -c1-8)), read mechanically by scripts/netv2_g4r1bf/check300-chain.sh"
  echo "$v1"; echo "$v2"; echo "--- $r46"; cat "$r46"; echo "--- $r500"; cat "$r500"; } > "$D/VERDICT.txt.tmp" && mv -f "$D/VERDICT.txt.tmp" "$D/VERDICT.txt" && chmod a-w "$D/VERDICT.txt"
st "VERDICT: $v1 | $v2"
st "done"
