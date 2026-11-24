#!/bin/bash
# wsp P3 (council session 1, 2026-11-24; PREREG $W/council/p3-gen4/PREREG.txt, main-signed): gen4 re-baseline at a
# FIXED 200,000 Foul Play iterations per state. Benchmark-slot job(s), ledger-claimed via $W/LEDGER-OUTBOX.md.
#   stage i : PREREG / code / input / smoke checks -> arm (i) the incumbent gen4 bot (fp4-04 config) -> netv2's
#             $C/council/p3-gen4/FAILED => FAILED; R1BF_CD_READY => stage ii in this allocation; neither => RELEASE
#             the slot and start the READY poller (p3_wait.sbatch every 30 min, at most 48 h after arm (i) ends;
#             on READY it CLAIMs the slot again and submits stage ii)
#   stage ii: READY md5 check -> arm (ii) raw R1b-F-cd at T 0.5 -> p3_read.py -> VERDICT.txt (r--) -> RELEASE
# Arm validity: ITERS_OK with N-count states, no ITERS_MISMATCH, >= 450 complete pairs. A void arm without a
# mismatch (infrastructure: Foul Play / server start, disk) is replayed once with the same seed; a mismatch or a
# second void => FAILED. Any failure writes $D/FAILED (+ STATUS) and releases the slot.
#   cd $W && sbatch --partition=training --dependency=afterany:<smoke job> council/p3-gen4/code/p3_chain.sbatch i
#SBATCH --job-name=ws-wsp-p3g4
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
STAGE=${1:-i}
W=/vol/bitbucket/fr1226/porygon-ws-wsp; C=/vol/bitbucket/fr1226/porygon-ws-netv2; G=/vol/bitbucket/fr1226/gloria
D=$W/council/p3-gen4; K=$D/code; NV=$C/council/p3-gen4; J=${SLURM_JOB_ID:?}
PY=$G/.venv/bin/python
PRE_MD5=a8e0d1ff2de75e449152855b71f44954
CODE_MD5="fp4_iters.sbatch:32433427888e494a928fddd91c1ead29 fp_fixed_iters.py:5b9ff869effb1e44042cd2935acb895b fp_iters_check.py:a1ff4886f5dc67720e4a421cd5659327 p3_read.py:139478fcfc145b1c5f233e22f6e9ab7d"
INC=$G/runs/20261025-195826-301040/final_model.zip
INC_SHA=e90aa0aed241290d7939a49027c4ce838051b73d44233587b3a621a43e9a68a3
EVAL=$G/src/porygon/data/eval_r1.json; EVAL_MD5=178ffcb1c572e652a7290d269d144169
# the incumbent's code (gloria, read only; unchanged since 11-07): src sha256 (the harness manifest's recipe), the
# vs_foulplay.py it runs and its .venv gen4 engine build
G_SRC=6a27438de4c7b399; G_VS=4e1f0c21e160a066b6af98577a3a1eb3; G_ENG=b2c234ec0b56d11ce716f6482061738a
G_ENG_SO=$G/.venv/lib/python3.12/site-packages/poke_engine/poke_engine.cpython-312-x86_64-linux-gnu.so
SEED=4031124; ITERS=200000; N=1000; MIN_PAIRS=450; WAIT_H=48
st() { echo "$(date -u +%FT%TZ) [$J $STAGE] $*" | tee -a "$D/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$W/LEDGER-OUTBOX.md" ) 9> "$W/LEDGER-OUTBOX.lock"; }
release() { outbox "$(date -u +%FT%H:%MZ) | wsp | RELEASE | $J | $1; slot FREE"; st "RELEASE: $1"; }
fail() { echo "$(date -u +%FT%TZ) job $J stage $STAGE: $1" >> "$D/FAILED"; st "FAILED: $1"; }
die() { fail "$1"; release "P3 stage $STAGE stopped: FAILED ($1)"; exit 1; }
summary() { head -n 1 "$1/result.txt" 2>/dev/null | cut -c1-200; }

checks() {
  local f m
  [ "$(md5sum < "$D/PREREG.txt" | cut -c1-32)" = "$PRE_MD5" ] && [ ! -w "$D/PREREG.txt" ] || die "PREREG md5 / mode wrong"
  for f in $CODE_MD5; do
    m=$(md5sum < "$K/${f%%:*}" 2>/dev/null | cut -c1-32)
    [ "$m" = "${f#*:}" ] || die "code ${f%%:*} md5 ${m:-missing} != ${f#*:}"
  done
  [ "$(sha256sum < "$INC" | cut -c1-64)" = "$INC_SHA" ] || die "incumbent prior net sha256 mismatch"
  [ "$(md5sum < "$EVAL" | cut -c1-32)" = "$EVAL_MD5" ] || die "fitted eval md5 mismatch"
  [ "${SLURM_CPUS_ON_NODE:-0}" = 24 ] || die "need 24 CPUs (the incumbent's fp4-04 environment), got ${SLURM_CPUS_ON_NODE:-?}"
  if [ "$STAGE" = i ] && [ ! -s "$D/ARM_i" ]; then
    [ "$(cd "$G" && find src/ -name '*.py' | sort | xargs cat | sha256sum | cut -c1-16)" = "$G_SRC" ] ||
      die "gloria's src changed since the PREREG (sha256 != $G_SRC): the incumbent would not be the pre-registered code"
    [ "$(md5sum < "$G/scripts/vs_foulplay.py" | cut -c1-32)" = "$G_VS" ] || die "gloria's scripts/vs_foulplay.py changed"
    [ "$(md5sum < "$G_ENG_SO" | cut -c1-32)" = "$G_ENG" ] || die "gloria's .venv engine build changed"
  fi
  st "checks OK (PREREG $PRE_MD5, code, nets, 24 CPUs; chain md5 $(md5sum < "$0" | cut -c1-8); netv2_common.sh $(md5sum < "$C/scripts/netv2_common.sh" | cut -c1-8), netv2 vs_foulplay.py $(md5sum < "$C/scripts/vs_foulplay.py" | cut -c1-8))"
}

# run_arm NAME PLAYER KFP NET [vs_foulplay args]: 1000 games vs Foul Play gen4 at ITERS; $D/ARM_<NAME> = the run dir
run_arm() {
  local name=$1 player=$2 kfp=$3 net=$4; shift 4
  local a R pairs off
  for a in 1 2; do
    if [ "$name" = i ]; then off=$(( a - 1 )); else off=$(( a + 1 )); fi  # one port per arm run (0-3)
    st "arm ($name) attempt $a: $player $net, K $kfp, Foul Play FIXED $ITERS iterations per 1000 ms, $N games, seed $SEED"
    ( cd "$C" && SLURM_SUBMIT_DIR=$C bash "$K/fp4_iters.sbatch" "$net" "$N" --gen 4 --player "$player" --k "$kfp" \
        --seed "$SEED" --fp-iters "$ITERS" --tag "$name" --port-offset "$off" "$@" ) > "$D/arm-$name-$J-$a.log" 2>&1
    R=$(ls -td "$D"/runs/foulplay-"$name"-*-"$J" 2>/dev/null | head -n 1)
    if [ -z "$R" ]; then st "arm ($name) attempt $a: VOID (no run dir; arm-$name-$J-$a.log)"; continue; fi
    if [ -f "$R/ITERS_MISMATCH" ]; then fail "arm ($name): iteration check: $(cut -c1-300 "$R/ITERS_MISMATCH")"; return 1; fi
    pairs=$("$PY" -I "$K/p3_read.py" pairs "$R" 2>/dev/null) || pairs=0
    if grep -qs "OK (.*N [1-9]" "$R/ITERS_OK" && [ "$pairs" -ge "$MIN_PAIRS" ]; then
      echo "$R" > "$D/ARM_$name"
      st "arm ($name) VALID: $pairs complete pairs; $(cat "$R/ITERS_OK"); $(summary "$R")"
      [ "$player" = search ] && st "arm ($name) player: $(grep -o 'SearchPlayer([^)]*)' "$R/vs.log" | head -n 1)"
      return 0
    fi
    st "arm ($name) attempt $a: VOID ($pairs complete pairs; ITERS_OK: $(cat "$R/ITERS_OK" 2>/dev/null || echo none)); kept as $(basename "$R")-void$a"
    mv "$R" "$R-void$a"
  done
  fail "arm ($name) void twice (see STATUS)"; return 1
}

stage_ii() {
  local net md5 step rest RI RII
  [ -s "$D/ARM_i" ] || die "no valid arm (i) (ARM_i)"
  RI=$(cat "$D/ARM_i")
  read -r net md5 step rest < "$NV/R1BF_CD_READY" || die "R1BF_CD_READY unreadable"
  [ -f "$net" ] || die "READY net $net does not exist"
  [ "${#md5}" -ge 8 ] && [[ "$(md5sum < "$net" | cut -c1-32)" == "$md5"* ]] || die "READY net md5 != $md5"
  st "READY: $net md5 $md5 step $step"
  if [ -s "$D/ARM_ii" ]; then st "arm (ii) already done: $(cat "$D/ARM_ii")"; else
    run_arm ii policy 20 "$net" --temperature 0.5 || { release "P3 arm (ii) FAILED (STATUS)"; exit 1; }
  fi
  RII=$(cat "$D/ARM_ii")
  "$PY" -I "$K/p3_read.py" verdict "$RI" "$RII" \
    "PREREG $D/PREREG.txt md5 $PRE_MD5; seed $SEED; Foul Play $ITERS iterations per 1000 ms state (fixed); jobs: (i) $(grep -o 'job: [0-9]*' "$RI/manifest.txt" | head -n 1 | cut -d' ' -f2), (ii) $J" \
    "R1b-F-cd (READY): $net md5 $md5 step $step" \
    "ITERS_OK (i): $(cat "$RI/ITERS_OK")" "ITERS_OK (ii): $(cat "$RII/ITERS_OK")" \
    "written $(date -u +%FT%TZ)" > "$D/VERDICT.txt.tmp" 2> "$D/read.err" || die "p3_read.py failed: $(head -c 300 "$D/read.err")"
  mv "$D/VERDICT.txt.tmp" "$D/VERDICT.txt" && chmod a-w "$D/VERDICT.txt"
  st "VERDICT: $(head -n 4 "$D/VERDICT.txt" | tr '\n' ' ' | cut -c1-700)"
  local dl si sii
  dl=$(sed -n 2p "$D/VERDICT.txt" | sed 's/ (paired.*//')
  si=$(sed -n 3p "$D/VERDICT.txt" | grep -o '[0-9.]*% \[[^]]*\]' | head -n 1)
  sii=$(sed -n 4p "$D/VERDICT.txt" | grep -o '[0-9.]*% \[[^]]*\]' | head -n 1)
  release "P3 done: $(head -n 1 "$D/VERDICT.txt"): $dl; (i) incumbent $si, (ii) raw R1b-F-cd $sii vs Foul Play gen4 at a fixed 200k iterations per state; $D/VERDICT.txt"
}

mkdir -p "$D/runs"
st "job $J on $(hostname -s) (${SLURM_CPUS_ON_NODE:-?} CPUs) start"
[ -f "$D/VERDICT.txt" ] && { st "VERDICT.txt exists: nothing to do"; exit 0; }
[ -f "$D/FAILED" ] && { st "FAILED exists: not running (remove it to replay)"; release "P3 stage $STAGE not run (FAILED exists)"; exit 1; }
checks
case "$STAGE" in
  i)
    grep -qs '^PASS' "$D/smoke/RESULT" || die "smoke did not pass: $(head -n 1 "$D/smoke/RESULT" 2>/dev/null || echo 'no smoke/RESULT')"
    if [ -s "$D/ARM_i" ]; then st "arm (i) already done: $(cat "$D/ARM_i")"; else
      run_arm i search 4 "$INC" --ms 1000 --samples 4 --prior-weight 0.25 --eval-config "$EVAL" ||
        { release "P3 arm (i) FAILED (STATUS)"; exit 1; }
    fi
    if [ -f "$NV/FAILED" ]; then
      fail "netv2's R1b-F-cd cooldown FAILED ($NV/FAILED: $(head -c 200 "$NV/FAILED")): no arm (ii)"
      release "P3 arm (i) done ($(summary "$(cat "$D/ARM_i")")); netv2 FAILED => no arm (ii), FAILED"; exit 1
    fi
    if [ -f "$NV/R1BF_CD_READY" ] && [ "$SECONDS" -lt $(( 22 * 3600 )) ]; then stage_ii; exit 0; fi
    [ -f "$D/WAIT_START" ] || date -u +%s > "$D/WAIT_START"
    jid=$(cd "$W" && sbatch --parsable --partition=training --begin=now+30minutes "$K/p3_wait.sbatch") ||
      die "could not submit the READY poller"
    st "READY absent: poller job $jid (every 30 min, until $(date -u -d "@$(( $(cat "$D/WAIT_START") + WAIT_H * 3600 ))" +%FT%TZ))"
    release "P3 arm (i) done ($(summary "$(cat "$D/ARM_i")")); arm (ii) waits for netv2's R1BF_CD_READY (poller $jid, <= ${WAIT_H} h), which CLAIMs the slot again"
    ;;
  ii) stage_ii ;;
  *) die "unknown stage '$STAGE'" ;;
esac
st "done"
