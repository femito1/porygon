#!/bin/bash
# BIGNET: the first bigger-network experiment, on the GPU Dojo VM (1x T4, 4 vCPU, 27 GB; Ubuntu 24.04).
# Design docs/netv2/BIGNET-DESIGN.md, pre-registration docs/netv2/BIGNET-PREREG.md (binding once main signs it off).
#
#   bash run_vm.sh install   # once, after staging (stage_vm.sh): systemd unit that (re)starts `run` at every boot
#   bash run_vm.sh run       # the pipeline (what the unit runs); idempotent: every stage has a marker, a restart
#                            # after a stop resumes (training from state.pt, the h2h by whole 500-pair shards)
#   bash run_vm.sh status    # stage markers, train step, h2h shards, last log lines
#
# Stages: setup -> data (wait for the cluster push, md5) -> T4 microbench (matched-time steps) -> h2h smoke ->
# arm A (D384L4, seed 2) train -> E1A / E2A -> READING A -> only if A reads BIGGER BETTER: control S (BC-P4's size,
# seed 3, trained here on the T4) -> E0 (S vs BC-P4) / E4 (A vs S) -> RESULT.txt -> `sudo shutdown -h now`.
# A failed stage writes FAILED, disables the unit and shuts the VM down (SHUTDOWN_ON_FAIL=1); with FAILED present
# `run` does nothing (inspect, fix, `rm FAILED`, start the unit). The budget guard writes HOLD instead (same effect;
# main removes HOLD to continue: training resumes from state.pt). Kill switch: `touch ~/bignet/NO_AUTORUN`.
# Everything lives on the persistent disk (/): /mnt is wiped when the VM is deallocated.
set -uo pipefail

B=${BIGNET_DIR:-$HOME/bignet}
# ---- pre-registered configuration (BIGNET-PREREG.md; do not edit after sign-off) ----
declare -A MJ=( [base]='{"n_layers":4,"ffn_mult":1}'                     # BC-P4's size (5,809,977 params)
                [D384L4]='{"d_model":384,"n_layers":4,"ffn_mult":1}' )    # arm A (11,606,329 params)
declare -A MAXH=( [D384L4]=24 [base]=16 )         # budget guard: projected training hours at the step-5200 check
ARM_A=D384L4; SEED=2                              # arm A: BC-P4's seed (wrapper --seed 2)
CTRL=base; SEED_S=3                               # control S: BC-P4's size, a new seed, the same T4 / code / data
H2H_SEED=4416017                                  # new seed (B1 used 4410041); every reading uses it (same team pairs)
E1_PAIRS=4000; SHARD=500                          # E1A / E4: 8 shards of 500 pairs
E2_PAIRS=2000                                     # E2A, E0 (informational): 4 shards
TEMP=0.5
BC_P4_MD5=${MOCK_BC_P4_MD5:-693c881e8be07ae6bfe09f5ecdf08145}       # $C/runs/pretrain/P4-g9-s2.pt
BC_P4_STEPS=140880                                # BC-P4's final step (6 epochs x 23,480)
BC_P4_VAL='policy_ce 1.13794 value_bce 0.58486 top1 0.57867'   # its final val line (same 767,203 val decisions)
SRC_COMMIT=745c479                                # BC code of record ($C/src-p2b)
DATA_WAIT_H=4                                     # max hours to wait for the cluster push
STALL_MIN=45                                      # train: no pretrain.log write for this long => FAILED
SHUTDOWN_ON_FAIL=${SHUTDOWN_ON_FAIL:-1}
# ---- paths ----
PY=$HOME/venv/bin/python
SRC=$B/src-p2b
DATA=$B/data/p4/v2pre-human-shuf
NET_P4=$B/nets/P4-g9-s2.pt
EV=$B/eval
LOG=$B/logs/run.log
WSP=$HOME/porygon/scripts/wsp                     # spd_h2h.py (VM setup 11-17)
V2SRC=$HOME/porygon/src-24c633f                   # netv2 src for the h2h nets (as in the 11-17 cal-vm1 check)
UNIT=porygon-bignet
run_dir() { echo "$B/run/g9-p4-$1-s$2"; }         # run_dir CFG SEED
keep_step() { cat "$B/mb/KEEP_$1"; }
keep_ckpt() { echo "$(run_dir "$ARM_A" $SEED)/keep-$(printf %07d "$(keep_step "$ARM_A")").pt"; }

mkdir -p "$B/logs" "$EV" "$B/mb"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
mark() { echo "$(date -u +%FT%TZ) $*" >> "$B/STATUS"; log "STATUS $*"; }
vmoff() {
  log "shutdown: $1 (boot $(uptime -s), up $(awk '{printf "%.2f h", $1/3600}' /proc/uptime))"
  echo "$(date -u +%FT%TZ) boot $(uptime -s) up_s $(awk '{print int($1)}' /proc/uptime) reason $1" >> "$B/logs/uptime.log"
  sync; sudo shutdown -h now
}
stop_with() {  # stop_with FAILED|HOLD message
  log "$1: $2"; echo "$(date -u +%FT%TZ) $2" > "$B/$1"; mark "$1 $2"
  sudo systemctl disable "$UNIT" >/dev/null 2>&1
  [ "$SHUTDOWN_ON_FAIL" = 1 ] && vmoff "$1 $2"
  exit 1
}
fail() { stop_with FAILED "$*"; }

h2h_env() {
  export PORYGON_PS=${PORYGON_PS:-$HOME/porygon/pokemon-showdown}
  export PORYGON_NODE=${PORYGON_NODE:-$(command -v node || ls $HOME/node/bin/node 2>/dev/null)}
  export SPD_FASTENV_JS=${SPD_FASTENV_JS:-$(dirname "$(ls -d $HOME/porygon/spd/fastenv-*/fastenv/js/host.js 2>/dev/null | head -1)" 2>/dev/null)}
  export PATH=$(dirname "$PORYGON_NODE"):$PATH
}

# h2h A B PAIRS OUTDIR [SHARDSIZE]: pairs [0, PAIRS) of seed H2H_SEED (smoke: from 900000) in shards; done shards kept
h2h() {
  local A=$1 Bn=$2 N=$3 OUT=$4 S=${5:-$SHARD} BASE=${6:-0} k first n
  mkdir -p "$OUT"; h2h_env
  for ((k = 0; k * S < N; k++)); do
    first=$((BASE + k * S)); n=$(( N - k * S < S ? N - k * S : S ))
    local f=$OUT/shard-$(printf %04d $k).jsonl
    if [ -s "$f" ] && [ "$(wc -l < "$f")" = "$n" ]; then continue; fi
    log "h2h $OUT shard $k pairs [$first, $((first + n)))"
    (cd "$HOME/porygon" && $PY -u "$WSP/spd_h2h.py" run --a "v2:$A,temperature=$TEMP" --b "v2:$Bn,temperature=$TEMP" \
       --pairs "$n" --first "$first" --shard "$k" --seed "$H2H_SEED" --format gen9randombattle --workers 3 \
       --device cuda --v2-src "$V2SRC" --out "$OUT" --overwrite --progress 300) >> "$OUT/h2h.log" 2>&1 9>&-
    [ $? = 0 ] && [ -s "$f" ] && [ "$(wc -l < "$f")" = "$n" ] || return 1
  done
  $PY -u "$WSP/spd_h2h.py" merge "$OUT" --out "$OUT/summary.json" >> "$OUT/h2h.log" 2>&1
}

# reading of a summary.json: "BIGGER BETTER" | "BIGGER WORSE" | "NO DIFFERENCE SHOWN" (A = side named first)
reading() {
  $PY -c 'import json,sys; s=json.load(open(sys.argv[1])); m,h=100*s["win_rate"],100*s["ci95"]
print("BIGGER BETTER" if m-h>50 else "BIGGER WORSE" if m+h<50 else "NO DIFFERENCE SHOWN")' "$1"
}

stage_setup() {
  local gpu; gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  [[ "$gpu" == *T4* ]] || fail "setup: GPU is '$gpu', not a T4"
  $PY -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' || fail "setup: torch has no CUDA"
  [ "$(cat "$SRC/GIT_COMMIT" 2>/dev/null)" = "$SRC_COMMIT" ] || fail "setup: $SRC/GIT_COMMIT is not $SRC_COMMIT"
  [ -f "$WSP/spd_h2h.py" ] && [ -f "$V2SRC/porygon/v2/model.py" ] || fail "setup: no $WSP/spd_h2h.py or $V2SRC"
  h2h_env
  [ -x "$PORYGON_NODE" ] && [ -f "$SPD_FASTENV_JS/host.js" ] && [ -d "$PORYGON_PS" ] || \
    fail "setup: h2h env incomplete (node '$PORYGON_NODE', fastenv '$SPD_FASTENV_JS', PS '$PORYGON_PS')"
  local free_gb mem_gb; free_gb=$(df -BG --output=avail "$B" | tail -1 | tr -dc 0-9)
  mem_gb=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
  [ "$free_gb" -ge 3 ] || fail "setup: only ${free_gb} GB free"
  [ "$mem_gb" -ge "${MEM_MIN_GB:-24}" ] || fail "setup: only ${mem_gb} GB RAM (BC-P4 peaked at 16.0 GB RSS)"
  log "setup ok: $gpu, torch $($PY -c 'import torch; print(torch.__version__)'), src $SRC_COMMIT, ${free_gb} GB free, ${mem_gb} GB RAM, node $($PORYGON_NODE --version), $(nproc) CPUs"
}

stage_data() {
  [ -f "$B/DATA_OK" ] && return 0
  local t0=$(date +%s)
  until [ -f "$B/data/PUSH_DONE" ]; do
    [ $(( $(date +%s) - t0 )) -gt $((DATA_WAIT_H * 3600)) ] && fail "data: no PUSH_DONE after ${DATA_WAIT_H} h"
    sleep 60
  done
  log "data: push done ($(cat "$B/data/PUSH_DONE")); md5 of $(grep -c 'shard-' "$B/data/p4/MD5SUMS.shuf") shards"
  (cd "$B/data/p4" && md5sum --quiet -c MD5SUMS.shuf) >> "$LOG" 2>&1 || fail "data: md5 mismatch (see $LOG)"
  [ "$(ls "$DATA"/shard-*.npz | wc -l)" = 533 ] || fail "data: not 533 shards"
  [ "$(md5sum < "$NET_P4" | cut -c1-32)" = "$BC_P4_MD5" ] || fail "data: $NET_P4 md5 is not $BC_P4_MD5"
  [ "$(cat "$SRC/GIT_COMMIT" 2>/dev/null)" = "$SRC_COMMIT" ] || fail "data: $SRC/GIT_COMMIT is not $SRC_COMMIT"
  touch "$B/DATA_OK"; mark "DATA_OK $(du -sb "$DATA" | cut -f1) bytes"
  # orchestrator 11-17: the cluster push key is needed only until the data is verified; revoke it on the VM now
  if grep -q 'porygon-vm-push' ~/.ssh/authorized_keys 2>/dev/null; then
    sed -i '/porygon-vm-push/d' ~/.ssh/authorized_keys && log "data: cluster push key revoked from authorized_keys"
  fi
}

stage_mb() {
  [ -f "$B/MB_OK" ] && return 0
  rm -f "$B/mb/t4.jsonl"
  PYTHONPATH=$SRC $PY -u "$B/microbench.py" --shard "$DATA/shard-00000.npz" --out "$B/mb/t4.jsonl" \
    --modes tf32,fp16 --cfgs base,$ARM_A >> "$B/mb/t4.log" 2>&1 || fail "microbench"
  for a in $ARM_A; do  # matched T4 time: the arm's step that used BC-P4's 140,880 steps of base step time
    $PY - "$B/mb/t4.jsonl" "$BC_P4_STEPS" "$a" > "$B/mb/KEEP_$a" <<'EOF' || fail "microbench parse"
import json, sys
r = {x["cfg_name"]: x for x in map(json.loads, open(sys.argv[1]))}
print(int(int(sys.argv[2]) * r["base"]["tf32/train_step_s"] / r[sys.argv[3]]["tf32/train_step_s"]) // 1000 * 1000)
EOF
  done
  touch "$B/MB_OK"; mark "MB_OK keep $ARM_A $(keep_step $ARM_A)"
}

stage_smoke() {
  [ -f "$B/SMOKE_OK" ] && return 0
  rm -rf "$EV/smoke"
  h2h "$NET_P4" "$NET_P4" 20 "$EV/smoke" 20 900000 || fail "smoke h2h (see $EV/smoke/h2h.log)"
  grep -q '"pairs": 20' "$EV/smoke/summary.json" || fail "smoke: summary has not 20 pairs"
  touch "$B/SMOKE_OK"; mark "SMOKE_OK $(grep -o '"line": "[^"]*"' "$EV/smoke/summary.json")"
}

stage_train() {  # stage_train CFG SEED [KEEP_STEPS]
  local a=$1 sd=$2 kp=${3:-} R; R=$(run_dir "$1" "$2")
  local tag=$a-s$sd
  [ -f "$B/TRAIN_DONE_$tag" ] && return 0
  mkdir -p "$R"
  log "train $tag ${MJ[$a]}: keep steps '$kp', $( [ -f "$R/state.pt" ] && echo resume || echo fresh)"
  PYTHONPATH=$SRC OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
    $PY -u "$B/pretrain_resumable.py" --out "$R" --shards "$DATA/shard-*.npz" --gen 9 --model-json "${MJ[$a]}" \
    --epochs 6 --seed $sd --value-sample one-per-game --val-frac 0.06 --no-bf16 --stream-gb 1.0 \
    --stream-parts shards --eval-every 5000 --state-every 1000 --keep-steps "$kp" >> "$R/stdout.log" 2>&1 9>&- &
  local pid=$! guard=0
  echo $pid > "$B/train.pid"
  while kill -0 $pid 2>/dev/null; do
    echo "$(date -u +%T) rss_kb $(ps -o rss= -p $pid | tr -d ' ') gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)" >> "$R/mem.log"
    local age=$(( $(date +%s) - $(stat -c %Y "$R/pretrain.log" 2>/dev/null || date +%s) ))
    if [ $age -gt $((STALL_MIN * 60)) ]; then kill -INT $pid; sleep 60; kill -9 $pid 2>/dev/null; fail "train $tag: stalled ${age}s"; fi
    if [ $guard = 0 ] && [ ! -f "$B/GUARD_OK_$tag" ]; then  # budget guard, once, on the first train line at step >= 5200
      local proj; proj=$($PY - "$R/pretrain.log" <<'EOF'
import json, sys
tr = [json.loads(l) for l in open(sys.argv[1]) if '"event": "train"' in l]
tr = [t for t in tr if t["step"] >= 5200]
print(f"{140880 * 512 / tr[-1]['samples_per_s'] / 3600 * 1.07:.1f}" if tr else "")
EOF
)
      if [ -n "$proj" ]; then
        guard=1
        if awk -v p="$proj" -v m="${MAXH[$a]}" 'BEGIN {exit !(p > m)}'; then
          kill -INT $pid; sleep 60; kill -9 $pid 2>/dev/null
          stop_with HOLD "budget guard $tag: projected ${proj} h > ${MAXH[$a]} h (state.pt kept; to continue: rm HOLD; touch GUARD_OK_$tag)"
        fi
        touch "$B/GUARD_OK_$tag"; mark "GUARD_OK $tag projected ${proj} h (limit ${MAXH[$a]} h)"
      fi
    fi
    sleep 60
  done
  wait $pid; local rc=$?
  [ $rc = 0 ] && [ -f "$R/final.pt" ] || fail "train $tag: exit $rc (tail: $(tail -c 300 "$R/stdout.log" | tr '\n' ' '))"
  [ -z "$kp" ] || [ -f "$R/keep-$(printf %07d "$kp").pt" ] || fail "train $tag: no keep-$kp checkpoint"
  touch "$B/TRAIN_DONE_$tag"
  mark "TRAIN_DONE $tag final md5 $(md5sum < "$R/final.pt" | cut -c1-8) $(grep '"event": "val"' "$R/pretrain.log" | tail -1 | grep -o '"policy_ce": [0-9.]*, "value_bce": [0-9.]*')"
}

stage_eval() {  # stage_eval NAME A B PAIRS
  [ -f "$EV/$1/summary.json" ] && return 0
  h2h "$2" "$3" "$4" "$EV/$1" || fail "$1 h2h (see $EV/$1/h2h.log)"
  mark "$1 $(grep -o '"line": "[^"]*"' "$EV/$1/summary.json") => $(reading "$EV/$1/summary.json")"
}

stage_result() {
  $PY - "$EV" "$B/mb/t4.jsonl" "$BC_P4_VAL" "$(run_dir $ARM_A $SEED)" "$(run_dir $CTRL $SEED_S)" "$ARM_A" > "$B/RESULT.txt" <<'EOF' || fail "result"
import json, os, sys
ev, mbp, p4v, runA, runS, A = sys.argv[1:7]
p4 = p4v.split(); p4 = dict(zip(p4[::2], map(float, p4[1::2])))
mb = {x["cfg_name"]: x for x in map(json.loads, open(mbp))}
def s(name):
    p = f"{ev}/{name}/summary.json"
    if not os.path.exists(p):
        return None
    x = json.load(open(p)); m, h = 100 * x["win_rate"], 100 * x["ci95"]
    rd = "BIGGER BETTER" if m - h > 50 else "BIGGER WORSE" if m + h < 50 else "NO DIFFERENCE SHOWN"
    return f"{m:.2f}% ± {h:.2f} [{m - h:.2f}, {m + h:.2f}] ({x['pairs']} pairs, {x['games']} games, void {x['void_games']}) => {rd}"
print("BIGNET RESULT (docs/netv2/BIGNET-PREREG.md); T 0.5, duplicate pairs, pair-mean 95% intervals, h2h seed 4416017")
e1a, e4 = s("e1a"), s("e4")
print(f"E1A (decision of record, arm A {A} final vs BC-P4): {e1a}")
print(f"E4  (confirmation, arm A vs control S = BC-P4's size, seed 3, same T4/code; run only if E1A = BIGGER BETTER): {e4 or 'not run'}")
if e4 is None:
    final = e1a.split("=> ")[1]
else:
    final = "CONFIRMED" if e4.endswith("BIGGER BETTER") else "NOT CONFIRMED"
print(f"OVERALL: {final}")
print(f"E2A (informational, {A} at matched T4 step time vs BC-P4): {s('e2a')}")
print(f"E0  (informational, control S vs BC-P4: seed + hardware noise; 'BIGGER' = S here): {s('e0') or 'not run'}")
for arm, rd in ((A, runA), ("S", runS)):
    lp = f"{rd}/pretrain.log"
    if not os.path.exists(lp) or not os.path.exists(f"{rd}/final.pt"):
        continue
    v = [json.loads(l) for l in open(lp) if '"event": "val"' in l][-1]
    print(f"held-out {arm} (step {v['step']}; same 767,203 val decisions): policy CE {v['policy_ce']:.5f} "
          f"(BC-P4 {p4['policy_ce']:.5f}, delta {v['policy_ce'] - p4['policy_ce']:+.5f}); value BCE {v['value_bce']:.5f} "
          f"(delta {v['value_bce'] - p4['value_bce']:+.5f}); top-1 {v['top1_vs_target_argmax']:.5f} (BC-P4 {p4['top1']:.5f})")
for k, r in mb.items():
    print(f"T4 microbench {k}: {r['params']:,} params, train {r['tf32/train_samples_per_s']:.0f} samples/s fp32 "
          f"({r['fp16/train_samples_per_s']:.0f} fp16), fwd bs512 {r['tf32/fwd_bs512_decisions_per_s']:.0f}/s, "
          f"bs64 {r['tf32/fwd_bs64_decisions_per_s']:.0f}/s, peak {r['tf32/peak_mem_gib']:.2f} GiB")
EOF
  cat "$B/logs/uptime.log" >> "$B/RESULT.txt"
  cat "$B/RESULT.txt" >> "$LOG"
  mark "RESULT written"
}

cmd=${1:-run}
case $cmd in
  install)
    sudo tee /etc/systemd/system/$UNIT.service > /dev/null <<EOF
[Unit]
Description=Porygon BIGNET pipeline (run_vm.sh run)
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
User=$(id -un)
WorkingDirectory=$B
ExecStart=/bin/bash $B/run_vm.sh run
Restart=no
[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload && sudo systemctl enable $UNIT && echo "installed + enabled $UNIT" ;;
  status)
    cat "$B/STATUS" 2>/dev/null; ls "$B"/FAILED "$B"/HOLD "$B"/NO_AUTORUN "$B"/DONE 2>/dev/null
    for r in "$(run_dir $ARM_A $SEED)" "$(run_dir $CTRL $SEED_S)"; do
      grep '"event": "train"' "$r/pretrain.log" 2>/dev/null | tail -1 | cut -c1-140 | sed "s|^|$(basename "$r"): |"
    done
    for d in "$EV"/e*; do [ -d "$d" ] && echo "$d: $(ls "$d"/shard-*.jsonl 2>/dev/null | grep -vc part) shards"; done
    tail -n 5 "$LOG" 2>/dev/null ;;
  run)
    exec 9> "$B/.lock"; flock -n 9 || { echo "another run_vm.sh holds $B/.lock"; exit 0; }
    for f in NO_AUTORUN FAILED HOLD DONE; do
      [ -f "$B/$f" ] && { log "$f present ($(head -c 200 "$B/$f")): exit without doing anything"; exit 0; }
    done
    echo "$(date -u +%FT%TZ) boot $(uptime -s) start" >> "$B/logs/uptime.log"
    log "run: boot $(uptime -s)"
    stage_setup; stage_data; stage_mb; stage_smoke
    stage_train $ARM_A $SEED "$(keep_step $ARM_A)"
    stage_eval e1a "$(run_dir $ARM_A $SEED)/final.pt" "$NET_P4" $E1_PAIRS
    stage_eval e2a "$(keep_ckpt)" "$NET_P4" $E2_PAIRS
    if [ "$(reading "$EV/e1a/summary.json")" = "BIGGER BETTER" ]; then
      stage_train $CTRL $SEED_S
      stage_eval e4 "$(run_dir $ARM_A $SEED)/final.pt" "$(run_dir $CTRL $SEED_S)/final.pt" $E1_PAIRS
      stage_eval e0 "$(run_dir $CTRL $SEED_S)/final.pt" "$NET_P4" $E2_PAIRS
    else
      mark "control S not run: E1A = $(reading "$EV/e1a/summary.json")"
    fi
    stage_result
    touch "$B/DONE"; mark "DONE"
    sudo systemctl disable $UNIT >/dev/null 2>&1
    vmoff "DONE" ;;
  *) echo "usage: $0 install|run|status"; exit 2 ;;
esac
