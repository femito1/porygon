#!/bin/bash
# Mock test of the cluster netv2.env (run on the laptop before every upload; the master lives on the cluster):
#   scp gpucluster2:/vol/bitbucket/fr1226/porygon-ws-netv2/netv2.env /tmp/x/netv2.env; edit; bash scripts/netv2_env_mock.sh /tmp/x/netv2.env
# Sources the file as slurm_train_v2.sbatch does at a restart in place (set -euo pipefail, fake NV2 / LOCAL trees in
# MOCK_ROOT) for the 2026-11-09 modes (r1 MODE g9s on the A30, g9 MODE g9p on the A16), the dormant swap modes, first
# trips -> each lineage's fallback, second trips -> HOLD, the INLINE_CMD -> MODE -> WAIT_FOR flow; prints FAIL lines and a
# pass count. Update the expected strings when you change netv2.env on purpose. Also diff the current modes' TRAIN_ARGS
# between the old and the new file before deploying (only the intended flags may differ).
# print what the trainer would get. Usage: bash mock.sh ENVFILE
ENVF=$(readlink -f "$1")
ROOTM=${MOCK_ROOT:-/tmp/netv2-env-mock}
pass=0; fail=0
check() {  # check NAME 'grep -E pattern' WHAT
  if printf '%s\n' "$3" | grep -qE -- "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [$1]: /$2/ not in: $3"; fi
}
checkno() {
  if printf '%s\n' "$3" | grep -qE -- "$2"; then fail=$((fail + 1)); echo "FAIL [$1]: /$2/ unexpectedly in: $3"; else pass=$((pass + 1)); fi
}
fresh() {  # a fresh fake cluster tree: NV2 + two job LOCAL dirs in the 06:50 state
  rm -rf "$ROOTM"; mkdir -p "$ROOTM"
  N=$ROOTM/nv2; mkdir -p "$N/runs/r1" "$N/runs/g9" "$N/runs/g9s" "$N/runs/g9a30" "$N/runs/pretrain" "$N/runs/r1/handover-g9p/pool/frozen"
  echo g9s > "$N/runs/r1/MODE"; echo g9p > "$N/runs/g9/MODE"
  touch "$N/runs/r1/G9_STARTED" "$N/runs/r1/LINEAGE_DIRS" "$N/runs/g9/LINEAGE_DIRS" "$N/runs/g9/G9P_STARTED" "$N/runs/g9/R1B_STARTED" "$N/runs/r1/R1B_PARKED"
  echo x > "$N/runs/pretrain/P1-g9h-dmg-s2.pt"; echo x > "$N/runs/g9s/init-g9s.pt"; echo x > "$N/runs/r1/handover-g9p/final_model.pt"
  for j in A30 A16; do
    L=$ROOTM/local-$j; mkdir -p "$L/run"; echo top > "$L/run/final_model.pt"; touch "$L/run/train.log"
    touch "$L/.toucher-G9s" "$L/.toucher-G9" "$L/.toucher-G9p" "$L/.toucher-R1b"
  done
  # A30: G9s lineage (and the stale G9 segment 6.5-13.0M); A16: G9 (= G9p) lineage
  L=$ROOTM/local-A30; mkdir -p "$L/run/G9s/pool/frozen" "$L/run/G9/pool"; echo g9s-final > "$L/run/G9s/final_model.pt"
  echo g9s-best > "$L/run/G9s/best_model.pt"; echo s > "$L/run/G9s/pool/00010000000.pt"; echo stale > "$L/run/G9/final_model.pt"
  L=$ROOTM/local-A16; mkdir -p "$L/run/G9/pool/frozen"; echo g9p-final > "$L/run/G9/final_model.pt"; echo g9p-best > "$L/run/G9/best_model.pt"
  echo p > "$L/run/G9/pool/00015000000.pt"; echo f > "$L/run/G9/pool/frozen/best-00005016943.pt"
}
handover() {  # handover DIR SRC: what the swap INLINE_CMD writes
  mkdir -p "$1"; cp -rp "$2/pool" "$1/pool"; for f in best_model.pt ema_model.pt; do [ -f "$2/$f" ] && cp -p "$2/$f" "$1/"; done
  cp -p "$2/final_model.pt" "$1/final_model.pt"
}
src() { _src "$@" > "$ROOTM.out" 2>&1; cat "$ROOTM.out"; }
_src() {  # _src RUN_NAME JOB(A30|A16) -> prints FORMAT|ANCHOR|TRAIN_ARGS after a restart-in-place source
  ( set -euo pipefail
    NV2=$ROOTM/nv2; LOCAL=$ROOTM/local-$2; RUN_NAME=$1; SLURM_JOB_NAME=ws-netv2-x
    NV2_HOLD_ITERS=${NV2_HOLD_ITERS:-0}; NV2_WAIT_ITERS=${NV2_WAIT_ITERS:-1}; NV2_WAIT_SLEEP=0
    source "$ENVF"
    echo "FORMAT=$FORMAT|ANCHOR=$ANCHOR|TRAIN_ARGS=$TRAIN_ARGS" )
}
N=$ROOTM/nv2
# 1-2: the current modes are unchanged
fresh
o=$(src r1 A30); check 1a "FORMAT=gen9randombattle" "$o"; check 1b "--guard-anchor-drop 0.10 --guard-entropy-rel 1.5 --guard-entropy-min 0.3" "$o"; check 1c "fastenv-workers 28" "$o"
check 1d "--run-dir $ROOTM/local-A30/run/G9s --pool-dir $ROOTM/local-A30/run/G9s/pool" "$o"; check 1e "--resume $ROOTM/local-A30/run/G9s/final_model.pt$" "$o"
checkno 1f "kl-anchor" "$o"; check 1g "--snapshot-cache 96 --stacked-opp --resume" "$o"
o=$(src g9 A16); check 2a "--kl-anchor $N/runs/pretrain/P1-g9h-dmg-s2.pt --kl-anchor-coef 0.1" "$o"; check 2b "fastenv-workers 8 " "$o"
check 2c "--run-dir $ROOTM/local-A16/run/G9 --pool-dir" "$o"; check 2d "--resume $ROOTM/local-A16/run/G9/final_model.pt$" "$o"; check 2e "guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5" "$o"; check 2f "--snapshot-cache 32 --resume" "$o"; checkno 2g "stacked-opp" "$o"
# 3-4: new modes set but no handover -> the current lineage continues
fresh; echo g9p > "$N/runs/r1/MODE"; echo g9s > "$N/runs/g9/MODE"
o=$(src r1 A30); check 3a "--run-dir $ROOTM/local-A30/run/G9s " "$o"; check 3b "--resume $ROOTM/local-A30/run/G9s/final_model.pt$" "$o"
check 3c "no G9p handover" "$(cat $ROOTM/local-A30/run/train.log)"; [ ! -e "$N/runs/r1/G9P_A30_STARTED" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 3d: STARTED written"; }
o=$(src g9 A16); check 4a "--run-dir $ROOTM/local-A16/run/G9 " "$o"; check 4b "kl-anchor-coef 0.1" "$o"; check 4c "no G9s handover" "$(cat $ROOTM/local-A16/run/train.log)"
check 4d "--resume $ROOTM/local-A16/run/G9/final_model.pt$" "$o"
# 5-8: handovers present -> first start of each new lineage dir, then a later restart
fresh; echo g9p > "$N/runs/r1/MODE"; echo g9s > "$N/runs/g9/MODE"
handover "$N/runs/g9/handover-g9p-a30" "$ROOTM/local-A16/run/G9"; handover "$N/runs/r1/handover-g9s-a16" "$ROOTM/local-A30/run/G9s"
o=$(src r1 A30); check 5a "kl-anchor-coef 0.1" "$o"; check 5b "fastenv-workers 28" "$o"; check 5c "--run-dir $ROOTM/local-A30/run/G9p --pool-dir $ROOTM/local-A30/run/G9p/pool" "$o"
check 5d "--resume $N/runs/g9/handover-g9p-a30/final_model.pt$" "$o"; check 5e "guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5" "$o"; check 5g "--snapshot-cache 96 --stacked-opp --resume" "$o"
[ -f "$ROOTM/local-A30/run/G9p/pool/00015000000.pt" ] && [ -f "$ROOTM/local-A30/run/G9p/pool/frozen/best-00005016943.pt" ] && [ "$(cat $ROOTM/local-A30/run/G9p/best_model.pt)" = g9p-best ] && [ -f "$N/runs/r1/G9P_A30_STARTED" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 5f: G9p seed"; }
o=$(src g9 A16); check 6a "guard-anchor-drop 0.10 --guard-entropy-rel 1.5 --guard-entropy-min 0.3" "$o"; check 6b "fastenv-workers 8 " "$o"; checkno 6c "kl-anchor" "$o"; check 6g "--snapshot-cache 32 --resume" "$o"
check 6d "--run-dir $ROOTM/local-A16/run/G9s --pool-dir $ROOTM/local-A16/run/G9s/pool" "$o"; check 6e "--resume $N/runs/r1/handover-g9s-a16/final_model.pt$" "$o"
[ -f "$ROOTM/local-A16/run/G9s/pool/00010000000.pt" ] && [ "$(cat $ROOTM/local-A16/run/G9s/best_model.pt)" = g9s-best ] && [ -f "$N/runs/g9/G9S_A16_STARTED" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 6f: G9s seed"; }
echo g9p-a30-final > "$ROOTM/local-A30/run/G9p/final_model.pt"; echo g9s-a16-final > "$ROOTM/local-A16/run/G9s/final_model.pt"
o=$(src r1 A30); check 7a "--resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"; check 7b "--run-dir $ROOTM/local-A30/run/G9p " "$o"
o=$(src g9 A16); check 8a "--resume $ROOTM/local-A16/run/G9s/final_model.pt$" "$o"; check 8b "--run-dir $ROOTM/local-A16/run/G9s " "$o"
# 9-10: first guard trips after the swap use each lineage's own fallback
echo "eval score 40.0% < best 60.0% - 15 points for 2 evals" > "$ROOTM/local-A30/run/G9p/GUARD_TRIPPED"; echo g9p-best2 > "$ROOTM/local-A30/run/G9p/best_model.pt"
o=$(src r1 A30); check 9a "--lr 3.5e-5" "$o"; check 9b "--resume $N/runs/g9a30/fu-start-best.pt$" "$o"; check 9c "^2 " "$(cat $N/runs/g9a30/FU.started)"
[ "$(cat $N/runs/g9a30/fu-start-best.pt)" = g9p-best2 ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 9d: fallback best"; }
echo "collapse: entropy < 0.3 for 5 updates" > "$ROOTM/local-A16/run/G9s/GUARD_TRIPPED"
o=$(src g9 A16); check 10a "--lr 3.5e-5" "$o"; check 10b "--resume $N/runs/g9s/fu-start-best.pt$" "$o"; check 10c "guard-entropy-min 0.3" "$o"
# 11-12: a second trip after the fallback holds the allocation
echo "collapse: P(top >= 0.9) > 0.6 for 5 updates" > "$ROOTM/local-A30/run/G9p/GUARD_TRIPPED"
o=$(src r1 A30); [ -f "$N/runs/r1/HOLD" ] && ls "$ROOTM"/local-A30/run/G9p/GUARD_TRIPPED.second-* >/dev/null 2>&1 && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 11: no HOLD on second G9p trip"; }
check 11b "--lr 3.5e-5" "$o"
echo "anchor drop" > "$ROOTM/local-A16/run/G9s/GUARD_TRIPPED"
o=$(src g9 A16); [ -f "$N/runs/g9/HOLD" ] && ls "$ROOTM"/local-A16/run/G9s/GUARD_TRIPPED.second-* >/dev/null 2>&1 && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 12: no HOLD on second G9s trip"; }
# 13: the full swap flow through nv2_pause: INLINE_CMD writes the handover, MODE and WAIT_FOR; the other side's handover exists
fresh; handover "$N/runs/g9/handover-g9p-a30" "$ROOTM/local-A16/run/G9"
cat > "$N/runs/r1/INLINE_CMD" <<EOF
set -u
L=\$LOCAL/run/G9s; H=$N/runs/r1/handover-g9s-a16
rm -rf "\$H.tmp"; mkdir -p "\$H.tmp"; cp -rp "\$L/pool" "\$H.tmp/pool"; cp -p "\$L/best_model.pt" "\$H.tmp/"; cp -p "\$L/final_model.pt" "\$H.tmp/final_model.pt"
mv "\$H.tmp" "\$H" && echo g9p > $N/runs/r1/MODE && echo $N/runs/g9/handover-g9p-a30/final_model.pt > $N/runs/r1/WAIT_FOR
EOF
o=$(src r1 A30); check 13a "--run-dir $ROOTM/local-A30/run/G9p " "$o"; check 13b "--resume $N/runs/g9/handover-g9p-a30/final_model.pt$" "$o"
[ -f "$N/runs/r1/handover-g9s-a16/final_model.pt" ] && ls "$N"/runs/r1/INLINE_CMD.done-* >/dev/null && ls "$N"/runs/r1/WAIT_FOR.done-* >/dev/null && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 13c: inline/wait files"; }
check 13d "WAIT_FOR done \(found: 1\)" "$(cat $ROOTM/local-A30/run/train.log)"
# 14: the same flow when the INLINE_CMD fails before MODE (copy failure): G9s continues at once, nothing waits
fresh
printf 'set -u\nfalse && echo g9p > %s/runs/r1/MODE\n' "$N" > "$N/runs/r1/INLINE_CMD"
o=$(src r1 A30); check 14a "--run-dir $ROOTM/local-A30/run/G9s " "$o"; check 14b "--resume $ROOTM/local-A30/run/G9s/final_model.pt$" "$o"
echo "passed $pass, failed $fail"
for f in /proc/[0-9]*/fd/1; do if [ "$(readlink "$f" 2>/dev/null)" = "$ROOTM.out" ]; then q=${f#/proc/}; q=${q%%/*}; echo "killing my leftover $(cat /proc/$q/cmdline | tr '\0' ' ') ($q)"; kill "$q" 2>/dev/null; fi; done
