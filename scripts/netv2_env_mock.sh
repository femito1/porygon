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
check 5d "--resume $N/runs/g9/handover-g9p-a30/final_model.pt$" "$o"; check 5e "guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5" "$o"; check 5g "--snapshot-cache 192 --stacked-opp --resume" "$o"
[ -f "$ROOTM/local-A30/run/G9p/pool/00015000000.pt" ] && [ -f "$ROOTM/local-A30/run/G9p/pool/frozen/best-00005016943.pt" ] && [ "$(cat $ROOTM/local-A30/run/G9p/best_model.pt)" = g9p-best ] && [ -f "$N/runs/r1/G9P_A30_STARTED" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 5f: G9p seed"; }
o=$(src g9 A16); check 6a "guard-anchor-drop 0.10 --guard-entropy-rel 1.5 --guard-entropy-min 0.3" "$o"; check 6b "fastenv-workers 8 " "$o"; checkno 6c "kl-anchor" "$o"; check 6g "--snapshot-cache 72 --resume" "$o"
check 6d "--run-dir $ROOTM/local-A16/run/G9s --pool-dir $ROOTM/local-A16/run/G9s/pool" "$o"; check 6e "--resume $N/runs/r1/handover-g9s-a16/final_model.pt$" "$o"
[ -f "$ROOTM/local-A16/run/G9s/pool/00010000000.pt" ] && [ "$(cat $ROOTM/local-A16/run/G9s/best_model.pt)" = g9s-best ] && [ -f "$N/runs/g9/G9S_A16_STARTED" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 6f: G9s seed"; }
echo g9p-a30-final > "$ROOTM/local-A30/run/G9p/final_model.pt"; echo g9s-a16-final > "$ROOTM/local-A16/run/G9s/final_model.pt"
o=$(src r1 A30); check 7a "--resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"; check 7b "--run-dir $ROOTM/local-A30/run/G9p " "$o"
o=$(src g9 A16); check 8a "--resume $ROOTM/local-A16/run/G9s/final_model.pt$" "$o"; check 8b "--run-dir $ROOTM/local-A16/run/G9s " "$o"
# 7c-8e (2026-11-09 evening): the opponent cache follows the GPU's memory (NV2_GPU_MIB; unset + no nvidia-smi = A30 / A16)
o=$(NV2_GPU_MIB=24576 src r1 A30); check 7c "--snapshot-cache 192 --stacked-opp --resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"
o=$(NV2_GPU_MIB=46068 src r1 A30); check 7d "--snapshot-cache 640 --stacked-opp --resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"; checkno 7e "snapshot-cache 192" "$o"
o=$(NV2_GPU_MIB=15356 src g9 A16); check 8c "--snapshot-cache 72 --resume $ROOTM/local-A16/run/G9s/final_model.pt$" "$o"; checkno 8d "stacked-opp" "$o"
o=$(NV2_GPU_MIB=46068 src g9 A16); check 8e "--snapshot-cache 512 --stacked-opp --resume $ROOTM/local-A16/run/G9s/final_model.pt$" "$o"
# 7f-8i (2026-11-09 19:30): successor jobs (an id other than 718701 / 741849 / 829660) train through the 9d0efe0
# wrapper with --snapshot-cache-cpu; the running jobs' restart in place is unchanged; no wrapper file -> no switch
mkdir -p "$N/bin"; printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-9d0efe0"; chmod +x "$N/bin/py-src-9d0efe0"
pyof() { ( set -euo pipefail; NV2=$ROOTM/nv2; LOCAL=$ROOTM/local-$3; RUN_NAME=$2; SLURM_JOB_ID=$1; NV2_HOLD_ITERS=0; source "$ENVF"; echo "PY=${PY:-unset}|PAT=$NV2_DEADLINE_PAT" ) 2>&1; }
o=$(SLURM_JOB_ID=810366 src r1 A30); check 7f "--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 7g "--snapshot-cache 192 --stacked-opp --resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"; checkno 7h "cache-cpu" "$o"
check 7i "^PY=$N/bin/py-src-9d0efe0\|PAT=\^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train\( \|\\\$\)$" "$(pyof 810366 r1 A30)"
check 7j "^PY=unset\|" "$(pyof 718701 r1 A30)"
o=$(SLURM_JOB_ID=810367 src g9 A16); check 8f "--snapshot-cache 72 --snapshot-cache-cpu 200 --resume $ROOTM/local-A16/run/G9s/final_model.pt$" "$o"
o=$(SLURM_JOB_ID=741849 src g9 A16); check 8g "--snapshot-cache 72 --resume $ROOTM/local-A16/run/G9s/final_model.pt$" "$o"; check 8h "^PY=unset\|" "$(pyof 741849 g9 A16)"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=46068 src g9 A16); check 8i "--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume" "$o"
rm -f "$N/bin/py-src-9d0efe0"; o=$(SLURM_JOB_ID=810366 src r1 A30); checkno 7k "cache-cpu" "$o"; check 7l "^PY=unset\|" "$(pyof 810366 r1 A30)"
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
# 15-20 (2026-11-09 evening): R1b-F resumed as run g4r1bf (lineage R1bF) on a dove T4
g4tree() {
  fresh; P=$N/runs/r1b-parked; mkdir -p "$P/pool" "$ROOTM/local-T4/run"; touch "$ROOTM/local-T4/run/train.log" "$ROOTM/local-T4/.toucher-R1bF"
  rm -f "$ROOTM/local-T4/run/final_model.pt"
  echo parked-final > "$P/final_model.pt"; echo parked-best > "$P/best_model.pt"; echo parked-ema > "$P/ema_model.pt"
  for x in 00002540836 00007500005 00020009055; do echo "s$x" > "$P/pool/$x.pt"; done
  echo best75 > "$N/runs/r1/restart-best-7.5M.pt"
}
g4tree; L=$ROOTM/local-T4
o=$(src g4r1bf T4); check 15a "FORMAT=gen4randombattle" "$o"; check 15b "ANCHOR=/vol/bitbucket/fr1226/gloria/runs/resume/pool/ppo_46259712_steps.pt" "$o"
check 15c "--n-layers 4 --ffn-mult 1 --lr 7e-5 --guard-anchor-drop 0.15 --guard-entropy-max 1.0 --eval-async --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 60 --pool-battles 30 --run-dir $L/run/R1bF --pool-dir $L/run/R1bF/pool --snapshot-cache 40 --guard-pg-mean 20 --guard-anchor-drop 0.0999 --resume $N/runs/r1b-parked/final_model.pt$" "$o"
checkno 15d "ent-coef|kl-anchor|frozen" "$o"
[ -f "$N/runs/g4r1bf/R1BF_STARTED" ] && [ -f "$L/run/R1bF/pool/00020009055.pt" ] && [ "$(cat "$L/run/R1bF/best_model.pt")" = parked-best ] && [ "$(cat "$L/run/R1bF/ema_model.pt")" = parked-ema ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 15e: R1b-F first-start seed"; }
o=$( ( set -euo pipefail; NV2=$ROOTM/nv2; LOCAL=$L; RUN_NAME=g4r1bf; SLURM_JOB_NAME=ws-netv2-g4; NV2_HOLD_ITERS=0; source "$ENVF"; echo "PS=$PORYGON_PS|CHAIN=$CHAIN|MIN=$MIN_CORES"; bash -c 'echo "child sees $PORYGON_PS"' ) 2>&1 )
check 15f "PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11\|CHAIN=3\|MIN=10" "$o"; check 15g "child sees /vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11" "$o"
o=$(src r1 A30); checkno 15h "porygon-ws-fastenv" "$o"  # the G9p branch never sees the gen4 settings (separate sourcing)
echo r1bf-final > "$L/run/R1bF/final_model.pt"
o=$(src g4r1bf T4); check 16a "--guard-anchor-drop 0.0999 --resume $L/run/R1bF/final_model.pt$" "$o"; checkno 16b "ent-coef" "$o"
# 17: an anchor-drop trip -> follow-up #1 from the 7.5M checkpoint (not follow-up 2), trip file renamed, snapshots > 7.5M parked
echo "eval score 50.0% < best 62.5% - 10% for 2 evals" > "$L/run/R1bF/GUARD_TRIPPED"; cp "$N/runs/r1b-parked/pool/00020009055.pt" "$L/run/R1bF/pool/"
o=$(src g4r1bf T4); check 17a "--ent-coef 0 --zero-avoid-coef 0 --resume $N/runs/r1/restart-best-7.5M.pt --guard-anchor-drop 0.0999$" "$o"; checkno 17b "guard-pg-mean|p-frozen" "$o"
check 17c "^1 " "$(cat "$N/runs/r1/FU.started" 2>/dev/null)"; check 17d "for 2 evals" "$(cat "$N/runs/g4r1bf/FU.reason" 2>/dev/null)"
[ ! -e "$L/run/R1bF/GUARD_TRIPPED" ] && ls "$L"/run/R1bF/GUARD_TRIPPED.fu-* > /dev/null 2>&1 && [ ! -e "$L/run/R1bF/pool/00020009055.pt" ] && [ -f "$L/run/R1bF/pool/00007500005.pt" ] && [ ! -e "$N/runs/g4r1bf/HOLD" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 17e: follow-up start files"; }
check 17f "follow-up #1 \(user rule\)" "$(cat "$L/run/train.log")"
# 18: the follow-up's next restart (no new trip): still follow-up 1, from its own save, no HOLD
o=$(src g4r1bf T4); check 18a "--ent-coef 0 --zero-avoid-coef 0 .*--guard-anchor-drop 0.0999 --resume $L/run/R1bF/final_model.pt$" "$o"; [ ! -e "$N/runs/g4r1bf/HOLD" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 18b: HOLD without a trip"; }
# 19: a trip of the follow-up -> HOLD (allocation kept), not a second follow-up
echo "eval score 40.0% < best 62.5% - 10% for 2 evals" > "$L/run/R1bF/GUARD_TRIPPED"
o=$(src g4r1bf T4); [ -f "$N/runs/g4r1bf/HOLD" ] && ls "$L"/run/R1bF/GUARD_TRIPPED.second-* > /dev/null 2>&1 && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 19a: no HOLD on the follow-up's trip"; }
check 19b "--ent-coef 0 --zero-avoid-coef 0" "$o"
# 20: a pool trip before any follow-up -> HOLD, no follow-up
g4tree; L=$ROOTM/local-T4; src g4r1bf T4 > /dev/null; echo r1bf-final > "$L/run/R1bF/final_model.pt"
echo "pool: snapshot 00020009055.pt failed to load" > "$L/run/R1bF/GUARD_TRIPPED"
o=$(src g4r1bf T4); [ -f "$N/runs/g4r1bf/HOLD" ] && [ ! -e "$N/runs/r1/FOLLOWUP1" ] && [ ! -e "$N/runs/r1/FU.started" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 20a: pool trip handling"; }
check 20b "--guard-pg-mean 20 --guard-anchor-drop 0.0999 --resume $L/run/R1bF/final_model.pt$" "$o"
echo "passed $pass, failed $fail"
for f in /proc/[0-9]*/fd/1; do if [ "$(readlink "$f" 2>/dev/null)" = "$ROOTM.out" ]; then q=${f#/proc/}; q=${q%%/*}; echo "killing my leftover $(cat /proc/$q/cmdline | tr '\0' ' ') ($q)"; kill "$q" 2>/dev/null; fi; done
