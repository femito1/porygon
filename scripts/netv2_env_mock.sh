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
    NV2_HOLD_ITERS=${NV2_HOLD_ITERS:-0}; NV2_WAIT_ITERS=${NV2_WAIT_ITERS:-1}; NV2_WAIT_SLEEP=0; NV2_SWAP_ITERS=${NV2_SWAP_ITERS:-1}; NV2_SWAP_SLEEP=${NV2_SWAP_SLEEP:-0}
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
# 7f-8i (2026-11-09 19:30; v5 2026-11-10): successor jobs (an id other than 718701 / 741849 / 829660) train through
# the 24c633f wrapper with --snapshot-cache-cpu; job 718701 (G9p) trains through the same wrapper with unchanged args;
# 741849 / 829660 unchanged; no wrapper file -> no switch
mkdir -p "$N/bin"; printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-24c633f"; chmod +x "$N/bin/py-src-24c633f"
pyof() { ( set -euo pipefail; NV2=$ROOTM/nv2; LOCAL=$ROOTM/local-$3; RUN_NAME=$2; SLURM_JOB_ID=$1; NV2_HOLD_ITERS=0; source "$ENVF"; echo "PY=${PY:-unset}|PAT=$NV2_DEADLINE_PAT" ) 2>&1; }
o=$(SLURM_JOB_ID=810366 src r1 A30); check 7f "--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 7g "--snapshot-cache 192 --stacked-opp --resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"; checkno 7h "cache-cpu" "$o"
check 7i "^PY=$N/bin/py-src-24c633f\|PAT=\^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train\( \|\\\$\)$" "$(pyof 810366 r1 A30)"
check 7j "^PY=$N/bin/py-src-24c633f\|PAT=\^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train\( \|\\\$\)$" "$(pyof 718701 r1 A30)"
o=$(SLURM_JOB_ID=810367 src g9 A16); check 8f "--snapshot-cache 72 --snapshot-cache-cpu 200 --resume $ROOTM/local-A16/run/G9s/final_model.pt$" "$o"
o=$(SLURM_JOB_ID=741849 src g9 A16); check 8g "--snapshot-cache 72 --resume $ROOTM/local-A16/run/G9s/final_model.pt$" "$o"; check 8h "^PY=unset\|" "$(pyof 741849 g9 A16)"
check 8j "^PY=$N/bin/py-src-24c633f\|" "$(pyof 810367 g9 A16)"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=46068 src g9 A16); check 8i "--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume" "$o"
# 7k-8o (v8, 2026-11-10; actor-stream (b) passed, orchestrator GO): with $NV2/bin/py-src-4f18108 present, the G9p
# lineage (718701 and the r1 successors) trains through it with --actor-stream --decoupled in place of --no-overlap
# (position kept, nothing else changes); the g9 lineage (741849, successor 810367) and R1b-F (15o) are unchanged
printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-4f18108"; chmod +x "$N/bin/py-src-4f18108"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 7k "--fastenv-workers 28 --fastenv-battles 48 --actor-stream --decoupled --n-battles 16 --pool-battles 16 .*--snapshot-cache 192 --stacked-opp --resume $ROOTM/local-A30/run/G9p/final_model.pt$" "$o"; checkno 7l "no-overlap|cache-cpu" "$o"
check 7m "^PY=$N/bin/py-src-4f18108\|PAT=\^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train\( \|\\\$\)$" "$(pyof 718701 r1 A30)"
o=$(SLURM_JOB_ID=810366 src r1 A30); check 7n "--fastenv-battles 48 --actor-stream --decoupled .*--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume" "$o"; checkno 7o "no-overlap" "$o"
check 7p "^PY=$N/bin/py-src-4f18108\|" "$(pyof 810366 r1 A30)"
o=$(SLURM_JOB_ID=741849 src g9 A16); check 8k "--fastenv-battles 48 --no-overlap " "$o"; checkno 8l "actor-stream|decoupled" "$o"; check 8m "^PY=unset\|" "$(pyof 741849 g9 A16)"
o=$(SLURM_JOB_ID=810367 src g9 A16); check 8n "--fastenv-battles 48 --no-overlap " "$o"; checkno 8o "actor-stream|decoupled" "$o"; check 8p "^PY=$N/bin/py-src-24c633f\|" "$(pyof 810367 g9 A16)"
rm -f "$N/bin/py-src-4f18108"
rm -f "$N/bin/py-src-24c633f"; o=$(SLURM_JOB_ID=810366 src r1 A30); checkno 7k "cache-cpu" "$o"; check 7l "^PY=unset\|" "$(pyof 810366 r1 A30)"
check 7m "^PY=unset\|" "$(pyof 718701 r1 A30)"
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
mkdir -p "$N/bin"; printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-24c633f"; chmod +x "$N/bin/py-src-24c633f"
# v7 (2026-11-10): R1b-F (829660 and its chained jobs) runs through the 24c633f wrapper with --stacked-opp when the
# wrapper exists; otherwise (15c above) $C/src and no --stacked-opp
check 15i "^PY=$N/bin/py-src-24c633f\|PAT=\^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train\( \|\\\$\)$" "$(pyof 829660 g4r1bf T4)"
check 15j "^PY=$N/bin/py-src-24c633f\|" "$(pyof 900099 g4r1bf T4)"
echo r1bf-final > "$L/run/R1bF/final_model.pt"
o=$(SLURM_JOB_ID=829660 src g4r1bf T4); check 15k "--no-overlap --n-battles 60 --pool-battles 30 --run-dir $L/run/R1bF --pool-dir $L/run/R1bF/pool --snapshot-cache 40 --stacked-opp --snapshot-cache-cpu 200 --guard-pg-mean 20 --guard-anchor-drop 0.0999 --resume $L/run/R1bF/final_model.pt$" "$o"
checkno 15l "ent-coef|kl-anchor|frozen" "$o"  # v11: R1b-F has the cpu tier (--snapshot-cache-cpu 200, 15k)
printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-4f18108"; chmod +x "$N/bin/py-src-4f18108"
o=$(SLURM_JOB_ID=829660 src g4r1bf T4); check 15o "--no-overlap --n-battles 60 .*--snapshot-cache 40 --stacked-opp --snapshot-cache-cpu 200 --guard-pg-mean 20" "$o"; checkno 15p "actor-stream" "$o"
check 15q "^PY=$N/bin/py-src-24c633f\|" "$(pyof 829660 g4r1bf T4)"
rm -f "$N/bin/py-src-4f18108"
o=$(SLURM_JOB_ID=741849 src g9 A16); checkno 15m "stacked-opp" "$o"  # the A16 run is untouched by the g4r1bf change
rm -f "$N/bin/py-src-24c633f"
o=$(SLURM_JOB_ID=829660 src g4r1bf T4); check 15n "--snapshot-cache 40 --guard-pg-mean 20 --guard-anchor-drop 0.0999 --resume $L/run/R1bF/final_model.pt$" "$o"  # no wrapper: as before
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
# 21-27 (2026-11-10, user decision: G9q = option A of runs/g9q/PROPOSAL-DRAFT.txt): g9 MODE g9q on the A16 (741849) =
# G9p's recipe with P3 in the init, KL-anchor and frozen roles (eval anchor P1), lineage dir G9q, guard state runs/g9q,
# November sets for the collector; the successor 810367 resumes the newest mirrored G9q file; G9s parked, never resumed
g9qtree() {
  fresh; echo g9q > "$N/runs/g9/MODE"; touch "$N/runs/g9/G9S_A16_STARTED"; echo p3 > "$N/runs/pretrain/P3-g9-s2.pt"
  L=$ROOTM/local-A16; mkdir -p "$L/run/G9s/pool/frozen"; echo g9s-final > "$L/run/G9s/final_model.pt"; touch "$L/.toucher-G9q"
}
g9qtree; L=$ROOTM/local-A16; P3=$N/runs/pretrain/P3-g9-s2.pt; P1=$N/runs/pretrain/P1-g9h-dmg-s2.pt
mkdir -p "$N/bin"; printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-24c633f"; chmod +x "$N/bin/py-src-24c633f"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16)
check 21a "^FORMAT=gen9randombattle\|ANCHOR=$P1\|TRAIN_ARGS=--no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 --kl-anchor $P3 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P3 --p-frozen 0.67 --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $L/run/G9q --pool-dir $L/run/G9q/pool --snapshot-cache 72 --resume $P3$" "$o"
checkno 21b "P1-g9h|stacked-opp|cache-cpu|G9s" "${o#*TRAIN_ARGS=}"
[ -f "$N/runs/g9q/G9Q_STARTED" ] && [ -d "$L/run/G9q/pool" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 21c: G9Q_STARTED / G9q dirs"; }
check 21d "G9q first start: from P3 $P3 \(steps 0\); November sets for the collector \(PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11\)" "$(cat "$L/run/train.log")"
check 21e "^PY=unset\|" "$(pyof 741849 g9 A16)"
o=$( ( set -euo pipefail; NV2=$ROOTM/nv2; LOCAL=$L; RUN_NAME=g9; SLURM_JOB_ID=741849; NV2_HOLD_ITERS=0; source "$ENVF"; bash -c 'echo "child sees ${PORYGON_PS:-nothing}"' ) 2>&1 )
check 21f "child sees /vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11" "$o"
# 22: restart in place of G9q: its own save (not P3 again, not G9s); 741849 has no P3 / G9q leftovers of G9s
echo g9q-final > "$L/run/G9q/final_model.pt"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); check 22a "--run-dir $L/run/G9q --pool-dir $L/run/G9q/pool --snapshot-cache 72 --resume $L/run/G9q/final_model.pt$" "$o"; check 22b "--kl-anchor $P3 .*--frozen $P3 " "$o"
[ "$(grep -c 'G9q first start' "$L/run/train.log")" = 1 ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 22c: first start logged twice"; }
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=46068 src g9 A16); check 22d "--snapshot-cache 512 --stacked-opp --resume $L/run/G9q/final_model.pt$" "$o"
# 23: the successor 810367 (a fresh job: no local state) resumes the newest MIRRORED G9q file, not the newer G9s mirror
K=$N/runs/g9/20261109-021316-741849; mkdir -p "$K/G9q/pool/frozen" "$K/G9q/checkpoints" "$K/G9s"
echo g9q-mirror-ckpt > "$K/G9q/checkpoints/ckpt_00005000000.pt"; touch -d '2026-11-12 01:00' "$K/G9q/checkpoints/ckpt_00005000000.pt"
echo g9q-mirror-final > "$K/G9q/final_model.pt"; touch -d '2026-11-12 01:45' "$K/G9q/final_model.pt"
echo g9s-mirror-final > "$K/G9s/final_model.pt"; touch -d '2026-11-12 01:50' "$K/G9s/final_model.pt"
echo b > "$K/G9q/best_model.pt"; echo e > "$K/G9q/ema_model.pt"; echo p3 > "$K/G9q/pool/frozen/P3-g9-s2.pt"; echo s > "$K/G9q/pool/00002500000.pt"
S=$ROOTM/local-S; rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9q"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 S); check 23a "--run-dir $S/run/G9q --pool-dir $S/run/G9q/pool --snapshot-cache 72 --snapshot-cache-cpu 200 --resume $K/G9q/final_model.pt$" "$o"
check 23b "--kl-anchor $P3 .*--frozen $P3 " "$o"; check 23c "^PY=$N/bin/py-src-24c633f\|" "$(pyof 810367 g9 S)"
[ -f "$S/run/G9q/best_model.pt" ] && [ -f "$S/run/G9q/pool/frozen/P3-g9-s2.pt" ] && [ -L "$S/run/final_model.pt" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 23d: successor reseed"; }
rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9q"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=46068 src g9 S); check 23e "--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume $K/G9q/final_model.pt$" "$o"
# 24: G9q's first trips follow G9p's fallback rules in runs/g9q (anchor drop -> best at lr 3.5e-5; collapse -> beta x3)
echo "eval score 40.0% < best 60.0% - 15% for 2 evals" > "$L/run/G9q/GUARD_TRIPPED"; echo g9q-best > "$L/run/G9q/best_model.pt"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); check 24a "--lr 3.5e-5 .*--resume $N/runs/g9q/fu-start-best.pt$" "$o"; check 24b "^2 " "$(cat "$N/runs/g9q/FU.started" 2>/dev/null)"
[ "$(cat "$N/runs/g9q/fu-start-best.pt" 2>/dev/null)" = g9q-best ] && [ ! -e "$N/runs/g9s/FU.started" ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 24c: G9q fallback state"; }
g9qtree; L=$ROOTM/local-A16; SLURM_JOB_ID=741849 src g9 A16 > /dev/null; echo g9q-final > "$L/run/G9q/final_model.pt"
echo "collapse: entropy < 0.5 for 5 updates" > "$L/run/G9q/GUARD_TRIPPED"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); check 24d "--kl-anchor-coef 0.3 .*--resume $L/run/G9q/final_model.pt$" "$o"; checkno 24e "lr 3.5e-5" "$o"
# 25: a second G9q trip after its fallback -> HOLD (allocation kept)
echo "eval score 40.0% < best 60.0% - 15% for 2 evals" > "$L/run/G9q/GUARD_TRIPPED"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); [ -f "$N/runs/g9/HOLD" ] && ls "$L"/run/G9q/GUARD_TRIPPED.second-* > /dev/null 2>&1 && pass=$((pass+1)) || { fail=$((fail+1)); echo "FAIL 25: no HOLD on a second G9q trip"; }
# 26: G9s resumed in the same job script later (MODE back to g9s after G9q): the November sets are dropped, G9s
# continues from its own (= parked) lineage file with its fallback-2 args
g9qtree; L=$ROOTM/local-A16; echo "2 2026-11-09T23:35:20" > "$N/runs/g9s/FU.started"; echo "eval score 66.5% < best 78.5% - 10% for 2 evals" > "$N/runs/g9s/FU.reason"
o=$( ( set -euo pipefail; NV2=$ROOTM/nv2; LOCAL=$L; RUN_NAME=g9; SLURM_JOB_ID=741849; NV2_GPU_MIB=15356; NV2_HOLD_ITERS=0
       source "$ENVF"; echo "first PS=${PORYGON_PS:-unset}"; echo g9s > "$NV2/runs/g9/MODE"; source "$ENVF"
       echo "then PS=${PORYGON_PS:-unset}|$TRAIN_ARGS"; bash -c 'echo "child sees ${PORYGON_PS:-nothing}"' ) 2>&1 )
check 26a "first PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11" "$o"; check 26b "then PS=unset\|.*--lr 3.5e-5 .*--run-dir $L/run/G9s --pool-dir $L/run/G9s/pool --snapshot-cache 72 --resume $L/run/G9s/final_model.pt$" "$o"
check 26c "child sees nothing" "$o"
# 27: the other runs never see G9q's settings
o=$(SLURM_JOB_ID=718701 src r1 A30); checkno 27a "P3-g9|porygon-ws-fastenv|G9q" "$o"
rm -f "$N/bin/py-src-24c633f"
# 28-29 (v9, 2026-11-10): the G9q <-> G9p GPU swap, two-phase and all-or-nothing. Phase 1 = the kit's INLINE_CMDs
# ($SWAP_KIT, default scripts/netv2_swap: inline-718701-g9p.sh in r1, inline-741849-g9q.sh in g9) copy each lineage into its
# handover dir with DONE (or FAIL-<run>); phase 2 = nv2_swap: one atomic runs/swap-50M/DECISION, COMMIT iff both DONE.
# COMMIT: r1 MODE g9q (the A30 trains G9q from runs/g9q/handover-a30, lineage dir G9q, guard state g9q, P3, November
# sets, 28 workers, cache 192 / 640 + stacked-opp, actor stream unless runs/g9q/NO_ACTOR_STREAM) and g9 MODE g9p2 (the
# A16 trains G9p from runs/r1/handover-g9p-a16, lineage dir G9p, guard state g9a30, P1, October sets, 8 workers,
# --no-overlap, cache 72 / 512 + stacked-opp, wrapper + cpu tier 200). ABORT / no decision: each job resumes its own lineage.
KIT=${SWAP_KIT:-$(dirname "$(readlink -f "$0")")/netv2_swap}
swaptree() {  # the 2026-11-10 state: r1 MODE g9p (A30 = G9p, 718701), g9 MODE g9q (A16 = G9q, 741849), both wrappers
  fresh; echo g9p > "$N/runs/r1/MODE"; echo g9q > "$N/runs/g9/MODE"; echo p3 > "$N/runs/pretrain/P3-g9-s2.pt"
  mkdir -p "$N/runs/g9q" "$N/bin"; touch "$N/runs/r1/G9P_A30_STARTED" "$N/runs/g9q/G9Q_STARTED" "$N/runs/g9/G9S_A16_STARTED"
  for w in py-src-24c633f py-src-4f18108; do printf '#!/bin/bash\nexit 0\n' > "$N/bin/$w"; chmod +x "$N/bin/$w"; done
  A=$ROOTM/local-A30; mkdir -p "$A/run/G9p/pool/frozen"; echo g9p-local > "$A/run/G9p/final_model.pt"; echo g9p-best > "$A/run/G9p/best_model.pt"
  echo g9p-ema > "$A/run/G9p/ema_model.pt"; for s in 00025000000 00050000000 00075000000; do echo "p$s" > "$A/run/G9p/pool/$s.pt"; done
  echo P1 > "$A/run/G9p/pool/frozen/P1-g9h-dmg-s2.pt"; echo bp > "$A/run/G9p/pool/frozen/best-00050000000.pt"
  echo "2026-11-10 15:00:00 upd 9 steps 80,000,000 (+43000) sps 2200" > "$A/run/G9p/train.log"
  B=$ROOTM/local-A16; mkdir -p "$B/run/G9q/pool/frozen"; echo g9q-local > "$B/run/G9q/final_model.pt"; echo g9q-best > "$B/run/G9q/best_model.pt"
  echo g9q-ema > "$B/run/G9q/ema_model.pt"; for s in 00002500000 00005000000; do echo "q$s" > "$B/run/G9q/pool/$s.pt"; done
  echo P3 > "$B/run/G9q/pool/frozen/P3-g9-s2.pt"; echo bq > "$B/run/G9q/pool/frozen/best-00002500000.pt"
  echo "2026-11-10 15:00:00 upd 9 steps 6,000,000 (+43000) sps 900" > "$B/run/G9q/train.log"
  touch "$A/.toucher-G9q" "$B/.toucher-G9q" "$A/.toucher-G9p" "$B/.toucher-G9p"
}
kit() { cp "$KIT/inline-718701-g9p.sh" "$N/runs/r1/INLINE_CMD" 2>/dev/null; cp "$KIT/inline-741849-g9q.sh" "$N/runs/g9/INLINE_CMD" 2>/dev/null; true; }
both() {  # both ITERS: r1 (A30, 718701) and g9 (A16, 741849) restart in place at the same time -> $ROOTM.r1 / $ROOTM.g9
  ( SLURM_JOB_ID=718701 NV2_SWAP_ITERS=$1 NV2_SWAP_SLEEP=0.2 _src r1 A30 > "$ROOTM.r1" 2>&1 ) & local p=$!
  /bin/sleep 0.5; ( SLURM_JOB_ID=741849 NV2_SWAP_ITERS=$1 NV2_SWAP_SLEEP=0.2 _src g9 A16 > "$ROOTM.g9" 2>&1 ); wait $p
}
psof() { ( set -euo pipefail; NV2=$ROOTM/nv2; LOCAL=$ROOTM/local-$3; RUN_NAME=$2; SLURM_JOB_ID=$1; NV2_HOLD_ITERS=0; NV2_SWAP_ITERS=0; source "$ENVF" > /dev/null; bash -c 'echo "child sees ${PORYGON_PS:-nothing}"' ) 2>&1; }
yes_() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [$1]: $2"; fi; }
P1=$N/runs/pretrain/P1-g9h-dmg-s2.pt; P3=$N/runs/pretrain/P3-g9-s2.pt; NOV=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
G9PARGS="--kl-anchor $P1 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P1 --p-frozen 0.67"
G9QARGS="--kl-anchor $P3 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P3 --p-frozen 0.67"
# 28a-28b: r1 MODE g9q without a committed swap (set by hand; or handover DONE but no DECISION) -> G9p continues here
swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; echo g9q > "$N/runs/r1/MODE"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 28a "$G9PARGS .*--fastenv-workers 28 --fastenv-battles 48 --actor-stream --decoupled --n-battles 16 --pool-battles 16 --run-dir $A/run/G9p --pool-dir $A/run/G9p/pool --snapshot-cache 192 --stacked-opp --resume $A/run/G9p/final_model.pt$" "$o"
check 28a2 "MODE g9q but no committed G9q handover .*: G9p continues" "$(cat "$A/run/train.log")"; yes_ 28a3 '[ ! -e "$N/runs/r1/G9Q_A30_STARTED" ] && [ "$(psof 718701 r1 A30)" = "child sees nothing" ]'
mkdir -p "$N/runs/swap-50M" "$N/runs/g9q/handover-a30/pool"; echo h > "$N/runs/g9q/handover-a30/final_model.pt"; echo d > "$N/runs/g9q/handover-a30/DONE"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 28b "--run-dir $A/run/G9p .*--resume $A/run/G9p/final_model.pt$" "$o"; yes_ 28b2 '[ ! -e "$N/runs/r1/G9Q_A30_STARTED" ]'
# 28c-28d: phase 1 on the A30 only, the A16 never answers: phase 2 times out -> ABORT; G9p resumes its own local save,
# MODE unchanged; the late A16 side then refuses phase 1 (a decision exists) and G9q continues there
swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; mkdir -p "$N/runs/swap-50M"; kit; rm -f "$N/runs/g9/INLINE_CMD"
o=$(SLURM_JOB_ID=718701 NV2_SWAP_ITERS=2 src r1 A30); check 28c "$G9PARGS .*--run-dir $A/run/G9p --pool-dir $A/run/G9p/pool --snapshot-cache 192 --stacked-opp --resume $A/run/G9p/final_model.pt$" "$o"
H=$N/runs/r1/handover-g9p-a16
yes_ 28c2 'cmp -s "$H/final_model.pt" "$A/run/G9p/final_model.pt" && [ -f "$H/DONE" ] && [ -f "$H/pool/00075000000.pt" ] && [ -f "$H/pool/frozen/best-00050000000.pt" ] && [ -f "$H/best_model.pt" ] && [ -f "$H/ema_model.pt" ] && [ -f "$H/HANDOVER.txt" ] && [ ! -e "$H.tmp" ]'
check 28c3 "upd 9 steps 80,000,000" "$(cat "$H/HANDOVER.txt" 2>/dev/null)"; check 28c4 "^ABORT by r1 718701 " "$(cat "$N/runs/swap-50M/DECISION" 2>/dev/null)"
check 28c5 "r1 718701 phase1 DONE: .*r1 718701 phase2 proposed ABORT" "$(cat "$N/runs/swap-50M/STATE" 2>/dev/null | tr '\n' ' ')"; yes_ 28c6 '[ "$(cat "$N/runs/r1/MODE")" = g9p ] && [ ! -e "$N/runs/r1/SWAP_APPLIED" ]'
cp "$KIT/inline-741849-g9q.sh" "$N/runs/g9/INLINE_CMD" 2>/dev/null
o=$(SLURM_JOB_ID=741849 src g9 A16); check 28d "$G9QARGS .*--fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $B/run/G9q --pool-dir $B/run/G9q/pool --snapshot-cache 72 --resume $B/run/G9q/final_model.pt$" "$o"
yes_ 28d2 '[ ! -e "$N/runs/g9q/handover-a30" ] && [ "$(cat "$N/runs/g9/MODE")" = g9q ]'; check 28d3 "g9 741849 phase1 REFUSED" "$(cat "$N/runs/swap-50M/STATE" 2>/dev/null)"
# 28e-28h: the full swap: both jobs at a restart in place at about the same time -> COMMIT, MODEs flipped, each first start
# from the other side's handover (A30: G9q, 718701 = actor stream through src-4f18108; A16: G9p, 741849 = 24c633f wrapper)
swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; mkdir -p "$N/runs/swap-50M"; kit; both 50
o=$(cat "$ROOTM.r1"); check 28e "^FORMAT=gen9randombattle\|ANCHOR=$P1\|TRAIN_ARGS=--no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 $G9QARGS --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 28 --fastenv-battles 48 --actor-stream --decoupled --n-battles 16 --pool-battles 16 --run-dir $A/run/G9q --pool-dir $A/run/G9q/pool --snapshot-cache 192 --stacked-opp --resume $N/runs/g9q/handover-a30/final_model.pt$" "$o"
o=$(cat "$ROOTM.g9"); check 29e "^FORMAT=gen9randombattle\|ANCHOR=$P1\|TRAIN_ARGS=--no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 $G9PARGS --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $B/run/G9p --pool-dir $B/run/G9p/pool --snapshot-cache 72 --snapshot-cache-cpu 200 --resume $N/runs/r1/handover-g9p-a16/final_model.pt$" "$o"
check 28e2 "^COMMIT by (r1 718701|g9 741849) " "$(cat "$N/runs/swap-50M/DECISION" 2>/dev/null)"
yes_ 28e3 '[ "$(cat "$N/runs/r1/MODE") $(cat "$N/runs/g9/MODE")" = "g9q g9p2" ] && [ -f "$N/runs/r1/SWAP_APPLIED" ] && [ -f "$N/runs/g9/SWAP_APPLIED" ] && [ -f "$N/runs/r1/G9Q_A30_STARTED" ] && [ -f "$N/runs/g9/G9P_A16_STARTED" ]'
yes_ 28e4 '[ "$(ls "$A/run/G9q/pool" | grep -c pt$) $(ls "$A/run/G9q/pool/frozen" | grep -c pt$) $(cat "$A/run/G9q/best_model.pt") $(cat "$A/run/G9q/ema_model.pt")" = "2 2 g9q-best g9q-ema" ]'
yes_ 29e2 '[ "$(ls "$B/run/G9p/pool" | grep -c pt$) $(ls "$B/run/G9p/pool/frozen" | grep -c pt$) $(cat "$B/run/G9p/best_model.pt") $(cat "$B/run/G9p/ema_model.pt")" = "3 2 g9p-best g9p-ema" ]'
yes_ 28e5 'cmp -s "$N/runs/g9q/handover-a30/final_model.pt" "$B/run/G9q/final_model.pt" && cmp -s "$N/runs/r1/handover-g9p-a16/final_model.pt" "$A/run/G9p/final_model.pt"'
check 28e6 "G9q first start on this A30 from $N/runs/g9q/handover-a30 .*November sets \(PORYGON_PS=$NOV\)" "$(cat "$A/run/train.log")"; check 29e3 "G9p first start on this A16 from $N/runs/r1/handover-g9p-a16 .*October sets" "$(cat "$B/run/train.log")"
check 28e7 "^PY=$N/bin/py-src-4f18108\|" "$(pyof 718701 r1 A30)"; check 29e4 "^PY=$N/bin/py-src-24c633f\|" "$(pyof 741849 g9 A16)"
check 28e8 "child sees $NOV" "$(psof 718701 r1 A30)"; check 29e5 "child sees nothing" "$(psof 741849 g9 A16)"
# 28f / 29f: restarts in place after the swap: each new lineage's own save (no second first start)
echo g9q-a30-final > "$A/run/G9q/final_model.pt"; echo g9p-a16-final > "$B/run/G9p/final_model.pt"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 28f "--run-dir $A/run/G9q --pool-dir $A/run/G9q/pool --snapshot-cache 192 --stacked-opp --resume $A/run/G9q/final_model.pt$" "$o"
o=$(SLURM_JOB_ID=741849 src g9 A16); check 29f "--run-dir $B/run/G9p --pool-dir $B/run/G9p/pool --snapshot-cache 72 --snapshot-cache-cpu 200 --resume $B/run/G9p/final_model.pt$" "$o"
yes_ 28f2 '[ "$(grep -c "G9q first start" "$A/run/train.log") $(grep -c "G9p first start" "$B/run/train.log") $(grep -c "phase2 applied COMMIT" "$N/runs/swap-50M/STATE")" = "1 1 2" ]'
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=46068 src g9 A16); check 29f2 "--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume $B/run/G9p/final_model.pt$" "$o"
o=$(SLURM_JOB_ID=718701 NV2_GPU_MIB=46068 src r1 A30); check 28f3 "--snapshot-cache 640 --stacked-opp --resume $A/run/G9q/final_model.pt$" "$o"
# 28g: runs/g9q/NO_ACTOR_STREAM -> the 24c633f wrapper and --no-overlap; 29g: no 24c633f wrapper -> $C/src, no cpu tier
touch "$N/runs/g9q/NO_ACTOR_STREAM"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 28g "--fastenv-battles 48 --no-overlap --n-battles 16 .*--snapshot-cache 192 --stacked-opp --resume $A/run/G9q/final_model.pt$" "$o"; checkno 28g2 "actor-stream|decoupled|cache-cpu" "$o"
check 28g3 "^PY=$N/bin/py-src-24c633f\|" "$(pyof 718701 r1 A30)"; rm -f "$N/runs/g9q/NO_ACTOR_STREAM"
mv "$N/bin/py-src-24c633f" "$N/bin/off"; o=$(SLURM_JOB_ID=741849 src g9 A16); check 29g "--snapshot-cache 72 --resume $B/run/G9p/final_model.pt$" "$o"; check 29g2 "^PY=unset\|" "$(pyof 741849 g9 A16)"; mv "$N/bin/off" "$N/bin/py-src-24c633f"
# 28h / 29h: successors after the swap (fresh jobs): the swapped lineages' newest MIRRORED files (810366: the A30's
# post-swap G9q, not the A16's pre-swap one; 810367: the A16's post-swap G9p, not the A30's pre-swap one)
J=$N/runs/r1/20261108-220401-718701; K=$N/runs/g9/20261109-021316-741849
for d in "$J/G9p" "$K/G9q" "$J/G9q" "$K/G9p"; do mkdir -p "$d/pool/frozen" "$d/checkpoints"; echo "${d##*/}-$(basename "$(dirname "$d")")" > "$d/final_model.pt"; echo b > "$d/best_model.pt"; echo s > "$d/pool/00001000000.pt"; done
touch -d '2026-11-10 16:00' "$J/G9p/final_model.pt" "$K/G9q/final_model.pt"; touch -d '2026-11-11 21:20' "$J/G9q/final_model.pt"; touch -d '2026-11-12 01:40' "$K/G9p/final_model.pt"
S=$ROOTM/local-S; rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9q" "$S/.toucher-G9p"
o=$(SLURM_JOB_ID=810366 NV2_GPU_MIB=24576 src r1 S); check 28h "$G9QARGS .*--fastenv-workers 28 --fastenv-battles 48 --actor-stream --decoupled --n-battles 16 --pool-battles 16 --run-dir $S/run/G9q --pool-dir $S/run/G9q/pool --snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume $J/G9q/final_model.pt$" "$o"
check 28h2 "^PY=$N/bin/py-src-4f18108\|" "$(pyof 810366 r1 S)"; yes_ 28h3 '[ -f "$S/run/G9q/best_model.pt" ] && [ -L "$S/run/final_model.pt" ]'
rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9q" "$S/.toucher-G9p"
o=$(SLURM_JOB_ID=810366 NV2_GPU_MIB=46068 src r1 S); check 28h4 "--snapshot-cache 640 --stacked-opp --snapshot-cache-cpu 400 --resume $J/G9q/final_model.pt$" "$o"
for g in 15356 46068; do
  rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9q" "$S/.toucher-G9p"
  o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=$g src g9 S); x="--snapshot-cache 72 --snapshot-cache-cpu 200"; [ $g = 46068 ] && x="--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200"
  check "29h-$g" "$G9PARGS .*--fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $S/run/G9p --pool-dir $S/run/G9p/pool $x --resume $K/G9p/final_model.pt$" "$o"
done
check 29h2 "^PY=$N/bin/py-src-24c633f\|" "$(pyof 810367 g9 S)"; check 29h3 "child sees nothing" "$(psof 810367 g9 S)"
# 28i / 29i: the swapped lineages' guard trips: first trip -> their own fallback state (G9q: runs/g9q, G9p: runs/g9a30);
# a trip after it -> HOLD in the job's run (r1 / g9)
echo "eval score 40.0% < best 60.0% - 15% for 2 evals" > "$A/run/G9q/GUARD_TRIPPED"; echo g9q-best2 > "$A/run/G9q/best_model.pt"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 28i "--lr 3.5e-5 .*--resume $N/runs/g9q/fu-start-best.pt$" "$o"; check 28i2 "^2 " "$(cat "$N/runs/g9q/FU.started" 2>/dev/null)"
echo "collapse: entropy" > "$A/run/G9q/GUARD_TRIPPED"; o=$(SLURM_JOB_ID=718701 src r1 A30); yes_ 28i3 '[ -f "$N/runs/r1/HOLD" ] && ls "$A"/run/G9q/GUARD_TRIPPED.second-* > /dev/null 2>&1'
echo "collapse: P(top >= 0.9) > 0.6 for 5 updates" > "$B/run/G9p/GUARD_TRIPPED"
o=$(SLURM_JOB_ID=741849 src g9 A16); check 29i "--kl-anchor-coef 0.3 .*--resume $B/run/G9p/final_model.pt$" "$o"; check 29i2 "^2 .*collapse: P\(top" "$(cat "$N/runs/g9a30/FU.started" "$N/runs/g9a30/FU.reason" 2>/dev/null | tr "\n" " ")"
echo "eval score" > "$B/run/G9p/GUARD_TRIPPED"; o=$(SLURM_JOB_ID=741849 src g9 A16); yes_ 29i3 '[ -f "$N/runs/g9/HOLD" ] && ls "$B"/run/G9p/GUARD_TRIPPED.second-* > /dev/null 2>&1'
# 28j: PORYGON_PS follows the r1 mode in one job script process (G9q -> back to G9p: the November sets are dropped)
o=$( ( set -euo pipefail; NV2=$ROOTM/nv2; LOCAL=$A; RUN_NAME=r1; SLURM_JOB_ID=718701; NV2_HOLD_ITERS=0; rm -f "$N/runs/r1/HOLD"
       source "$ENVF"; echo "first PS=${PORYGON_PS:-unset}"; echo g9p > "$NV2/runs/r1/MODE"; source "$ENVF"; echo "then PS=${PORYGON_PS:-unset}|$TRAIN_ARGS" ) 2>&1 )
check 28j "first PS=$NOV" "$o"; check 28j2 "then PS=unset\|.*--kl-anchor $P1 .*--run-dir $A/run/G9p .*--resume $A/run/G9p/final_model.pt$" "$o"
# 29a: g9 MODE g9p2 without a committed swap -> G9q continues here (its own save, November sets)
swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; echo g9p2 > "$N/runs/g9/MODE"
o=$(SLURM_JOB_ID=741849 src g9 A16); check 29a "$G9QARGS .*--run-dir $B/run/G9q --pool-dir $B/run/G9q/pool --snapshot-cache 72 --resume $B/run/G9q/final_model.pt$" "$o"
check 29a2 "MODE g9p2 but no committed G9p handover .*: G9q continues" "$(cat "$B/run/train.log")"; yes_ 29a3 '[ ! -e "$N/runs/g9/G9P_A16_STARTED" ] && [ "$(psof 741849 g9 A16)" = "child sees $NOV" ]'
# 29b (a): one side fails mid-copy (an unreadable G9q pool net on the A16): FAIL-g9, both ABORT at once, no handover
# published by the A16, both MODEs unchanged, each job resumes its own lineage's local save
swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; mkdir -p "$N/runs/swap-50M"; kit; chmod 000 "$B/run/G9q/pool/00005000000.pt"
both 50; chmod 644 "$B/run/G9q/pool/00005000000.pt"
check 29b "--run-dir $A/run/G9p --pool-dir $A/run/G9p/pool --snapshot-cache 192 --stacked-opp --resume $A/run/G9p/final_model.pt$" "$(cat "$ROOTM.r1")"
check 29b2 "--run-dir $B/run/G9q --pool-dir $B/run/G9q/pool --snapshot-cache 72 --resume $B/run/G9q/final_model.pt$" "$(cat "$ROOTM.g9")"
check 29b3 "^ABORT by " "$(cat "$N/runs/swap-50M/DECISION" 2>/dev/null)"; yes_ 29b4 '[ -f "$N/runs/swap-50M/FAIL-g9" ] && [ ! -e "$N/runs/g9q/handover-a30" ] && [ "$(cat "$N/runs/r1/MODE") $(cat "$N/runs/g9/MODE")" = "g9p g9q" ] && [ ! -e "$N/runs/r1/SWAP_APPLIED" ] && [ ! -e "$N/runs/g9/SWAP_APPLIED" ]'
# 29c (b): one side "times out" after the other has decided. (1) The A30's phase 1 is DONE but its job stalls; the A16
# completes phase 1, COMMITs and flips; the A30 then gets no wait (iters 0) -> it still follows COMMIT (no fork).
# (2) The A30 timed out first (ABORT); the A16's phase 1 lands afterwards: it follows ABORT, G9q continues there.
swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; mkdir -p "$N/runs/swap-50M"; kit
( cd "$N" && LOCAL=$A bash runs/r1/INLINE_CMD > /dev/null 2>&1 ); mv "$N/runs/r1/INLINE_CMD" "$N/runs/r1/INLINE_CMD.done-mock"
o=$(SLURM_JOB_ID=741849 NV2_SWAP_ITERS=2 src g9 A16); check 29c "--run-dir $B/run/G9p .*--resume $N/runs/r1/handover-g9p-a16/final_model.pt$" "$o"
o=$(SLURM_JOB_ID=718701 NV2_SWAP_ITERS=0 src r1 A30); check 29c2 "--run-dir $A/run/G9q .*--resume $N/runs/g9q/handover-a30/final_model.pt$" "$o"
check 29c3 "^COMMIT by g9 741849 " "$(cat "$N/runs/swap-50M/DECISION")"
swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; mkdir -p "$N/runs/swap-50M"; kit
o=$(SLURM_JOB_ID=718701 NV2_SWAP_ITERS=1 src r1 A30); ( cd "$N" && LOCAL=$B bash runs/g9/INLINE_CMD > /dev/null 2>&1 )  # (its phase 1 refuses: a decision exists)
rm -rf "$N/runs/g9q/handover-a30"; mkdir -p "$N/runs/g9q/handover-a30"; cp -p "$B/run/G9q/final_model.pt" "$N/runs/g9q/handover-a30/"; echo d > "$N/runs/g9q/handover-a30/DONE"  # as if it landed just before
mv "$N/runs/g9/INLINE_CMD" "$N/runs/g9/INLINE_CMD.done-mock"; o=$(SLURM_JOB_ID=741849 src g9 A16); check 29c4 "--run-dir $B/run/G9q .*--resume $B/run/G9q/final_model.pt$" "$o"
yes_ 29c5 '[ "$(cat "$N/runs/r1/MODE") $(cat "$N/runs/g9/MODE")" = "g9p g9q" ] && grep -q "^ABORT by r1" "$N/runs/swap-50M/DECISION"'
# 29d (c): restart after an interrupted swap, state on disk only. (1) Both phase 1 DONE, both jobs died before any
# decision: the fresh successor 810366 decides COMMIT and first-starts G9q from the handover (successor args); the
# fresh 810367 applies that COMMIT (its MODE was never flipped) and first-starts G9p from its handover. (2) Only the
# A30's phase 1 is DONE: 810366 times out -> ABORT and resumes the newest mirrored G9p (MODE g9p unchanged).
swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; mkdir -p "$N/runs/swap-50M"; kit
( cd "$N" && LOCAL=$A bash runs/r1/INLINE_CMD > /dev/null 2>&1; LOCAL=$B bash runs/g9/INLINE_CMD > /dev/null 2>&1 ); rm -f "$N/runs/r1/INLINE_CMD" "$N/runs/g9/INLINE_CMD"
J=$N/runs/r1/20261108-220401-718701; K=$N/runs/g9/20261109-021316-741849; mkdir -p "$J/G9p" "$K/G9q"; echo jp > "$J/G9p/final_model.pt"; echo kq > "$K/G9q/final_model.pt"
S=$ROOTM/local-S; rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9q" "$S/.toucher-G9p"
o=$(SLURM_JOB_ID=810366 NV2_GPU_MIB=24576 NV2_SWAP_ITERS=1 src r1 S); check 29d "--actor-stream --decoupled .*--run-dir $S/run/G9q --pool-dir $S/run/G9q/pool --snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume $N/runs/g9q/handover-a30/final_model.pt$" "$o"
yes_ 29d2 '[ "$(ls "$S/run/G9q/pool" | grep -c pt$)" = 2 ] && grep -q "^COMMIT by r1 810366" "$N/runs/swap-50M/DECISION" && [ "$(cat "$N/runs/g9/MODE")" = g9q ]'
T=$ROOTM/local-T; rm -rf "$T"; mkdir -p "$T/run"; touch "$T/run/train.log" "$T/.toucher-G9q" "$T/.toucher-G9p"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 T); check 29d3 "$G9PARGS .*--run-dir $T/run/G9p --pool-dir $T/run/G9p/pool --snapshot-cache 72 --snapshot-cache-cpu 200 --resume $N/runs/r1/handover-g9p-a16/final_model.pt$" "$o"
yes_ 29d4 '[ "$(cat "$N/runs/g9/MODE")" = g9p2 ] && [ -f "$N/runs/g9/G9P_A16_STARTED" ] && [ "$(ls "$T/run/G9p/pool" | grep -c pt$)" = 3 ]'
swaptree; A=$ROOTM/local-A30; mkdir -p "$N/runs/swap-50M"; kit; ( cd "$N" && LOCAL=$A bash runs/r1/INLINE_CMD > /dev/null 2>&1 ); rm -f "$N/runs/r1/INLINE_CMD" "$N/runs/g9/INLINE_CMD"
J=$N/runs/r1/20261108-220401-718701; mkdir -p "$J/G9p"; echo jp > "$J/G9p/final_model.pt"
S=$ROOTM/local-S; rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9q" "$S/.toucher-G9p"
o=$(SLURM_JOB_ID=810366 NV2_SWAP_ITERS=1 src r1 S); check 29d5 "$G9PARGS .*--run-dir $S/run/G9p .*--snapshot-cache-cpu 400 --resume $J/G9p/final_model.pt$" "$o"
yes_ 29d6 'grep -q "^ABORT by r1 810366" "$N/runs/swap-50M/DECISION" && [ "$(cat "$N/runs/r1/MODE")" = g9p ]'
# 29j: the other runs never see the swap: R1b-F (g4r1bf) unchanged with swap state on disk
g4tree; L=$ROOTM/local-T4; mkdir -p "$N/runs/swap-50M" "$N/bin"; echo "COMMIT by x" > "$N/runs/swap-50M/DECISION"; printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-24c633f"; chmod +x "$N/bin/py-src-24c633f"
o=$(SLURM_JOB_ID=829660 src g4r1bf T4); check 29j "--snapshot-cache 40 --stacked-opp --snapshot-cache-cpu 200 --guard-pg-mean 20 --guard-anchor-drop 0.0999 --resume $N/runs/r1b-parked/final_model.pt$" "$o"
# 30-33 (v10, 2026-11-11; G9r, approved .orchestra/netv2/G9R-PROPOSAL.md): g9 MODE g9r = G9q's recipe + --lr-schedule power
# --lr-horizon 300000000, entropy floor 0.35 (replaced), lineage dir G9r, guard state runs/g9r, P3 init / anchor / frozen,
# P1 eval anchor, November sets, PY = $NV2/bin/py-src-ff32b71 everywhere, cache 72 / 512 + stacked-opp, cpu tier 200.
# MODE g9r comes from the park INLINE_CMD ($G9R_KIT, default scripts/netv2_g9r/inline-a16-park-g9r.sh): it parks the A16's
# lineage (MODE g9q -> G9q -> runs/g9q-parked; g9p2 -> G9p -> runs/g9p-parked-a16) whole, then flips MODE; any failure
# or refusal changes nothing. A swap COMMIT applied afterwards never re-flips MODE g9r.
G9R_KIT=${G9R_KIT:-$(dirname "$(readlink -f "$0")")/netv2_g9r}; G9R_SHA=ff32b71
WT=$(dirname "$(dirname "$(readlink -f "$0")")")
printf '#!/bin/bash\n# mock: the park script'"'"'s checkpoint load\necho "counters: {'"'"'steps'"'"': 50004575, '"'"'updates'"'"': 1205, '"'"'games'"'"': 1}"; echo "keys: ['"'"'args'"'"', '"'"'counters'"'"', '"'"'optimizer'"'"']"\n' > "$ROOTM.parkpy"; chmod +x "$ROOTM.parkpy"
export NV2_PARK_PY=$ROOTM.parkpy
g9rtree() {  # the 2026-11-11 state before the park: swaptree (r1 g9p, g9 g9q) + v10 on disk, the G9r wrapper + src, PREREG
  swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16
  cp "$ENVF" "$N/netv2.env"; mkdir -p "$N/runs/g9r" "$N/src-$G9R_SHA/porygon/v2"; echo prereg > "$N/runs/g9r/PREREG.txt"
  git -C "$WT" show "$G9R_SHA:src/porygon/v2/train.py" > "$N/src-$G9R_SHA/porygon/v2/train.py" 2>/dev/null || cp "$WT/src/porygon/v2/train.py" "$N/src-$G9R_SHA/porygon/v2/train.py"
  printf '#!/bin/bash\nexport PYTHONPATH=%s/src-%s${PYTHONPATH:+:$PYTHONPATH}\nexit 0\n' "$N" "$G9R_SHA" > "$N/bin/py-src-$G9R_SHA"; chmod +x "$N/bin/py-src-$G9R_SHA"
  for x in G9q G9p; do mkdir -p "$B/run/$x"; done
  printf '2026-11-11 10:00:00 v2 trainer: gen9randombattle ...\n2026-11-11 12:00:00 upd 1205 steps 50,004,575 (+41500) sps 455\n2026-11-11 12:00:30 saved final_model.pt at 50,004,575 steps\n' > "$B/run/G9q/train.log"
  mkdir -p "$B/run/G9q/checkpoints" "$B/run/G9q/tb" "$B/run/G9q/eval"; echo c > "$B/run/G9q/checkpoints/ckpt_00045004575.pt"; echo t > "$B/run/G9q/tb/events.x"; echo e > "$B/run/G9q/eval/eval.log"
  touch "$B/.toucher-G9r" "$B/.toucher-G9q"
}
park() { cp "$G9R_KIT/inline-a16-park-g9r.sh" "$N/runs/g9/INLINE_CMD" 2>/dev/null || echo 'echo "mock: no park kit"; exit 1' > "$N/runs/g9/INLINE_CMD"; }
W=$N/bin/py-src-$G9R_SHA
G9RA() {  # G9RA RUNDIR OPP RESUME: the exact G9r TRAIN_ARGS
  echo "--no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.35 --guard-ptop-max 0.6 $G9QARGS --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $1 --pool-dir $1/pool $2 --lr-schedule power --lr-horizon 300000000 --resume $3"
}
# 30a: the park from MODE g9q in 741849 (no swap armed): G9q copied whole to runs/g9q-parked, MODE g9r, the same sourcing
# is G9r's first start from P3 (exact args), November sets, the G9r wrapper; G9q's own files stay in place
g9rtree; park
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); W=$N/bin/py-src-$G9R_SHA; Q=$N/runs/g9q-parked
check 30a "^FORMAT=gen9randombattle\|ANCHOR=$P1\|TRAIN_ARGS=$(G9RA "$B/run/G9r" "--snapshot-cache 72 --snapshot-cache-cpu 200" "$P3")$" "$o"
checkno 30a2 "guard-entropy-min 0.5|P1-g9h-dmg-s2.pt --kl|G9q" "${o#*TRAIN_ARGS=}"
yes_ 30a3 '[ "$(cat "$N/runs/g9/MODE")" = g9r ] && [ -f "$N/runs/g9r/G9R_STARTED" ] && [ -d "$B/run/G9r/pool" ] && [ ! -e "$Q.tmp" ]'
yes_ 30a4 'cmp -s "$Q/final_model.pt" "$B/run/G9q/final_model.pt" && [ "$(ls "$Q/pool" | grep -c pt$) $(ls "$Q/pool/frozen" | grep -c pt$) $(cat "$Q/best_model.pt") $(cat "$Q/ema_model.pt")" = "2 2 g9q-best g9q-ema" ] && [ -f "$Q/checkpoints/ckpt_00045004575.pt" ] && [ -f "$Q/tb/events.x" ] && [ -f "$Q/eval/eval.log" ] && [ -f "$Q/train.log" ]'
yes_ 30a5 '(cd "$Q" && md5sum -c --quiet MD5SUMS) && [ "$(wc -l < "$Q/MD5SUMS")" -ge 10 ] && [ -f "$B/run/G9q/final_model.pt" ]'
check 30a6 "^G9q PARKED .* in job 741849 .*MODE g9q -> g9r.*source: $B/run/G9q .*guard state runs/g9q.*counters: \{'steps': 50004575.*last update: .*upd 1205 steps 50,004,575.*last save: .*saved final_model.pt at 50,004,575.*resume: " "$(tr '\n' ' ' < "$Q/PARKED.txt" 2>/dev/null)"
check 30a7 "G9r first start: from P3 $P3 \(steps 0\); lr schedule power H=300000000, entropy floor 0.35; November sets for the collector \(PORYGON_PS=$NOV\)" "$(cat "$B/run/train.log")"
check 30a8 "park DONE: G9q parked in $Q .*MODE g9q -> g9r" "$(cat "$N/runs/g9r/park.log")"
check 30a9 "^PY=$W\|PAT=\^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train\( \|\\\$\)$" "$(pyof 741849 g9 A16)"; check 30a10 "child sees $NOV" "$(psof 741849 g9 A16)"
# 30b: G9r restart in place: its own save, no second first start, no second park; >= 40 GB: cache 512 + stacked-opp
echo g9r-final > "$B/run/G9r/final_model.pt"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); check 30b "TRAIN_ARGS=$(G9RA "$B/run/G9r" "--snapshot-cache 72 --snapshot-cache-cpu 200" "$B/run/G9r/final_model.pt")$" "$o"
yes_ 30b2 '[ "$(grep -c "G9r first start" "$B/run/train.log")" = 1 ] && [ "$(grep -c "park DONE" "$N/runs/g9r/park.log")" = 1 ]'
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=46068 src g9 A16); check 30b3 "TRAIN_ARGS=$(G9RA "$B/run/G9r" "--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200" "$B/run/G9r/final_model.pt")$" "$o"
# 30c: G9r's guard trips follow G9q's fallback rules in runs/g9r: a non-collapse trip -> best_model.pt with the WHOLE
# power curve halved (--lr 3.5e-5, schedule kept, floor 0.35); a trip after that -> HOLD; a collapse trip -> KL x3
echo "eval score 40.0% < best 60.0% - 15% for 2 evals" > "$B/run/G9r/GUARD_TRIPPED"; echo g9r-best > "$B/run/G9r/best_model.pt"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16)
check 30c "--batch-size 4096 --lr 3.5e-5 --eval-async .*--guard-entropy-min 0.35 .*--run-dir $B/run/G9r .*--snapshot-cache-cpu 200 --lr-schedule power --lr-horizon 300000000 --resume $N/runs/g9r/fu-start-best.pt$" "$o"
yes_ 30c2 '[ "$(cut -c1-2 "$N/runs/g9r/FU.started")" = "2 " ] && [ "$(cat "$N/runs/g9r/fu-start-best.pt")" = g9r-best ] && [ ! -e "$N/runs/g9q/FU.started" ]'
echo "collapse: entropy < 0.35 for 5 updates" > "$B/run/G9r/GUARD_TRIPPED"; o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16)
yes_ 30c3 '[ -f "$N/runs/g9/HOLD" ] && ls "$B"/run/G9r/GUARD_TRIPPED.second-* > /dev/null 2>&1'; rm -f "$N/runs/g9/HOLD"
g9rtree; park; SLURM_JOB_ID=741849 src g9 A16 > /dev/null; echo g9r-final > "$B/run/G9r/final_model.pt"; echo "collapse: P(top >= 0.9) > 0.6 for 5 updates" > "$B/run/G9r/GUARD_TRIPPED"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); check 30c4 "--lr 7e-5 .*--kl-anchor-coef 0.3 .*--guard-entropy-min 0.35|--guard-entropy-min 0.35 .*--kl-anchor-coef 0.3 .*--lr-schedule power --lr-horizon 300000000 --resume $B/run/G9r/final_model.pt$" "$o"
check 30c5 "--lr-schedule power --lr-horizon 300000000 --resume $B/run/G9r/final_model.pt$" "$o"
# 30d: 741849 without a G9r final_model.pt (its first trainer crashed): its newest local checkpoint; none anywhere: P3
g9rtree; park; SLURM_JOB_ID=741849 src g9 A16 > /dev/null; mkdir -p "$B/run/G9r/checkpoints"; echo k1 > "$B/run/G9r/checkpoints/ckpt_00005000000.pt"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); check 30d "--lr-schedule power --lr-horizon 300000000 --resume $B/run/G9r/checkpoints/ckpt_00005000000.pt$" "$o"
rm -rf "$B/run/G9r/checkpoints"; o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); check 30d2 "--lr-schedule power --lr-horizon 300000000 --resume $P3$" "$o"
check 30d3 "G9r: no final_model.pt of its own here: resuming P3 again \(no G9r save anywhere\)" "$(cat "$B/run/train.log")"
# 30e: no G9r wrapper -> HOLD (<= 4 h; here 0 iterations), never the default src (it has no --lr-schedule)
rm -f "$W"; o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); yes_ 30e '[ -f "$N/runs/g9/HOLD" ]'; check 30e2 "MODE g9r but no $W .*HOLD" "$(cat "$B/run/train.log")"; rm -f "$N/runs/g9/HOLD"
# 30f: the successor 810367 (fresh job) after G9r started: the newest MIRRORED G9r file, never the newer parked G9q /
# G9p dirs or a newer G9q mirror; A16 and >= 40 GB settings; wrapper; November sets
g9rtree; echo g9r > "$N/runs/g9/MODE"; date > "$N/runs/g9r/G9R_STARTED"; K=$N/runs/g9/20261109-021316-741849
mkdir -p "$K/G9r/pool/frozen" "$K/G9r/checkpoints"; echo g9r-mirror-ckpt > "$K/G9r/checkpoints/ckpt_00001000000.pt"; touch -d '2026-11-12 01:00' "$K/G9r/checkpoints/ckpt_00001000000.pt"
echo g9r-mirror-final > "$K/G9r/final_model.pt"; touch -d '2026-11-12 01:44' "$K/G9r/final_model.pt"; echo b > "$K/G9r/best_model.pt"; echo P3 > "$K/G9r/pool/frozen/P3-g9-s2.pt"; echo s > "$K/G9r/pool/00000500000.pt"
for d in "$N/runs/g9q-parked" "$N/runs/g9p-parked-a16" "$K/G9q"; do mkdir -p "$d/pool"; echo newer > "$d/final_model.pt"; touch -d '2026-11-12 01:50' "$d/final_model.pt"; done
for g in 15356 46068; do
  S=$ROOTM/local-S; rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9r"
  x="--snapshot-cache 72 --snapshot-cache-cpu 200"; [ $g = 46068 ] && x="--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200"
  o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=$g src g9 S); check "30f-$g" "TRAIN_ARGS=$(G9RA "$S/run/G9r" "$x" "$K/G9r/final_model.pt")$" "$o"
done
yes_ 30f2 '[ -f "$S/run/G9r/best_model.pt" ] && [ -f "$S/run/G9r/pool/frozen/P3-g9-s2.pt" ] && [ -L "$S/run/final_model.pt" ] && [ "$(cat "$N/runs/g9/MODE")" = g9r ]'
check 30f3 "^PY=$W\|" "$(pyof 810367 g9 S)"; check 30f4 "child sees $NOV" "$(psof 810367 g9 S)"
# 30g: a swap COMMIT applied AFTER g9r is set (a fresh job: SWAP_APPLIED absent) never re-flips MODE: G9r continues
g9rtree; echo g9r > "$N/runs/g9/MODE"; date > "$N/runs/g9r/G9R_STARTED"; mkdir -p "$B/run/G9r/pool"; echo g9r-final > "$B/run/G9r/final_model.pt"
mkdir -p "$N/runs/swap-50M"; echo "COMMIT by r1 718701 2026-11-11T16:00:00" > "$N/runs/swap-50M/DECISION"
for h in "$N/runs/g9q/handover-a30" "$N/runs/r1/handover-g9p-a16"; do mkdir -p "$h/pool"; echo h > "$h/final_model.pt"; echo d > "$h/DONE"; done
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); check 30g "TRAIN_ARGS=$(G9RA "$B/run/G9r" "--snapshot-cache 72 --snapshot-cache-cpu 200" "$B/run/G9r/final_model.pt")$" "$o"
yes_ 30g2 '[ "$(cat "$N/runs/g9/MODE")" = g9r ] && [ -f "$N/runs/g9/SWAP_APPLIED" ] && [ ! -e "$N/runs/g9/G9P_A16_STARTED" ]'
check 30g3 "applied COMMIT: MODE g9r -> g9r" "$(cat "$N/runs/swap-50M/STATE")"
rm -f "$N/runs/g9/SWAP_APPLIED"; S=$ROOTM/local-S; rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9r"
K=$N/runs/g9/20261109-021316-741849; mkdir -p "$K/G9r"; echo g9r-mirror > "$K/G9r/final_model.pt"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 S); check 30g4 "--run-dir $S/run/G9r .*--resume $K/G9r/final_model.pt$" "$o"; yes_ 30g5 '[ "$(cat "$N/runs/g9/MODE")" = g9r ]'
# 31a: the park from MODE g9p2 (after a committed + applied swap: G9p on this A16) -> runs/g9p-parked-a16, MODE g9r, G9r from P3
g9rtree; mkdir -p "$N/runs/swap-50M"; echo "COMMIT by g9 741849 2026-11-11T12:00:00" > "$N/runs/swap-50M/DECISION"; echo g9p2 > "$N/runs/g9/MODE"; touch "$N/runs/g9/SWAP_APPLIED" "$N/runs/g9/G9P_A16_STARTED"
mkdir -p "$B/run/G9p/pool/frozen"; echo g9p-a16-final > "$B/run/G9p/final_model.pt"; echo g9p-best > "$B/run/G9p/best_model.pt"; echo P1 > "$B/run/G9p/pool/frozen/P1-g9h-dmg-s2.pt"
for s in 00250000000 00260000000 00270000000; do echo "p$s" > "$B/run/G9p/pool/$s.pt"; done; echo "eval score 1 < best" > "$B/run/G9p/GUARD_TRIPPED.fu-010101"
printf '2026-11-11 16:00:00 v2 trainer: x\n2026-11-11 20:00:00 upd 6000 steps 290,000,000 (+41500)\n2026-11-11 20:00:20 saved final_model.pt at 290,000,000 steps\n' > "$B/run/G9p/train.log"
park; o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); Q=$N/runs/g9p-parked-a16
check 31a "TRAIN_ARGS=$(G9RA "$B/run/G9r" "--snapshot-cache 72 --snapshot-cache-cpu 200" "$P3")$" "$o"
yes_ 31a2 '[ "$(cat "$N/runs/g9/MODE")" = g9r ] && cmp -s "$Q/final_model.pt" "$B/run/G9p/final_model.pt" && [ "$(ls "$Q/pool" | grep -c pt$) $(ls "$Q/pool/frozen" | grep -c pt$)" = "3 1" ] && [ -f "$Q/GUARD_TRIPPED.fu-010101" ] && (cd "$Q" && md5sum -c --quiet MD5SUMS) && [ ! -e "$N/runs/g9q-parked" ]'
check 31a3 "^G9p PARKED .*MODE g9p2 -> g9r.*guard state runs/g9a30" "$(tr '\n' ' ' < "$Q/PARKED.txt" 2>/dev/null)"; check 31a4 "^PY=$W\|" "$(pyof 741849 g9 A16)"
# 31b: an injected copy failure (an unreadable G9q pool net): FAIL, MODE g9q, no park dir, G9q relaunches from its own save
g9rtree; park; chmod 000 "$B/run/G9q/pool/00005000000.pt"
o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); chmod 644 "$B/run/G9q/pool/00005000000.pt"
check 31b "$G9QARGS .*--fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $B/run/G9q --pool-dir $B/run/G9q/pool --snapshot-cache 72 --resume $B/run/G9q/final_model.pt$" "$o"; checkno 31b2 "lr-schedule|G9r" "$o"
yes_ 31b3 '[ "$(cat "$N/runs/g9/MODE")" = g9q ] && [ ! -e "$N/runs/g9q-parked" ] && [ ! -e "$N/runs/g9r/G9R_STARTED" ]'; check 31b4 "park FAIL \(MODE untouched: g9q\): copy" "$(cat "$N/runs/g9r/park.log")"
check 31b5 "^PY=unset\|" "$(pyof 741849 g9 A16)"
# 31c: a failed MODE write after the publish: the park dir is voided (renamed), MODE stays g9q
g9rtree; mkdir -p "$N/runs/g9/MODE.tmp"; park; o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 src g9 A16); rmdir "$N/runs/g9/MODE.tmp" 2>/dev/null
yes_ 31c '[ "$(cat "$N/runs/g9/MODE")" = g9q ] && [ ! -e "$N/runs/g9q-parked" ] && ls -d "$N"/runs/g9q-parked.void-* > /dev/null 2>&1 && [ ! -e "$N/runs/g9r/G9R_STARTED" ]'; check 31c2 "--run-dir $B/run/G9q .*--resume $B/run/G9q/final_model.pt$" "$o"
# 32: refusals (nothing done, MODE untouched, G9q continues): (a) the swap armed but unresolved, (b) a COMMIT this side
# has not applied, (c) a stale final_model.pt, (d) no PREREG, (e) v9 on disk, (f) MODE g9p2 without G9P_A16_STARTED,
# (g) G9r already started, (h) the wrong src
for c in a b c d e f g h; do
  g9rtree; m=g9q; ma=g9q
  case $c in
    a) mkdir -p "$N/runs/swap-50M"; why="without a DECISION" ;;
    b) mkdir -p "$N/runs/swap-50M"; echo "COMMIT by r1 x" > "$N/runs/swap-50M/DECISION"; why="has not applied it"; ma=g9p2 ;;  # then nv2_swap applies it
    c) touch -d '2 hours ago' "$B/run/G9q/final_model.pt"; why="s old" ;;
    d) rm -f "$N/runs/g9r/PREREG.txt"; why="no runs/g9r/PREREG.txt" ;;
    e) grep -v 'G9_MODE" = g9r' "$ENVF" > "$N/netv2.env"; why="netv2.env has no g9r mode" ;;
    f) m=g9p2; ma=g9p2; echo g9p2 > "$N/runs/g9/MODE"; why="no runs/g9/G9P_A16_STARTED" ;;
    g) date > "$N/runs/g9r/G9R_STARTED"; why="G9r started already" ;;
    h) echo "# other" >> "$N/src-$G9R_SHA/porygon/v2/train.py"; why="is not ws/netv2 $G9R_SHA" ;;
  esac
  park; o=$(SLURM_JOB_ID=741849 NV2_GPU_MIB=15356 NV2_SWAP_ITERS=0 src g9 A16)
  yes_ "32$c" '[ "$(cat "$N/runs/g9/MODE")" = $ma ] && [ ! -e "$N/runs/g9q-parked" ] && [ ! -e "$N/runs/g9p-parked-a16" ] && ls "$N"/runs/g9/INLINE_CMD.done-* > /dev/null 2>&1'
  check "32$c-2" "park FAIL \(MODE untouched: $m\): .*$why" "$(cat "$N/runs/g9r/park.log" 2>/dev/null)"
  [ $c = g ] || checkno "32$c-3" "lr-schedule" "$o"
done
# 33: the other runs never see G9r: r1 (718701) and R1b-F unchanged with MODE g9r and G9r state on disk
g9rtree; echo g9r > "$N/runs/g9/MODE"; date > "$N/runs/g9r/G9R_STARTED"
o=$(SLURM_JOB_ID=718701 src r1 A30); check 33a "$G9PARGS .*--run-dir $A/run/G9p .*--snapshot-cache 192 --stacked-opp --resume $A/run/G9p/final_model.pt$" "$o"; checkno 33b "lr-schedule|G9r|entropy-min 0.35" "$o"
check 33c "^PY=$N/bin/py-src-4f18108\|" "$(pyof 718701 r1 A30)"
if [ -z "${MOCK_SKIP_G9T:-}" ]; then
# 34-41 (g9t DRAFT, not deployed; skipped with MOCK_SKIP_G9T=1 for an env without g9t; .orchestra/netv2/NEXT-RECIPE-PROPOSAL.md 3 / 5): MODE g9t = control C's recipe + ONE
# change, --gae-lambda LAMBDA; C and LAMBDA from runs/g9t/RECIPE ("g9q|g9r 0.5|0.6", one non-comment line), recorded at
# the first start in runs/g9t/G9T_STARTED ("TIME C LAMBDA RUN JOB": later starts use the record, only in RUN). Lineage
# dir G9t, guard state runs/g9t, P3 init / anchor / frozen, P1 eval anchor, November sets, PY = the ff32b71 wrapper.
# A16 (g9): 8 workers, cache 72 / 512 + stacked-opp, cpu 200; A30 (r1): 28 workers, 192 / 640 + stacked-opp, cpu 400.
# Refused (no / malformed RECIPE, unreadable record, started in the other run) -> the run's current lineage continues
# (g9: G9r, r1: G9q) with a train.log note. The tree is the 2026-11-12 cluster state (swap COMMITted and applied: r1
# MODE g9q = G9q on the A30, G9r started on the A16; 4f18108 disabled) with MODE g9t set on one side.
g9ttree() {  # g9ttree [r1]: MODE g9t on g9 (default) or on r1 (then g9 MODE g9r)
  swaptree; A=$ROOTM/local-A30; B=$ROOTM/local-A16; rm -f "$N/bin/py-src-4f18108"
  mkdir -p "$N/runs/swap-50M" "$N/runs/g9r" "$N/runs/g9t/lam"; echo "COMMIT by r1 718701 2026-11-11T11:05:44" > "$N/runs/swap-50M/DECISION"
  touch "$N/runs/r1/SWAP_APPLIED" "$N/runs/g9/SWAP_APPLIED" "$N/runs/r1/G9Q_A30_STARTED" "$N/runs/g9/G9P_A16_STARTED"; date > "$N/runs/g9r/G9R_STARTED"
  printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-ff32b71"; chmod +x "$N/bin/py-src-ff32b71"
  echo g9q > "$N/runs/r1/MODE"; echo g9t > "$N/runs/g9/MODE"; if [ "${1:-}" = r1 ]; then echo g9t > "$N/runs/r1/MODE"; echo g9r > "$N/runs/g9/MODE"; fi
  mkdir -p "$A/run/G9q/pool/frozen" "$B/run/G9r/pool/frozen"; echo g9q-a30-final > "$A/run/G9q/final_model.pt"; echo g9r-final > "$B/run/G9r/final_model.pt"; echo g9r-best > "$B/run/G9r/best_model.pt"
  touch "$A/.toucher-G9t" "$B/.toucher-G9t" "$A/.toucher-G9q" "$B/.toucher-G9r"
}
recipe() { printf "$1" > "$N/runs/g9t/RECIPE"; }
G9TA() {  # G9TA C LAMBDA WORKERS RUNDIR OPP RESUME: the exact G9t TRAIN_ARGS
  local fl=0.5 cx=""; if [ "$1" = g9r ]; then fl=0.35; cx=" --lr-schedule power --lr-horizon 300000000"; fi
  echo "--no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min $fl --guard-ptop-max 0.6 $G9QARGS --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers $3 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $4 --pool-dir $4/pool $5$cx --gae-lambda $2 --resume $6"
}
norm() { printf '%s\n' "${1#*TRAIN_ARGS=}" | sed -E 's#/G9[qrt]([/ ])#/G9X\1#g; s# --gae-lambda [0-9.]+##; s# --resume .*$##'; }
W9=$N/bin/py-src-ff32b71; A16X="--snapshot-cache 72 --snapshot-cache-cpu 200"; A16Y="--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200"
# 34: A16 first start (job 810367, runs/g9t/G9T_STARTED absent), both C and both lambda: exact args from P3, the record,
# the log line, wrapper, November sets; "one change": the args equal C's own (G9r: MODE g9r; G9q: MODE g9q, the same job)
# without --gae-lambda; G9r's lineage files untouched
for cl in "g9q 0.5" "g9q 0.6" "g9r 0.5" "g9r 0.6"; do
  c=${cl% *}; l=${cl#* }; g9ttree; recipe "$c $l\n"
  o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
  check "34a-$c-$l" "^FORMAT=gen9randombattle\|ANCHOR=$P1\|TRAIN_ARGS=$(G9TA "$c" "$l" 8 "$B/run/G9t" "$A16X" "$P3")$" "$o"
  check "34b-$c-$l" "^[0-9T:-]+ $c $l g9 810367$" "$(cat "$N/runs/g9t/G9T_STARTED" 2>/dev/null)"
  check "34c-$c-$l" "G9t first start in run g9: from P3 $P3 \(steps 0\); C = $c's recipe .*\+ --gae-lambda $l \(runs/g9t/RECIPE\); November sets for the collector \(PORYGON_PS=$NOV\)" "$(cat "$B/run/train.log")"
  check "34d-$c-$l" "^PY=$W9\|" "$(pyof 810367 g9 A16)"; check "34e-$c-$l" "child sees $NOV" "$(psof 810367 g9 A16)"
  yes_ "34f-$c-$l" '[ -d "$B/run/G9t/pool" ] && [ "$(cat "$B/run/G9r/final_model.pt")" = g9r-final ] && [ "$(cat "$N/runs/g9/MODE")" = g9t ] && [ ! -e "$N/runs/g9/HOLD" ]'
  echo "$c" > "$N/runs/g9/MODE"; oc=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); echo g9t > "$N/runs/g9/MODE"
  yes_ "34g-$c-$l" '[ -n "$(norm "$o")" ] && [ "$(norm "$o")" = "$(norm "$oc")" ] && [ "$(printf "%s" "$o" | grep -o -- "--gae-lambda [0-9.]*")" = "--gae-lambda $l" ]'
done
check 34c2 "lr schedule power H=300000000, entropy floor 0.35\) \+ --gae-lambda 0.6" "$(cat "$B/run/train.log")"
# 35: restart in place (C g9r, lambda 0.5): its own save, no second first start; >= 40 GB: 512 + stacked-opp; a later
# RECIPE edit (or its removal) never changes the started recipe (note); G9T_STARTED unchanged
g9ttree; recipe "g9r 0.5\n"; SLURM_JOB_ID=810367 src g9 A16 > /dev/null; echo g9t-final > "$B/run/G9t/final_model.pt"; s0=$(cat "$N/runs/g9t/G9T_STARTED")
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 35a "TRAIN_ARGS=$(G9TA g9r 0.5 8 "$B/run/G9t" "$A16X" "$B/run/G9t/final_model.pt")$" "$o"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=46068 src g9 A16); check 35b "TRAIN_ARGS=$(G9TA g9r 0.5 8 "$B/run/G9t" "$A16Y" "$B/run/G9t/final_model.pt")$" "$o"
recipe "g9q 0.6\n"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 35c "TRAIN_ARGS=$(G9TA g9r 0.5 8 "$B/run/G9t" "$A16X" "$B/run/G9t/final_model.pt")$" "$o"
check 35d "G9t: runs/g9t/RECIPE now says 'g9q 0.6': ignored, G9t keeps its started recipe 'g9r 0.5'" "$(cat "$B/run/train.log")"
rm -f "$N/runs/g9t/RECIPE"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 35e "TRAIN_ARGS=$(G9TA g9r 0.5 8 "$B/run/G9t" "$A16X" "$B/run/G9t/final_model.pt")$" "$o"
yes_ 35f '[ "$(grep -c "G9t first start" "$B/run/train.log")" = 1 ] && [ "$(cat "$N/runs/g9t/G9T_STARTED")" = "$s0" ] && ! grep -q "g9t refused" "$B/run/train.log"'
# 36: a fresh job (successor) after G9t started: the newest MIRRORED G9t file (a16 / a40), never a newer G9r mirror,
# a parked dir or a G9t-named file outside a lineage mirror; best / pool / frozen re-seeded; wrapper; November sets
g9ttree; recipe "g9q 0.6\n"; echo "2026-11-13T09:00:00 g9q 0.6 g9 810367" > "$N/runs/g9t/G9T_STARTED"; K=$N/runs/g9/20261112-015035-810367
mkdir -p "$K/G9t/pool/frozen" "$K/G9t/checkpoints" "$K/G9r"; echo g9t-ckpt > "$K/G9t/checkpoints/ckpt_00001000000.pt"; touch -d '2026-11-13 10:00' "$K/G9t/checkpoints/ckpt_00001000000.pt"
echo g9t-mirror > "$K/G9t/final_model.pt"; touch -d '2026-11-13 11:00' "$K/G9t/final_model.pt"; echo b > "$K/G9t/best_model.pt"; echo P3 > "$K/G9t/pool/frozen/P3-g9-s2.pt"; echo s > "$K/G9t/pool/00000500000.pt"
echo g9r-newer > "$K/G9r/final_model.pt"; touch -d '2026-11-13 12:00' "$K/G9r/final_model.pt"
for d in "$N/runs/g9r-parked" "$N/runs/g9t-parked"; do mkdir -p "$d/pool"; echo parked > "$d/final_model.pt"; touch -d '2026-11-13 12:00' "$d/final_model.pt"; done
for g in 15356 46068; do
  S=$ROOTM/local-S; rm -rf "$S"; mkdir -p "$S/run"; touch "$S/run/train.log" "$S/.toucher-G9t"
  x=$A16X; [ $g = 46068 ] && x=$A16Y
  o=$(SLURM_JOB_ID=1100001 NV2_GPU_MIB=$g src g9 S); check "36a-$g" "TRAIN_ARGS=$(G9TA g9q 0.6 8 "$S/run/G9t" "$x" "$K/G9t/final_model.pt")$" "$o"
done
yes_ 36b '[ -f "$S/run/G9t/best_model.pt" ] && [ -f "$S/run/G9t/pool/frozen/P3-g9-s2.pt" ] && [ -L "$S/run/final_model.pt" ] && ! grep -q "first start" "$S/run/train.log"'
check 36c "^PY=$W9\|" "$(pyof 1100001 g9 S)"; check 36d "child sees $NOV" "$(psof 1100001 g9 S)"
# 37: the job has no G9t final_model.pt (its first trainer crashed): the newest checkpoint (local); none anywhere: P3 again
g9ttree; recipe "g9r 0.6\n"; SLURM_JOB_ID=810367 src g9 A16 > /dev/null; mkdir -p "$B/run/G9t/checkpoints"; echo k1 > "$B/run/G9t/checkpoints/ckpt_00005000000.pt"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 37a "TRAIN_ARGS=$(G9TA g9r 0.6 8 "$B/run/G9t" "$A16X" "$B/run/G9t/checkpoints/ckpt_00005000000.pt")$" "$o"
rm -rf "$B/run/G9t/checkpoints"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 37b "TRAIN_ARGS=$(G9TA g9r 0.6 8 "$B/run/G9t" "$A16X" "$P3")$" "$o"
check 37c "G9t: no final_model.pt of its own here: resuming P3 again \(no G9t save anywhere\)" "$(cat "$B/run/train.log")"
# 38: refusals on the A16 (no G9T_STARTED written, MODE file kept, G9r continues from its own save with G9r's exact
# args, wrapper and November sets, train.log note): no RECIPE; malformed ones; a record of another run; an unreadable
# record. 38z: comments, blank lines and CRLF around one valid line are accepted
G9RCONT="TRAIN_ARGS=$(G9RA "$B/run/G9r" "$A16X" "$B/run/G9r/final_model.pt")$"
i=0
for r in NONE "" "g9q 0.754\n" "g9s 0.5\n" "g9q\n" "g9q 0.5 x\n" "G9Q 0.5\n" "g9q 0.5\ng9r 0.6\n" "g9q,0.5\n" "0.5 g9q\n"; do
  i=$((i + 1)); g9ttree; if [ "$r" != NONE ]; then recipe "$r"; fi
  o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); why="runs/g9t/RECIPE malformed"; [ "$r" = NONE ] && why="no runs/g9t/RECIPE"
  check "38a-$i" "$G9RCONT" "$o"; checkno "38b-$i" "gae-lambda|/G9t" "$o"
  check "38c-$i" "MODE g9t refused: $why.*: G9r continues" "$(cat "$B/run/train.log")"
  yes_ "38d-$i" '[ ! -e "$N/runs/g9t/G9T_STARTED" ] && [ "$(cat "$N/runs/g9/MODE")" = g9t ] && [ ! -d "$B/run/G9t" ]'
  check "38e-$i" "^PY=$W9\|" "$(pyof 810367 g9 A16)"
done
g9ttree; recipe "g9q 0.5\n"; echo "2026-11-13T09:00:00 g9q 0.5 r1 1135184" > "$N/runs/g9t/G9T_STARTED"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 38f "$G9RCONT" "$o"; check 38f2 "MODE g9t refused: G9t started in run r1 \(2026-11-13T09:00:00\), not g9: G9r continues" "$(cat "$B/run/train.log")"
for rec in "garbage" "2026-11-13T09:00:00 g9q 0.9 g9 1" "2026-11-13T09:00:00 g9q 0.5"; do
  g9ttree; recipe "g9q 0.5\n"; echo "$rec" > "$N/runs/g9t/G9T_STARTED"
  o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check "38g-${rec// /_}" "$G9RCONT" "$o"; check "38g2-${rec// /_}" "MODE g9t refused: runs/g9t/G9T_STARTED unreadable" "$(cat "$B/run/train.log")"
done
g9ttree; recipe "# G9t recipe (pre-check verdict)\n\n  \ng9r 0.6\r\n# end\n"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
check 38z "TRAIN_ARGS=$(G9TA g9r 0.6 8 "$B/run/G9t" "$A16X" "$P3")$" "$o"; check 38z2 "^[0-9T:-]+ g9r 0.6 g9 810367$" "$(cat "$N/runs/g9t/G9T_STARTED" 2>/dev/null)"
# 39: guard trips in runs/g9t (C's rules): an anchor-drop trip -> best_model.pt with --lr 3.5e-5 (C g9r: schedule and
# floor 0.35 kept, lambda kept); a second trip -> HOLD; a collapse trip (C g9q) -> KL x3 from its own save; no wrapper -> HOLD
g9ttree; recipe "g9r 0.6\n"; SLURM_JOB_ID=810367 src g9 A16 > /dev/null; echo g9t-final > "$B/run/G9t/final_model.pt"
echo "eval score 40.0% < best 60.0% - 15% for 2 evals" > "$B/run/G9t/GUARD_TRIPPED"; echo g9t-best > "$B/run/G9t/best_model.pt"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
check 39a "TRAIN_ARGS=$(G9TA g9r 0.6 8 "$B/run/G9t" "$A16X" "$N/runs/g9t/fu-start-best.pt" | sed 's/--lr 7e-5/--lr 3.5e-5/')$" "$o"
yes_ 39b '[ "$(cut -c1-2 "$N/runs/g9t/FU.started")" = "2 " ] && [ "$(cat "$N/runs/g9t/fu-start-best.pt")" = g9t-best ] && [ ! -e "$N/runs/g9r/FU.started" ] && ls "$B"/run/G9t/GUARD_TRIPPED.fu-* > /dev/null 2>&1'
echo "collapse: entropy < 0.35 for 5 updates" > "$B/run/G9t/GUARD_TRIPPED"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
yes_ 39c '[ -f "$N/runs/g9/HOLD" ] && ls "$B"/run/G9t/GUARD_TRIPPED.second-* > /dev/null 2>&1'; rm -f "$N/runs/g9/HOLD"
g9ttree; recipe "g9q 0.5\n"; SLURM_JOB_ID=810367 src g9 A16 > /dev/null; echo g9t-final > "$B/run/G9t/final_model.pt"; echo "collapse: P(top >= 0.9) > 0.6 for 5 updates" > "$B/run/G9t/GUARD_TRIPPED"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 39d "TRAIN_ARGS=$(G9TA g9q 0.5 8 "$B/run/G9t" "$A16X" "$B/run/G9t/final_model.pt" | sed 's/--kl-anchor-coef 0.1/--kl-anchor-coef 0.3/')$" "$o"
check 39e "^1 |^2 .*collapse" "$(cat "$N/runs/g9t/FU.started" "$N/runs/g9t/FU.reason" 2>/dev/null | tr '\n' ' ')"
rm -f "$W9"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); yes_ 39f '[ -f "$N/runs/g9/HOLD" ]'; check 39f2 "MODE g9t but no $W9 .*HOLD" "$(cat "$B/run/train.log")"; rm -f "$N/runs/g9/HOLD"
# 40: the A30 (r1 MODE g9t, job 1135184): first start (both C), restart in place, >= 40 GB, refusal -> G9q continues
# (its own A30 save, its args); the g9 side then refuses a G9t start (the record says r1) and G9r continues there
for c in g9q g9r; do
  g9ttree r1; recipe "$c 0.5\n"
  o=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=24576 src r1 A30); check "40a-$c" "^FORMAT=gen9randombattle\|ANCHOR=$P1\|TRAIN_ARGS=$(G9TA $c 0.5 28 "$A/run/G9t" "--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400" "$P3")$" "$o"
  check "40b-$c" "^[0-9T:-]+ $c 0.5 r1 1135184$" "$(cat "$N/runs/g9t/G9T_STARTED" 2>/dev/null)"; check "40c-$c" "^PY=$W9\|" "$(pyof 1135184 r1 A30)"; check "40d-$c" "child sees $NOV" "$(psof 1135184 r1 A30)"
done
echo g9t-a30-final > "$A/run/G9t/final_model.pt"
o=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=24576 src r1 A30); check 40e "TRAIN_ARGS=$(G9TA g9r 0.5 28 "$A/run/G9t" "--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400" "$A/run/G9t/final_model.pt")$" "$o"
o=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=46068 src r1 A30); check 40f "TRAIN_ARGS=$(G9TA g9r 0.5 28 "$A/run/G9t" "--snapshot-cache 640 --stacked-opp --snapshot-cache-cpu 400" "$A/run/G9t/final_model.pt")$" "$o"
echo g9t > "$N/runs/g9/MODE"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 40g "$G9RCONT" "$o"; check 40g2 "MODE g9t refused: G9t started in run r1 .*: G9r continues" "$(cat "$B/run/train.log")"
G9QCONT="TRAIN_ARGS=--no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 $G9QARGS --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 28 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir $A/run/G9q --pool-dir $A/run/G9q/pool --snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume $A/run/G9q/final_model.pt$"
g9ttree r1; o=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=24576 src r1 A30); check 40h "$G9QCONT" "$o"; check 40h2 "MODE g9t refused: no runs/g9t/RECIPE: G9q continues" "$(cat "$A/run/train.log")"
echo g9q > "$N/runs/r1/MODE"; oq=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=24576 src r1 A30); yes_ 40h3 '[ "$o" = "$oq" ] && [ ! -e "$N/runs/g9t/G9T_STARTED" ]'
g9ttree r1; recipe "g9r 0.6\n"; echo "2026-11-13T09:00:00 g9r 0.6 g9 810367" > "$N/runs/g9t/G9T_STARTED"; o=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=24576 src r1 A30)
check 40i "$G9QCONT" "$o"; check 40i2 "MODE g9t refused: G9t started in run g9 .*: G9q continues" "$(cat "$A/run/train.log")"
# 41: isolation: with G9t training on one side, the other side's lineage and R1b-F are unchanged; a swap COMMIT that a
# fresh job would apply late (SWAP_APPLIED absent) never flips MODE g9t on either side
g9ttree; recipe "g9r 0.5\n"; SLURM_JOB_ID=810367 src g9 A16 > /dev/null
o=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=24576 src r1 A30); check 41a "$G9QCONT" "$o"; checkno 41b "gae-lambda|G9t|lr-schedule" "$o"
rm -f "$N/runs/g9/SWAP_APPLIED"; echo g9t-final > "$B/run/G9t/final_model.pt"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
check 41c "TRAIN_ARGS=$(G9TA g9r 0.5 8 "$B/run/G9t" "$A16X" "$B/run/G9t/final_model.pt")$" "$o"; yes_ 41d '[ "$(cat "$N/runs/g9/MODE")" = g9t ] && [ -f "$N/runs/g9/SWAP_APPLIED" ]'
g9ttree r1; recipe "g9q 0.6\n"; rm -f "$N/runs/r1/SWAP_APPLIED"; o=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=24576 src r1 A30)
check 41e "TRAIN_ARGS=$(G9TA g9q 0.6 28 "$A/run/G9t" "--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400" "$P3")$" "$o"; yes_ 41f '[ "$(cat "$N/runs/r1/MODE")" = g9t ]'
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 41g "$G9RCONT" "$o"; checkno 41h "gae-lambda|G9t" "$o"
g4tree; L=$ROOTM/local-T4; mkdir -p "$N/runs/g9t" "$N/bin"; echo "g9q 0.5" > "$N/runs/g9t/RECIPE"; echo g9t > "$N/runs/g9/MODE"; echo g9t > "$N/runs/r1/MODE"
printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-24c633f"; chmod +x "$N/bin/py-src-24c633f"
o=$(SLURM_JOB_ID=829660 src g4r1bf T4); check 41i "--snapshot-cache 40 --stacked-opp --snapshot-cache-cpu 200 --guard-pg-mean 20 --guard-anchor-drop 0.0999 --resume $N/runs/r1b-parked/final_model.pt$" "$o"; checkno 41j "gae-lambda" "$o"
fi
# 42-49 (v12, 2026-11-12, orchestrator-approved): the successor watchdog (nv2_watchdog, at a job's FIRST source, ids >=
# 1135476, runs r1 / g9 / g4r1bf). Stubs on PATH (scripts/netv2_wd_stubs: squeue from $WDQ, sbatch with the cluster's
# banner that registers its job PENDING, scontrol, df from $WDQ.df, hostname nodex). The tree is the 2026-11-12 state.
WDB=$(dirname "$(readlink -f "$0")")/netv2_wd_stubs; WDQ=$ROOTM.wdq; export WDQ
wdtree() {  # swaptree + now: COMMIT applied, r1 MODE g9q (G9q on the A30), g9 MODE g9r (G9r on the A16), R1b-F parked
  swaptree; rm -f "$N/bin/py-src-4f18108"; rm -f "$WDQ"*; echo 1300001 > "$WDQ.next"
  mkdir -p "$N/runs/swap-50M" "$N/runs/g9r" "$N/runs/g4r1bf" "$N/runs/r1b-parked/pool"; echo "COMMIT by r1 718701 2026-11-11T11:05:44" > "$N/runs/swap-50M/DECISION"
  printf '2026-11-11T11:05:44 r1 718701 phase2 applied COMMIT: MODE g9p -> g9q\n' > "$N/runs/swap-50M/STATE"
  touch "$N/runs/r1/SWAP_APPLIED" "$N/runs/g9/SWAP_APPLIED" "$N/runs/r1/G9Q_A30_STARTED" "$N/runs/g9/G9P_A16_STARTED" "$N/runs/g4r1bf/R1BF_STARTED"; date > "$N/runs/g9r/G9R_STARTED"
  printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-ff32b71"; chmod +x "$N/bin/py-src-ff32b71"; echo g9q > "$N/runs/r1/MODE"; echo g9r > "$N/runs/g9/MODE"
  echo 0 > "$N/runs/r1/chain_left"; echo 0 > "$N/runs/g9/chain_left"; echo 3 > "$N/runs/g4r1bf/chain_left"; echo x > "$N/runs/r1b-parked/final_model.pt"; echo b75 > "$N/runs/r1/restart-best-7.5M.pt"
  J2=$N/runs/r1/20261112-112408-1135184/G9q; K2=$N/runs/g9/20261112-015035-810367/G9r; Q2=$N/runs/g4r1bf/20261109-190000-829660/R1bF
  for d in "$J2" "$K2" "$Q2"; do mkdir -p "$d/pool/frozen"; echo "mirror-${d##*/}" > "$d/final_model.pt"; done
  rm -rf "$ROOTM"/wd-*
}
wdsrc() {  # wdsrc RUN JOB NAME [JOBARGS...]: a job's first source (fresh LOCAL, as the job script makes it) -> "RC=n|TRAIN_ARGS=..."
  local run=$1 job=$2 name=$3; shift 3
  mkdir -p "$ROOTM/wd-$job/run" "$ROOTM/wd-$job/pool"; touch "$ROOTM/wd-$job/.toucher-G9q" "$ROOTM/wd-$job/.toucher-G9r" "$ROOTM/wd-$job/.toucher-R1bF" "$ROOTM/wd-$job/.toucher-G9p"
  ( set -euo pipefail; PATH=$WDB:$PATH; NV2=$N; LOCAL=$ROOTM/wd-$job; RUN_NAME=$run; SLURM_JOB_ID=$job; SLURM_JOB_NAME=$name
    SLURM_JOB_PARTITION=${WD_PART:-a30}; SLURM_CPUS_ON_NODE=${WD_CPUS:-30}; SLURM_MEM_PER_NODE=${WD_MEM:-98304}; NV2_HOLD_ITERS=0; NV2_SWAP_ITERS=0
    NV2_TMPWATCH_POLL_S=0; NV2_WD_RETRY_S=0; NV2_GPU_MIB=${WD_GPU:-24576}
    source "$ENVF"; echo "TRAIN_ARGS=$TRAIN_ARGS" ) > "$ROOTM.wdout" 2>&1
  echo "RC=$?|$(grep '^TRAIN_ARGS=' "$ROOTM.wdout" || true)"
}
sq() { echo "$1 $2 $3 $4 $5" >> "$WDQ"; }  # sq ID NAME STATE START DEP
nsb() { grep -c . "$WDQ.sbatch" 2> /dev/null || echo 0; }
wdlog() { cat "$N/runs/$1/WATCHDOG.log" 2> /dev/null | tr '\n' ' '; }
RES="--job-name=ws-netv2-r1 --partition=a30 --gres=gpu:1 --cpus-per-task=30 --mem=98304M --time=72:00:00"
# 42: inactive: the running jobs (1135184 r1, 810367 g9, 829660 g4r1bf) at a restart in place or a first source,
# ids < 1135476, other runs, the kill switch -> no squeue / sbatch / JOBSTARTS / WATCHDOG.log; TRAIN_ARGS as without it
wdtree; sq 1135184 ws-netv2-r1 RUNNING 2026-11-12T11:12:24 "(null)"
for c in "r1 1135184 ws-netv2-r1" "g9 810367 ws-netv2-g9" "g4r1bf 829660 ws-netv2-g4" "r1 1135475 ws-netv2-r1" "smoke 1200009 ws-netv2-smoke"; do
  read -r r j nm <<< "$c"; o=$(wdsrc $r $j $nm); o2=$(NV2_WD_OFF=1 wdsrc $r $j $nm)
  yes_ "42a-$j" '[ "${o%%|*}" = RC=0 ] && [ "$o" = "$o2" ] && [ ! -e "$WDQ.sbatch" ] && [ ! -e "$WDQ.calls" ] && [ ! -e "$N/runs/$r/JOBSTARTS" ] && [ ! -e "$N/runs/$r/WATCHDOG.log" ] && [ ! -e "$ROOTM/wd-$j/.nv2-watchdog" ]'
done
touch "$N/runs/WATCHDOG_OFF"; o=$(wdsrc r1 1200001 ws-netv2-r1); yes_ 42b '[ "${o%%|*}" = RC=0 ] && [ ! -e "$WDQ.sbatch" ] && [ ! -e "$N/runs/r1/JOBSTARTS" ]'; rm -f "$N/runs/WATCHDOG_OFF"
# 43: r1's pending successor 1135476 starts (fresh job; 1135184 gone): JOBSTARTS, a successor afterany:1135476 with this
# job's resources and job args, its id parsed past the banner, chain_left stays 0, MODE / TRAIN_ARGS as without the watchdog
wdtree; o=$(wdsrc r1 1135476 ws-netv2-r1 --extra 1); o2=$(NV2_WD_OFF=1 wdsrc r1 1135476 ws-netv2-r1 --extra 1)
check 43a "^sbatch $RES --dependency=afterany:1135476 scripts/slurm_train_v2.sbatch r1 --extra 1$" "$(cat "$WDQ.sbatch" 2> /dev/null)"
yes_ 43b '[ "$(nsb)" = 1 ] && [ "${o%%|*}" = RC=0 ] && [ "${o#*|}" = "${o2#*|}" ] && [ -n "${o#*|TRAIN_ARGS=}" ]'
check 43c "queued successor 1300001 \(afterany:1135476\)" "$(wdlog r1)"; check 43d "^1135476 [0-9]+ nodex$" "$(cat "$N/runs/r1/JOBSTARTS" 2> /dev/null)"
check 43e "watchdog: queued successor 1300001" "$(cat "$ROOTM/wd-1135476/run/train.log")"; yes_ 43f '[ "$(cat "$N/runs/r1/chain_left") $(cat "$N/runs/r1/MODE")" = "0 g9q" ]'
check 43g "TRAIN_ARGS=.*--run-dir $ROOTM/wd-1135476/run/G9q .*--resume $J2/final_model.pt$" "$o"
# 43h: g4r1bf's first chained job (chain_left 3): chain_left -> 0, its successor on the training partition
wdtree; o=$(WD_PART=training WD_CPUS=10 WD_MEM=40960 wdsrc g4r1bf 1200100 ws-netv2-g4)
check 43h "^sbatch --job-name=ws-netv2-g4 --partition=training --gres=gpu:1 --cpus-per-task=10 --mem=40960M --time=72:00:00 --dependency=afterany:1200100 scripts/slurm_train_v2.sbatch g4r1bf$" "$(cat "$WDQ.sbatch" 2> /dev/null)"
check 43i "chain_left 3 -> 0" "$(wdlog g4r1bf)"; yes_ 43j '[ "$(cat "$N/runs/g4r1bf/chain_left")" = 0 ]'; check 43k "TRAIN_ARGS=.*--resume $Q2/final_model.pt$" "$o"
# 44 (b): at most one pending successor: a second source in the same job (a restart in place), the successor's own
# start while another job of the name is pending, a requeue of the same id; two jobs racing (both RUNNING)
wdtree; wdsrc r1 1135476 ws-netv2-r1 > /dev/null; o=$(wdsrc r1 1135476 ws-netv2-r1); yes_ 44a '[ "$(nsb)" = 1 ] && [ "$(grep -c . "$N/runs/r1/JOBSTARTS")" = 1 ] && [ "${o%%|*}" = RC=0 ]'
wdtree; sq 1135500 ws-netv2-r1 PENDING N/A "(null)"; o=$(wdsrc r1 1135476 ws-netv2-r1); yes_ 44b '[ "$(nsb)" = 0 ] && [ "${o%%|*}" = RC=0 ]'; check 44b2 "successor already pending: 1135500" "$(wdlog r1)"
wdtree; wdsrc r1 1135476 ws-netv2-r1 > /dev/null; rm -rf "$ROOTM/wd-1135476"; o=$(wdsrc r1 1135476 ws-netv2-r1)  # requeued (same id, new LOCAL)
yes_ 44c '[ "$(nsb)" = 1 ] && [ "$(grep -c . "$N/runs/r1/JOBSTARTS")" = 2 ]'
wdtree; sq 1135476 ws-netv2-r1 RUNNING 2026-11-13T10:00:00 "(null)"; sq 1135600 ws-netv2-r1 RUNNING 2026-11-13T10:05:00 "(null)"; sq 1300001 ws-netv2-r1 PENDING N/A afterany:1135476; echo 1300002 > "$WDQ.next"
o=$(wdsrc r1 1135600 ws-netv2-r1); yes_ 44d '[ "$o" = "RC=0|" ] && [ "$(nsb)" = 0 ] && [ ! -e "$N/runs/r1/JOBSTARTS" ] && [ ! -d "$ROOTM/wd-1135600" ]'
check 44d2 "exit without training or chaining: job 1135476 of run r1 is already running \(started 2026-11-13T10:00:00\); pending: 1300001" "$(wdlog r1)"
o=$(wdsrc r1 1135476 ws-netv2-r1); yes_ 44e '[ "${o%%|*}" = RC=0 ] && [ -n "${o#*|TRAIN_ARGS=}" ] && [ "$(nsb)" = 0 ]'  # the earlier job trains on
wdtree; sq 1135476 ws-netv2-r1 RUNNING 2026-11-13T10:00:00 "(null)"; sq 1135600 ws-netv2-r1 RUNNING 2026-11-13T10:00:00 "(null)"  # same second, nothing pending
o=$(wdsrc r1 1135600 ws-netv2-r1); yes_ 44f '[ "$o" = "RC=0|" ]'; check 44f2 "^sbatch $RES --dependency=afterany:1135476 scripts/slurm_train_v2.sbatch r1$" "$(cat "$WDQ.sbatch" 2> /dev/null)"
o=$(wdsrc r1 1135476 ws-netv2-r1); yes_ 44g '[ -n "${o#*|TRAIN_ARGS=}" ] && [ "$(nsb)" = 1 ]'; check 44g2 "successor already pending: 1300001" "$(wdlog r1)"
# 45 (a): deliberate stops: the job exits 0 at once (no training, no successor, no JOBSTARTS, LOCAL removed, reason logged)
k=0
for c in "r1 HOLD" "g9 PARKED" "g4r1bf CHAIN_STOPPED" "g9 INLINE_CMD.running" "g9 park" "r1 swapkit" "r1 swapstate" "g9 swapfail"; do
  read -r r what <<< "$c"; k=$((k + 1)); wdtree; nm=ws-netv2-$r; [ $r = g4r1bf ] && nm=ws-netv2-g4
  case $what in
    park) cp "$G9R_KIT/inline-a16-park-g9r.sh" "$N/runs/g9/INLINE_CMD"; why="a park / swap INLINE_CMD is queued" ;;
    swapkit) cp "$KIT/inline-718701-g9p.sh" "$N/runs/r1/INLINE_CMD"; why="a park / swap INLINE_CMD is queued" ;;
    swapstate) rm -f "$N/runs/swap-50M/DECISION"; mkdir -p "$N/runs/r1/handover-g9p-a16"; echo d > "$N/runs/r1/handover-g9p-a16/DONE"; why="the 50M swap is in progress" ;;
    swapfail) rm -f "$N/runs/swap-50M/DECISION" "$N/runs/swap-50M/STATE"; echo x > "$N/runs/swap-50M/FAIL-r1"; why="the 50M swap is in progress" ;;
    *) echo stop > "$N/runs/$r/$what"; why="runs/$r/$what exists" ;;
  esac
  o=$(wdsrc $r 1200200 $nm)
  yes_ "45a-$k" '[ "$o" = "RC=0|" ] && [ "$(nsb)" = 0 ] && [ ! -e "$N/runs/$r/JOBSTARTS" ] && [ ! -d "$ROOTM/wd-1200200" ]'; check "45b-$k" "exit without training or chaining: $why" "$(wdlog $r)"
done
# 45c: not stops: a decided swap long ago (above), a finished park's PARKED.txt, runs/r1/R1B_PARKED, a failed park's leftover
# *.tmp, a generic queued INLINE_CMD, an undecided swap for g4r1bf
wdtree; mkdir -p "$N/runs/g9p-parked-a16" "$N/runs/g9q-parked.tmp"; echo p > "$N/runs/g9p-parked-a16/PARKED.txt"; touch "$N/runs/r1/R1B_PARKED"; echo 'echo hello' > "$N/runs/g9/INLINE_CMD"
o=$(WD_PART=a16 WD_GPU=15356 wdsrc g9 1200201 ws-netv2-g9); yes_ 45c '[ "${o%%|*}" = RC=0 ] && [ -n "${o#*|TRAIN_ARGS=}" ] && [ "$(nsb)" = 1 ]'
o=$(wdsrc r1 1200202 ws-netv2-r1); yes_ 45c2 '[ -n "${o#*|TRAIN_ARGS=}" ]'
rm -f "$N/runs/swap-50M/DECISION"; echo x > "$N/runs/swap-50M/FAIL-r1"; o=$(WD_PART=training wdsrc g4r1bf 1200203 ws-netv2-g4); yes_ 45c3 '[ -n "${o#*|TRAIN_ARGS=}" ]'
# 46: crash-loop cap: the previous two jobs each lived < 30 min -> CHAIN_STOPPED (reason, JOBSTARTS), no successor, it trains;
# the next job then exits at once; long-lived predecessors -> no cap
wdtree; t=$(date +%s); printf '1200300 %s n1\n1200301 %s n2\n' $((t - 1500)) $((t - 600)) > "$N/runs/r1/JOBSTARTS"
o=$(wdsrc r1 1200302 ws-netv2-r1); yes_ 46a '[ "${o%%|*}" = RC=0 ] && [ -n "${o#*|TRAIN_ARGS=}" ] && [ "$(nsb)" = 0 ]'
check 46b "the previous two jobs each lived < 30 min \(900 s, 6[0-9][0-9] s\): crash loop; this job keeps training" "$(head -n 1 "$N/runs/r1/CHAIN_STOPPED" 2> /dev/null)"
check 46c "JOBSTARTS \(last 3\): 1200300 .* 1200302 " "$(tr '\n' ' ' < "$N/runs/r1/CHAIN_STOPPED" 2> /dev/null)"
o=$(wdsrc r1 1200303 ws-netv2-r1); yes_ 46d '[ "$o" = "RC=0|" ] && [ "$(nsb)" = 0 ]'
wdtree; printf '1200300 %s n1\n1200301 %s n2\n' $((t - 9000)) $((t - 600)) > "$N/runs/r1/JOBSTARTS"; o=$(wdsrc r1 1200302 ws-netv2-r1); yes_ 46e '[ "$(nsb)" = 1 ] && [ ! -e "$N/runs/r1/CHAIN_STOPPED" ]'
# 47: /tmp guard (< 20 GB at the start): resubmit with --exclude=<node>[,previous], no --dependency, exit 1 before
# training, MODE kept, TMP_ALERT; a pending job instead gets the node in its ExcNodeList; with a crash loop: no resubmit
wdtree; echo $((10 * 1048576)) > "$WDQ.df"; o=$(SLURM_JOB_EXC_NODES=hopper wdsrc r1 1200400 ws-netv2-r1 --extra 1)
yes_ 47a '[ "$o" = "RC=1|" ] && [ "$(nsb)" = 1 ] && [ "$(cat "$N/runs/r1/MODE")" = g9q ] && [ ! -d "$ROOTM/wd-1200400" ] && [ -f "$N/runs/r1/TMP_ALERT" ]'
check 47b "^sbatch $RES --exclude=nodex,hopper scripts/slurm_train_v2.sbatch r1 --extra 1$" "$(cat "$WDQ.sbatch" 2> /dev/null)"
check 47c "/tmp has 10 GB < 20 GB: resubmitted as 1300001 excluding nodex,hopper; exit before training \(MODE stays g9q\)" "$(wdlog r1)"
echo $((25 * 1048576)) > "$WDQ.df"; o=$(wdsrc r1 1300001 ws-netv2-r1); yes_ 47d '[ -n "${o#*|TRAIN_ARGS=}" ] && [ "$(nsb)" = 2 ]'; check 47d2 "TRAIN_ARGS=.*--run-dir $ROOTM/wd-1300001/run/G9q " "$o"
wdtree; sq 1300009 ws-netv2-r1 PENDING N/A afterany:1135184; echo hopper > "$WDQ.exc-1300009"; echo $((5 * 1048576)) > "$WDQ.df"; o=$(wdsrc r1 1200401 ws-netv2-r1)
yes_ 47e '[ "$o" = "RC=1|" ] && [ "$(nsb)" = 0 ]'; check 47e2 "scontrol update jobid=1300009 ExcNodeList=hopper,nodex" "$(cat "$WDQ.scontrol" 2> /dev/null)"
wdtree; printf '1200300 %s n1\n1200301 %s n2\n' $((t - 1500)) $((t - 600)) > "$N/runs/r1/JOBSTARTS"; echo $((5 * 1048576)) > "$WDQ.df"; o=$(wdsrc r1 1200402 ws-netv2-r1)
yes_ 47f '[ "$o" = "RC=1|" ] && [ "$(nsb)" = 0 ]'; check 47f2 "/tmp has 5 GB < 20 GB on nodex and the previous two jobs" "$(head -n 1 "$N/runs/r1/CHAIN_STOPPED" 2> /dev/null)"
# 48-49: squeue failing (3 tries) -> no duplicate check and no successor (logged); sbatch failing -> logged, it trains
wdtree; touch "$WDQ.fail"; o=$(wdsrc r1 1200500 ws-netv2-r1); yes_ 48a '[ -n "${o#*|TRAIN_ARGS=}" ] && [ "$(nsb)" = 0 ] && [ "$(grep -c "^squeue" "$WDQ.calls")" = 3 ]'
check 48b "squeue failed 3 times.*no successor queued \(squeue failed\)" "$(wdlog r1)"; rm -f "$WDQ.fail"
wdtree; touch "$WDQ.sbfail"; o=$(wdsrc r1 1200501 ws-netv2-r1); yes_ 49a '[ -n "${o#*|TRAIN_ARGS=}" ]'; check 49b "queued successor NOTHING \(sbatch failed\)" "$(wdlog r1)"
if [ -z "${MOCK_SKIP_G9T:-}" ] && [ -z "${MOCK_SKIP_V14:-}" ]; then
# 50-55 (v14, 2026-11-12; G9t takes the A16, G9r PARKED resumable; skipped with MOCK_SKIP_V14=1 or MOCK_SKIP_G9T=1). The
# park INLINE_CMD ($G9T_KIT, default scripts/netv2_g9t/inline-a16-park-g9r-start-g9t.sh) runs in 810367 (g9 MODE g9r)
# after the G9r trainer's SIGINT: G9r's lineage dir copied whole to runs/g9r-parked (MD5SUMS, load check, PARKED.txt),
# then MODE g9t; the same sourcing is G9t's first start (RECIPE "g9q 0.5": G9q's recipe + --gae-lambda 0.5 from P3).
# v14's one change: MODE g9t refused while runs/g9r-parked/PARKED.txt exists -> HOLD (allocation kept), re-check, never
# G9r; not parked -> v13 (G9r continues). The tree is the 2026-11-12 cluster state (swap COMMITted + applied, r1 g9q on
# the A30, g9 g9r on the A16 = G9r in 810367 at 60,000,000 steps; 4f18108 disabled).
G9T_KIT=${G9T_KIT:-$(dirname "$(readlink -f "$0")")/netv2_g9t}; PK0=$NV2_PARK_PY
printf '#!/bin/bash\n# mock: the park script'"'"'s checkpoint load (PARK_MOCK_STEPS / PARK_MOCK_FAIL)\n[ -z "${PARK_MOCK_FAIL:-}" ] || { echo "RuntimeError: mock load failure"; exit 1; }\necho "counters: {'"'"'steps'"'"': ${PARK_MOCK_STEPS:-60000000}, '"'"'updates'"'"': 1440, '"'"'games'"'"': 1}"; echo "keys: ['"'"'args'"'"', '"'"'counters'"'"', '"'"'optimizer'"'"']"; echo "pp: $PYTHONPATH"\n' > "$ROOTM.parkpy14"; chmod +x "$ROOTM.parkpy14"
export NV2_PARK_PY=$ROOTM.parkpy14
t14tree() {  # g9rtree (v-env on disk, ff32b71 src + wrapper) + the 2026-11-12 state; G9r live in 810367 ($B/run/G9r)
  g9rtree; rm -f "$N/bin/py-src-4f18108"
  echo "COMMIT by r1 718701 2026-11-11T11:05:44" > "$N/runs/swap-50M/DECISION"; touch "$N/runs/r1/SWAP_APPLIED" "$N/runs/g9/SWAP_APPLIED" "$N/runs/r1/G9Q_A30_STARTED" "$N/runs/g9/G9P_A16_STARTED"
  echo g9q > "$N/runs/r1/MODE"; echo g9r > "$N/runs/g9/MODE"; date > "$N/runs/g9r/G9R_STARTED"; echo "old trip" > "$N/runs/g9r/FU.reason"
  mkdir -p "$A/run/G9q/pool" "$N/runs/g9t/lam"; echo g9q-a30-final > "$A/run/G9q/final_model.pt"
  GR=$B/run/G9r; mkdir -p "$GR/pool/frozen" "$GR/checkpoints" "$GR/tb" "$GR/eval"
  echo g9r-final-60M > "$GR/final_model.pt"; echo g9r-best > "$GR/best_model.pt"; echo g9r-ema > "$GR/ema_model.pt"
  for s in 00050000000 00055000000; do echo "r$s" > "$GR/pool/$s.pt"; done; echo P3 > "$GR/pool/frozen/P3-g9-s2.pt"; echo br > "$GR/pool/frozen/best-00045000000.pt"
  echo c1 > "$GR/checkpoints/ckpt_00050003097.pt"; echo c2 > "$GR/checkpoints/ckpt_00055000000.pt"; echo t > "$GR/tb/events.x"; echo e > "$GR/eval/eval.log"; echo old > "$GR/GUARD_TRIPPED.fu-010101"
  printf '2026-11-12 18:49:30 v2 trainer: gen9randombattle ...\n2026-11-12 23:40:00 upd 1440 steps 60,000,000 (+41600) sps 440\n2026-11-12 23:40:01 [eval @ 60,000,000] vs P1 61.0%%\n2026-11-12 23:41:30 saved final_model.pt at 60,000,000 steps\n' > "$GR/train.log"
  echo "g9q 0.5" > "$N/runs/g9t/RECIPE"; echo prereg-g9t > "$N/runs/g9t/PREREG.txt"; chmod 444 "$N/runs/g9t/PREREG.txt"
  touch "$B/.toucher-G9t" "$B/.toucher-G9r"
}
park14() {  # the netv2 agent installs the INLINE_CMD (10 min ago), then the trainer's SIGINT save lands (now)
  cp "$G9T_KIT/inline-a16-park-g9r-start-g9t.sh" "$N/runs/g9/INLINE_CMD" 2>/dev/null || echo 'echo "mock: no g9t park kit"; exit 1' > "$N/runs/g9/INLINE_CMD"
  touch -d '10 minutes ago' "$N/runs/g9/INLINE_CMD"; touch "$B/run/G9r/final_model.pt"
}
Z=$N/runs/g9r-parked
# 50 (i): the park in 810367's restart in place: G9r parked whole, MODE g9t, the same sourcing = G9t's first start from
# P3 with G9q's recipe + --gae-lambda 0.5 (exact args: wsd, constant lr 7e-5, entropy floor 0.5, KL anchor + frozen P3,
# P1 eval anchor, 8 workers, cache 72 + cpu 200), PY = the ff32b71 wrapper, November sets, G9T_STARTED "TIME g9q 0.5 g9 810367"
t14tree; park14; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); G9TP="TRAIN_ARGS=$(G9TA g9q 0.5 8 "$B/run/G9t" "$A16X" "$P3")$"
check 50a "^FORMAT=gen9randombattle\|ANCHOR=$P1\|$G9TP" "$o"; checkno 50a2 "lr-schedule|entropy-min 0.35|/G9r|P1-g9h-dmg-s2.pt --kl" "${o#*TRAIN_ARGS=}"
check 50b "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8} g9q 0.5 g9 810367$" "$(cat "$N/runs/g9t/G9T_STARTED" 2>/dev/null)"
check 50c "^PY=$W9\|PAT=\^/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train\( \|\\\$\)$" "$(pyof 810367 g9 A16)"; check 50c2 "child sees $NOV" "$(psof 810367 g9 A16)"
yes_ 50d '[ "$(cat "$N/runs/g9/MODE")" = g9t ] && [ ! -e "$N/runs/g9/PARKED" ] && [ ! -e "$N/runs/g9/HOLD" ] && [ ! -e "$Z.tmp" ] && [ -d "$B/run/G9t/pool" ] && ls "$N"/runs/g9/INLINE_CMD.done-* > /dev/null 2>&1'
yes_ 50e 'cmp -s "$Z/final_model.pt" "$B/run/G9r/final_model.pt" && [ "$(ls "$Z/pool" | grep -c pt$) $(ls "$Z/pool/frozen" | grep -c pt$) $(ls "$Z/checkpoints" | grep -c pt$) $(cat "$Z/best_model.pt") $(cat "$Z/ema_model.pt")" = "2 2 2 g9r-best g9r-ema" ] && [ -f "$Z/tb/events.x" ] && [ -f "$Z/eval/eval.log" ] && cmp -s "$Z/train.log" "$B/run/G9r/train.log" && [ -f "$Z/state/G9R_STARTED" ] && [ -f "$Z/state/FU.reason" ]'
yes_ 50f '(cd "$Z" && md5sum -c --quiet MD5SUMS) && [ "$(wc -l < "$Z/MD5SUMS")" -ge 14 ] && [ -f "$Z/PARKED.txt" ] && [ "$(cat "$B/run/G9r/final_model.pt")" = g9r-final-60M ] && [ -f "$B/run/G9r/checkpoints/ckpt_00055000000.pt" ]'
check 50g "^G9r PARKED .* in job 810367 .*MODE g9r -> g9t; runs/g9t/RECIPE 'g9q 0.5'.*steps: 60000000 \(final_model.pt saved .*source: $B/run/G9r .*guard state runs/g9r.*counters: \{'steps': 60000000.*pp: $N/src-ff32b71:$N/src .*last update: .*upd 1440 steps 60,000,000.*last save: .*saved final_model.pt at 60,000,000 steps.*\[eval @ 60,000,000\].*FU.started '', FU.reason 'old trip'.*resume: " "$(tr '\n' ' ' < "$Z/PARKED.txt" 2>/dev/null)"
check 50h "park-g9r DONE: G9r parked in $Z at 60000000 steps .*MODE g9r -> g9t" "$(cat "$N/runs/g9t/park.log" 2>/dev/null)"
check 50i "G9t first start in run g9: from P3 $P3 \(steps 0\); C = g9q's recipe \(entropy floor 0.5\) \+ --gae-lambda 0.5 \(runs/g9t/RECIPE\); November sets for the collector \(PORYGON_PS=$NOV\)" "$(cat "$B/run/train.log")"
yes_ 50j '[ -f "$B/run/G9r/GUARD_TRIPPED.fu-010101" ] && [ -f "$Z/GUARD_TRIPPED.fu-010101" ] && ! grep -q "MODE g9t refused" "$B/run/train.log"'
# 50k: "one change": G9t's args = MODE g9q's (G9q's recipe, same job and GPU) + --gae-lambda 0.5 (lineage dir and --resume aside)
cp -a "$N/runs/g9" "$ROOTM.g9save"; echo g9q > "$N/runs/g9/MODE"; oq=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); rm -rf "$N/runs/g9"; mv "$ROOTM.g9save" "$N/runs/g9"
yes_ 50k '[ -n "$(norm "$o")" ] && [ "$(norm "$o")" = "$(norm "$oq")" ] && [ "$(printf "%s" "$o" | grep -o -- "--gae-lambda [0-9.]*")" = "--gae-lambda 0.5" ]'
# 50l: G9t's restart in place (in 810367 and on >= 40 GB): its own save; no second first start, no second park
echo g9t-final > "$B/run/G9t/final_model.pt"; s0=$(cat "$N/runs/g9t/G9T_STARTED")
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); check 50l "TRAIN_ARGS=$(G9TA g9q 0.5 8 "$B/run/G9t" "$A16X" "$B/run/G9t/final_model.pt")$" "$o"
o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=46068 src g9 A16); check 50l2 "TRAIN_ARGS=$(G9TA g9q 0.5 8 "$B/run/G9t" "$A16Y" "$B/run/G9t/final_model.pt")$" "$o"
yes_ 50l3 '[ "$(grep -c "G9t first start" "$B/run/train.log")" = 1 ] && [ "$(grep -c "park-g9r DONE" "$N/runs/g9t/park.log")" = 1 ] && [ "$(cat "$N/runs/g9t/G9T_STARTED")" = "$s0" ]'
# 50m: the other runs are untouched: r1 (1135184 G9q on the A30) and R1b-F (1136569)
o=$(SLURM_JOB_ID=1135184 NV2_GPU_MIB=24576 src r1 A30); check 50m "--run-dir $A/run/G9q .*--snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume $A/run/G9q/final_model.pt$" "$o"; checkno 50m2 "gae-lambda|G9t|G9r" "$o"
# 51 (iv): the park refuses (FAIL in runs/g9t/park.log, MODE untouched, no runs/g9r-parked, no G9T_STARTED; G9r relaunches
# from its own save): RECIPE missing / "g9q 0.6" / "g9r 0.5" / two lines; PREREG missing / writable; G9T_STARTED exists;
# runs/g9r-parked exists; a live trainer; a stale save (older than the queued INLINE_CMD; too old; an update or a
# checkpoint past the last save; counters != the save line); the load fails; MODE g9q; v13 on disk
G9RC="--run-dir $B/run/G9r --pool-dir $B/run/G9r/pool $A16X --lr-schedule power --lr-horizon 300000000 --resume $B/run/G9r/final_model.pt$"
for c in norecipe r06 rg9r r2 noprereg prw started parked live olderq old upd ckpt counters noload mode env13; do
  t14tree; park14; ma=g9r; lp=""; pe=""; why=""
  case $c in
    norecipe) rm -f "$N/runs/g9t/RECIPE"; why="no runs/g9t/RECIPE" ;;
    r06) echo "g9q 0.6" > "$N/runs/g9t/RECIPE"; why="RECIPE is not exactly 'g9q 0.5' \(1 line\(s\), first: 'g9q 0.6'\)" ;;
    rg9r) echo "g9r 0.5" > "$N/runs/g9t/RECIPE"; why="RECIPE is not exactly 'g9q 0.5'" ;;
    r2) printf 'g9q 0.5\ng9q 0.5\n' > "$N/runs/g9t/RECIPE"; why="RECIPE is not exactly 'g9q 0.5' \(2 line" ;;
    noprereg) rm -f "$N/runs/g9t/PREREG.txt"; why="no runs/g9t/PREREG.txt" ;;
    prw) chmod 644 "$N/runs/g9t/PREREG.txt"; why="PREREG.txt is not read-only \(mode 644\)" ;;
    started) echo "2026-11-12T20:00:00 g9q 0.5 g9 810367" > "$N/runs/g9t/G9T_STARTED"; why="G9T_STARTED exists" ;;
    parked) mkdir -p "$Z"; echo earlier > "$Z/PARKED.txt"; why="$Z exists already" ;;
    live) ( exec -a "/x/python -u -m porygon.v2.train --run-dir $B/run/G9r --pool-dir $B/run/G9r/pool" /bin/sleep 120 ) & lp=$!; /bin/sleep 0.3; why="a trainer with --run-dir under $B/run is still running" ;;
    olderq) touch -d '20 minutes ago' "$B/run/G9r/final_model.pt"; why="is not newer than the queued INLINE_CMD .*stale" ;;
    old) touch -d '3 hours ago' "$N/runs/g9/INLINE_CMD"; touch -d '1 hour ago' "$B/run/G9r/final_model.pt"; why="final_model.pt is [0-9]+ s old" ;;
    upd) echo "2026-11-12 23:43:00 upd 1441 steps 60,041,600 (+41600) sps 440" >> "$B/run/G9r/train.log"; why="the last save \(60000000 steps\) is not the newest step \(last update 60041600" ;;
    ckpt) echo c3 > "$B/run/G9r/checkpoints/ckpt_00060041600.pt"; why="is not the newest step \(.*newest checkpoint 00060041600\)" ;;
    counters) pe="PARK_MOCK_STEPS=55000000"; why="counters say steps 55000000, its train.log save line 60000000" ;;
    noload) pe="PARK_MOCK_FAIL=1"; why="does not load with optimizer \+ counters: .*mock load failure" ;;
    mode) ma=g9q; echo g9q > "$N/runs/g9/MODE"; why="runs/g9/MODE is 'g9q' \(expected g9r\)" ;;
    env13) grep -v 'runs/g9r-parked/PARKED.txt' "$ENVF" > "$N/netv2.env"; why="netv2.env has no v14 g9t mode" ;;
  esac
  o=$(if [ -n "$pe" ]; then export "${pe?}"; fi; SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
  if [ -n "$lp" ]; then kill "$lp" 2>/dev/null; wait "$lp" 2>/dev/null; fi
  yes_ "51$c" '[ "$(cat "$N/runs/g9/MODE")" = $ma ] && [ ! -e "$N/runs/g9/PARKED" ] && ls "$N"/runs/g9/INLINE_CMD.done-* > /dev/null 2>&1 && [ "$(cat "$B/run/G9r/final_model.pt")" = g9r-final-60M ] && [ ! -d "$B/run/G9t" ]'
  case $c in
    parked) yes_ "51$c-2" '[ "$(cat "$Z/PARKED.txt")" = earlier ] && [ "$(ls -A "$Z")" = PARKED.txt ] && [ ! -e "$N/runs/g9t/G9T_STARTED" ]' ;;
    started) yes_ "51$c-2" '[ ! -e "$Z" ] && [ "$(cat "$N/runs/g9t/G9T_STARTED")" = "2026-11-12T20:00:00 g9q 0.5 g9 810367" ]' ;;
    *) yes_ "51$c-2" '[ ! -e "$Z" ] && [ ! -e "$N/runs/g9t/G9T_STARTED" ]' ;;
  esac
  check "51$c-3" "park-g9r FAIL \(MODE untouched: $ma\): .*$why" "$(cat "$N/runs/g9t/park.log" 2>/dev/null)"
  checkno "51$c-4" "gae-lambda|/G9t" "$o"
  if [ $ma = g9r ]; then check "51$c-5" "$G9RC" "$o"; fi
done
# 52 (ii): the pending g9 successor 1137377 (afterany:810367; id >= NV2_WD_MIN_JOB) starts fresh after the park and G9t's
# first segment (MODE g9t, G9T_STARTED, runs/g9r-parked/PARKED.txt): not blocked by the park, queues exactly ONE
# successor, resumes G9t's newest MIRRORED file (never the newer G9r mirror or the parked dir); 52b: while a park
# INLINE_CMD is queued (it names runs/g9r-parked) a fresh g9 job exits at once (intended)
wdtree; echo g9t > "$N/runs/g9/MODE"; mkdir -p "$N/runs/g9t"; echo "2026-11-12T23:45:00 g9q 0.5 g9 810367" > "$N/runs/g9t/G9T_STARTED"; rm -f "$N/runs/g9t/RECIPE"
KT=$N/runs/g9/20261112-015035-810367/G9t; mkdir -p "$KT/pool/frozen" "$KT/checkpoints" "$Z/pool" "$Z/checkpoints"
echo g9t-ckpt > "$KT/checkpoints/ckpt_00004000000.pt"; touch -d '2026-11-14 23:00' "$KT/checkpoints/ckpt_00004000000.pt"
echo g9t-mirror > "$KT/final_model.pt"; touch -d '2026-11-15 01:31' "$KT/final_model.pt"; echo b > "$KT/best_model.pt"; echo P3 > "$KT/pool/frozen/P3-g9-s2.pt"; echo s > "$KT/pool/00004000000.pt"
touch -d '2026-11-15 01:40' "$K2/final_model.pt"; echo parked > "$Z/final_model.pt"; echo pk > "$Z/checkpoints/ckpt_00060000000.pt"; echo p > "$Z/PARKED.txt"; touch -d '2026-11-15 01:50' "$Z/final_model.pt" "$Z/checkpoints/ckpt_00060000000.pt"
sq 1137377 ws-netv2-g9 RUNNING 2026-11-15T01:46:00 "(null)"
mkdir -p "$ROOTM/wd-1137377"; touch "$ROOTM/wd-1137377/.toucher-G9t"
o=$(WD_PART=a16 WD_CPUS=10 WD_MEM=65536 WD_GPU=15356 wdsrc g9 1137377 ws-netv2-g9); W1=$ROOTM/wd-1137377
check 52a "^RC=0\|TRAIN_ARGS=$(G9TA g9q 0.5 8 "$W1/run/G9t" "$A16X" "$KT/final_model.pt")$" "$o"; checkno 52a2 "/G9r|g9r-parked|lr-schedule" "$o"
check 52a3 "^sbatch --job-name=ws-netv2-g9 --partition=a16 --gres=gpu:1 --cpus-per-task=10 --mem=65536M --time=72:00:00 --dependency=afterany:1137377 scripts/slurm_train_v2.sbatch g9$" "$(cat "$WDQ.sbatch" 2> /dev/null)"
yes_ 52a4 '[ "$(nsb)" = 1 ] && [ "$(grep -c 1137377 "$N/runs/g9/JOBSTARTS")" = 1 ] && [ "$(cat "$N/runs/g9/chain_left")" = 0 ] && [ ! -e "$N/runs/g9/HOLD" ] && [ "$(cat "$N/runs/g9/MODE")" = g9t ]'
check 52a5 "first start on nodex: .*MODE g9: g9t .*queued successor 1300001 \(afterany:1137377\)" "$(wdlog g9)"; checkno 52a6 "exit without training" "$(wdlog g9)"
yes_ 52a7 '[ -f "$W1/run/G9t/best_model.pt" ] && [ -f "$W1/run/G9t/pool/frozen/P3-g9-s2.pt" ] && [ -L "$W1/run/final_model.pt" ] && grep -q "fresh job: G9t resumes from $KT/final_model.pt" "$W1/run/train.log" && ! grep -q "G9t first start" "$W1/run/train.log"'
o=$(WD_PART=a16 WD_CPUS=10 WD_MEM=65536 WD_GPU=15356 wdsrc g9 1137377 ws-netv2-g9); yes_ 52a8 '[ "$(nsb)" = 1 ] && [ "${o%%|*}" = RC=0 ]'  # its restart in place: no second successor
wdtree; echo g9r > "$N/runs/g9/MODE"; cp "$G9T_KIT/inline-a16-park-g9r-start-g9t.sh" "$N/runs/g9/INLINE_CMD"; o=$(WD_PART=a16 WD_GPU=15356 wdsrc g9 1137377 ws-netv2-g9)
yes_ 52b '[ "$o" = "RC=0|" ] && [ "$(nsb)" = 0 ] && [ -f "$N/runs/g9/INLINE_CMD" ]'; check 52b2 "exit without training or chaining: a park / swap INLINE_CMD is queued \(runs/g9/INLINE_CMD\)" "$(wdlog g9)"
# 53 (iii): MODE g9t refused WITH G9r parked -> HOLD rounds (the mock caps them: NV2_G9T_HOLD_ROUNDS), never G9r:
# (a) RECIPE removed and G9T_STARTED corrupted; (b) the same, fixed by a human during round 1 (an INLINE_CMD that a
# queued one installs: nv2_pause runs it inside the HOLD round) -> G9t continues from its own save; (c) the record says
# r1; (d) outside a job (LOCAL / SLURM_JOB_ID unset): no HOLD possible, no G9r either (the generic branch)
t53() { t14tree; park14; SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16 > /dev/null; echo g9t-final > "$B/run/G9t/final_model.pt"; rm -f "$N/runs/g9t/RECIPE"; echo "corrupt" > "$N/runs/g9t/G9T_STARTED"; : > "$B/run/train.log"; }
t53; o=$(NV2_G9T_HOLD_ROUNDS=2 SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); l=$(cat "$B/run/train.log")
checkno 53a "TRAIN_ARGS=" "$o"; yes_ 53a2 '[ -f "$N/runs/g9/HOLD" ] && [ "$(cat "$N/runs/g9/MODE")" = g9t ] && [ "$(cat "$N/runs/g9t/G9T_STARTED")" = corrupt ] && [ -f "$Z/PARKED.txt" ]'
check 53a3 "MODE g9t refused: runs/g9t/G9T_STARTED unreadable \('corrupt'\): HOLD: G9r is parked \(runs/g9r-parked/PARKED.txt\) and never resumes on a refusal; nothing continues" "$l"
check 53a4 "MODE g9t refused with G9r parked \(round 1\): HOLD .*MODE g9t refused with G9r parked \(round 2\): HOLD .*MODE g9t held 2 rounds \(mock cap\): exit" "$(tr '\n' ' ' <<< "$l")"
checkno 53a5 "G9r continues" "$l"; check 53a6 "^3$" "$(grep -c 'MODE g9t refused: ' <<< "$l")"
t53; printf 'echo "2026-11-12T23:45:00 g9q 0.5 g9 810367" > runs/g9t/G9T_STARTED\nrm -f runs/g9/HOLD\n' > "$ROOTM.fixer"
echo "cp $ROOTM.fixer runs/g9/INLINE_CMD" > "$N/runs/g9/INLINE_CMD"
o=$(NV2_G9T_HOLD_ROUNDS=3 SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); l=$(cat "$B/run/train.log")
check 53b "TRAIN_ARGS=$(G9TA g9q 0.5 8 "$B/run/G9t" "$A16X" "$B/run/G9t/final_model.pt")$" "$o"; checkno 53b2 "G9r continues|round 2" "$l"
yes_ 53b3 '[ ! -e "$N/runs/g9/HOLD" ] && [ "$(grep -c "refused with G9r parked (round 1)" <<< "$l")" = 1 ] && [ "$(cut -d" " -f2- "$N/runs/g9t/G9T_STARTED")" = "g9q 0.5 g9 810367" ]'
t53; echo "2026-11-12T23:45:00 g9q 0.5 r1 1135184" > "$N/runs/g9t/G9T_STARTED"; o=$(NV2_G9T_HOLD_ROUNDS=1 SLURM_JOB_ID=810367 src g9 A16)
checkno 53c "TRAIN_ARGS=" "$o"; check 53c2 "MODE g9t refused: G9t started in run r1 .*: HOLD: G9r is parked" "$(cat "$B/run/train.log")"; yes_ 53c3 '[ -f "$N/runs/g9/HOLD" ]'
t53; o=$( ( set -euo pipefail; NV2=$N; RUN_NAME=g9; unset LOCAL SLURM_JOB_ID; NV2_GPU_MIB=15356; source "$ENVF"; echo "TRAIN_ARGS=$TRAIN_ARGS" ) 2>&1 )
check 53d "TRAIN_ARGS=.*--kl-anchor $P1 " "$o"; checkno 53d2 "/G9r|lr-schedule|gae-lambda|entropy-min 0.35" "$o"; yes_ 53d3 '[ ! -e "$N/runs/g9/HOLD" ]'
# 54 (iii): refused WITHOUT G9r parked (no runs/g9r-parked/PARKED.txt: never parked, a failed park's .tmp, a voided park)
# -> v13: G9r continues from its own save with G9r's exact args, train.log "G9r continues", no HOLD
for v in none tmp void; do
  t14tree; rm -f "$N/runs/g9t/RECIPE"; echo g9t > "$N/runs/g9/MODE"
  case $v in tmp) mkdir -p "$Z.tmp"; echo p > "$Z.tmp/PARKED.txt" ;; void) mkdir -p "$Z.void-1112-235959"; echo p > "$Z.void-1112-235959/PARKED.txt" ;; esac
  o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
  check "54$v" "TRAIN_ARGS=$(G9RA "$B/run/G9r" "$A16X" "$B/run/G9r/final_model.pt")$" "$o"
  check "54$v-2" "MODE g9t refused: no runs/g9t/RECIPE: G9r continues" "$(cat "$B/run/train.log")"; yes_ "54$v-3" '[ ! -e "$N/runs/g9/HOLD" ] && [ ! -e "$N/runs/g9t/G9T_STARTED" ]'
done
# 55: G9r parked and the recipe valid: no HOLD, no refusal note (the guard is silent when G9t may start)
t14tree; park14; SLURM_JOB_ID=810367 src g9 A16 > /dev/null; echo g9t-final > "$B/run/G9t/final_model.pt"; o=$(SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
yes_ 55 '[ ! -e "$N/runs/g9/HOLD" ] && ! grep -q "refused" "$B/run/train.log"'; check 55b "--run-dir $B/run/G9t .*--gae-lambda 0.5 --resume $B/run/G9t/final_model.pt$" "$o"
# 56 (2026-11-14, G9t chain; orchestrator: a rule STOP parks G9t persistently): scripts/netv2_g9t/park-g9t.sh as the
# queued INLINE_CMD after G9t's SIGINT save: G9t copied whole to runs/g9t-parked (MD5SUMS, PARKED.txt), G9T_STARTED run
# field -> parked (original kept), runs/g9/PARKED; the same sourcing HOLDs round after round (never G9t, never G9r);
# a refusal (stale save) changes nothing and G9t relaunches from its own save.
t56() { t14tree; park14; SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16 > /dev/null; GT=$B/run/G9t; mkdir -p "$GT/pool/frozen" "$GT/checkpoints"
  echo g9t-final-55M > "$GT/final_model.pt"; echo g9t-best > "$GT/best_model.pt"; echo s > "$GT/pool/00055000000.pt"; echo P3 > "$GT/pool/frozen/P3-g9-s2.pt"; echo c > "$GT/checkpoints/ckpt_00055000000.pt"
  printf '2026-11-14 09:00:00 v2 trainer: gen9randombattle ...\n2026-11-14 15:20:00 upd 1340 steps 55,700,000 (+41600) sps 520\n2026-11-14 15:21:00 saved final_model.pt at 55,700,000 steps\n' > "$GT/train.log"
  printf 'set -u\nC=%s\nPARK_RULE=1 PARK_QUEUED_FILE="$C/runs/g9/INLINE_CMD.running" bash %s/park-g9t.sh\n' "$N" "$G9T_KIT" > "$N/runs/g9/INLINE_CMD"
  touch -d '10 minutes ago' "$N/runs/g9/INLINE_CMD"; touch "$GT/final_model.pt"; : > "$B/run/train.log"; }
t56; o=$(PARK_MOCK_STEPS=55700000 NV2_G9T_HOLD_ROUNDS=2 SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16); l=$(cat "$B/run/train.log"); Y=$N/runs/g9t-parked
checkno 56a "TRAIN_ARGS=" "$o"; checkno 56a2 "G9r continues" "$l"
yes_ 56b '[ -f "$Y/PARKED.txt" ] && (cd "$Y" && md5sum -c --quiet MD5SUMS) && cmp -s "$Y/final_model.pt" "$B/run/G9t/final_model.pt" && [ -f "$Y/state/G9T_STARTED" ] && [ -f "$N/runs/g9/PARKED" ] && [ -f "$N/runs/g9/HOLD" ]'
yes_ 56c '[ "$(cut -d" " -f2-4 "$N/runs/g9t/G9T_STARTED")" = "g9q 0.5 parked" ] && [ "$(cut -d" " -f2-4 "$N/runs/g9t/G9T_STARTED.orig-parked")" = "g9q 0.5 g9" ] && [ "$(cat "$N/runs/g9/MODE")" = g9t ]'
check 56d "MODE g9t refused: G9t started in run parked .*not g9: HOLD: G9r is parked" "$l"; check 56d2 "MODE g9t held 2 rounds \(mock cap\): exit" "$l"
check 56e "G9t PARKED .*STOPPED by runs/g9t/PREREG.txt rule 1.*steps: 55700000" "$(tr '\n' ' ' < "$Y/PARKED.txt")"
check 56f "park-g9t DONE: G9t parked in $Y at 55700000 steps" "$(cat "$N/runs/g9t/park.log")"
t56; touch -d '20 minutes ago' "$B/run/G9t/final_model.pt"; o=$(PARK_MOCK_STEPS=55700000 SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)
check 56g "--run-dir $B/run/G9t .*--gae-lambda 0.5 --resume $B/run/G9t/final_model.pt$" "$o"
yes_ 56g2 '[ ! -e "$N/runs/g9t-parked" ] && [ ! -e "$N/runs/g9/PARKED" ] && [ ! -e "$N/runs/g9/HOLD" ] && [ "$(cut -d" " -f4 "$N/runs/g9t/G9T_STARTED")" = g9 ]'
check 56g3 "park-g9t FAIL .*not newer than the queued INLINE" "$(cat "$N/runs/g9t/park.log")"
t56; o=$(PARK_MOCK_STEPS=55600000 SLURM_JOB_ID=810367 NV2_GPU_MIB=15356 src g9 A16)  # counters disagree with the save line
check 56h "--resume $B/run/G9t/final_model.pt$" "$o"; yes_ 56h2 '[ ! -e "$N/runs/g9t-parked" ] && [ ! -e "$N/runs/g9/PARKED" ]'; check 56h3 "park-g9t FAIL .*counters say steps 55600000" "$(cat "$N/runs/g9t/park.log")"
chmod -R u+w "$N/runs/g9t" 2>/dev/null; rm -f "$ROOTM.parkpy14" "$ROOTM.fixer"; export NV2_PARK_PY=$PK0
fi
# 57 (v15, 2026-11-15, USER decision ~14:05; runs/g9q2/PREREG.txt): g9 MODE g9q2 = G9q2 = G9q's recipe (the MODE g9q
# branch) with only the seed changed (--seed 7272027 on the FIRST start only), lineage G9q2, guard state runs/g9q2,
# PY = the 24c633f wrapper (missing -> HOLD), cache 72 + cpu 200 on the A16 (512 + stacked-opp on >= 40 GB), November sets
if [ -z "${MOCK_SKIP_G9Q2:-}" ]; then
q2tree() {  # the 2026-11-15 state: wdtree + G9t parked (runs/g9/PARKED removed at deploy), g9 MODE g9q2, a16 job
  wdtree; echo g9q2 > "$N/runs/g9/MODE"; mkdir -p "$N/runs/g9t-parked" "$N/runs/g9t"; echo "parked" > "$N/runs/g9t-parked/PARKED.txt"
  echo "2026-11-13T09:11:25 g9q 0.5 parked 810367" > "$N/runs/g9t/G9T_STARTED"; echo "g9q 0.5" > "$N/runs/g9t/RECIPE"
}
q2src() {  # q2src JOB [first]: a source in job JOB (LOCAL wd-JOB; "first" = the job's first source, watchdog included) -> PY|PS|TRAIN_ARGS
  local job=$1
  mkdir -p "$ROOTM/wd-$job/run" "$ROOTM/wd-$job/pool"; touch "$ROOTM/wd-$job/run/train.log" "$ROOTM/wd-$job/.toucher-G9q2" "$ROOTM/wd-$job/.toucher-G9q"
  ( set -euo pipefail; PATH=$WDB:$PATH; NV2=$N; LOCAL=$ROOTM/wd-$job; RUN_NAME=g9; SLURM_JOB_ID=$job; SLURM_JOB_NAME=ws-netv2-g9
    SLURM_JOB_PARTITION=a16; SLURM_CPUS_ON_NODE=10; SLURM_MEM_PER_NODE=65536; NV2_HOLD_ITERS=0; NV2_SWAP_ITERS=0
    NV2_TMPWATCH_POLL_S=0; NV2_WD_RETRY_S=0; NV2_GPU_MIB=${WD_GPU:-15356}
    source "$ENVF"; echo "PY=${PY:-unset}|PS=${PORYGON_PS:-unset}|child=$(bash -c 'echo ${PORYGON_PS:-nothing}')|TRAIN_ARGS=$TRAIN_ARGS" ) > "$ROOTM.wdout" 2>&1
  echo "RC=$?|$(grep '^PY=' "$ROOTM.wdout" || true)"
}
QBASE="--no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 $G9QARGS --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16"
# 57a: first start in a fresh a16 job: G9q's A16 branch args + --seed, from P3; G9Q2_STARTED; one successor queued
q2tree; W=$ROOTM/wd-1300100; o=$(q2src 1300100)
check 57a "^RC=0\|PY=$N/bin/py-src-24c633f\|PS=$NOV\|child=$NOV\|TRAIN_ARGS=$QBASE --run-dir $W/run/G9q2 --pool-dir $W/run/G9q2/pool --snapshot-cache 72 --snapshot-cache-cpu 200 --seed 7272027 --resume $P3$" "$o"
checkno 57a2 "stacked-opp|lr-schedule|gae-lambda|actor-stream|/G9q/|/G9t|/G9r" "$o"
check 57a3 "^20[0-9T:-]+ seed 7272027 g9 1300100$" "$(cat "$N/runs/g9q2/G9Q2_STARTED" 2>/dev/null)"
check 57a4 "G9q2 first start: from P3 $P3 \(steps 0\); G9q's recipe \+ --seed 7272027 \(this start only\); November sets for the collector \(PORYGON_PS=$NOV\)" "$(cat "$W/run/train.log")"
yes_ 57a5 '[ "$(nsb)" = 1 ] && grep -q -- "--partition=a16 .*--cpus-per-task=10 --mem=65536M --time=72:00:00 --dependency=afterany:1300100" "$WDQ.sbatch" && [ "$(cat "$N/runs/g9/MODE")" = g9q2 ] && [ ! -e "$N/runs/g9/HOLD" ]'
check 57a6 "first start on nodex: .*MODE g9: g9q2 .*queued successor 1300001 \(afterany:1300100\)" "$(wdlog g9)"
# 57a7: the same first start under MODE g9q (G9q's branch, a successor job) gives the same args up to the lineage name
# and the seed: the argsdiff the user asked for ("only the seed + run dir")
x=$(sed -E 's#/G9q2([/ ])#/G9q\1#g; s# --seed 7272027##' <<< "${o#*TRAIN_ARGS=}")
q2tree; echo g9q > "$N/runs/g9/MODE"; rm -f "$N/runs/g9q/G9Q_STARTED"; y=$(q2src 1300100)
yes_ 57a7 '[ "$x" = "${y#*TRAIN_ARGS=}" ] && [ "${o%%|TRAIN_ARGS=*}" = "${y%%|TRAIN_ARGS=*}" ]'
# 57b: a restart in place of the same job after G9q2 saved: its own final, no --seed, no second first start
q2tree; W=$ROOTM/wd-1300100; q2src 1300100 > /dev/null; echo g9q2-final > "$W/run/G9q2/final_model.pt"; s0=$(cat "$N/runs/g9q2/G9Q2_STARTED")
o=$(q2src 1300100); check 57b "^RC=0\|PY=$N/bin/py-src-24c633f\|PS=$NOV\|.*TRAIN_ARGS=$QBASE --run-dir $W/run/G9q2 --pool-dir $W/run/G9q2/pool --snapshot-cache 72 --snapshot-cache-cpu 200 --resume $W/run/G9q2/final_model.pt$" "$o"
checkno 57b2 "--seed" "$o"; yes_ 57b3 '[ "$(cat "$N/runs/g9q2/G9Q2_STARTED")" = "$s0" ] && [ "$(grep -c "G9q2 first start" "$W/run/train.log")" = 1 ] && [ "$(nsb)" = 1 ]'
# 57c: the fresh successor resumes G9q2's newest MIRRORED file (never G9q's, G9t's or G9r's), no --seed
K3=$N/runs/g9/20261115-143000-1300100/G9q2; mkdir -p "$K3/pool/frozen" "$K3/checkpoints"; echo m > "$K3/final_model.pt"; echo b > "$K3/best_model.pt"; echo s > "$K3/pool/00001000000.pt"
touch -d '2026-11-17 10:00' "$K3/final_model.pt"; K4=$N/runs/g9/20261109-021316-741849/G9q; mkdir -p "$K4"; echo q > "$K4/final_model.pt"; touch -d '2026-11-17 11:00' "$K4/final_model.pt"
W2=$ROOTM/wd-1300001; o=$(q2src 1300001); check 57c "^RC=0\|.*TRAIN_ARGS=$QBASE --run-dir $W2/run/G9q2 --pool-dir $W2/run/G9q2/pool --snapshot-cache 72 --snapshot-cache-cpu 200 --resume $K3/final_model.pt$" "$o"
checkno 57c2 "--seed" "$o"; yes_ 57c3 '[ -f "$W2/run/G9q2/best_model.pt" ] && [ "$(cat "$N/runs/g9q2/G9Q2_STARTED")" = "$s0" ]'
# 57c4: a successor with a checkpoint newer than the mirrored final (a crash): the newest file of G9q2 either way
rm -rf "$ROOTM/wd-1300002"; echo c > "$K3/checkpoints/ckpt_00031000000.pt"; touch -d '2026-11-17 12:00' "$K3/checkpoints/ckpt_00031000000.pt"
o=$(q2src 1300002); check 57c4 "--resume $K3/checkpoints/ckpt_00031000000.pt$" "$o"
# 57d: no wrapper -> runs/g9/HOLD + note (the job holds; the mock passes the hold at once)
q2tree; mv "$N/bin/py-src-24c633f" "$N/bin/off"; W=$ROOTM/wd-1300100; o=$(q2src 1300100); mv "$N/bin/off" "$N/bin/py-src-24c633f"
yes_ 57d '[ -f "$N/runs/g9/HOLD" ]'; check 57d2 "MODE g9q2 but no $N/bin/py-src-24c633f \(G9q2's src\): HOLD" "$(cat "$W/run/train.log")"
# 57e: a >= 40 GB GPU: 512 + --stacked-opp (G9q's A16-branch rule; the a16 job never gets one)
q2tree; W=$ROOTM/wd-1300100; o=$(WD_GPU=46068 q2src 1300100); check 57e "--pool-dir $W/run/G9q2/pool --snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --seed 7272027 --resume $P3$" "$o"
# 57e2 (orchestrator 11-15 16:4x: G9q's A16 TRAIN_ARGS on whatever GPU, at least until 50M; only speed-only,
# logits-equivalent options may differ): on an A40 (46068) / A100 (40960, 81920) the first-start args equal the A16's
# with "--snapshot-cache 72" replaced by "--snapshot-cache 512 --stacked-opp" and nothing else; 8 workers everywhere
q2tree; W=$ROOTM/wd-1300100; a16=$(WD_GPU=15356 q2src 1300100)
for g in 46068 40960 81920; do q2tree; o=$(WD_GPU=$g q2src 1300100)
  yes_ "57e2-$g" '[ "${o#*TRAIN_ARGS=}" = "$(sed "s# --snapshot-cache 72 # --snapshot-cache 512 --stacked-opp #" <<< "${a16#*TRAIN_ARGS=}")" ] && [ "${o%%|TRAIN_ARGS=*}" = "${a16%%|TRAIN_ARGS=*}" ]'
  check "57e3-$g" "--fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 " "$o"; done
# 57f: guard trips: collapse -> KL x3 from the trip state (state runs/g9q2 only); then a second trip -> HOLD
q2tree; W=$ROOTM/wd-1300100; q2src 1300100 > /dev/null; echo g9q2-final > "$W/run/G9q2/final_model.pt"; echo "collapse: entropy 0.4 < 0.5" > "$W/run/G9q2/GUARD_TRIPPED"
o=$(q2src 1300100); check 57f "--kl-anchor-coef 0.3 .*--run-dir $W/run/G9q2 .*--resume $W/run/G9q2/final_model.pt$" "$o"
yes_ 57f2 'grep -q "^1 " "$N/runs/g9q2/FU.started" && grep -q collapse "$N/runs/g9q2/FU.reason" && [ ! -e "$N/runs/g9q/FU.started" ] && ls "$W"/run/G9q2/GUARD_TRIPPED.fu-* > /dev/null 2>&1'
echo "eval score 40% < best 60% - 15%" > "$W/run/G9q2/GUARD_TRIPPED"; o=$(q2src 1300100); yes_ 57f3 '[ -f "$N/runs/g9/HOLD" ] && ls "$W"/run/G9q2/GUARD_TRIPPED.second-* > /dev/null 2>&1'
# 57g: a non-collapse first trip -> best_model.pt at --lr 3.5e-5
q2tree; W=$ROOTM/wd-1300100; q2src 1300100 > /dev/null; echo g9q2-final > "$W/run/G9q2/final_model.pt"; echo g9q2-best > "$W/run/G9q2/best_model.pt"; echo "eval score 40% < best 60% - 15%" > "$W/run/G9q2/GUARD_TRIPPED"
o=$(q2src 1300100); check 57g "--lr 3.5e-5 .*--kl-anchor-coef 0.1 .*--resume $N/runs/g9q2/fu-start-best.pt$" "$o"; yes_ 57g2 '[ "$(cat "$N/runs/g9q2/fu-start-best.pt")" = g9q2-best ]'
# 57h: deploy blockers: runs/g9/HOLD or runs/g9/PARKED (G9t's park marker) -> a fresh job exits without training
for f in HOLD PARKED; do q2tree; touch "$N/runs/g9/$f"; o=$(q2src 1300100); check "57h-$f" "^RC=0\|$" "$o"; yes_ "57h2-$f" '[ ! -e "$N/runs/g9q2/G9Q2_STARTED" ] && [ "$(nsb)" = 0 ]'; done
# 57i: the other runs never see g9q2: r1 (G9q on the A30) and g4r1bf restarts unchanged by MODE g9 = g9q2 (argsdiff too)
fi
# 58 (2026-11-15 20:34, orchestrator): R1b-F's stale follow-up #1 (at 360M it would resume runs/r1/restart-best-7.5M.pt
# and park the pool > 7.5M) disarmed by runs/r1/FU.started = "0 <time> orchestrator: ..." (first field 0). A restart in
# place of the live job (1137378, wrapper, watchdog already run): an anchor-drop trip -> runs/g4r1bf/HOLD (nv2_second_trip),
# the trip file renamed .second-*, no FOLLOWUP1 / FU.reason, no 7.5M resume, the pool kept, TRAIN_ARGS identical to the
# no-trip restart; after rm HOLD the same config continues; G9q (r1 MODE g9q) never reads runs/r1/FU.started.
g4tree; L=$ROOTM/local-T4; mkdir -p "$N/bin"; printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-24c633f"; chmod +x "$N/bin/py-src-24c633f"
src g4r1bf T4 > /dev/null; echo r1bf-final > "$L/run/R1bF/final_model.pt"; touch "$L/.nv2-watchdog"
base=$(SLURM_JOB_ID=1137378 src g4r1bf T4)
check 58a "--snapshot-cache 40 --stacked-opp --snapshot-cache-cpu 200 --guard-pg-mean 20 --guard-anchor-drop 0.0999 --resume $L/run/R1bF/final_model.pt$" "$base"
echo "0 2026-11-15T20:34:00 orchestrator: follow-up #1 disarmed (stale at 360M)" > "$N/runs/r1/FU.started"
o=$(SLURM_JOB_ID=1137378 src g4r1bf T4); yes_ 58b '[ "$o" = "$base" ] && [ ! -e "$N/runs/g4r1bf/HOLD" ]'
echo "eval score 74.0% < best 84.5% - 10% for 2 evals" > "$L/run/R1bF/GUARD_TRIPPED"
o=$(SLURM_JOB_ID=1137378 src g4r1bf T4); yes_ 58c '[ "$o" = "$base" ]'
yes_ 58d '[ -f "$N/runs/g4r1bf/HOLD" ] && [ ! -e "$L/run/R1bF/GUARD_TRIPPED" ] && ls "$L"/run/R1bF/GUARD_TRIPPED.second-* > /dev/null 2>&1'
yes_ 58e '[ ! -e "$N/runs/r1/FOLLOWUP1" ] && [ ! -e "$N/runs/g4r1bf/FU.reason" ] && [ -f "$L/run/R1bF/pool/00020009055.pt" ] && [ ! -d "$N/runs/r1/pool-before-FU" ] && grep -q "^0 2026-11-15T20:34:00 orchestrator" "$N/runs/r1/FU.started"'
checkno 58f "ent-coef|restart-best-7.5M|zero-avoid" "$o"; check 58g "second guard trip of R1bF after its fallback: HOLD" "$(cat "$L/run/train.log")"
rm -f "$N/runs/g4r1bf/HOLD"; o=$(SLURM_JOB_ID=1137378 src g4r1bf T4); yes_ 58h '[ "$o" = "$base" ] && [ ! -e "$N/runs/g4r1bf/HOLD" ]'
wdtree; o1=$(wdsrc r1 1300500 ws-netv2-r1); wdtree; echo "0 2026-11-15T20:34:00 orchestrator" > "$N/runs/r1/FU.started"; o2=$(wdsrc r1 1300500 ws-netv2-r1)
yes_ 58i '[ "$o1" = "$o2" ] && [ "${o1#RC=0|TRAIN_ARGS=}" != "$o1" ]'; check 58j "--lr 7e-5 .*--kl-anchor-coef 0.1 " "$o2"; checkno 58k "ent-coef|3.5e-5|kl-anchor-coef 0.3" "$o2"
rm -f "$N/runs/r1/FU.started"
if [ -z "${MOCK_SKIP_V16:-}" ]; then
# 59 (v16, 2026-11-15; runs/g4r1bf/HOLD-PREREG.txt): the R1b-F ROLLBACK hook. runs/g4r1bf/ROLLBACK "PATH STEP MD5" valid
# -> the first relaunch resumes PATH at --lr 3.5e-5 and parks snapshots > STEP (local + this job's mirror) once; later
# restarts keep --lr 3.5e-5 from R1b-F's own final; a bad file -> HOLD, the same config; r1 / g9 never see it.
rbtree() {
  g4tree; L=$ROOTM/local-T4; mkdir -p "$N/bin"; printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-24c633f"; chmod +x "$N/bin/py-src-24c633f"
  src g4r1bf T4 > /dev/null; echo r1bf-final > "$L/run/R1bF/final_model.pt"; touch "$L/.nv2-watchdog"
  echo "0 2026-11-15T20:34:58 orchestrator" > "$N/runs/r1/FU.started"
  MP=$N/runs/g4r1bf/20261115-175703-1137378/R1bF/pool; mkdir -p "$MP" "$N/runs/g4r1bf-hold1"
  for x in 00350016623 00360021860 00365000000; do echo "s$x" > "$L/run/R1bF/pool/$x.pt"; echo "s$x" > "$MP/$x.pt"; done
  RB=$N/runs/g4r1bf-hold1/rollback-00355016707.pt; echo r355 > "$RB"
}
rbtree; base=$(SLURM_JOB_ID=1137378 src g4r1bf T4)
check 59a "--lr 7e-5 .*--snapshot-cache 40 --stacked-opp --snapshot-cache-cpu 200 --guard-pg-mean 20 --guard-anchor-drop 0.0999 --resume $L/run/R1bF/final_model.pt$" "$base"
echo "$RB 355016707 $(md5sum < "$RB" | cut -c1-8)" > "$N/runs/g4r1bf/ROLLBACK"
o=$(SLURM_JOB_ID=1137378 src g4r1bf T4)
yes_ 59b '[ "$o" = "$(sed -e "s# --lr 7e-5 # --lr 3.5e-5 #" -e "s#--guard-anchor-drop 0.0999 --resume $L/run/R1bF/final_model.pt\$#--guard-anchor-drop 0.0999 --resume $RB#" <<< "$base")" ]'
checkno 59c "--lr 7e-5|R1bF/final_model.pt" "$o"
yes_ 59d 'grep -q " $RB 355016707 " "$N/runs/g4r1bf/ROLLBACK.started" && [ ! -e "$N/runs/g4r1bf/HOLD" ]'
yes_ 59e '[ -f "$L/run/R1bF/pool/00350016623.pt" ] && [ -f "$MP/00350016623.pt" ] && [ -f "$L/run/R1bF/pool/00020009055.pt" ] && [ ! -e "$L/run/R1bF/pool/00360021860.pt" ] && [ ! -e "$MP/00365000000.pt" ] && [ -f "$N/runs/g4r1bf/pool-after-rollback/00360021860.pt" ] && [ -f "$N/runs/g4r1bf/pool-after-rollback/00365000000.pt" ]'
check 59f "R1b-F ROLLBACK \(runs/g4r1bf/ROLLBACK\): resume $RB \(steps 355016707\) at --lr 3.5e-5" "$(cat "$L/run/train.log")"
echo s370 > "$L/run/R1bF/pool/00370000000.pt"; o=$(SLURM_JOB_ID=1137378 src g4r1bf T4)
yes_ 59g '[ "$o" = "$(sed "s# --lr 7e-5 # --lr 3.5e-5 #" <<< "$base")" ] && [ -f "$L/run/R1bF/pool/00370000000.pt" ]'
for bad in "md5 $RB 355016707 deadbeef" "step $RB 35501x707 $(md5sum < "$RB" | cut -c1-8)" "path $N/nope.pt 355016707 deadbeef"; do
  rbtree; read -r w x <<< "$bad"; echo "$x" > "$N/runs/g4r1bf/ROLLBACK"; o=$(SLURM_JOB_ID=1137378 src g4r1bf T4)
  yes_ "59h-$w" '[ "$o" = "$base" ] && [ -f "$N/runs/g4r1bf/HOLD" ] && [ ! -e "$N/runs/g4r1bf/ROLLBACK.started" ] && [ -f "$L/run/R1bF/pool/00365000000.pt" ]'
  check "59i-$w" "R1b-F ROLLBACK refused" "$(cat "$L/run/train.log")"; done
wdtree; o1=$(wdsrc r1 1300500 ws-netv2-r1); q1=$(wdsrc g9 1300501 ws-netv2-g9); wdtree; mkdir -p "$N/runs/g4r1bf"; echo "/x 355016707 deadbeef" > "$N/runs/g4r1bf/ROLLBACK"; touch "$N/runs/g4r1bf/ROLLBACK.started"
o2=$(wdsrc r1 1300500 ws-netv2-r1); q2=$(wdsrc g9 1300501 ws-netv2-g9); yes_ 59j '[ "$o1" = "$o2" ] && [ "$q1" = "$q2" ] && [ "${o1#RC=0|TRAIN_ARGS=}" != "$o1" ]'
rm -f "$N/runs/g4r1bf/ROLLBACK" "$N/runs/g4r1bf/ROLLBACK.started" "$N/runs/r1/FU.started"
fi
if [ -z "${MOCK_SKIP_V17:-}" ] && [ -z "${MOCK_SKIP_G9Q2:-}" ]; then
# 60 (v17, 2026-11-15; runs/g9qbf/PREREG.txt): the bf16 A/B on the A40 (g9). Park guards (MODE g9q2 + runs/g9q2-parked/
# PARKED.txt, MODE g9qbf + runs/g9qbf-parked/PARKED.txt => HOLD rounds, MODE re-read); the bench INLINE -> MODE g9qbf ->
# the fork's first start = G9q2's restart args with ONLY --no-bf16 dropped (dirs G9qbf, resume = the fork point); its
# restarts / a fresh job; refusals (bad FORK_POINT, no staged pool) => HOLD rounds; a guard trip => TRIPPED + HOLD, same args.
qbtree() {  # q2tree + G9q2 started and saved on an A40 job 1300200 (restart args = base2)
  q2tree; W=$ROOTM/wd-1300200; WD_GPU=46068 q2src 1300200 > /dev/null; echo g9q2-final > "$W/run/G9q2/final_model.pt"
  FP=$N/runs/g9qbf/fork-point-00830040000.pt; mkdir -p "$N/runs/g9qbf"; echo fp830 > "$FP"
}
fpok() { echo "$FP 830040000 $(md5sum < "$FP" | cut -c1-8)" > "$N/runs/g9qbf/FORK_POINT"; mkdir -p "$W/run/G9qbf/pool"; touch "$W/run/G9qbf/.staged-830040000"; }
qbtree; base2=$(WD_GPU=46068 q2src 1300200); check 60a "^RC=0\|PY=$N/bin/py-src-24c633f\|.*TRAIN_ARGS=--no-bf16 .*--run-dir $W/run/G9q2 .*--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume $W/run/G9q2/final_model.pt$" "$base2"; checkno 60a2 "--seed" "$base2"
mkdir -p "$N/runs/g9q2-parked"; echo parked > "$N/runs/g9q2-parked/PARKED.txt"
o=$(NV2_PARK_HOLD_ROUNDS=2 WD_GPU=46068 q2src 1300200); check 60b "^RC=3\|$" "$o"; yes_ 60b2 '[ -f "$N/runs/g9/HOLD" ] && [ "$(grep -c "MODE g9q2 is parked" "$W/run/train.log")" = 2 ]'
# the bench INLINE flow: an INLINE_CMD (run by nv2_pause before the MODE read) parks, stages and writes MODE g9qbf
qbtree; fpok_cmd="echo parked > $N/runs/g9q2-parked/PARKED.txt; echo g9qbf > $N/runs/g9/MODE"; mkdir -p "$N/runs/g9q2-parked"; fpok
echo "$fpok_cmd" > "$N/runs/g9/INLINE_CMD"; o=$(WD_GPU=46068 q2src 1300200)
exp="${base2/--no-bf16 /}"; exp="${exp//$W\/run\/G9q2/$W/run/G9qbf}"; exp="${exp%--resume *}--resume $FP"
yes_ 60c '[ "$o" = "$exp" ]'; yes_ 60c2 'grep -q "^20.* 830040000 g9 " "$N/runs/g9qbf/G9QBF_STARTED" && [ ! -e "$N/runs/g9/HOLD" ] && [ "$(cat "$N/runs/g9/MODE")" = g9qbf ] && ls "$N"/runs/g9/INLINE_CMD.done-* > /dev/null 2>&1'
check 60c3 "G9qbf first start: the fork of G9q at 830040000" "$(cat "$W/run/train.log")"; checkno 60c4 "no-bf16|--seed|G9q2" "${o#*TRAIN_ARGS=}"
echo qbf-final > "$W/run/G9qbf/final_model.pt"; o=$(WD_GPU=46068 q2src 1300200); yes_ 60d '[ "$o" = "${exp%--resume *}--resume $W/run/G9qbf/final_model.pt" ]'
# a guard trip: no fallback (TRIPPED + HOLD), the same args
echo "collapse: entropy 0.4 < 0.5" > "$W/run/G9qbf/GUARD_TRIPPED"; o2=$(WD_GPU=46068 q2src 1300200)
yes_ 60e '[ "$o2" = "$o" ] && [ -f "$N/runs/g9/HOLD" ] && grep -q collapse "$N/runs/g9qbf/TRIPPED" && ls "$W"/run/G9qbf/GUARD_TRIPPED.ab-* > /dev/null 2>&1 && [ ! -e "$N/runs/g9qbf/FU.started" ] && [ ! -e "$N/runs/g9q2/FU.started" ]'
checkno 60e2 "kl-anchor-coef 0.3|lr 3.5e-5" "$o2"
# the fork parked => HOLD rounds
rm -f "$N/runs/g9/HOLD"; mkdir -p "$N/runs/g9qbf-parked"; echo parked > "$N/runs/g9qbf-parked/PARKED.txt"
o=$(NV2_PARK_HOLD_ROUNDS=1 WD_GPU=46068 q2src 1300200); check 60f "^RC=3\|$" "$o"; check 60f2 "MODE g9qbf is parked" "$(cat "$W/run/train.log")"
# refusals at the first start: bad md5, no staged pool => HOLD rounds, no G9QBF_STARTED
for bad in md5 staged; do
  qbtree; echo g9qbf > "$N/runs/g9/MODE"; fpok
  if [ $bad = md5 ]; then echo "$FP 830040000 deadbeef" > "$N/runs/g9qbf/FORK_POINT"; else rm -f "$W/run/G9qbf/.staged-830040000"; fi
  o=$(NV2_PARK_HOLD_ROUNDS=1 WD_GPU=46068 q2src 1300200); check "60g-$bad" "^RC=3\|$" "$o"
  yes_ "60g2-$bad" '[ ! -e "$N/runs/g9qbf/G9QBF_STARTED" ] && [ -f "$N/runs/g9/HOLD" ] && grep -q "MODE g9qbf: no valid first start" "$W/run/train.log"'; done
# a fresh A40 job (the successor) under MODE g9qbf after the first start: the newest mirrored G9qbf file
qbtree; echo g9qbf > "$N/runs/g9/MODE"; echo "2026-11-16T03:00:00 830040000 g9 1300200" > "$N/runs/g9qbf/G9QBF_STARTED"
MQ=$N/runs/g9/20261116-030000-1300200/G9qbf; mkdir -p "$MQ/pool"; echo mq > "$MQ/final_model.pt"; W2=$ROOTM/wd-1300210
o=$(WD_GPU=46068 q2src 1300210); check 60h "^RC=0\|PY=$N/bin/py-src-24c633f\|.*TRAIN_ARGS=--n-epochs 3 .*--run-dir $W2/run/G9qbf .*--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume $MQ/final_model.pt$" "$o"
checkno 60h2 "no-bf16|--seed" "$o"
rm -rf "$N/runs/g9qbf" "$N/runs/g9qbf-parked" "$N/runs/g9q2-parked"; echo g9q2 > "$N/runs/g9/MODE"; rm -f "$N/runs/g9/HOLD"
fi
if [ -z "${MOCK_SKIP_V18:-}" ] && [ -z "${MOCK_SKIP_V17:-}" ] && [ -z "${MOCK_SKIP_G9Q2:-}" ]; then
# 61 (v18, 2026-11-15; runs/g9qent/PREREG.txt): the entropy-schedule arm G9qent on the A40. From the parked-G9q2 HOLD
# (after SKIP / the bf16 park) an INLINE_CMD stages it and writes MODE g9qent => first start = G9q2's restart args
# (fp32 kept) with the G9qent dirs, --guard-entropy-min 0.35, + --ent-schedule linear --ent-final 0.001 --ent-start S0
# --ent-horizon 50000000, resume = the fork point, PY = py-src-8892762; restarts / a fresh job; trip => TRIPPED + HOLD.
qetree() { qbtree; printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-8892762"; chmod +x "$N/bin/py-src-8892762"
  EP=$N/runs/g9qent/fork-point-00880050000.pt; mkdir -p "$N/runs/g9qent" "$N/runs/g9q2-parked"; echo fp880 > "$EP"; echo parked > "$N/runs/g9q2-parked/PARKED.txt"; }
epok() { echo "$EP 880050000 $(md5sum < "$EP" | cut -c1-8)" > "$N/runs/g9qent/FORK_POINT"; mkdir -p "$W/run/G9qent/pool"; touch "$W/run/G9qent/.staged-880050000"; }
qetree; epok; echo "echo g9qent > $N/runs/g9/MODE" > "$N/runs/g9/INLINE_CMD"; o=$(WD_GPU=46068 q2src 1300200)
ex="${base2//$W\/run\/G9q2/$W/run/G9qent}"; ex="${ex/ --guard-entropy-min 0.5 / --guard-entropy-min 0.35 }"
ex="${ex/--snapshot-cache-cpu 200 --resume*/--snapshot-cache-cpu 200 --ent-schedule linear --ent-final 0.001 --ent-start 880050000 --ent-horizon 50000000 --resume $EP}"
ex="${ex/PY=$N\/bin\/py-src-24c633f/PY=$N/bin/py-src-8892762}"
yes_ 61a '[ "$o" = "$ex" ]'; yes_ 61a2 'grep -q " 880050000 g9 1300200" "$N/runs/g9qent/G9QENT_STARTED" && [ "$(cat "$N/runs/g9/MODE")" = g9qent ] && [ ! -e "$N/runs/g9/HOLD" ]'
check 61a3 "TRAIN_ARGS=--no-bf16 " "$o"; checkno 61a4 "--seed|G9q2|G9qbf|guard-entropy-min 0.5" "${o#*TRAIN_ARGS=}"; check 61a5 "G9qent first start: the fork of G9q at 880050000" "$(cat "$W/run/train.log")"
echo qe-final > "$W/run/G9qent/final_model.pt"; o=$(WD_GPU=46068 q2src 1300200); yes_ 61b '[ "$o" = "${ex%--resume *}--resume $W/run/G9qent/final_model.pt" ]'
echo "eval score 40% < best 82.5% - 15% for 2 evals" > "$W/run/G9qent/GUARD_TRIPPED"; o2=$(WD_GPU=46068 q2src 1300200)
yes_ 61c '[ "$o2" = "$o" ] && [ -f "$N/runs/g9/HOLD" ] && grep -q "for 2 evals" "$N/runs/g9qent/TRIPPED" && [ ! -e "$N/runs/g9qent/FU.started" ] && ls "$W"/run/G9qent/GUARD_TRIPPED.ab-* > /dev/null 2>&1'
rm -f "$N/runs/g9/HOLD"; mkdir -p "$N/runs/g9qent-parked"; echo parked > "$N/runs/g9qent-parked/PARKED.txt"
o=$(NV2_PARK_HOLD_ROUNDS=1 WD_GPU=46068 q2src 1300200); check 61d "^RC=3\|$" "$o"; check 61d2 "MODE g9qent is parked" "$(cat "$W/run/train.log")"
qetree; echo g9qent > "$N/runs/g9/MODE"; epok; echo "$EP 880050000 deadbeef" > "$N/runs/g9qent/FORK_POINT"
o=$(NV2_PARK_HOLD_ROUNDS=1 WD_GPU=46068 q2src 1300200); check 61e "^RC=3\|$" "$o"; yes_ 61e2 '[ ! -e "$N/runs/g9qent/G9QENT_STARTED" ] && grep -q "MODE g9qent: no valid first start" "$W/run/train.log"'
qetree; echo g9qent > "$N/runs/g9/MODE"; echo "2026-11-16T04:00:00 880050000 g9 1300200" > "$N/runs/g9qent/G9QENT_STARTED"; epok
MQ=$N/runs/g9/20261116-040000-1300200/G9qent; mkdir -p "$MQ/pool"; echo mq > "$MQ/final_model.pt"; W2=$ROOTM/wd-1300220
o=$(WD_GPU=46068 q2src 1300220); check 61f "^RC=0\|PY=$N/bin/py-src-8892762\|.*TRAIN_ARGS=--no-bf16 .*--guard-entropy-min 0.35 .*--run-dir $W2/run/G9qent .*--ent-schedule linear --ent-final 0.001 --ent-start 880050000 --ent-horizon 50000000 --resume $MQ/final_model.pt$" "$o"
rm -rf "$N/runs/g9qent" "$N/runs/g9qent-parked" "$N/runs/g9q2-parked" "$N/runs/g9qbf"; rm -f "$N/bin/py-src-8892762" "$N/runs/g9/HOLD"; echo g9q2 > "$N/runs/g9/MODE"
fi
if [ -z "${MOCK_SKIP_V19:-}" ] && [ -z "${MOCK_SKIP_V18:-}" ] && [ -z "${MOCK_SKIP_V17:-}" ] && [ -z "${MOCK_SKIP_G9Q2:-}" ]; then
# 62 (v19, 2026-11-16; runs/p4/PREREG.txt): MODE g9p4 = G9-P4 on the A40 = G9q2's restart args on this GPU with the
# BC net P4 in all three roles (kl-anchor, frozen, first --resume), lineage G9P4, no --seed, first start only from a
# read-only BC-P4 whose md5 = runs/g9p4/P4.md5; the bridge state (G9qbf started and saved, then MODE g9p4) -> first
# start; restarts / a fresh job; park guard; refusals => HOLD rounds; G9q's guard fallbacks with state runs/g9p4.
p4tree() {  # qetree + the 11-16 state (G9qent parked, MODE g9qent) + BC-P4 r-- with its md5 in runs/g9p4/P4.md5
  qetree; echo g9qent > "$N/runs/g9/MODE"; mkdir -p "$N/runs/g9qent-parked" "$N/runs/g9p4"; echo parked > "$N/runs/g9qent-parked/PARKED.txt"
  P4=$N/runs/pretrain/P4-g9-s2.pt; rm -f "$P4"; echo p4net > "$P4"; chmod a-w "$P4"; md5sum < "$P4" | cut -c1-8 > "$N/runs/g9p4/P4.md5"; }
p4bridge() {  # what the bridge leaves: G9qbf started and saved in this job, MODE g9p4 (written by the chain)
  mkdir -p "$W/run/G9qbf/pool"; echo qbf-bridge-final > "$W/run/G9qbf/final_model.pt"; echo "2026-11-17T04:00:00 820017467 g9 1300200" > "$N/runs/g9qbf/G9QBF_STARTED"; echo g9p4 > "$N/runs/g9/MODE"; }
p4tree; p4bridge; o=$(WD_GPU=46068 q2src 1300200)
ex="${base2//$W\/run\/G9q2/$W/run/G9P4}"; ex="${ex//$P3/$P4}"; ex="${ex%--resume *}--resume $P4"
yes_ 62a '[ "$o" = "$ex" ]'; yes_ 62a2 'grep -q "^20[0-9T:-]* $(cat "$N/runs/g9p4/P4.md5") g9 1300200$" "$N/runs/g9p4/G9P4_STARTED" && [ "$(cat "$N/runs/g9/MODE")" = g9p4 ] && [ ! -e "$N/runs/g9/HOLD" ]'
check 62a3 "TRAIN_ARGS=--no-bf16 .*--kl-anchor $P4 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P4 --p-frozen 0.67 .*--fastenv-workers 8 .*--snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume $P4$" "$o"
check 62a4 "^RC=0\|PY=$N/bin/py-src-24c633f\|PS=$NOV\|child=$NOV\|" "$o"; checkno 62a5 "P3-g9-s2|--seed|G9q2|G9qbf|G9qent|ent-schedule|lr-schedule|gae-lambda|actor-stream" "${o#*TRAIN_ARGS=}"
check 62a6 "G9-P4 first start: from BC-P4 $P4 \(md5 $(cat "$N/runs/g9p4/P4.md5"); steps 0\)" "$(cat "$W/run/train.log")"
echo p4-final > "$W/run/G9P4/final_model.pt"; o=$(WD_GPU=46068 q2src 1300200); yes_ 62b '[ "$o" = "${ex%--resume *}--resume $W/run/G9P4/final_model.pt" ]'
yes_ 62b2 '[ "$(grep -c "G9-P4 first start" "$W/run/train.log")" = 1 ]'
MQ=$N/runs/g9/20261117-040000-1300200/G9P4; mkdir -p "$MQ/pool"; echo mq > "$MQ/final_model.pt"; W2=$ROOTM/wd-1300230
o=$(WD_GPU=46068 q2src 1300230); check 62c "^RC=0\|PY=$N/bin/py-src-24c633f\|.*TRAIN_ARGS=--no-bf16 .*--kl-anchor $P4 .*--frozen $P4 .*--run-dir $W2/run/G9P4 --pool-dir $W2/run/G9P4/pool --snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume $MQ/final_model.pt$" "$o"; checkno 62c2 "--seed|P3-g9-s2" "$o"
# a first trip: collapse -> KL x3 (state runs/g9p4 only), then a second trip -> HOLD (G9q's rules)
p4tree; p4bridge; WD_GPU=46068 q2src 1300200 > /dev/null; echo p4-final > "$W/run/G9P4/final_model.pt"; echo "collapse: entropy 0.4 < 0.5" > "$W/run/G9P4/GUARD_TRIPPED"
o=$(WD_GPU=46068 q2src 1300200); check 62d "--kl-anchor $P4 --kl-anchor-coef 0.3 .*--run-dir $W/run/G9P4 .*--resume $W/run/G9P4/final_model.pt$" "$o"
yes_ 62d2 'grep -q "^1 " "$N/runs/g9p4/FU.started" && grep -q collapse "$N/runs/g9p4/FU.reason" && [ ! -e "$N/runs/g9q2/FU.started" ] && [ ! -e "$N/runs/g9q/FU.started" ]'
echo "eval score 40% < best 60% - 15%" > "$W/run/G9P4/GUARD_TRIPPED"; o=$(WD_GPU=46068 q2src 1300200); yes_ 62d3 '[ -f "$N/runs/g9/HOLD" ] && ls "$W"/run/G9P4/GUARD_TRIPPED.second-* > /dev/null 2>&1'
# the park guard: MODE g9p4 + runs/g9p4-parked/PARKED.txt => HOLD rounds
rm -f "$N/runs/g9/HOLD"; mkdir -p "$N/runs/g9p4-parked"; echo parked > "$N/runs/g9p4-parked/PARKED.txt"
o=$(NV2_PARK_HOLD_ROUNDS=1 WD_GPU=46068 q2src 1300200); check 62e "^RC=3\|$" "$o"; check 62e2 "MODE g9p4 is parked" "$(cat "$W/run/train.log")"
# refusals at the first start (P4 missing / writable / md5 mismatch / no P4.md5) => HOLD rounds, no G9P4_STARTED
for bad in missing writable md5 nomd5; do
  p4tree; p4bridge
  case $bad in missing) rm -f "$P4" ;; writable) chmod u+w "$P4" ;; md5) echo deadbeef > "$N/runs/g9p4/P4.md5" ;; nomd5) rm -f "$N/runs/g9p4/P4.md5" ;; esac
  o=$(NV2_PARK_HOLD_ROUNDS=1 WD_GPU=46068 q2src 1300200); check "62f-$bad" "^RC=3\|$" "$o"
  yes_ "62f2-$bad" '[ ! -e "$N/runs/g9p4/G9P4_STARTED" ] && [ -f "$N/runs/g9/HOLD" ] && grep -q "MODE g9p4: no valid BC-P4" "$W/run/train.log"'; done
# v19 keeps v18's park HOLD of the 11-16 state (MODE g9qent parked), and the bridge itself (MODE g9qbf, v18's 60c path)
p4tree; o=$(NV2_PARK_HOLD_ROUNDS=1 WD_GPU=46068 q2src 1300200); check 62g "^RC=3\|$" "$o"; check 62g2 "MODE g9qent is parked" "$(cat "$W/run/train.log")"
p4tree; FP=$N/runs/g9qbf/fork-point-00830040000.pt; fpok; echo g9qbf > "$N/runs/g9/MODE"; o=$(WD_GPU=46068 q2src 1300200)
ex="${base2/--no-bf16 /}"; ex="${ex//$W\/run\/G9q2/$W/run/G9qbf}"; ex="${ex%--resume *}--resume $FP"; yes_ 62h '[ "$o" = "$ex" ]'
chmod -R u+w "$N/runs/pretrain"; rm -rf "$N/runs/g9qent" "$N/runs/g9qent-parked" "$N/runs/g9q2-parked" "$N/runs/g9qbf" "$N/runs/g9p4" "$N/runs/g9p4-parked"; rm -f "$P4" "$N/bin/py-src-8892762" "$N/runs/g9/HOLD"; echo g9q2 > "$N/runs/g9/MODE"
fi
rm -f "$WDQ"* "$ROOTM.wdout"; rm -rf "$ROOTM"/wd-*
rm -f "$ROOTM.parkpy"; unset NV2_PARK_PY
rm -f "$N/bin/py-src-24c633f" "$N/bin/py-src-4f18108" "$ROOTM.r1" "$ROOTM.g9"
echo "passed $pass, failed $fail"
for f in /proc/[0-9]*/fd/1; do if [ "$(readlink "$f" 2>/dev/null)" = "$ROOTM.out" ]; then q=${f#/proc/}; q=${q%%/*}; echo "killing my leftover $(cat /proc/$q/cmdline | tr '\0' ' ') ($q)"; kill "$q" 2>/dev/null; fi; done
