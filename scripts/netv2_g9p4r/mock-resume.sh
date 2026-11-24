#!/bin/bash
# netv2 2026-11-24 (user decision 11-24): laptop mock of the G9-P4 RESUME on the A40 (job 1175374) after G9E finished
# (runs/g9e/DONE, READING ROBUST), with the LIVE netv2.env v21 (1ad5bb53) unchanged. Mechanism under test (no deploy):
#   mv runs/g9p4-parked -> runs/g9p4-parked.resumed-<time>; echo g9p4 > runs/g9/MODE; rm runs/g9/HOLD
# done while the job sits in a HOLD round (nv2_pause), whose end re-reads MODE. Simulated here by a two-stage
# INLINE_CMD (stage 1 = the pre-loop nv2_pause, stage 2 = round 1's nv2_pause does the three steps), so round 1 HOLDs
# for "MODE g9e is finished" and the loop then dispatches MODE g9p4 = v21's g9p4 branch (unchanged since v19).
# Checks: TRAIN_ARGS = G9-P4's last launch in 1173283 (11-21 15:18, train.log) with only the job paths and --resume
# changed; --resume = the parked POINTER (mirrored final, newer than ckpt_640M); reseed of best / ema / frozen / pool;
# TOTAL 1e9; the wrapper; November sets; the sbatch restart branch's argv (its own --resume first, TRAIN_ARGS' last);
# runs/g9e untouched; a fresh successor resumes 1175374's G9P4 mirror (or, without one, the POINTER again).
# Usage: bash scripts/netv2_g9p4r/mock-resume.sh scripts/netv2_g9e/netv2.env.v21
set -u
ENVF=$(readlink -f "$1"); R=${MOCK_ROOT:-/home/nand/.cache/nv2s/mock-g9p4r}
pass=0; fail=0
ok() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [$1]: $2"; fi; }
rm -rf "$R"; mkdir -p "$R"; N=$R/nv2; L=$R/tmp-1175374
mkdir -p "$N/runs/g9" "$N/runs/g9e/r100" "$N/runs/g9e/cd100/final" "$N/runs/g9p4" "$N/runs/pretrain" "$N/bin" "$N/scripts/netv2_g9e" "$N/src-24c633f"
printf '#!/bin/bash\nexit 0\n' > "$N/bin/py-src-24c633f"; chmod 555 "$N/bin/py-src-24c633f"
echo g9e > "$N/runs/g9/MODE"; touch "$N/runs/g9/LINEAGE_DIRS" "$N/runs/g9/R1B_STARTED"; echo 0 > "$N/runs/g9/chain_left"
for f in P1-g9h-dmg-s2.pt P3-g9-s2.pt; do echo x > "$N/runs/pretrain/$f"; done
P4=$N/runs/pretrain/P4-g9-s2.pt; echo p4net > "$P4"; chmod a-w "$P4"; md5sum < "$P4" | cut -c1-8 > "$N/runs/g9p4/P4.md5"
echo "2026-11-16T21:11:00 $(cat "$N/runs/g9p4/P4.md5") g9 1173274" > "$N/runs/g9p4/G9P4_STARTED"
# G9E finished (the record that must stay untouched)
echo "FINAL: ROBUST (E1@100M; runs/g9e/r100/READING.txt)" > "$N/runs/g9e/DONE"; printf 'hdr\nFINAL: ROBUST (E1@100M; runs/g9e/r100/READING.txt)\n' > "$N/runs/g9e/READING.txt"
echo "E1@100M VERDICT ROBUST (upper 95%% 51.9046 < 55)" > "$N/runs/g9e/r100/READING.txt"; touch "$N/runs/g9e/r100/DONE" "$N/runs/g9e/G9E_STARTED"
echo cooled > "$N/runs/g9e/cd100/final/01460078184.pt"; echo "status" > "$N/runs/g9e/STATUS"
# G9-P4's parked state (pointer park): the 1173283 mirror
D=$N/runs/g9/20261118-164715-1173283/G9P4; mkdir -p "$D/checkpoints" "$D/pool/frozen"
echo ck640 > "$D/checkpoints/ckpt_00640016576.pt"; touch -d '2026-11-21 16:08:15' "$D/checkpoints/ckpt_00640016576.pt"
echo p4final > "$D/final_model.pt"; touch -d '2026-11-21 16:27:55' "$D/final_model.pt"
echo best > "$D/best_model.pt"; echo ema > "$D/ema_model.pt"; echo "2026-11-21 16:27:55 saved final_model.pt at 641,887,817 steps" > "$D/train.log"
for s in 00002534650 00637500000 00640000000; do echo "$s" > "$D/pool/$s.pt"; done
for f in P4-g9-s2.pt best-00002534650.pt best-00232521384.pt; do echo "$f" > "$D/pool/frozen/$f"; done
mkdir -p "$N/runs/g9p4-parked"; echo "G9P4 PARKED ..." > "$N/runs/g9p4-parked/PARKED.txt"; echo "$D/final_model.pt 641887817 848c8888" > "$N/runs/g9p4-parked/POINTER"
# 1175374's mirror (G9E lineages only) and its node-local state (G9E + G9Ecd100 finals, no top-level final_model.pt)
J=$N/runs/g9/20261123-103220-1175374; mkdir -p "$J/G9E" "$J/G9Ecd100"; echo e > "$J/G9E/final_model.pt"; echo c > "$J/G9Ecd100/final_model.pt"
mkdir -p "$L/run/G9E/pool" "$L/run/G9Ecd100/pool" "$L/pool"; echo e > "$L/run/G9E/final_model.pt"; echo c > "$L/run/G9Ecd100/final_model.pt"
touch -d '2026-11-24 05:48' "$L/run/G9E/final_model.pt"; touch -d '2026-11-24 06:46' "$L/run/G9Ecd100/final_model.pt"
touch "$L/run/train.log" "$L/.nv2-watchdog" "$L/.toucher-G9E" "$L/.toucher-G9Ecd100" "$L/.toucher-G9P4"
# the two-stage INLINE_CMD = main acting during the HOLD round
cat > "$N/runs/g9/INLINE_CMD" << EOF
cat > "$N/runs/g9/INLINE_CMD" << 'EOS'
mv -T "$N/runs/g9p4-parked" "$N/runs/g9p4-parked.resumed-20261124" && echo g9p4 > "$N/runs/g9/MODE" && rm -f "$N/runs/g9/HOLD"
EOS
EOF
g9e_sum() { (cd "$N/runs/g9e" && find . -type f -print0 | sort -z | xargs -0 md5sum | md5sum); }
E0=$(g9e_sum)
src() {  # src JOB LOCAL: one netv2.env source as the job script's restart branch does it; prints the trainer argv
  ( set -euo pipefail; NV2=$N; LOCAL=$2; RUN_NAME=g9; export SLURM_JOB_ID=$1; SLURM_JOB_NAME=ws-netv2-g9; SLURM_JOB_PARTITION=a40
    SLURM_CPUS_ON_NODE=10; NV2_GPU_MIB=46068; NV2_HOLD_ITERS=0; NV2_SWAP_ITERS=0; NV2_PARK_HOLD_ROUNDS=3; NV2_TMPWATCH_POLL_S=0; NV2_WD_OFF=1
    source "$ENVF"
    LATEST=$LOCAL/run/final_model.pt; [ -f "$LATEST" ] || LATEST=$(ls -t "$LOCAL"/run/*/final_model.pt 2>/dev/null | head -n 1 || true)
    echo "TOTAL=$TOTAL|PY=$PY|PS=${PORYGON_PS:-unset}|ANCHOR=$ANCHOR|FORMAT=$FORMAT"
    echo "TRAIN_ARGS=$TRAIN_ARGS"
    echo "ARGV=-u -m porygon.v2.train --format $FORMAT --ports X --n-battles 16 --pool-battles 16 --pool-dir $LOCAL/pool --run-dir $LOCAL/run ${LATEST:+--resume $LATEST} --anchor $ANCHOR --total-steps $TOTAL --max-hours 70.5 --collector fastenv --fastenv-workers 8 --fastenv-battles 48 $TRAIN_ARGS"
  ) > "$R/out" 2>&1; echo "RC=$?"; cat "$R/out"
}
o=$(src 1175374 "$L")
# G9-P4's last launch (1173283 train.log 11-21 15:18:08, under v21), paths mapped to the mock
REF='--no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 --kl-anchor /vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/P4-g9-s2.pt --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen /vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/P4-g9-s2.pt --p-frozen 0.67 --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir /tmp/ws-netv2-1173283/run/G9P4 --pool-dir /tmp/ws-netv2-1173283/run/G9P4/pool --snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --resume /tmp/ws-netv2-1173283/run/G9P4/final_model.pt'
EXP=${REF//\/vol\/bitbucket\/fr1226\/porygon-ws-netv2/$N}; EXP=${EXP//\/tmp\/ws-netv2-1173283/$L}; EXP=${EXP/--resume $L\/run\/G9P4\/final_model.pt/--resume $D/final_model.pt}
TA=$(grep '^TRAIN_ARGS=' <<< "$o" | cut -d= -f2-)
ok 1-rc '[ "$(head -n 1 <<< "$o")" = RC=0 ]'
ok 2-args '[ "$TA" = "$EXP" ]' || { echo "  got: $TA"; echo "  exp: $EXP"; }
ok 3-env 'grep -qx "TOTAL=1000000000|PY=$N/bin/py-src-24c633f|PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11|ANCHOR=$N/runs/pretrain/P1-g9h-dmg-s2.pt|FORMAT=gen9randombattle" <<< "$o"'
ok 4-lastresume '[ "$(grep "^ARGV=" <<< "$o" | grep -oE -- "--resume [^ ]+" | tail -n 1)" = "--resume $D/final_model.pt" ]'
ok 5-hold1 '[ "$(grep -c "MODE g9e is finished (FINAL: ROBUST (E1@100M; runs/g9e/r100/READING.txt)): the A40 is for main, round 1: HOLD" "$L/run/train.log")" = 1 ] && ! grep -q "round 2" "$L/run/train.log"'
ok 6-markers '[ ! -e "$N/runs/g9/HOLD" ] && [ ! -e "$N/runs/g9p4-parked" ] && [ -f "$N/runs/g9p4-parked.resumed-20261124/PARKED.txt" ] && [ "$(cat "$N/runs/g9/MODE")" = g9p4 ] && [ ! -e "$N/runs/g9/INLINE_CMD" ]'
ok 7-reseed '[ "$(cat "$L/run/G9P4/best_model.pt")" = best ] && [ "$(cat "$L/run/G9P4/ema_model.pt")" = ema ] && [ "$(ls "$L/run/G9P4/pool/frozen" | wc -l)" = 3 ] && [ "$(readlink "$L/run/final_model.pt")" = G9P4/final_model.pt ]'
sleep 1; ok 8-pool '[ "$(ls "$L/run/G9P4/pool" | grep -c "\.pt$")" = 3 ]'
ok 9-log 'grep -q "fresh job: G9P4 resumes from $D/final_model.pt" "$L/run/train.log"'
ok 10-g9e '[ "$(g9e_sum)" = "$E0" ] && [ ! -e "$N/runs/g9e/FAILED" ]'
ok 11-nofirst 'grep -qx "2026-11-16T21:11:00 $(cat "$N/runs/g9p4/P4.md5") g9 1173274" "$N/runs/g9p4/G9P4_STARTED"'
checkno() { if grep -qE -- "$2" <<< "$3"; then fail=$((fail + 1)); echo "FAIL [$1]: /$2/ in $3"; else pass=$((pass + 1)); fi; }
checkno 12-no-g9e "G9E|cd1350|--p-frozen 1.0|--cooldown|--total-steps" "$TA"
# the restart branch again (trainer saved its own final in 1175374): restart in place from the local G9P4 final
echo p4-local > "$L/run/G9P4/final_model.pt"; o=$(src 1175374 "$L")
ok 13-restart '[ "$(grep "^TRAIN_ARGS=" <<< "$o" | grep -oE -- "--resume [^ ]+$")" = "--resume $L/run/G9P4/final_model.pt" ] && [ "$(grep "^ARGV=" <<< "$o" | grep -oE -- "--resume [^ ]+" | head -n 1)" = "--resume $L/run/final_model.pt" ]'
# a fresh successor (1552709): 1175374's G9P4 mirror (newer) -> resumes it, MODE g9p4 kept, no HOLD
mkdir -p "$J/G9P4/pool/frozen" "$J/G9P4/checkpoints"; echo m > "$J/G9P4/final_model.pt"; echo b2 > "$J/G9P4/best_model.pt"
L2=$R/tmp-1552709; mkdir -p "$L2/run" "$L2/pool"; touch "$L2/run/train.log" "$L2/.toucher-G9P4"; o=$(src 1552709 "$L2")
ok 14-succ '[ "$(head -n 1 <<< "$o")" = RC=0 ] && [ "$(grep "^TRAIN_ARGS=" <<< "$o" | grep -oE -- "--resume [^ ]+$")" = "--resume $J/G9P4/final_model.pt" ] && [ "$(cat "$L2/run/G9P4/best_model.pt")" = b2 ] && [ ! -e "$N/runs/g9/HOLD" ]'
# a successor when 1175374 left no G9P4 mirror: the POINTER again (never G9E)
rm -rf "$J/G9P4" "$L2"; mkdir -p "$L2/run" "$L2/pool"; touch "$L2/run/train.log" "$L2/.toucher-G9P4"; o=$(src 1552709 "$L2")
ok 15-succ2 '[ "$(grep "^TRAIN_ARGS=" <<< "$o" | grep -oE -- "--resume [^ ]+$")" = "--resume $D/final_model.pt" ]'
ok 16-g9e-end '[ "$(g9e_sum)" = "$E0" ]'
echo "mock-resume: $pass passed, $fail failed"; [ "$fail" = 0 ]
