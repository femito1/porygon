#!/bin/bash
# laptop mock of cd-driver.sh (= mock-cd2030.sh for R1b-F): a fake job loop (sim.sh = the "job script", JOBPP) runs a fake
# trainer (a sleeping python3 with R1b-F-like argv), lands the SIGINT, runs INLINE_CMD with a stub trainer wrapper,
# relaunches with --total-steps 2e9 (v22); squeue stubbed.
#   bash mock-cd.sh scripts/netv2_g4r1bf/cd-driver.sh
set -u
DRV=$(readlink -f "$1"); W=/home/nand/.cache/nv2s/g4cd/mkd; pass=0; failn=0; SIMS=()
ok() { if eval "$2"; then pass=$((pass+1)); echo "PASS $1"; else failn=$((failn+1)); echo "FAIL $1"; tail -n 3 "$W/C/runs/g4r1bf-cd/STATUS" 2>/dev/null | sed 's/^/     /'; fi; }
TAG=nv2mocktrainer$$
S=635038608
setup() {  # $1 = mode: ok | bpbad | bpw | short | hold | prew | noanchor | tot | argvdiff | totnew | stop | shortcd
  rm -rf "$W"; mkdir -p "$W/C/runs/g4r1bf-cd" "$W/C/runs/g4r1bf" "$W/C/bin" "$W/bin" "$W/LT/run/R1bF/checkpoints" "$W/LT/run/R1bF/pool/frozen"
  C=$W/C; LT=$W/LT
  echo prereg > $C/runs/g4r1bf-cd/PREREG.txt; [ "$1" = prew ] || chmod a-w $C/runs/g4r1bf-cd/PREREG.txt
  echo bp > $C/runs/g4r1bf-cd/branch-point-00635038608.pt; [ "$1" = bpw ] || chmod a-w $C/runs/g4r1bf-cd/branch-point-00635038608.pt
  echo p1 > $LT/run/R1bF/pool/00632535155.pt; echo p2 > $LT/run/R1bF/pool/00635038608.pt; echo p3 > $LT/run/R1bF/pool/00637540000.pt
  echo fm > $LT/run/R1bF/final_model.pt; ln -s R1bF/final_model.pt $LT/run/final_model.pt
  : > $LT/run/R1bF/train.log
  [ "$1" = hold ] && touch $C/runs/g4r1bf/HOLD
  [ "$1" = stop ] && touch $C/runs/g4r1bf-cd/CD_STOP
  local cut=6466; [ "$1" = shortcd ] && cut=-100000
  # the stub trainer wrapper run by INLINE_CMD: writes the branch's final + log lines + a checkpoint and a pool snapshot
  cat > $C/bin/py-src-24c633f <<STUB
#!/bin/bash
B=""; T=""; P=""; while [ \$# -gt 0 ]; do case \$1 in --run-dir) B=\$2; shift ;; --total-steps) T=\$2; shift ;; --pool-dir) P=\$2; shift ;; esac; shift; done
echo "args-ok" > "\$B/stub-args"; echo "upd 1 steps \$((T - 40000)) lr 7.00e-06 ent 0.540" >> "\$B/train.log"
mkdir -p "\$B/checkpoints"; echo ck > "\$B/checkpoints/ckpt_x.pt"; echo snap > "\$P/00637538608.pt.tmp" && mv -f "\$P/00637538608.pt.tmp" "\$P/00637538608.pt"
echo net > "\$B/final_model.pt"; echo "saved final_model.pt at \$(python3 -c "print(f'{\$((T + $cut)):,}')") steps" >> "\$B/train.log"
STUB
  chmod +x $C/bin/py-src-24c633f
  local end; if [ "$1" = short ]; then end=$(date -d "+5 hour" +%FT%T); else end=$(date -d "+2 days" +%FT%T); fi
  printf '#!/bin/bash\necho %s\n' "$end" > $W/bin/squeue; chmod +x $W/bin/squeue
  local anc="--guard-anchor-drop 0.0999"; [ "$1" = noanchor ] && anc="--guard-anchor-drop 0.15"
  local t1=1000000000; [ "$1" = tot ] && t1=2000000000
  local t2=2000000000; [ "$1" = totnew ] && t2=3000000000
  local x2=""; [ "$1" = argvdiff ] && x2="--extra 1"
  local base="--format gen4randombattle --ports 42224-42226 --n-battles 16 --pool-battles 16 --run-dir $LT/run"
  cat > $W/sim.sh <<SIM
#!/bin/bash
LOG=$LT/run/R1bF/train.log
python3 -c "import signal,time; signal.signal(2, signal.default_int_handler); time.sleep(120)" $TAG $base --resume /old/final_model.pt --total-steps $t1 --max-hours 70.5 --lr 7e-5 --run-dir $LT/run/R1bF --pool-dir $LT/run/R1bF/pool --guard-pg-mean 20 $anc --resume /old/R1bF/final_model.pt > /dev/null 2>&1 &
p=\$!; echo "v2 trainer: first start" >> \$LOG
wait \$p; echo "signal 2: saved final_model.pt at 636,000,000 steps" >> \$LOG
if [ -f $C/runs/g4r1bf/INLINE_CMD ]; then mv -f $C/runs/g4r1bf/INLINE_CMD $C/runs/g4r1bf/INLINE_CMD.running
  (cd $C && export LOCAL=$LT && bash $C/runs/g4r1bf/INLINE_CMD.running) > $C/runs/g4r1bf/inline.log 2>&1; mv -f $C/runs/g4r1bf/INLINE_CMD.running $C/runs/g4r1bf/INLINE_CMD.done; fi
python3 -c "import signal,time; signal.signal(2, signal.default_int_handler); time.sleep(120)" $TAG $base --resume $LT/run/final_model.pt --total-steps $t2 --max-hours 70.5 --lr 7e-5 --run-dir $LT/run/R1bF --pool-dir $LT/run/R1bF/pool --guard-pg-mean 20 $anc $x2 --resume $LT/run/R1bF/final_model.pt > /dev/null 2>&1 &
p=\$!; echo "v2 trainer: relaunch" >> \$LOG; sleep 0.3; echo "x upd 1 steps 636,040,000 (+40000) sps 380" >> \$LOG
wait \$p
SIM
  bash $W/sim.sh & SIM=$!; SIMS+=($SIM)
  sleep 0.5
  export MOCK_C=$C MOCK_LT=$LT MOCK_PRE_MD5=$(md5sum < $C/runs/g4r1bf-cd/PREREG.txt | cut -c1-8) MOCK_S=$S
  export MOCK_BP_MD5=$(md5sum < $C/runs/g4r1bf-cd/branch-point-00635038608.pt | cut -c1-8)
  export MOCK_PAT="^[^ ]*python3[.0-9]* -c import signal,time; signal.signal\(2, signal.default_int_handler\); time.sleep\(120\) $TAG( |\$)" SLURM_JOB_ID=4242 JOBPP=$SIM
  [ "$1" = bpbad ] && export MOCK_BP_MD5=deadbeef
  return 0
}
run() { PATH=$W/bin:$PATH timeout 120 bash "$DRV" > $W/out.log 2>&1; echo $?; }
cleanup() { for s in "${SIMS[@]}"; do pkill -P "$s" 2>/dev/null; kill "$s" 2>/dev/null; done; SIMS=(); }
alive() { pgrep -f "$MOCK_PAT" > /dev/null; }
D() { echo "$W/C/runs/g4r1bf-cd"; }
P3() { echo "$W/C/council/p3-gen4"; }
nothing() { [ ! -e $W/C/runs/g4r1bf/INLINE_CMD ] && [ ! -e $W/C/runs/g4r1bf/INLINE_CMD.done ] && alive; }
# 1 the normal path
setup ok; rc=$(run)
ok "1a rc 0, driver done" '[ $rc = 0 ] && grep -q "driver done" $(D)/STATUS'
ok "1b INLINE ran once with the branch dir outside run/" '[ -f $W/C/runs/g4r1bf/INLINE_CMD.done ] && [ -f $W/LT/cd/R1bF-cd/stub-args ] && grep -q "R1b-F-cd exit 0" $(D)/notes.log && [ ! -e $W/LT/run/R1bF-cd ]'
ok "1c INLINE argv: resume, cooldown, total, frac, floor; S and the branch point" 'grep -q -- "--resume \"\$CK\" --cooldown-start \"\$S\" --total-steps \"\$((S + 5000000))\" --final-lr-frac 0.1 --guard-entropy-min 0.35" $W/C/runs/g4r1bf/INLINE_CMD.done && grep -q "S=635038608; CK=$(D)/branch-point-00635038608.pt" $W/C/runs/g4r1bf/INLINE_CMD.done'
ok "1d final r-- at S+5M+ with md5 line" '[ -f $(D)/final/00640045074.pt ] && [ ! -w $(D)/final/00640045074.pt ] && grep -q "R1b-F-cd final at 640045074 steps -> .* (md5 [0-9a-f]\{8\})" $(D)/STATUS'
ok "1e R1BF_CD_READY r-- = path md5 step" 'm=$(md5sum < $(D)/final/00640045074.pt | cut -c1-8); [ "$(cat $(P3)/R1BF_CD_READY)" = "$(D)/final/00640045074.pt $m 640045074" ] && [ ! -w $(P3)/R1BF_CD_READY ] && [ ! -e $(P3)/FAILED ]'
ok "1f pool: snapshot > S moved aside in the copy; R1b-F pool untouched (incl. the hard-linked files)" 'grep -q "pool 2 + frozen 0, moved aside 1" $(D)/notes.log && [ -f $W/LT/run/R1bF/pool/00637540000.pt ] && [ "$(cat $W/LT/run/R1bF/pool/00635038608.pt)" = p2 ] && [ ! -e $W/LT/run/R1bF/pool/00637538608.pt ] && [ ! -e $W/LT/cd/R1bF-cd/pool ]'
ok "1g relaunch check PASS incl. TOTAL 1e9 -> 2e9, restart re-armed" 'grep -q "relaunch check PASS" $(D)/STATUS && grep -q "relaunch --total-steps 2000000000 (was 1000000000)" $(D)/STATUS && [ -e $W/C/runs/g4r1bf/restart ] && [ ! -e $(D)/relaunch/FAILED ]'
ok "1h no FAILED anywhere" '[ ! -e $(D)/FAILED ] && [ ! -e $(P3)/FAILED ]'
cleanup
# 2 branch point md5 mismatch / writable
setup bpbad; rc=$(run)
ok "2a bp md5 bad => FAILED (both), nothing queued, trainer alive" '[ $rc = 3 ] && grep -q "md5 not the pinned" $(D)/FAILED && [ -s $(P3)/FAILED ] && [ ! -e $(P3)/R1BF_CD_READY ] && nothing'
cleanup
setup bpw; rc=$(run)
ok "2b bp writable => FAILED" '[ $rc = 3 ] && grep -q "missing, writable" $(D)/FAILED && nothing'
cleanup
# 3 < 10 h left
setup short; rc=$(run)
ok "3 short job => FAILED, nothing queued" '[ $rc = 3 ] && grep -q "need >= 10 h" $(D)/FAILED && nothing'
cleanup
# 4 HOLD in runs/g4r1bf
setup hold; rc=$(run)
ok "4 HOLD => FAILED" '[ $rc = 3 ] && grep -q "runs/g4r1bf/HOLD present" $(D)/FAILED && nothing'
cleanup
# 5 writable PREREG
setup prew; rc=$(run)
ok "5 writable PREREG => FAILED" '[ $rc = 3 ] && grep -q "no read-only runs/g4r1bf-cd/PREREG.txt" $(D)/FAILED && nothing'
cleanup
# 6 argv without --guard-anchor-drop 0.0999
setup noanchor; rc=$(run)
ok "6 argv check (anchor drop) => FAILED" '[ $rc = 3 ] && grep -q "lacks --guard-anchor-drop 0.0999" $(D)/FAILED && nothing'
cleanup
# 7 recorded --total-steps not 1e9
setup tot; rc=$(run)
ok "7 recorded TOTAL 2e9 => FAILED" '[ $rc = 3 ] && grep -q "total-steps is .2000000000., not 1000000000" $(D)/FAILED && nothing'
cleanup
# 8 relaunched argv differs beyond --resume / TOTAL
setup argvdiff; rc=$(run)
ok "8 argv diff => relaunch/FAILED, final + READY kept" '[ -e $(D)/relaunch/FAILED ] && grep -q "differs beyond" $(D)/relaunch/FAILED && ls $(D)/final/*.pt > /dev/null && [ -e $(P3)/R1BF_CD_READY ] && [ ! -e $(D)/FAILED ]'
cleanup
# 9 relaunched TOTAL not 2e9
setup totnew; rc=$(run)
ok "9 relaunch TOTAL 3e9 => relaunch/FAILED" '[ -e $(D)/relaunch/FAILED ] && grep -q "total-steps is .3000000000., not 2000000000" $(D)/relaunch/FAILED && [ -e $(P3)/R1BF_CD_READY ]'
cleanup
# 10 CD_STOP
setup stop; rc=$(run)
ok "10 CD_STOP => exit 0, nothing queued" '[ $rc = 0 ] && grep -q "CD_STOP: exiting" $(D)/STATUS && nothing'
cleanup
# 11 a second run with a final present refuses
setup ok; mkdir -p $(D)/final; echo x > $(D)/final/00640045074.pt; rc=$(run)
ok "11 final present => exit 2" '[ $rc = 2 ] && grep -q "final is not empty" $(D)/STATUS'
cleanup
# 12 branch cut short
setup shortcd; rc=$(run)
ok "12 short branch => FAILED (both), no READY" '[ $rc = 3 ] && grep -q "cut short" $(D)/FAILED && [ -s $(P3)/FAILED ] && [ ! -e $(P3)/R1BF_CD_READY ] && ! ls $(D)/final/*.pt 2>/dev/null'
cleanup
echo "mock-g4cd: $pass passed, $failn failed"
