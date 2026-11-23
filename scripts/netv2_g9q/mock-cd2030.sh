#!/bin/bash
# laptop mock of cd2030-driver.sh: a fake job loop (sim.sh = the "job script", JOBPP) runs a fake trainer (a sleeping
# python3 with G9q-like argv), lands the SIGINT, runs INLINE_CMD with a stub trainer wrapper, relaunches; squeue stubbed.
#   bash mock-cd2030.sh scripts/netv2_g9q/cd2030-driver.sh
set -u
DRV=$(readlink -f "$1"); W=/home/nand/.cache/nv2s/cd2030/mkd; pass=0; failn=0; SIMS=()
ok() { if eval "$2"; then pass=$((pass+1)); echo "PASS $1"; else failn=$((failn+1)); echo "FAIL $1"; tail -n 3 "$W/C/runs/g9q-cd2030/STATUS" 2>/dev/null | sed 's/^/     /'; fi; }
TAG=nv2mocktrainer$$
setup() {  # $1 = mode: ok | keepbad | short | hold | nokeep | prew | noptop | argvdiff | stop
  rm -rf "$W"; mkdir -p "$W/C/runs/g9q-cd2030" "$W/C/runs/r1" "$W/C/bin" "$W/bin" "$W/LT/run/G9q/checkpoints" "$W/LT/run/G9q/pool/frozen" "$W/LT/cd2030-keep"
  C=$W/C; LT=$W/LT
  echo prereg > $C/runs/g9q-cd2030/PREREG.txt; [ "$1" = prew ] || chmod a-w $C/runs/g9q-cd2030/PREREG.txt
  echo ck2030 > $LT/cd2030-keep/ckpt_02030019516.pt; echo ck2035 > $LT/run/G9q/checkpoints/ckpt_02035019516.pt
  [ "$1" = nokeep ] && rm -f $LT/cd2030-keep/*.pt
  echo p1 > $LT/run/G9q/pool/02027500000.pt; echo p2 > $LT/run/G9q/pool/02030000000.pt; echo p3 > $LT/run/G9q/pool/02032500000.pt
  echo f1 > $LT/run/G9q/pool/frozen/best-02000000000.pt; echo f2 > $LT/run/G9q/pool/frozen/best-02034000000.pt
  echo fm > $LT/run/G9q/final_model.pt; ln -s G9q/final_model.pt $LT/run/final_model.pt
  : > $LT/run/G9q/train.log
  [ "$1" = hold ] && touch $C/runs/r1/HOLD
  [ "$1" = stop ] && touch $C/runs/g9q-cd2030/CD_STOP
  # the stub trainer wrapper run by INLINE_CMD: writes the branch's final + log lines
  cat > $C/bin/py-src-24c633f <<'STUB'
#!/bin/bash
B=""; T=""; while [ $# -gt 0 ]; do case $1 in --run-dir) B=$2; shift ;; --total-steps) T=$2; shift ;; esac; shift; done
echo "args-ok" > "$B/stub-args"; echo "upd 1 steps $((T - 40000)) lr 7.00e-06 ent 0.540" >> "$B/train.log"
echo net > "$B/final_model.pt"; echo "saved final_model.pt at $(python3 -c "print(f'{$((T + 6466)):,}')") steps" >> "$B/train.log"
STUB
  chmod +x $C/bin/py-src-24c633f
  local end; if [ "$1" = short ]; then end=$(date -d "+1 hour" +%FT%T); else end=$(date -d "+2 days" +%FT%T); fi
  printf '#!/bin/bash\necho %s\n' "$end" > $W/bin/squeue; chmod +x $W/bin/squeue
  local ptop="--guard-ptop-max 0.6"; [ "$1" = noptop ] && ptop="--guard-ptop-max 0.7"
  local rl2="$LT/run/G9q/final_model.pt"; [ "$1" = argvdiff ] && rl2="--extra"
  cat > $W/sim.sh <<SIM
#!/bin/bash
LOG=$LT/run/G9q/train.log
python3 -c "import signal,time; signal.signal(2, signal.default_int_handler); time.sleep(120)" $TAG --n-envs 28 $ptop --resume /old/final_model.pt --resume $LT/run/G9q/final_model.pt > /dev/null 2>&1 &
p=\$!; echo "v2 trainer: first start" >> \$LOG
wait \$p; echo "signal 2: saved final_model.pt at 2,036,000,000 steps" >> \$LOG
if [ -f $C/runs/r1/INLINE_CMD ]; then mv -f $C/runs/r1/INLINE_CMD $C/runs/r1/INLINE_CMD.running
  (cd $C && export LOCAL=$LT && bash $C/runs/r1/INLINE_CMD.running) > $C/runs/r1/inline.log 2>&1; mv -f $C/runs/r1/INLINE_CMD.running $C/runs/r1/INLINE_CMD.done; fi
python3 -c "import signal,time; signal.signal(2, signal.default_int_handler); time.sleep(120)" $TAG --n-envs 28 $ptop --resume /old/final_model.pt --resume $rl2 > /dev/null 2>&1 &
p=\$!; echo "v2 trainer: relaunch" >> \$LOG; sleep 0.3; echo "x upd 1 steps 2,036,040,000 (+40000) sps 2000" >> \$LOG
wait \$p
SIM
  bash $W/sim.sh & SIM=$!; SIMS+=($SIM)
  sleep 0.5
  export MOCK_C=$C MOCK_LT=$LT MOCK_PRE_MD5=$(md5sum < $C/runs/g9q-cd2030/PREREG.txt | cut -c1-8) MOCK_KEEP_MD5=$(md5sum < $LT/cd2030-keep/ckpt_02030019516.pt 2>/dev/null | cut -c1-8)
  export MOCK_PAT="^[^ ]*python3[.0-9]* -c import signal,time; signal.signal\(2, signal.default_int_handler\); time.sleep\(120\) $TAG( |\$)" SLURM_JOB_ID=4242 JOBPP=$SIM
  [ "$1" = keepbad ] && export MOCK_KEEP_MD5=deadbeef
  return 0
}
run() { PATH=$W/bin:$PATH timeout 120 bash "$DRV" > $W/out.log 2>&1; echo $?; }
cleanup() { for s in "${SIMS[@]}"; do pkill -P "$s" 2>/dev/null; kill "$s" 2>/dev/null; done; SIMS=(); }
alive() { pgrep -f "$MOCK_PAT" > /dev/null; }
D() { echo "$W/C/runs/g9q-cd2030"; }
# 1 the normal path
setup ok; rc=$(run)
ok "1a rc 0, driver done" '[ $rc = 0 ] && grep -q "driver done" $(D)/STATUS'
ok "1b earliest checkpoint = the keep copy 2030019516" 'grep -q "checkpoint $W/LT/cd2030-keep/ckpt_02030019516.pt (S = 2030019516)" $(D)/STATUS && [ -f $(D)/branch-point-02030019516.pt ]'
ok "1c INLINE ran once with the branch dirs" '[ -f $W/C/runs/r1/INLINE_CMD.done ] && [ -f $W/LT/run/G9q-cd2030/stub-args ] && grep -q "G9q-cd2030 exit 0" $(D)/notes.log'
ok "1d INLINE argv: resume, cooldown, total, frac, floor" 'grep -q -- "--resume \"\$CK\" --cooldown-start \"\$S\" --total-steps \"\$((S + 5000000))\" --final-lr-frac 0.1 --guard-entropy-min 0.35" $W/C/runs/r1/INLINE_CMD.done && grep -q "S=2030019516; CK=$(D)/branch-point-02030019516.pt" $W/C/runs/r1/INLINE_CMD.done'
ok "1e final r-- at S+5M+ with md5 line" '[ -f $(D)/final/02035025982.pt ] && [ ! -w $(D)/final/02035025982.pt ] && grep -q "G9q-cd2030 final at 2035025982 steps -> .* (md5 [0-9a-f]\{8\})" $(D)/STATUS'
ok "1f pool snapshots / frozen bests newer than S moved out; G9q's own pool untouched" 'grep -q "pool 2 + frozen 1" $(D)/notes.log && [ -f $W/LT/run/G9q/pool/02032500000.pt ] && [ ! -e $W/LT/run/G9q-cd2030/pool ]'
ok "1g relaunch check PASS, restart re-armed" 'grep -q "relaunch check PASS" $(D)/STATUS && [ -e $W/C/runs/r1/restart ] && [ ! -e $(D)/relaunch/FAILED ]'
ok "1h no FAILED" '[ ! -e $(D)/FAILED ]'
cleanup
# 2 keep copy md5 mismatch
setup keepbad; rc=$(run)
ok "2 keep md5 bad => FAILED, nothing queued, trainer alive" '[ $rc = 3 ] && grep -q "md5 is not the pinned" $(D)/FAILED && [ ! -e $W/C/runs/r1/INLINE_CMD ] && [ ! -e $W/C/runs/r1/INLINE_CMD.done ] && alive'
cleanup
# 3 < 3 h left
setup short; rc=$(run)
ok "3 short job => FAILED, nothing queued" '[ $rc = 3 ] && grep -q "need >= 3 h" $(D)/FAILED && [ ! -e $W/C/runs/r1/INLINE_CMD ] && alive'
cleanup
# 4 HOLD in runs/r1
setup hold; rc=$(run)
ok "4 HOLD => FAILED" '[ $rc = 3 ] && grep -q "runs/r1/HOLD present" $(D)/FAILED && alive'
cleanup
# 5 no keep copy => the earliest live one (no md5 pin applies)
setup nokeep; rc=$(run)
ok "5 no keep => branch at 2035019516" '[ $rc = 0 ] && grep -q "(S = 2035019516)" $(D)/STATUS && ls $(D)/final/02040025982.pt > /dev/null'
cleanup
# 6 writable PREREG
setup prew; rc=$(run)
ok "6 writable PREREG => FAILED" '[ $rc = 3 ] && grep -q "no read-only runs/g9q-cd2030/PREREG.txt" $(D)/FAILED && alive'
cleanup
# 7 argv without --guard-ptop-max 0.6
setup noptop; rc=$(run)
ok "7 argv check => FAILED" '[ $rc = 3 ] && grep -q "lacks --guard-ptop-max 0.6" $(D)/FAILED && [ ! -e $W/C/runs/r1/INLINE_CMD ] && alive'
cleanup
# 8 relaunched argv differs
setup argvdiff; rc=$(run)
ok "8 argv diff => relaunch/FAILED, final kept" '[ -e $(D)/relaunch/FAILED ] && ls $(D)/final/*.pt > /dev/null && [ ! -e $(D)/FAILED ]'
cleanup
# 9 CD_STOP
setup stop; rc=$(run)
ok "9 CD_STOP => exit 0, nothing copied" '[ $rc = 0 ] && grep -q "CD_STOP: exiting" $(D)/STATUS && ! ls $(D)/branch-point-*.pt 2>/dev/null && alive'
cleanup
# 10 a second run with a final present refuses
setup ok; mkdir -p $(D)/final; echo x > $(D)/final/02035025982.pt; rc=$(run)
ok "10 final present => exit 2" '[ $rc = 2 ] && grep -q "final is not empty" $(D)/STATUS'
cleanup
echo "mock-cd2030: $pass passed, $failn failed"
