#!/bin/bash
# laptop mock of r2030-chain.sh (host picking, host hand-over retry, F2030 gate) and conf2030.sbatch (R2030 gate, fixed
# iterations args, VOID replay once, iteration check): stubs cmp_step, srun, squeue, scontrol, scancel, fp9_iters.
#   bash mock-r2030.sh scripts/netv2_g9q/r2030-chain.sh scripts/netv2_conf2030/conf2030.sbatch
set -u
CH=$(readlink -f "$1"); SB=$(readlink -f "$2"); W=/home/nand/.cache/nv2s/cd2030/mkr; pass=0; failn=0
ok() { if eval "$2"; then pass=$((pass+1)); echo "PASS $1"; else failn=$((failn+1)); echo "FAIL $1"; tail -n 3 "$W/C/runs/g9q-cd2030/r2030/STATUS" "$W/C/runs/conf2030/STATUS" 2>/dev/null | sed 's/^/     /'; fi; }
setup() {  # $1 = x exit, $2 = y exit, $3 = mode (ok|branchfail|md5bad|nojob|twohosts|hostend|h2herr|nohost)
  rm -rf "$W"; mkdir -p "$W/C/runs/g9q-cd2030/final" "$W/C/runs/g9q-cd1350/final" "$W/C/runs/conf2030" "$W/C/scripts" "$W/bin"
  C=$W/C; echo prereg > $C/runs/g9q-cd2030/PREREG.txt; chmod a-w $C/runs/g9q-cd2030/PREREG.txt
  echo b1350 > $C/runs/g9q-cd1350/final/01355036714.pt
  echo net2030 > $C/runs/g9q-cd2030/final/02035025982.pt
  m=$(md5sum < $C/runs/g9q-cd2030/final/02035025982.pt | cut -c1-8); [ "$3" = md5bad ] && m=deadbeef
  echo "13:40:00 G9q-cd2030 final at 2035025982 steps -> x (md5 $m); lr at the end: lr 7.00e-06" > $C/runs/g9q-cd2030/STATUS
  [ "$3" = branchfail ] && { rm $C/runs/g9q-cd2030/final/*.pt; echo "x FAILED" > $C/runs/g9q-cd2030/FAILED; }
  [ "$3" = nojob ] || echo 7777777 > $C/runs/conf2030/JOBID
  A=$C/runs/g9q-cd2030/final/02035025982.pt; B=$C/runs/g9q-cd1350/final/01355036714.pt
  local crc=0; [ "$3" = hostend ] && crc=1; [ "$3" = h2herr ] && crc=2
  cat > $C/scripts/netv2_cmp_step.sh <<CMP
K=\$3; n=\$(cat $W/cmpcalls 2>/dev/null || echo 0); echo \$((n + 1)) > $W/cmpcalls
if [ "$crc" = 1 ] && [ "\$n" = 0 ]; then touch $W/host1-ended; exit 143; fi
if [ "$crc" = 2 ]; then exit 1; fi
echo "G9s $A" > \$K/NETS.txt; echo "G9p $B" >> \$K/NETS.txt
echo "md5 g9s \$(md5sum < $A | cut -d' ' -f1) g9p \$(md5sum < $B | cut -d' ' -f1)" >> \$K/NETS.txt
{ echo "DECISION OF RECORD (nested pair-id order): test G9s > G9p exit $1 at look 9; test G9p > G9s exit $2 at look 3 (10 = SUCCESS, 20 = FUTILITY, 0 = undecided)"; echo "alongside x"
  echo "--- nested, G9s > G9p:"; echo "DECISION X"; echo "  estimate: median-unbiased 52.0%"; echo "--- nested, G9p > G9s:"; echo "--- round-robin, G9s > G9p:"
  echo "--- all pairs played (900):"; echo "A win rate 52.6% ± 1.7%"; } > \$K/RESULT.txt
echo "args \$* workers \${CMP_WORKERS:-?}" > \$K/args.txt
CMP
  # squeue: data hosts by mode; job ends; scontrol / scancel record calls
  local h1 h2; h1=$(date -d "+4 hours" +%FT%T); h2=$(date -d "+3 days" +%FT%T)
  [ "$3" = twohosts ] && h1=$(date -d "+1 hour" +%FT%T)
  cat > $W/bin/squeue <<SQ
#!/bin/bash
echo "squeue \$*" >> $W/calls.log
case "\$*" in
  *"-n ws-data-host -t RUNNING"*)
    case "$3" in nohost) ;; twohosts) echo "1174320 $h1"; echo "1545813 $h2" ;;
      hostend) if [ -e $W/host1-ended ]; then echo "1545813 $h2"; else echo "1174320 $h1"; fi ;;
      *) echo "1174320 $h1" ;; esac ;;
  *"-o %T %r"*) echo "PENDING JobHeldUser" ;;
  *"-j 1174320 -o %T"*) if [ -e $W/host1-ended ]; then echo COMPLETING; else echo RUNNING; fi ;;
  *"-o %T"*) echo RUNNING ;;
  *"-o %e"*) echo "$h1" ;;
esac
SQ
  cat > $W/bin/srun <<'SR'
#!/bin/bash
echo "srun $* | step ${SLURM_STEP_ID:-unset} job ${SLURM_JOB_ID:-unset}" >> MOCKW/calls.log
while [ $# -gt 0 ]; do case $1 in -J) shift 2; break ;; *) shift ;; esac; done
exec "$@"
SR
  sed -i "s|MOCKW|$W|" $W/bin/srun
  for c in scontrol scancel; do printf '#!/bin/bash\necho "%s $*" >> %s/calls.log\nexit 0\n' $c $W > $W/bin/$c; done
  chmod +x $W/bin/*
  export MOCK_C=$C MOCK_PRE_MD5=$(md5sum < $C/runs/g9q-cd2030/PREREG.txt | cut -c1-8) MOCK_B1350_MD5=$(md5sum < $B | cut -c1-8) MOCK_CMP_MD5=$(md5sum < $C/scripts/netv2_cmp_step.sh | cut -c1-8) MOCK_WAIT_S=4
}
run() { PATH=$W/bin:$PATH SLURM_JOB_ID=1483048 SLURM_STEP_ID=9 timeout 120 bash "$CH" > $W/out.log 2>&1; echo $?; }
gate() { sed -n 2p "$W/C/runs/g9q-cd2030/READING.txt" 2>/dev/null | grep -q '^R2030 IMPROVING: SUCCESS'; }
K() { echo "$W/C/runs/g9q-cd2030/r2030"; }
# 1 SUCCESS
setup 10 20 ok; rc=$(run)
ok "1a success rc 0" '[ $rc = 0 ]'
ok "1b READING r-- with SUCCESS line" '[ ! -w $W/C/runs/g9q-cd2030/READING.txt ] && gate'
ok "1c released 7777777, no scancel, no outbox" 'grep -qx "scontrol release 7777777" $W/calls.log && ! grep -q "^scancel" $W/calls.log && [ ! -e $W/C/LEDGER-OUTBOX.md ]'
ok "1d harness args step 0 seed 8203061 dirs, 5 workers" 'grep -qx "args 0 8203061 $(K) $W/C/runs/g9q-cd2030/final $W/C/runs/g9q-cd1350/final workers 5" $(K)/args.txt'
ok "1e h2h srun in host 1174320 with the GPU, clean Slurm env" 'grep -q "^srun --jobid=1174320 --overlap --ntasks=1 -c 6 --gres=gpu:1 --mem=14G -J nv2-r2030-h2h env CMP_WORKERS=5 timeout [0-9]* bash $(K)/cmp-step-copy.sh .* | step unset job unset$" $W/calls.log'
ok "1f reading names cd2030 / cd1350" 'grep -q "A = G9q-cd@2030 02035025982.pt" $W/C/runs/g9q-cd2030/READING.txt && grep -q "B = G9q-cd@1350 01355036714.pt" $W/C/runs/g9q-cd2030/READING.txt'
# 2 FUTILITY
setup 20 20 ok; rc=$(run)
ok "2a futility: gate closed" '[ $rc = 0 ] && ! gate && grep -q "^R2030 NOT SHOWN: \"cd2030 > cd1350\" FUTILITY" $W/C/runs/g9q-cd2030/READING.txt'
ok "2b scancel + WITHDRAWN" 'grep -qx "scancel 7777777" $W/calls.log && ! grep -q "scontrol release" $W/calls.log && grep -q "| netv2 | WITHDRAWN | 7777777 | F2030" $W/C/LEDGER-OUTBOX.md'
# 3 cd2030 weaker
setup 20 10 ok; rc=$(run)
ok "3 weaker warning + cancel" 'grep -q "cd2030 is weaker" $W/C/runs/g9q-cd2030/READING.txt && grep -qx "scancel 7777777" $W/calls.log'
# 4 undecided
setup 0 20 ok; rc=$(run)
ok "4 undecided => cancel" 'grep -q "undecided" $W/C/runs/g9q-cd2030/READING.txt && grep -qx "scancel 7777777" $W/calls.log'
# 5 branch FAILED
setup 10 20 branchfail; rc=$(run)
ok "5 branch FAILED => r2030/FAILED, no release/cancel" '[ $rc = 1 ] && [ -e $(K)/FAILED ] && ! grep -qE "^(scontrol|scancel)" $W/calls.log 2>/dev/null'
# 6 md5 mismatch
setup 10 20 md5bad; rc=$(run)
ok "6 md5 mismatch => FAILED, no READING" '[ $rc = 1 ] && [ -e $(K)/FAILED ] && [ ! -e $W/C/runs/g9q-cd2030/READING.txt ]'
# 7 no job id
setup 10 20 nojob; rc=$(run)
ok "7 no JOBID => READING, nothing released" '[ $rc = 0 ] && gate && ! grep -qE "^(scontrol|scancel)" $W/calls.log'
# 8 re-run with READING present => nothing
rc=$(run); ok "8 rerun is a no-op" '[ $rc = 0 ] && grep -q "READING.txt exists: nothing to do" $W/out.log'
# 9 two hosts: the one with < 1.5 h left is skipped
setup 10 20 twohosts; rc=$(run)
ok "9 picks the host with enough time" '[ $rc = 0 ] && grep -q "^srun --jobid=1545813 " $W/calls.log && ! grep -q "^srun --jobid=1174320 " $W/calls.log'
# 10 host ends during attempt 1 => attempt 2 in the successor
setup 10 20 hostend; rc=$(run)
ok "10 host hand-over: once more in 1545813, SUCCESS" '[ $rc = 0 ] && grep -q "^srun --jobid=1174320 " $W/calls.log && grep -q "^srun --jobid=1545813 " $W/calls.log && grep -q "once more in the next host" $(K)/STATUS && gate && grep -qx "scontrol release 7777777" $W/calls.log'
# 11 harness error with the host still running => FAILED, held
setup 10 20 h2herr; rc=$(run)
ok "11 h2h error => FAILED, one attempt, nothing released" '[ $rc = 1 ] && [ "$(cat $W/cmpcalls)" = 1 ] && grep -q "h2h exited 1 in host 1174320" $(K)/FAILED && ! grep -qE "^(scontrol|scancel)" $W/calls.log'
# 12 no data host at all => FAILED after the host wait
setup 10 20 nohost; rc=$(run)
ok "12 no host => FAILED" '[ $rc = 1 ] && grep -q "no RUNNING ws-data-host job" $(K)/FAILED && [ ! -e $W/cmpcalls ]'
# ---------------- conf2030.sbatch
csetup() {  # $1 = R2030 reading (success|futility), $2.. = harness modes per attempt (ok|low|void|mismatch|crash)
  setup 10 20 ok; [ "$1" = success ] && run > /dev/null; [ "$1" = futility ] && { setup 20 20 ok; run > /dev/null; }
  shift; printf '%s\n' "$@" > $W/fpmodes
  FQ=$W/FQ; mkdir -p $FQ/code $FQ/runs; echo chk > $FQ/code/fp_iters_check.py; echo wrap > $FQ/code/fp_fixed_iters.py
  cat > $FQ/code/fp9_iters.sbatch <<FPI
mode=\$(head -n 1 $W/fpmodes); sed -i 1d $W/fpmodes
echo "fp9 \$* | submit \$SLURM_SUBMIT_DIR | pwd \$(pwd)" >> $W/fpargs.log
sleep 1.1; OUT=$FQ/runs/foulplay-net-\$(date +%H%M%S%N)-\$SLURM_JOB_ID-200000; mkdir -p \$OUT
case \$mode in
  ok|low) p=59.1; [ \$mode = low ] && p=51.0; seq 906 > \$OUT/results.jsonl; echo "x all iteration counts OK" > \$OUT/ITERS_OK
     echo "\$OUT: \$p% ± 2.9% (453 duplicate pairs, 906 games; won both 138, split 259, lost both 56)" > \$OUT/result.txt ;;
  void) echo "Foul Play instance 0 did not start:" >&2; echo "TimeoutError: timed out during handshake" >&2; exit 1 ;;
  mismatch) seq 40 > \$OUT/results.jsonl; echo "x expected 200000, saw 7000" > \$OUT/ITERS_MISMATCH; echo "\$OUT: 55.0% ± 9.9% (20 duplicate pairs)" > \$OUT/result.txt ;;
  crash) echo "Traceback: boom" >&2; exit 1 ;;
esac
FPI
  export MOCK_FQ=$FQ MOCK_FPI_MD5=$(md5sum < $FQ/code/fp9_iters.sbatch | cut -c1-8) MOCK_FCHK_MD5=$(md5sum < $FQ/code/fp_iters_check.py | cut -c1-8) MOCK_FWRAP_MD5=$(md5sum < $FQ/code/fp_fixed_iters.py | cut -c1-8)
}
crun() { SLURM_JOB_ID=8888888 SLURM_CPUS_ON_NODE=24 timeout 120 bash "$SB" > $W/cout.log 2>&1; echo $?; }
CD() { echo "$W/C/runs/conf2030"; }
OB() { grep "| 8888888 |" $W/C/LEDGER-OUTBOX.md 2>/dev/null | tail -n 1; }
# C1 normal
csetup success ok; rc=$(crun)
ok "C1a CONFIRMED, RESULTS r--" '[ ! -w $(CD)/RESULTS.txt ] && grep -q "59.1% ± 2.9% .* => \[56.2, 62.0\] => CONFIRMED" $(CD)/RESULTS.txt'
ok "C1b launcher args: fixed 200000, seed 8203222, T 0.5, submit dir C" 'grep -q "^fp9 $W/C/runs/g9q-cd2030/final/02035025982.pt 1000 --gen 9 --seed 8203222 --fp-iters 200000 --temperature 0.5 | submit $W/C | pwd $W/C$" $W/fpargs.log && [ "$(wc -l < $W/fpargs.log)" = 1 ]'
ok "C1c RELEASE line with the budget" 'OB | grep -q "RELEASE | 8888888 | F2030 .*fixed 200k iterations) 59.1% ± 2.9% \[56.2, 62.0\] => CONFIRMED.*voids replayed 0; failed stages: none"'
# C2 VOID then OK
csetup success void ok; rc=$(crun)
ok "C2 void replayed once => CONFIRMED from attempt 2" '[ -e $(CD)/VOID-1 ] && [ ! -e $(CD)/VOID-2 ] && [ "$(wc -l < $W/fpargs.log)" = 2 ] && grep -q "CONFIRMED" $(CD)/RESULTS.txt && grep -q "infrastructure voids replayed: 1" $(CD)/RESULTS.txt && OB | grep -q "voids replayed 1; failed stages: none"'
# C3 two voids
csetup success void void; rc=$(crun)
ok "C3 two voids => FAILED-ii, nothing read" '[ -e $(CD)/FAILED-ii ] && [ -e $(CD)/VOID-2 ] && grep -q "two infrastructure voids" $(CD)/RESULTS.txt && ! grep -q "CONFIRMED" $(CD)/RESULTS.txt && OB | grep -q "failed stages: FAILED-ii"'
# C4 iteration mismatch: void, not replayed
csetup success mismatch ok; rc=$(crun)
ok "C4 ITERS_MISMATCH => FAILED-ii, not replayed" '[ -e $(CD)/FAILED-ii ] && [ "$(wc -l < $W/fpargs.log)" = 1 ] && grep -q "iteration check" $(CD)/RESULTS.txt && ! grep -q "CONFIRMED" $(CD)/RESULTS.txt'
# C5 gate closed
csetup futility ok; rc=$(crun)
ok "C5 R2030 not SUCCESS => nothing run, RELEASE did not run" '[ $rc = 1 ] && [ -e $(CD)/FAILED-gate ] && [ ! -e $W/fpargs.log ] && OB | grep -q "did not run"'
# C6 launcher md5 changed
csetup success ok; echo changed >> $W/FQ/code/fp_iters_check.py; rc=$(crun)
ok "C6 launcher md5 => FAILED-ii, nothing run" '[ -e $(CD)/FAILED-ii ] && [ ! -e $W/fpargs.log ] && grep -q "launcher md5" $(CD)/RESULTS.txt'
# C7 NOT CONFIRMED
csetup success low; rc=$(crun)
ok "C7 51.0 ± 2.9 => NOT CONFIRMED" 'grep -q "\[48.1, 53.9\] => NOT CONFIRMED" $(CD)/RESULTS.txt'
# C8 other harness failure: no replay
csetup success crash ok; rc=$(crun)
ok "C8 crash => FAILED-ii, no replay" '[ -e $(CD)/FAILED-ii ] && [ ! -e $(CD)/VOID-1 ] && [ "$(wc -l < $W/fpargs.log)" = 1 ]'
# C9 PREREG writable
csetup success ok; chmod u+w $W/C/runs/g9q-cd2030/PREREG.txt; rc=$(crun)
ok "C9 writable PREREG => nothing run" '[ $rc = 1 ] && [ -e $(CD)/FAILED-prereg ] && [ ! -e $W/fpargs.log ]'
echo "mock-r2030: $pass passed, $failn failed"
