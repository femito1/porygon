#!/bin/bash
# laptop mock of r1350-chain.sh (+ conf1350's R1350 gate grep): stubs cmp_step, squeue, scontrol, scancel
set -u
CH=$1; SB=$2; W=/home/nand/.cache/nv2s/cd1350/mk; pass=0; failn=0
ok() { if eval "$2"; then pass=$((pass+1)); echo "PASS $1"; else failn=$((failn+1)); echo "FAIL $1"; fi; }
setup() {  # $1 = x exit, $2 = y exit, $3 = mode (ok|branchfail|md5bad|nojob)
  rm -rf "$W"; mkdir -p "$W/C/runs/g9q-cd1350/final" "$W/C/runs/g9q-cd800/final" "$W/C/runs/conf1350" "$W/C/scripts" "$W/bin"
  C=$W/C; echo prereg > $C/runs/g9q-cd1350/PREREG.txt; chmod a-w $C/runs/g9q-cd1350/PREREG.txt
  echo b805 > $C/runs/g9q-cd800/final/00805029873.pt
  echo net1350 > $C/runs/g9q-cd1350/final/01355012345.pt
  m=$(md5sum < $C/runs/g9q-cd1350/final/01355012345.pt | cut -c1-8); [ "$3" = md5bad ] && m=deadbeef
  echo "11:40:00 G9q-cd1350 final at 1355012345 steps -> x (md5 $m); lr at the end: lr 7.00e-06" > $C/runs/g9q-cd1350/STATUS
  [ "$3" = branchfail ] && { rm $C/runs/g9q-cd1350/final/*.pt; echo "x FAILED" > $C/runs/g9q-cd1350/FAILED; }
  [ "$3" = nojob ] || echo 7777777 > $C/runs/conf1350/JOBID
  A=$C/runs/g9q-cd1350/final/01355012345.pt; B=$C/runs/g9q-cd800/final/00805029873.pt
  cat > $C/scripts/netv2_cmp_step.sh <<CMP
K=\$3; echo "G9s $A" > \$K/NETS.txt; echo "G9p $B" >> \$K/NETS.txt
echo "md5 g9s \$(md5sum < $A | cut -d' ' -f1) g9p \$(md5sum < $B | cut -d' ' -f1)" >> \$K/NETS.txt
{ echo "DECISION OF RECORD (nested pair-id order): test G9s > G9p exit $1 at look 9; test G9p > G9s exit $2 at look 3 (10 = SUCCESS, 20 = FUTILITY, 0 = undecided)"; echo "alongside x"
  echo "--- nested, G9s > G9p:"; echo "DECISION X"; echo "  estimate: median-unbiased 52.0%"; echo "--- nested, G9p > G9s:"; echo "--- round-robin, G9s > G9p:"
  echo "--- all pairs played (900):"; echo "A win rate 52.6% ± 1.7%"; } > \$K/RESULT.txt
echo "args \$*" > \$K/args.txt
CMP
  for c in squeue scontrol scancel; do printf '#!/bin/bash\necho "%s $*" >> %s/calls.log\n[ "%s" = squeue ] && echo "PENDING JobHeldUser"\nexit 0\n' $c $W $c > $W/bin/$c; chmod +x $W/bin/$c; done
  export MOCK_C=$C MOCK_PRE_MD5=$(md5sum < $C/runs/g9q-cd1350/PREREG.txt | cut -c1-8) MOCK_B805_MD5=$(md5sum < $B | cut -c1-8) MOCK_CMP_MD5=$(md5sum < $C/scripts/netv2_cmp_step.sh | cut -c1-8) MOCK_WAIT_S=5
}
run() { PATH=$W/bin:$PATH bash "$CH" > $W/out.log 2>&1; echo $?; }
gate() { sed -n 2p "$W/C/runs/g9q-cd1350/READING.txt" 2>/dev/null | grep -q '^R1350 IMPROVING: SUCCESS'; }
# 1 SUCCESS
setup 10 20 ok; rc=$(run)
ok "1a success rc 0" '[ $rc = 0 ]'
ok "1b READING r-- with SUCCESS line" '[ ! -w $W/C/runs/g9q-cd1350/READING.txt ] && gate'
ok "1c released 7777777" 'grep -qx "scontrol release 7777777" $W/calls.log'
ok "1d no scancel" '! grep -q scancel $W/calls.log'
ok "1e no outbox line" '[ ! -e $W/C/LEDGER-OUTBOX.md ]'
ok "1f harness args step 0 seed 8135061" 'grep -q "^args 0 8135061 $W/C/runs/g9q-cd1350/r1350 $W/C/runs/g9q-cd1350/final $W/C/runs/g9q-cd800/final$" $W/C/runs/g9q-cd1350/r1350/args.txt'
ok "1g reading names cd1350/cd805" 'grep -q "A = G9q-cd@1350 01355012345.pt" $W/C/runs/g9q-cd1350/READING.txt && grep -q "B = G9q-cd@805 00805029873.pt" $W/C/runs/g9q-cd1350/READING.txt'
# 2 FUTILITY
setup 20 20 ok; rc=$(run)
ok "2a futility rc 0" '[ $rc = 0 ]'
ok "2b gate closed" '! gate && grep -q "^R1350 NOT SHOWN: \"cd1350 > cd805\" FUTILITY" $W/C/runs/g9q-cd1350/READING.txt'
ok "2c scancel 7777777" 'grep -qx "scancel 7777777" $W/calls.log && ! grep -q "scontrol release" $W/calls.log'
ok "2d WITHDRAWN line" 'grep -q "| netv2 | WITHDRAWN | 7777777 | F1350" $W/C/LEDGER-OUTBOX.md'
# 3 cd1350 weaker
setup 20 10 ok; rc=$(run)
ok "3 weaker warning + cancel" 'grep -q "cd1350 is weaker" $W/C/runs/g9q-cd1350/READING.txt && grep -qx "scancel 7777777" $W/calls.log'
# 4 undecided
setup 0 20 ok; rc=$(run)
ok "4 undecided => cancel" 'grep -q "undecided" $W/C/runs/g9q-cd1350/READING.txt && grep -qx "scancel 7777777" $W/calls.log'
# 5 branch FAILED
setup 10 20 branchfail; rc=$(run)
ok "5 branch FAILED => r1350/FAILED, no release/cancel" '[ $rc = 1 ] && [ -e $W/C/runs/g9q-cd1350/r1350/FAILED ] && [ ! -e $W/calls.log ] || ! grep -qE "release|scancel" $W/calls.log'
# 6 md5 mismatch
setup 10 20 md5bad; rc=$(run)
ok "6 md5 mismatch => FAILED, no READING" '[ $rc = 1 ] && [ -e $W/C/runs/g9q-cd1350/r1350/FAILED ] && [ ! -e $W/C/runs/g9q-cd1350/READING.txt ]'
# 7 no job id
setup 10 20 nojob; rc=$(run)
ok "7 no JOBID => READING, nothing released" '[ $rc = 0 ] && gate && ! grep -qE "release|scancel" $W/calls.log 2>/dev/null'
# 8 re-run with READING present => nothing
rc=$(run); ok "8 rerun is a no-op" '[ $rc = 0 ] && grep -q "READING.txt exists: nothing to do" $W/out.log'
# 9 conf1350 gate regex on the real sbatch text
ok "9 conf1350 has the same gate regex" "grep -qF \"grep -q '^R1350 IMPROVING: SUCCESS'\" $SB"
echo "mock-r1350: $pass passed, $failn failed"
