#!/bin/bash
# Laptop mock of run_vm.sh's control flow (no GPU, no torch, no data): a fake $HOME with stub python / nvidia-smi /
# sudo / node, fake shards and fake h2h results. Cases:
#   1 E1A BIGGER BETTER, E4 BIGGER BETTER        => OVERALL CONFIRMED, control S trained, DONE, shutdown called
#   2 E1A NO DIFFERENCE                          => no control, OVERALL NO DIFFERENCE SHOWN, DONE
#   3 resume: stop after the train stage of case-2 layout (kill), rerun => no retraining, eval shards kept
#   4 budget guard: projected time over the limit => HOLD, no further stage, rerun does nothing
#   5 md5 mismatch => FAILED, rerun does nothing
#   bash mock_vm.sh   (prints PASS/FAIL per case; scratch under ~/.cache/nv2s/bignet-mock)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${MOCK_ROOT:-$HOME/.cache/nv2s/bignet-mock}
REALPY=$(command -v python3)
pass=0; failn=0
export MOCK_BC_P4_MD5=$(printf fake-bcp4 | md5sum | cut -c1-32)
setup_home() {
  local H=$ROOT/$1; rm -rf "$H"; mkdir -p "$H/bin" "$H/venv/bin" "$H/porygon/scripts/wsp" "$H/porygon/src-24c633f/porygon/v2" \
    "$H/porygon/pokemon-showdown" "$H/porygon/spd/fastenv-x/fastenv/js" "$H/node/bin" "$H/bignet/src-p2b" "$H/bignet/nets" \
    "$H/bignet/data/p4/v2pre-human-shuf"
  touch "$H/porygon/scripts/wsp/spd_h2h.py" "$H/porygon/src-24c633f/porygon/v2/model.py" "$H/porygon/spd/fastenv-x/fastenv/js/host.js"
  printf '#!/bin/bash\necho v22.22.0\n' > "$H/node/bin/node"
  printf '#!/bin/bash\ncase "$*" in *name*) echo "Tesla T4";; *) echo "1000 MiB, 50 %%";; esac\n' > "$H/bin/nvidia-smi"
  printf '#!/bin/bash\necho "sudo $*" >> %s/sudo.log; case "$1" in tee) cat > /dev/null;; esac; exit 0\n' "$H" > "$H/bin/sudo"
  echo 745c479 > "$H/bignet/src-p2b/GIT_COMMIT"
  printf 'fake-bcp4' > "$H/bignet/nets/P4-g9-s2.pt"
  ( cd "$H/bignet/data/p4/v2pre-human-shuf" && for i in $(seq -f %05g 0 500); do echo $i > shard-$i.npz; done
    for i in $(seq -f %04g 0 31); do echo v$i > shard-val-$i.npz; done; echo '{}' > meta.json )
  ( cd "$H/bignet/data/p4" && md5sum v2pre-human-shuf/* > MD5SUMS.shuf )
  echo "mock push" > "$H/bignet/data/PUSH_DONE"
  cp "$HERE/run_vm.sh" "$H/bignet/"; touch "$H/bignet/microbench.py" "$H/bignet/pretrain_resumable.py"
  cat > "$H/venv/bin/python" <<EOF
#!/bin/bash
# stub: real python for inline code, fakes for the three scripts
if [ "\$1" = -c ] || [ "\$1" = - ]; then
  if [ "\$1" = -c ] && [[ "\$2" == *torch* ]]; then [[ "\$2" == *version* ]] && echo 2.15-mock; exit 0; fi
  exec $REALPY "\$@"; fi
[ "\$1" = -u ] && shift
case "\$1" in
  */microbench.py) out=\$(echo "\$@" | grep -o -- '--out [^ ]*' | cut -d' ' -f2)
    for c in base D384L4; do st=0.25; [ \$c = D384L4 ] && st=0.36
      echo "{\"cfg_name\": \"\$c\", \"params\": 1, \"tf32/train_step_s\": \$st, \"tf32/train_samples_per_s\": 2000, \"fp16/train_samples_per_s\": 3000, \"tf32/fwd_bs512_decisions_per_s\": 5000, \"tf32/fwd_bs64_decisions_per_s\": 2000, \"tf32/peak_mem_gib\": 2.0}" >> "\$out"; done ;;
  */pretrain_resumable.py) out=\$(echo "\$@" | grep -o -- '--out [^ ]*' | cut -d' ' -f2)
    keep=\$(echo "\$@" | grep -o -- '--keep-steps [0-9]*' | cut -d' ' -f2)
    echo "train \$*" >> $H/train_calls.log
    sps=\${MOCK_SPS:-2500}
    echo "{\"event\": \"train\", \"step\": 5200, \"samples_per_s\": \$sps}" >> "\$out/pretrain.log"
    [ -n "\${MOCK_TRAIN_SLEEP:-}" ] && sleep "\$MOCK_TRAIN_SLEEP"
    echo '{"event": "val", "step": 140880, "policy_ce": 1.12, "value_bce": 0.58, "top1_vs_target_argmax": 0.58}' >> "\$out/pretrain.log"
    echo final > "\$out/final.pt"; [ -n "\$keep" ] && echo k > "\$out/keep-\$(printf %07d \$keep).pt"; exit 0 ;;
  */spd_h2h.py) shift
    if [ "\$1" = run ]; then n=\$(echo "\$@" | grep -o -- '--pairs [0-9]*' | cut -d' ' -f2); k=\$(echo "\$@" | grep -o -- '--shard [0-9]*' | cut -d' ' -f2)
      out=\$(echo "\$@" | grep -o -- '--out [^ ]*' | cut -d' ' -f2); echo "h2h \$*" >> $H/h2h_calls.log; [ -n "\${MOCK_H2H_SLEEP:-}" ] && sleep "\$MOCK_H2H_SLEEP"
      for i in \$(seq 1 \$n); do echo '{}'; done > "\$out/shard-\$(printf %04d \$k).jsonl"
    else d=\$2; nm=\$(basename \$d); wr=\$(eval echo \\\${MOCK_WR_\$nm:-0.5})
      n=\$(cat \$d/shard-*.jsonl | wc -l)
      echo "{\"pairs\": \$n, \"games\": \$((2*n)), \"win_rate\": \$wr, \"ci95\": 0.012, \"void_games\": 0, \"line\": \"mock\"}" > "\$d/summary.json"; fi ;;
esac
EOF
  chmod +x "$H/bin/"* "$H/venv/bin/python" "$H/node/bin/node"
  echo "$H"
}
runit() { HOME=$1 PATH=$1/bin:$1/node/bin:$PATH MEM_MIN_GB=1 timeout 300 bash "$1/bignet/run_vm.sh" run > "$1/run.out" 2>&1; }
check() { if eval "$2"; then echo "PASS $1"; pass=$((pass+1)); else echo "FAIL $1"; failn=$((failn+1)); fi; }

H=$(setup_home c1); export MOCK_WR_e1a=0.53 MOCK_WR_e4=0.52; runit "$H"
check "1 confirmed" 'grep -q "OVERALL: CONFIRMED" $H/bignet/RESULT.txt && [ -f $H/bignet/DONE ] && grep -q "shutdown -h now" $H/sudo.log && [ $(grep -c "^train" $H/train_calls.log) = 2 ] && grep -q -- "--seed 3" $H/train_calls.log && [ $(cat $H/bignet/eval/e1a/shard-*.jsonl | wc -l) = 4000 ] && [ -f $H/bignet/eval/e0/summary.json ]'
check "1 keep step = 140880*0.25/0.36 floored to 1000" 'grep -q -- "--keep-steps 97000" $H/train_calls.log && [ -f $H/bignet/eval/e2a/summary.json ]'
unset MOCK_WR_e1a MOCK_WR_e4

H=$(setup_home c2); export MOCK_WR_e1a=0.505; runit "$H"
check "2 no difference" 'grep -q "OVERALL: NO DIFFERENCE SHOWN" $H/bignet/RESULT.txt && [ $(grep -c "^train" $H/train_calls.log) = 1 ] && [ ! -d $H/bignet/eval/e4 ] && grep -q "control S not run" $H/bignet/STATUS'
unset MOCK_WR_e1a

H=$(setup_home c3); export MOCK_WR_e1a=0.47
HOME=$H PATH=$H/bin:$H/node/bin:$PATH MEM_MIN_GB=1 MOCK_TRAIN_SLEEP=2 MOCK_H2H_SLEEP=2 bash "$H/bignet/run_vm.sh" run > "$H/run1.out" 2>&1 &
p=$!; for i in $(seq 600); do [ -f "$H/bignet/TRAIN_DONE_D384L4-s2" ] && [ -s "$H/bignet/eval/e1a/shard-0000.jsonl" ] && break; sleep 0.2; done; kill $p; wait $p 2>/dev/null; sleep 3; rm -f "$H/bignet/eval/e1a/shard-0003.jsonl"; : > "$H/h2h_calls.log"
runit "$H"
check "3 resume" '[ $(grep -c "^train" $H/train_calls.log) = 1 ] && grep -q "OVERALL: BIGGER WORSE" $H/bignet/RESULT.txt && ! grep -- "--shard 0 " $H/h2h_calls.log | grep -q e1a && grep -- "--shard 3 " $H/h2h_calls.log | grep -q e1a && [ -f $H/bignet/DONE ]'
unset MOCK_WR_e1a

H=$(setup_home c4); MOCK_SPS=500 MOCK_TRAIN_SLEEP=90 runit "$H"
check "4 budget hold" '[ -f $H/bignet/HOLD ] && [ ! -f $H/bignet/DONE ] && grep -q "projected" $H/bignet/HOLD && grep -q "shutdown -h now" $H/sudo.log'
: > "$H/sudo.log"; runit "$H"
check "4 rerun with HOLD does nothing" '! grep -q shutdown $H/sudo.log && grep -q "HOLD present" $H/run.out'

H=$(setup_home c5); echo corrupt > "$H/bignet/data/p4/v2pre-human-shuf/shard-00007.npz"; runit "$H"
check "5 md5 fail" '[ -f $H/bignet/FAILED ] && grep -q "md5 mismatch" $H/bignet/FAILED && [ ! -f $H/bignet/MB_OK ]'
echo "mock: $pass pass, $failn fail"
[ $failn = 0 ]
