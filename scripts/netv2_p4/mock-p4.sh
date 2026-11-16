#!/bin/bash
# Laptop mock of scripts/netv2_p4/inline-bridge.sh and argvcheck.py (fake NV2 / LOCAL trees):  bash mock-p4.sh [ROOT]
H=$(cd "$(dirname "$0")" && pwd); M=${1:-/tmp/nv2-p4-mock}; pass=0; fail=0
ok() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [$1]: $2"; fi; }
base() {
  rm -rf "$M"; N=$M/nv2; L=$M/local; mkdir -p "$N/runs/g9" "$N/runs/g9qent-parked" "$N/runs/g9qbf" "$L/run"
  echo g9qent > "$N/runs/g9/MODE"; echo parked > "$N/runs/g9qent-parked/PARKED.txt"; touch "$N/runs/g9/HOLD"
  FP=$N/runs/g9qbf/fork-point-00820017467.pt; echo fp > "$FP"; echo "$FP 820017467 $(md5sum < "$FP" | cut -c1-8)" > "$N/runs/g9qbf/FORK_POINT"; }
br() { ( cd "$N" && LOCAL=$L NV2=$N bash "$H/inline-bridge.sh" ) > "$M/br.log" 2>&1; }
unchanged() { [ "$(cat "$N/runs/g9/MODE")" = g9qent ] && [ -f "$N/runs/g9/HOLD" ] && [ -f "$N/runs/p4/FAILED" ] && [ ! -e "$N/runs/g9qbf/BRIDGE.txt" ]; }
base; br
ok b1 '[ "$(cat "$N/runs/g9/MODE")" = g9qbf ] && [ ! -e "$N/runs/g9/HOLD" ] && [ -f "$L/run/G9qbf/.staged-820017467" ] && [ -d "$L/run/G9qbf/pool/frozen" ] && [ -z "$(ls "$L/run/G9qbf/pool" | grep -v frozen)" ]'
ok b1b '[ "$(cat "$N/runs/g9/MODE.g9qent-before-p4bridge")" = g9qent ] && grep -q "BRIDGE" "$N/runs/g9qbf/BRIDGE.txt" && [ ! -e "$N/runs/p4/FAILED" ] && grep -q "MODE g9qbf written" "$N/runs/p4/STATUS"'
base; echo x > "$N/runs/g9qbf/G9QBF_STARTED"; br; ok b2 unchanged
base; echo g9q2 > "$N/runs/g9/MODE"; br; ok b3 '[ "$(cat "$N/runs/g9/MODE")" = g9q2 ] && [ -f "$N/runs/g9/HOLD" ] && [ -f "$N/runs/p4/FAILED" ]'
base; rm "$N/runs/g9qent-parked/PARKED.txt"; br; ok b4 unchanged
base; echo "$FP 820017467 deadbeef" > "$N/runs/g9qbf/FORK_POINT"; br; ok b5 unchanged
base; mkdir -p "$L/run/G9qbf"; br; ok b6 unchanged
base; mkdir -p "$N/runs/g9qbf-parked"; br; ok b7 unchanged
base; ( cd "$N" && NV2=$N bash "$H/inline-bridge.sh" ) > "$M/br.log" 2>&1; ok b8 unchanged
# argvcheck: the expected G9-P4 argv passes; a stray P3 role, a recipe change or a kept seed fail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; P3=$C/runs/pretrain/P3-g9-s2.pt; P4=$C/runs/pretrain/P4-g9-s2.pt
Q2="/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train --format gen9randombattle --ports 40384-40386 --n-battles 16 --pool-battles 16 --pool-dir /tmp/ws-netv2-1173274/pool --run-dir /tmp/ws-netv2-1173274/run --resume $C/runs/g9/20261112-015035-810367/final_model.pt --anchor $C/runs/pretrain/P1-g9h-dmg-s2.pt --total-steps 1000000000 --max-hours 70.5 --no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 --kl-anchor $P3 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P3 --p-frozen 0.67 --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 8 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir /tmp/ws-netv2-1173274/run/G9q2 --pool-dir /tmp/ws-netv2-1173274/run/G9q2/pool --snapshot-cache 512 --stacked-opp --snapshot-cache-cpu 200 --seed 7272027 --resume $P3"
QL="/vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train --format gen9randombattle --ports 43616-43625 --n-battles 60 --pool-battles 30 --pool-dir /tmp/ws-netv2-1135476/pool --run-dir /tmp/ws-netv2-1135476/run --resume /tmp/ws-netv2-1135476/run/final_model.pt --anchor $C/runs/pretrain/P1-g9h-dmg-s2.pt --total-steps 1000000000 --max-hours 70.5 --no-bf16 --n-epochs 3 --batch-size 4096 --lr 7e-5 --eval-async --guard-anchor-drop 0.15 --guard-entropy-rel 1.5 --guard-entropy-min 0.5 --guard-ptop-max 0.6 --kl-anchor $P3 --kl-anchor-coef 0.1 --kl-anchor-decay-steps 50000000 --critic-warmup-steps 1000000 --frozen $P3 --p-frozen 0.67 --p-baseline 0.08 --frozen-best --collector fastenv --fastenv-workers 28 --fastenv-battles 48 --no-overlap --n-battles 16 --pool-battles 16 --run-dir /tmp/ws-netv2-1135476/run/G9q --pool-dir /tmp/ws-netv2-1135476/run/G9q/pool --snapshot-cache 192 --stacked-opp --snapshot-cache-cpu 400 --resume /tmp/ws-netv2-1135476/run/G9q/final_model.pt"
mkdir -p "$M"; echo "$Q2 " > "$M/q2.txt"; echo "$QL " > "$M/ql.txt"
mk() { sed "$1" <<< "$Q2" | tr -d '\n' | tr ' ' '\0' > "$M/a.bin"; python3 -I "$H/argvcheck.py" "$M/a.bin" "$M/q2.txt" "$M/ql.txt" "$P4" "$P3" > "$M/ac.log" 2>&1; }
good="s#$P3#$P4#g; s#/G9q2#/G9P4#g; s# --seed 7272027##; s#--ports 40384-40386#--ports 41000-41002#; s#--resume $C/runs/g9/20261112-015035-810367/final_model.pt#--resume /tmp/ws-netv2-1173274/run/G9qbf/final_model.pt#"
mk "$good"; ok a1 '[ $? = 0 ] && grep -q "ARGV CHECK PASS" "$M/ac.log"'
mk "$good; s#--frozen $P4#--frozen $P3#"; ok a2 'grep -q "ARGV CHECK FAIL" "$M/ac.log"'
mk "$good; s#--kl-anchor-coef 0.1#--kl-anchor-coef 0.3#"; ok a3 'grep -q "ARGV CHECK FAIL" "$M/ac.log"'
mk "s#$P3#$P4#g; s#/G9q2#/G9P4#g"; ok a4 'grep -q "ARGV CHECK FAIL" "$M/ac.log" && grep -q "^--seed tokens: 1" "$M/ac.log"'  # a kept seed fails
mk "$good; s#--resume $P4\$#--resume /tmp/ws-netv2-1173274/run/G9P4/final_model.pt#"; ok a5 'grep -q "ARGV CHECK FAIL" "$M/ac.log" && grep -q "last --resume: /tmp" "$M/ac.log"'
mk "$good; s#--snapshot-cache-cpu 200#--snapshot-cache-cpu 200 --ent-schedule linear#"; ok a6 'grep -q "ARGV CHECK FAIL" "$M/ac.log"'
echo "mock-p4: passed $pass, failed $fail"; if [ "$fail" = 0 ]; then rm -rf "$M"; fi
