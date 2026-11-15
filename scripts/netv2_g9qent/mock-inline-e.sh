#!/bin/bash
# Laptop mock of scripts/netv2_g9qent/inline-e1.sh and inline-e2.sh (+ scripts/netv2_g9qbf/park.sh): fake NV2 / LOCAL
# trees, a fake trainer wrapper (py-src-8892762) and torch check.   bash scripts/netv2_g9qent/mock-inline-e.sh [ROOT]
H=$(cd "$(dirname "$0")" && pwd); M=${1:-/tmp/nv2-qent-mock}; pass=0; fail=0
ok() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [$1]: $2"; fi; }
base() {
  rm -rf "$M"; N=$M/nv2; L=$M/local; mkdir -p "$N/runs/g9" "$N/bin" "$L/run"
  GQ=$N/runs/r1/20261115-111201-1135476/G9q; mkdir -p "$GQ/checkpoints" "$GQ/pool/frozen"
  echo ck875 > "$GQ/checkpoints/ckpt_00875040000.pt"; sleep 1; echo ck880 > "$GQ/checkpoints/ckpt_00880050000.pt"
  for x in 00877500000 00880037000 00882500000; do echo "s$x" > "$GQ/pool/$x.pt"; done
  for x in P3-g9-s2.pt best-00390042157.pt best-00881000000.pt; do echo "$x" > "$GQ/pool/frozen/$x"; done
  echo b > "$GQ/best_model.pt"; echo e > "$GQ/ema_model.pt"
  cat > "$N/bin/py-src-8892762" <<'T'
#!/bin/bash
rd=""; tot=""; prev=""; for a in "$@"; do [ "$prev" = --run-dir ] && rd=$a; [ "$prev" = --total-steps ] && tot=$a; prev=$a; done
mkdir -p "$rd"; echo "v2 trainer: fake" > "$rd/train.log"; echo "2026 upd 1 steps 1 (+1) sps 1 ent 0.41" >> "$rd/train.log"
echo "2026 saved final_model.pt at $((tot + 1234)) steps" >> "$rd/train.log"; echo x > "$rd/final_model.pt"; echo "$*" > "$rd/args.txt"
T
  chmod +x "$N/bin/py-src-8892762"
  printf '#!/bin/bash\necho "counters: {%ssteps%s: $FAKE_STEPS}"\n' "'" "'" > "$M/fakepy"; chmod +x "$M/fakepy"
}
e1() { ( cd "$N" && LOCAL=$L NV2=$N bash "$H/inline-e1.sh" ) > "$M/e1.log" 2>&1; }
# E1: after SKIP (MODE g9q2, G9q2 parked, HOLD) -> fork point, staged pool, MODE g9qent, HOLD lifted
base; echo g9q2 > "$N/runs/g9/MODE"; mkdir -p "$N/runs/g9q2-parked"; echo p > "$N/runs/g9q2-parked/PARKED.txt"; touch "$N/runs/g9/HOLD"; e1
E=$N/runs/g9qent; fp=""; s0=""; fm=""; read -r fp s0 fm < "$E/FORK_POINT"; D=$L/run/G9qent
ok e1a '[ "$(cat "$N/runs/g9/MODE")" = g9qent ] && [ ! -e "$N/runs/g9/HOLD" ] && [ "$s0" = 880050000 ] && [ "$fm" = "$(md5sum < "$fp" | cut -c1-8)" ] && [ ! -w "$fp" ] && [ "$(cat "$fp")" = ck880 ]'
ok e1b '[ -f "$D/.staged-880050000" ] && [ -f "$D/pool/00880037000.pt" ] && [ ! -e "$D/pool/00882500000.pt" ] && [ -f "$D/pool/frozen/best-00390042157.pt" ] && [ ! -e "$D/pool/frozen/best-00881000000.pt" ] && [ -f "$D/pool/frozen/P3-g9-s2.pt" ] && [ -f "$D/best_model.pt" ] && [ -f "$N/runs/g9/MODE.g9q2-before-g9qent" ]'
# E1 refusal: G9q2 not parked => FAILED, MODE and HOLD unchanged
base; echo g9q2 > "$N/runs/g9/MODE"; touch "$N/runs/g9/HOLD"; e1
ok e1c '[ "$(cat "$N/runs/g9/MODE")" = g9q2 ] && [ -f "$N/runs/g9/HOLD" ] && [ -f "$N/runs/g9qent-chain/FAILED" ] && [ ! -e "$N/runs/g9qent/FORK_POINT" ]'
# E2: the arm at its reading point -> park, arm-cd (schedule kept), G9q-cd (--ent-* stripped)
e2setup() {
  base; echo g9qent > "$N/runs/g9/MODE"; E=$N/runs/g9qent; mkdir -p "$E" "$N/runs/g9qent-chain"
  FP=$E/fork-point-00830040000.pt; echo fp > "$FP"; echo "$FP 830040000 $(md5sum < "$FP" | cut -c1-8)" > "$E/FORK_POINT"; echo x > "$E/PREREG.txt"
  touch "$N/runs/g9/INLINE_CMD.running"; sleep 1
  A=$L/run/G9qent; mkdir -p "$A/checkpoints" "$A/pool/frozen"; echo ck > "$A/checkpoints/ckpt_00880045000.pt"; echo fin > "$A/final_model.pt"
  for x in 00877500000 00880000000 00882500000; do echo "a$x" > "$A/pool/$x.pt"; done; echo f > "$A/pool/frozen/P3-g9-s2.pt"; echo f > "$A/pool/frozen/best-00881000000.pt"
  printf 'v2 trainer: x\n2026 upd 9 steps 880,100,000 (+42000) sps 1\nsaved final_model.pt at 880,100,000 steps\n' > "$A/train.log"
  printf '%s\0' /vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train --no-bf16 --guard-entropy-min 0.35 --snapshot-cache-cpu 200 --ent-schedule linear --ent-final 0.001 --ent-start 830040000 --ent-horizon 50000000 --resume "$FP" > "$N/runs/g9qent-chain/arm-argv.bin"
}
e2() { ( cd "$N" && LOCAL=$L NV2=$N QE_QUEUED=$N/runs/g9/INLINE_CMD.running QE_PARK=$M/../park.sh NV2_PARK_PY=$M/fakepy FAKE_STEPS=880100000 bash "$H/inline-e2.sh" ) > "$M/e2.log" 2>&1; }
cp "$H/../netv2_g9qbf/park.sh" "$M/../park.sh" 2>/dev/null || { mkdir -p "$M"; cp "$H/../netv2_g9qbf/park.sh" "$M/../park.sh"; }
e2setup; ( cd "$N" && LOCAL=$L NV2=$N QE_QUEUED=$N/runs/g9/INLINE_CMD.running QE_PARK=$M/../park.sh NV2_PARK_PY=$M/fakepy FAKE_STEPS=880100000 bash "$H/inline-e2.sh" ) > "$M/e2.log" 2>&1
ok e2a '[ -f "$N/runs/g9qent-parked/PARKED.txt" ] && [ ! -e "$N/runs/g9qent-chain/FAILED" ] && grep -q "INLINE #E2 done" "$N/runs/g9qent-chain/STATUS"'
fa=$(ls "$N"/runs/g9qent-cd/final/*.pt 2>/dev/null); fq=$(ls "$N"/runs/g9qent-g9qcd/final/*.pt 2>/dev/null)
ok e2b '[ "$(basename "$fa")" = 00885046234.pt ] && [ "$(basename "$fq")" = 00885051234.pt ] && [ ! -w "$fa" ] && [ ! -w "$fq" ]'
ok e2c 'grep -q -- "--ent-schedule linear --ent-final 0.001 --ent-start 830040000 --ent-horizon 50000000" "$N/runs/g9qent-cd/argv.txt" && ! grep -q -- "--ent-" "$N/runs/g9qent-g9qcd/argv.txt" && grep -q -- "--cooldown-start 880050000 --total-steps 885050000 --final-lr-frac 0.1 --guard-entropy-min 0.35" "$N/runs/g9qent-g9qcd/argv.txt" && grep -q -- "--resume $N/runs/g9qent-cd/branch-point-00880045000.pt --cooldown-start 880045000" "$N/runs/g9qent-cd/argv.txt"'
ok e2d 'grep -q "g9qent-cd final at 885046234 steps" "$N/runs/g9qent-chain/STATUS" && grep -q "g9qent-g9qcd final at 885051234 steps" "$N/runs/g9qent-chain/STATUS" && [ -f "$A/pool/00882500000.pt" ] && [ ! -d "$L/run/g9qent-cd/pool" ]'
# E2 park only (a guard trip): parked, no branches
e2setup; ( cd "$N" && LOCAL=$L NV2=$N QE_PARK_ONLY=1 QE_QUEUED=$N/runs/g9/INLINE_CMD.running QE_PARK=$M/../park.sh NV2_PARK_PY=$M/fakepy FAKE_STEPS=880100000 bash "$H/inline-e2.sh" ) > "$M/e2.log" 2>&1
ok e2e '[ -f "$N/runs/g9qent-parked/PARKED.txt" ] && [ ! -d "$N/runs/g9qent-cd/final" ] && grep -q "park only" "$N/runs/g9qent-chain/STATUS"'
echo "mock-inline-e: passed $pass, failed $fail"; if [ "$fail" = 0 ]; then rm -rf "$M" "$M/../park.sh"; fi
