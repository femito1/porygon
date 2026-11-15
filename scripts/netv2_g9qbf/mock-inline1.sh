#!/bin/bash
# Laptop mock of scripts/netv2_g9qbf/inline1.sh + park.sh (fake NV2 / LOCAL tree, fake trainer wrapper and torch check):
# GO, SKIP, too-few-updates FAIL, stale-final park refusal.   bash scripts/netv2_g9qbf/mock-inline1.sh [ROOT]
H=$(cd "$(dirname "$0")" && pwd); M=${1:-/tmp/nv2-qbf-mock}; pass=0; fail=0
ok() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [$1]: $2"; fi; }
tree() {  # tree SPS_BF16 SPS_FP32 NUPD
  rm -rf "$M"; mkdir -p "$M/nv2/runs/g9" "$M/nv2/runs/g9q2" "$M/nv2/bin" "$M/local/run/G9q2/pool/frozen" "$M/local/run/G9q2/checkpoints"
  N=$M/nv2; L=$M/local; echo g9q2 > "$N/runs/g9/MODE"; echo x > "$N/runs/g9q2/PREREG.txt"
  GQ=$N/runs/r1/20261115-111201-1135476/G9q; mkdir -p "$GQ/checkpoints" "$GQ/pool/frozen"
  echo ck825 > "$GQ/checkpoints/ckpt_00825040000.pt"; sleep 1; echo ck830 > "$GQ/checkpoints/ckpt_00830040000.pt"
  for x in 00827500000 00830037000 00832500000; do echo "s$x" > "$GQ/pool/$x.pt"; done
  for x in P3-g9-s2.pt best-00390042157.pt best-00831000000.pt; do echo "$x" > "$GQ/pool/frozen/$x"; done
  echo b > "$GQ/best_model.pt"; echo e > "$GQ/ema_model.pt"
  touch "$N/runs/g9/INLINE_CMD.running"; sleep 1
  Q=$L/run/G9q2; echo q2 > "$Q/final_model.pt"; echo s > "$Q/pool/00050000000.pt"; echo f > "$Q/pool/frozen/P3-g9-s2.pt"; echo c > "$Q/checkpoints/ckpt_00050000000.pt"
  printf 'v2 trainer: x\n2026 upd 1200 steps 50,000,000 (+42000) sps 1\nsaved final_model.pt at 50,040,000 steps\n' > "$Q/train.log"
  mkdir -p "$N/runs/g9qbf-bench"
  printf '%s\0' /vol/bitbucket/fr1226/gloria/.venv/bin/python -u -m porygon.v2.train --format gen9randombattle --no-bf16 --n-epochs 3 --seed 7272027 --run-dir "$Q" --resume P3 > "$N/runs/g9qbf-bench/g9q2-argv.bin"
  cat > "$N/bin/py-src-24c633f" <<T
#!/bin/bash
rd=""; bf=1; prev=""; for a in "\$@"; do [ "\$prev" = --run-dir ] && rd=\$a; [ "\$a" = --no-bf16 ] && bf=0; [ "\$a" = --seed ] && echo SEED-PASSED >&2; prev=\$a; done
mkdir -p "\$rd"; s=$1; [ \$bf = 0 ] && s=$2
echo "v2 trainer: fake" > "\$rd/train.log"
for i in \$(seq 1 $3); do echo "2026 upd \$i steps \$((830040000 + i * 42000)) (+42000) sps \$s collect_sps 2600 games/s 30 lr 7e-05 upd_s 12.5 | x" >> "\$rd/train.log"; done
echo x > "\$rd/final_model.pt"
T
  chmod +x "$N/bin/py-src-24c633f"
  printf '#!/bin/bash\necho "counters: {%ssteps%s: 50040000}"\n' "'" "'" > "$M/fakepy"; chmod +x "$M/fakepy"
}
run() { ( cd "$N" && LOCAL=$L NV2=$N QBF_QUEUED=$N/runs/g9/INLINE_CMD.running QBF_PARK=$H/park.sh QBF_PY=python3 NV2_PARK_PY=$M/fakepy bash "$H/inline1.sh" ) > "$M/out.log" 2>&1; }
# GO: 1700 / 1450 = 1.172
tree 1700 1450 26; run
F=$N/runs/g9qbf; fp=""; s0=""; fm=""; read -r fp s0 fm < "$F/FORK_POINT"
ok go1 '[ "$(cat "$N/runs/g9/MODE")" = g9qbf ] && grep -q "^DECISION GO" "$N/runs/g9qbf-bench/RESULTS.txt" && [ ! -w "$N/runs/g9qbf-bench/RESULTS.txt" ]'
ok go2 '[ -f "$N/runs/g9q2-parked/PARKED.txt" ] && [ -f "$N/runs/g9q2-parked/MD5SUMS" ] && grep -q "PLACEMENT DECISION" "$N/runs/g9q2/notes.log" && grep -q "at 50040000 steps" "$N/runs/g9q2-parked/PARKED.txt"'
ok go3 '[ "$s0" = 830040000 ] && [ "$fp" = "$F/fork-point-00830040000.pt" ] && [ "$fm" = "$(md5sum < "$fp" | cut -c1-8)" ] && [ ! -w "$fp" ] && [ "$(cat "$fp")" = ck830 ]'
D=$L/run/G9qbf
ok go4 '[ -f "$D/.staged-830040000" ] && [ -f "$D/pool/00827500000.pt" ] && [ -f "$D/pool/00830037000.pt" ] && [ ! -e "$D/pool/00832500000.pt" ] && [ -f "$D/pool/frozen/P3-g9-s2.pt" ] && [ -f "$D/pool/frozen/best-00390042157.pt" ] && [ ! -e "$D/pool/frozen/best-00831000000.pt" ] && [ -f "$D/best_model.pt" ] && [ -f "$D/ema_model.pt" ]'
ok go5 'grep -q "RATIO median sps bf16 / fp32 = 1700 / 1450 = 1.172" "$N/runs/g9qbf-bench/RESULTS.txt" && grep -q "bf16: 23 measured updates (of 26" "$N/runs/g9qbf-bench/RESULTS.txt"'
ok go6 '! grep -q -- "--no-bf16" "$N/runs/g9qbf-bench/bf16-argv.txt" && grep -q -- "--no-bf16" "$N/runs/g9qbf-bench/fp32-argv.txt" && ! grep -q -- "--seed" "$N/runs/g9qbf-bench/bf16-argv.txt" && ! grep -q SEED-PASSED "$M/out.log" && grep -q -- "--resume $fp --total-steps 831140000" "$N/runs/g9qbf-bench/bf16-argv.txt"'
ok go7 '[ ! -d "$L/bench-bf16" ] && [ ! -d "$L/bench-fp32" ] && [ ! -e "$N/runs/g9qbf-bench/FAILED" ] && [ ! -e "$F/SKIP.txt" ]'
# SKIP: 1600 / 1450 = 1.103
tree 1600 1450 26; run
ok skip1 '[ "$(cat "$N/runs/g9/MODE")" = g9q2 ] && grep -q "^DECISION SKIP" "$N/runs/g9qbf-bench/RESULTS.txt" && [ -f "$N/runs/g9qbf/SKIP.txt" ] && [ ! -d "$L/run/G9qbf" ] && [ -f "$N/runs/g9q2-parked/PARKED.txt" ]'
# FAIL: only 10 updates per arm
tree 1700 1450 10; run
ok fail1 '[ "$(cat "$N/runs/g9/MODE")" = g9q2 ] && [ -f "$N/runs/g9qbf-bench/FAILED" ] && grep -q "DECISION FAIL" "$N/runs/g9qbf-bench/RESULTS.txt" && [ ! -e "$N/runs/g9qbf/SKIP.txt" ]'
# park refusal (a stale final, older than the queued INLINE) => nothing parked, MODE g9q2, FAILED, no fork point
tree 1700 1450 26; touch -d '2 hours ago' "$L/run/G9q2/final_model.pt"; run
ok stale1 '[ ! -e "$N/runs/g9q2-parked" ] && [ "$(cat "$N/runs/g9/MODE")" = g9q2 ] && [ -f "$N/runs/g9qbf-bench/FAILED" ] && [ ! -e "$N/runs/g9qbf/FORK_POINT" ]'
echo "mock-inline1: passed $pass, failed $fail"; if [ "$fail" = 0 ]; then rm -rf "$M"; fi
