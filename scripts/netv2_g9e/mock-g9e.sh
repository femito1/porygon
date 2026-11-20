#!/bin/bash
# Laptop mock of scripts/netv2_g9e/read.sh (the reading rule, resume, refusals) and park-p4.sh (the pointer park):
#   MOCK_ROOT=/home/nand/.cache/nv2s/g9e/mock SPD_H2H=<copy of vendor/wsp-ee76994/spd_h2h.py> bash scripts/netv2_g9e/mock-g9e.sh
# A stub spd (score pattern per experiment from MOCK_SCORES) writes the shards; the real spd_h2h.py merge pools them.
set -u
H=${MOCK_H:-$(cd "$(dirname "$0")" && pwd)}; M=${MOCK_ROOT:?}; SPDPY=${SPD_H2H:?}; PY3=${MOCK_PY:-python3}
pass=0; fail=0
ok() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL [$1]: $2"; fi; }
mkstub() {  # the stub spd: run --a A --b B --pairs P --shard J --seed S --out D ... -> D/shard-JJJJ.jsonl
  cat > "$M/spd-stub.sh" <<'STUB'
#!/bin/bash
shift; while [ $# -gt 0 ]; do case $1 in --a) a=$2; shift ;; --b) b=$2; shift ;; --pairs) p=$2; shift ;; --shard) j=$2; shift ;; --seed) s=$2; shift ;; --out) d=$2; shift ;; esac; shift; done
echo "$a|$b|$s" >> "$MOCK_CALLS"
n=$(basename "$d"); k=$(basename "$(dirname "$d")")
[ -e "$MOCK_DIE_AT" ] && [ "$(cat "$MOCK_DIE_AT")" = "$k/$n/$j" ] && { rm -f "$MOCK_DIE_AT"; exit 9; }
m=$(printf '%s\n' "$MOCK_SCORES" | tr ' ' '\n' | sed -n "s#^$k/$n:##p"); m=${m:-0.5}
python3 - "$d" "$j" "$p" "$s" "$m" "$a" "$b" <<'PYS'
import json, sys
d, j, p, s, m, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7]
with open(f"{d}/shard-{j:04d}.jsonl", "w") as f:
    for q in range(j * p, (j + 1) * p):
        x = ((q * 7919) % 2000) / 2000.0
        sc = 1.0 if x < m else 0.0
        g = [{"winner": "A" if sc == 1.0 else "B", "turns": 30, "void": False, "truncated": False}] * 2
        f.write(json.dumps({"exp": f"{a}|{b}|{s}", "pair": q, "seed": [s, q, 0, 0], "games": g, "score": sc}) + "\n")
PYS
STUB
  chmod +x "$M/spd-stub.sh"
}
tree() {  # tree K...: a fake cluster tree with the nets, PREREG and the cooled finals for the given K
  rm -rf "$M/c"; C=$M/c; mkdir -p "$C/runs/g9e" "$C/runs/g9q-cd1350/final" "$C/runs/g9q-cd800/final" "$C/runs/r1/20261120-112200-1174225/G9q/checkpoints" "$C/runs/g9/20261121-170000-1175374/G9E"
  echo cd1350 > "$C/runs/g9q-cd1350/final/01355036714.pt"; echo cd805 > "$C/runs/g9q-cd800/final/00805029873.pt"
  echo q1 > "$C/runs/r1/20261120-112200-1174225/G9q/checkpoints/ckpt_01560000000.pt"; echo q2 > "$C/runs/r1/20261120-112200-1174225/G9q/checkpoints/ckpt_01565000000.pt"
  echo prereg > "$C/runs/g9e/PREREG.txt"; chmod a-w "$C/runs/g9e/PREREG.txt"
  printf '2026-11-22 10:00:00 [eval @ 1,357,512,000] anchor=80.0%%  heuristic=90.0%%\n2026-11-22 12:00:00 [eval @ 1,380,100,000] anchor=79.5%%\n2026-11-22 14:00:00 [eval @ 1,455,100,000] anchor=78.0%%\n' > "$C/runs/g9/20261121-170000-1175374/G9E/train.log"
  for k in "$@"; do mkdir -p "$C/runs/g9e/cd$k/final"; f=$C/runs/g9e/cd$k/final/0145$k.pt; echo "cooled$k" > "$f"; chmod a-w "$f"; echo "$(basename "$f") $(md5sum < "$f" | cut -c1-8)" > "$C/runs/g9e/cd$k/FINAL.md5"; done
  B1=$(md5sum < "$C/runs/g9q-cd1350/final/01355036714.pt" | cut -c1-8); B8=$(md5sum < "$C/runs/g9q-cd800/final/00805029873.pt" | cut -c1-8)
  sed -e "s/B1350_MD5=6cf40c0c/B1350_MD5=$B1/" -e "s/B805_MD5=d0c5b6cf/B805_MD5=$B8/" "$H/read.sh" > "$M/read.sh"
}
rd() {  # rd K: run the (md5-patched) read.sh as netv2.env does
  ( cd "$C" && NV2=$C LOCAL=$M/local G9E_SPD=$M/spd-stub.sh G9E_SPD_MD5=$(md5sum < "$M/spd-stub.sh" | cut -c1-8) G9E_MERGE="$PY3 $SPDPY merge" \
    G9E_JPY=$PY3 MOCK_CALLS=$M/calls MOCK_DIE_AT=$M/die MOCK_SCORES=$MOCK_SCORES SLURM_JOB_ID=1175374 bash "$M/read.sh" "$1" ) > "$M/rd.out" 2>&1
}
v() { sed -n 's/^E1@[0-9]*M VERDICT \([A-Z]*\).*/\1/p' "$C/runs/g9e/r$1/READING.txt" 2> /dev/null; }
mkdir -p "$M"; mkstub; rm -f "$M/die"
# R1: E1 lower > 55 at 100M => EXPLOITABLE, final; info readings played; DONE; READING.txt r--
tree 100; : > "$M/calls"; MOCK_SCORES="r100/E1:0.62 r100/T1:0.66 r100/Q:0.52 r100/B805:0.55"; rd 100
ok R1a '[ "$(v 100)" = EXPLOITABLE ] && [ -f "$C/runs/g9e/DONE" ] && [ -f "$C/runs/g9e/r100/DONE" ] && [ ! -w "$C/runs/g9e/READING.txt" ] && [ ! -w "$C/runs/g9e/r100/READING.txt" ]'
ok R1b 'sed -n 2p "$C/runs/g9e/READING.txt" | grep -q "^FINAL: EXPLOITABLE" && [ ! -e "$C/runs/g9e/FAILED" ]'
ok R1c '[ "$(grep -c . "$M/calls")" = 80 ] && [ "$(grep -c "temperature=0.5|v2:$C/runs/g9q-cd1350/final/01355036714.pt,temperature=0.5|9150101$" "$M/calls")" = 20 ] && [ "$(grep -c "temperature=1.0|v2:$C/runs/g9q-cd1350/final/01355036714.pt,temperature=1.0|9150102$" "$M/calls")" = 20 ]'
ok R1d '[ "$(grep -c "ckpt_01565000000.pt,temperature=0.5|9150103$" "$M/calls")" = 20 ] && [ "$(grep -c "00805029873.pt,temperature=0.5|9150104$" "$M/calls")" = 20 ]'
ok R1e 'grep -q "^E1@100M 62.00% \[5[0-9.]*, 6[0-9.]*\] (pair-mean 95%, 2000 pairs = 4000 games, 0 void)" "$C/runs/g9e/r100/READING.txt"'
ok R1f 'grep -q "G9E +0M: \[eval @ 1357512000\]" "$C/runs/g9e/r100/INFO.txt" && grep -q "G9E +25M: \[eval @ 1380100000\]" "$C/runs/g9e/r100/INFO.txt" && grep -q "G9E +100M: \[eval @ 1455100000\]" "$C/runs/g9e/r100/INFO.txt"'
ok R1g 'grep -q "vs G9q.s newest checkpoint ckpt_01565000000.pt" "$C/runs/g9e/r100/INFO.txt" && grep -q "^specificity: E1 62.0000 vs G9q-now 52.0000 vs cd@805 55.0000" "$C/runs/g9e/r100/INFO.txt"'
# R2: upper < 55 at 100M => ROBUST, final
tree 100; : > "$M/calls"; MOCK_SCORES="r100/E1:0.50"; rd 100
ok R2a '[ "$(v 100)" = ROBUST ] && [ -f "$C/runs/g9e/DONE" ] && sed -n 2p "$C/runs/g9e/READING.txt" | grep -q "^FINAL: ROBUST" && grep -q "upper 95% 5[0-4]" "$C/runs/g9e/r100/READING.txt"'
# R3: 55 inside the interval at 100M => CONTINUE: interim, no runs/g9e/DONE; then 250M: lower > 55 => EXPLOITABLE / else ROBUST
tree 100; : > "$M/calls"; MOCK_SCORES="r100/E1:0.555 r250/E1:0.60"; rd 100
ok R3a '[ "$(v 100)" = CONTINUE ] && [ ! -e "$C/runs/g9e/DONE" ] && [ -f "$C/runs/g9e/r100/DONE" ] && sed -n 2p "$C/runs/g9e/READING.txt" | grep -q "^INTERIM: CONTINUE"'
mkdir -p "$C/runs/g9e/cd250/final"; f=$C/runs/g9e/cd250/final/01605040000.pt; echo cooled250 > "$f"; chmod a-w "$f"; echo "01605040000.pt $(md5sum < "$f" | cut -c1-8)" > "$C/runs/g9e/cd250/FINAL.md5"
: > "$M/calls"; rd 250
ok R3b '[ "$(v 250)" = EXPLOITABLE ] && [ -f "$C/runs/g9e/DONE" ] && sed -n 2p "$C/runs/g9e/READING.txt" | grep -q "^FINAL: EXPLOITABLE (E1@250M" && grep -q "^--- E1@100M:" "$C/runs/g9e/READING.txt" && grep -q "^--- E1@250M:" "$C/runs/g9e/READING.txt"'
ok R3c '[ "$(grep -c "01605040000.pt,temperature=0.5|v2:$C/runs/g9q-cd1350/final/01355036714.pt,temperature=0.5|9150101$" "$M/calls")" = 20 ]'
tree 100; MOCK_SCORES="r100/E1:0.555 r250/E1:0.56"; rd 100; mkdir -p "$C/runs/g9e/cd250/final"; f=$C/runs/g9e/cd250/final/01605040000.pt; echo cooled250 > "$f"; chmod a-w "$f"; echo "01605040000.pt $(md5sum < "$f" | cut -c1-8)" > "$C/runs/g9e/cd250/FINAL.md5"; rd 250
ok R3d '[ "$(v 250)" = ROBUST ] && grep -q "the 250M reading is final" "$C/runs/g9e/r250/READING.txt" && [ -f "$C/runs/g9e/DONE" ]'
# R4: cut off in the 7th E1 shard, then resumed: only the missing shards replayed, same verdict
tree 100; : > "$M/calls"; MOCK_SCORES="r100/E1:0.62"; echo r100/E1/6 > "$M/die"; rd 100
ok R4a '[ -e "$C/runs/g9e/FAILED" ] && [ ! -e "$C/runs/g9e/r100/READING.txt" ] && [ "$(ls "$C/runs/g9e/r100/E1"/shard-*.jsonl | wc -l)" = 6 ]'
rm -f "$C/runs/g9e/FAILED"; : > "$M/calls"; rd 100
ok R4b '[ "$(v 100)" = EXPLOITABLE ] && [ "$(grep -c "|9150101$" "$M/calls")" = 14 ] && [ ! -e "$C/runs/g9e/FAILED" ]'
# R5: refusals => runs/g9e/FAILED, no READING: writable PREREG, cooled final md5 mismatch, two finals, cd@1350 md5, spd md5
for bad in prereg finalmd5 twofinals b1350 spd; do
  tree 100; MOCK_SCORES="r100/E1:0.62"
  case $bad in prereg) chmod u+w "$C/runs/g9e/PREREG.txt" ;; finalmd5) echo "0145100.pt deadbeef" > "$C/runs/g9e/cd100/FINAL.md5" ;; twofinals) echo x > "$C/runs/g9e/cd100/final/0145999.pt" ;; b1350) chmod u+w "$C/runs/g9q-cd1350/final/01355036714.pt"; echo changed > "$C/runs/g9q-cd1350/final/01355036714.pt" ;; spd) echo "# changed" >> "$M/spd-stub.sh" ;; esac
  if [ $bad = spd ]; then ( cd "$C" && NV2=$C G9E_SPD=$M/spd-stub.sh G9E_SPD_MD5=00000000 bash "$M/read.sh" 100 ) > /dev/null 2>&1; else rd 100; fi
  ok "R5-$bad" '[ -e "$C/runs/g9e/FAILED" ] && [ ! -e "$C/runs/g9e/r100/READING.txt" ] && [ ! -e "$C/runs/g9e/READING.txt" ]'
  mkstub
done
# R6: the G9q checkpoint is chosen once per reading (a resumed reading keeps it although a newer one appeared)
tree 100; MOCK_SCORES="r100/E1:0.62"; echo r100/Q/3 > "$M/die"; rd 100; echo q3 > "$C/runs/r1/20261120-112200-1174225/G9q/checkpoints/ckpt_01570000000.pt"; rm -f "$C/runs/g9e/FAILED"; rd 100
ok R6 'grep -q "ckpt_01565000000.pt" "$C/runs/g9e/r100/NETS.txt" && grep -q "newest checkpoint ckpt_01565000000.pt" "$C/runs/g9e/r100/INFO.txt" && [ ! -e "$C/runs/g9e/FAILED" ]'
# --- park-p4.sh (pointer park)
ptree() {  # a fake tree: G9-P4's last job dir with final_model.pt at 645,000,000 + ckpt 640M, an older job dir
  rm -rf "$M/c"; C=$M/c; J=$C/runs/g9/20261118-164715-1173283/G9P4; mkdir -p "$J/checkpoints" "$J/pool/frozen" "$C/runs/g9/20261115-170149-1173274/G9P4/checkpoints" "$M/bin"
  echo old > "$C/runs/g9/20261115-170149-1173274/G9P4/checkpoints/ckpt_00240000000.pt"; touch -d '2026-11-18 16:40' "$C/runs/g9/20261115-170149-1173274/G9P4/checkpoints/ckpt_00240000000.pt"
  echo ck > "$J/checkpoints/ckpt_00640000000.pt"; touch -d '2026-11-21 15:00' "$J/checkpoints/ckpt_00640000000.pt"; echo s > "$J/pool/00640000000.pt"; echo b > "$J/best_model.pt"; echo e > "$J/ema_model.pt"
  printf '2026-11-18 16:50:00 v2 trainer: gen9 ...\n2026-11-21 16:26:00 upd 1 steps 645,000,000 (+41000) sps 1500\n2026-11-21 16:26:30 saved final_model.pt at 645,000,000 steps\n' > "$J/train.log"
  echo fin > "$J/final_model.pt"; touch -d '2026-11-21 16:27' "$J/final_model.pt"
  printf '#!/bin/bash\necho "${MOCK_SQ:-}"\n' > "$M/bin/squeue"; chmod +x "$M/bin/squeue"
  printf '#!/bin/bash\n[ -n "${MOCK_PYFAIL:-}" ] && exit 1\necho "steps ${MOCK_PYSTEPS:-645000000}"\n' > "$M/pypark"; chmod +x "$M/pypark"
}
pk() { ( cd "$C" && PATH=$M/bin:$PATH NV2=$C NV2_PARK_PY=$M/pypark SLURM_JOB_ID=1175374 bash "$H/park-p4.sh" ) > "$M/pk.out" 2>&1; }
ptree; pk; P=$C/runs/g9p4-parked
ok P1a '[ -f "$P/PARKED.txt" ] && [ ! -e "$P.tmp" ] && grep -qx "$J/final_model.pt 645000000 $(md5sum < "$J/final_model.pt" | cut -c1-8)" "$P/POINTER" && grep -q "G9P4 PARKED .* at 645000000 steps (final)" "$P/PARKED.txt"'
ok P1b '[ "$(grep -c . "$P/MD5SUMS")" = 3 ] && grep -q " best_model.pt$" "$P/MD5SUMS" && [ -w "$J/final_model.pt" ] && [ -w "$J/best_model.pt" ] && [ ! -e "$C/runs/g9e/FAILED" ]'
pk; ok P1c 'grep -q "already parked" "$C/runs/g9e/STATUS" && [ ! -e "$C/runs/g9e/FAILED" ]'
ptree; rm -f "$J/final_model.pt"; MOCK_PYSTEPS=640000000 pk
ok P2 'grep -q "at 640000000 steps (checkpoint)" "$C/runs/g9p4-parked/PARKED.txt" && grep -q "checkpoints/ckpt_00640000000.pt$" "$C/runs/g9p4-parked/MD5SUMS"'
for bad in stale running pyfail steps; do
  ptree
  case $bad in stale) printf '2026-11-21 16:30:00 upd 2 steps 645,041,000 (+41000)\n' >> "$J/train.log"; pk ;; running) MOCK_SQ=1188888 pk ;; pyfail) MOCK_PYFAIL=1 pk ;; steps) MOCK_PYSTEPS=1 pk ;; esac
  ok "P3-$bad" '[ ! -e "$C/runs/g9p4-parked" ] && [ -e "$C/runs/g9e/FAILED" ]'
done
[ -n "${MOCK_KEEP:-}" ] || rm -rf "$M/c" "$M/local"
echo "mock-g9e: passed $pass, failed $fail"
