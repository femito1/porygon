#!/bin/bash
# laptop mock of verdict-chain.sh: fake council/p1-tera (PREREG, finals, ARMS_DONE), a fake spd_h2h_tera.py (run = fake
# shard records with a chosen score pattern; merge = the REAL spd_h2h_tera.merge), stub squeue / srun (two data hosts).
#   bash mock-verdict.sh scripts/netv2_p1tera/verdict-chain.sh scripts/netv2_p1tera/spd_h2h_tera.py PYTHON
set -u
DRV=$(readlink -f "$1"); REAL=$(readlink -f "$2"); PYB=$3; W=/home/nand/.cache/nv2s/p1tera/mkv; pass=0; failn=0
ok() { if eval "$2"; then pass=$((pass+1)); echo "PASS $1"; else failn=$((failn+1)); echo "FAIL $1"; tail -n 3 "$W/C/council/p1-tera/h2h/STATUS" 2>/dev/null | sed 's/^/     /'; fi; }
setup() {  # $1 = scores per pair (e.g. "1 0.5" cycles) ; $2 = mode: ok | prew | md5bad | failed | armsfail | hostdie | nodone
  rm -rf "$W"; mkdir -p "$W/C/council/p1-tera/E/final" "$W/C/council/p1-tera/C/final" "$W/C/scripts/netv2_p1tera" "$W/bin" "$W/T/src" "$W/T/fastenv/js"
  P=$W/C/council/p1-tera
  echo prereg > $P/PREREG.txt; [ "$2" = prew ] || chmod a-w $P/PREREG.txt
  echo e > $P/E/final/02245050000.pt; echo c > $P/C/final/02245060000.pt; chmod a-w $P/*/final/*.pt
  me=$(md5sum < $P/E/final/02245050000.pt | cut -c1-8); mc=$(md5sum < $P/C/final/02245060000.pt | cut -c1-8)
  [ "$2" = md5bad ] && me=deadbeef
  echo "$P/E/final/02245050000.pt $me 2245050000" > $P/E/FINAL; echo "$P/C/final/02245060000.pt $mc 2245060000" > $P/C/FINAL
  [ "$2" = nodone ] || [ "$2" = armsfail ] || cat $P/E/FINAL $P/C/FINAL > $P/ARMS_DONE
  [ "$2" = failed ] && echo "x" > $P/FAILED
  # the fake harness: run -> records; merge -> the real merge
  cat > $W/C/scripts/netv2_p1tera/spd_h2h_tera.py <<PYF
import importlib.util, json, os, sys
from pathlib import Path
a = sys.argv[1:]
if a[0] == "merge":
    sp = importlib.util.spec_from_file_location("real", "$REAL"); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    s = m.merge([a[1]], out=a[a.index("--out") + 1]); print(s["line"]); sys.exit(0)
g = lambda k: a[a.index(k) + 1]
shard, n, out = int(g("--shard")), int(g("--pairs")), Path(g("--out")); out.mkdir(parents=True, exist_ok=True)
if os.environ.get("MOCK_DIE_AT") == str(shard) and not Path("$W/died").exists():
    Path("$W/died").write_text("x"); sys.exit(3)
sc = [float(x) for x in os.environ["MOCK_SCORES"].split()]
with open(out / f"shard-{shard:04d}.jsonl", "w") as f:
    for p in range(shard * n, (shard + 1) * n):
        s = sc[p % len(sc)]
        gm = [{"winner": "A" if s == 1 else "B" if s == 0 else ("A" if p % 2 else "B"), "turns": 30, "void": False, "truncated": False}] * 2
        if s == 0.5: gm = [{"winner": "A", "turns": 30, "void": False, "truncated": False}, {"winner": "B", "turns": 30, "void": False, "truncated": False}]
        f.write(json.dumps({"exp": "x", "pair": p, "seed": [p, 0, 0, 0], "team_seed": [[0]*4, [0]*4], "games": gm, "score": s, "shard": shard, "worker": 0}) + "\n")
print("shard", shard, "ok")
PYF
  sed -i "1i #!$PYB" $W/C/scripts/netv2_p1tera/spd_h2h_tera.py
  local end1 end2; end1=$(date -d "+20 hours" +%FT%T); end2=$(date -d "+40 hours" +%FT%T)
  printf '#!/bin/bash\nif [ "$1" = -h ] && [ "$2" = -j ]; then echo RUNNING; exit 0; fi\nif [ -f %s/host1-gone ]; then echo "222 %s"; else echo "111 %s"; echo "222 %s"; fi\n' "$W" "$end2" "$end1" "$end2" > $W/bin/squeue
  # srun stub: the command after -J NAME; host 222 = the longest-lived; MOCK_DIE_AT makes host 222's first pass die (then host 222 "ends": next pick 111? no: 222 stays)
  printf '#!/bin/bash\nwhile [ $# -gt 0 ] && [ "$1" != -J ]; do echo "$1" >> %s/srun.args; shift; done; shift 2\nexec "$@"\n' "$W" > $W/bin/srun
  chmod +x $W/bin/*
  export MOCK_C=$W/C MOCK_PRE_MD5=$(md5sum < $P/PREREG.txt | cut -c1-8) MOCK_WAIT_S=6 MOCK_SRUN=$W/bin/srun MOCK_T=$W/T MOCK_SCORES="$1"
  export MOCK_H2H_MD5=$(md5sum < $W/C/scripts/netv2_p1tera/spd_h2h_tera.py | cut -c1-8)
  unset MOCK_DIE_AT; [ "$2" = hostdie ] && export MOCK_DIE_AT=4
  return 0
}
run() { PATH=$W/bin:$PATH PY_FAKE=1 timeout 300 bash -c "sed 's#^G=/vol/bitbucket/fr1226/gloria; PY=\$G/.venv/bin/python#G=/x; PY=$PYB#' '$DRV' > '$W/drv.sh'; bash '$W/drv.sh'" > $W/out.log 2>&1; echo $?; }
V() { echo "$W/C/council/p1-tera/VERDICT.txt"; }
# 1 GO: scores 1,1,0.5,0 -> mean 62.5%, lower far above 50.5
setup "1 1 0.5 0" ok; rc=$(run)
ok "1a GO: rc 0, VERDICT line 1 GO, r--" '[ $rc = 0 ] && [ "$(head -n 1 $(V))" = GO ] && [ ! -w $(V) ]'
ok "1b line 2 = pair-mean and CI over 4000 pairs" 'sed -n 2p $(V) | grep -q "^E-cd vs C-cd pair-mean 62.50% \[.*\] (95%, +- .*; 4000 duplicate pairs = 8000 games"'
ok "1c 10 shards of 400, harness md5 checked, srun into host 222 (most time left) with 6 CPUs + GPU" '[ $(ls $W/C/council/p1-tera/h2h/shards/shard-*.jsonl | wc -l) = 10 ] && grep -q -- "--jobid=222" $W/srun.args && grep -q -- "-c" $W/srun.args && grep -q -- "--gres=gpu:1" $W/srun.args'
ok "1d NETS.txt names both finals + md5s, seed 9241103" 'grep -q "^A E-cd .*/E/final/02245050000.pt" $W/C/council/p1-tera/h2h/NETS.txt && grep -q "seed 9241103 pairs 4000 shards 10x400" $W/C/council/p1-tera/h2h/NETS.txt'
ok "1e no FAILED" '[ ! -e $W/C/council/p1-tera/FAILED ]'
# 2 REPLICATE: mean just above 50: lower in (50.0, 50.5]
setup "$(python3 -c "print(' '.join(['1']*1036 + ['0']*964))")" ok; rc=$(run)
ok "2 REPLICATE when the lower bound is in (50.0, 50.5]" '[ $rc = 0 ] && [ "$(head -n 1 $(V))" = REPLICATE ] && lo=$(sed -n 2p $(V) | sed -n "s/.*\[\([0-9.]*\),.*/\1/p") && python3 -c "import sys; lo=float(\"$lo\"); sys.exit(0 if 50.0 < lo <= 50.5 else 1)"'
# 3 PARK: mean 50
setup "1 0" ok; rc=$(run)
ok "3 PARK at 50%" '[ $rc = 0 ] && [ "$(head -n 1 $(V))" = PARK ]'
# 4 FAILED present before: refuse
setup "1 0" failed; rc=$(run)
ok "4 FAILED present => exit 1, no VERDICT" '[ $rc = 1 ] && [ ! -e $(V) ]'
# 5 arms fail while waiting
setup "1 0" armsfail; ( sleep 2; echo "arm E: x" >> $W/C/council/p1-tera/FAILED ) & rc=$(run)
ok "5 arms FAILED while waiting => exit 1, no VERDICT" '[ $rc = 1 ] && [ ! -e $(V) ] && grep -q "arms FAILED" $W/C/council/p1-tera/h2h/STATUS'
# 6 md5 mismatch of a final
setup "1 0" md5bad; rc=$(run)
ok "6 final md5 mismatch => FAILED" '[ $rc = 1 ] && grep -q "md5 is not deadbeef" $W/C/council/p1-tera/FAILED && [ ! -e $(V) ]'
# 7 a shard fails on host 222's first pass: retry once (host 222 still RUNNING per the stub => that is a failure, not a retry)
setup "1 1 0.5 0" hostdie; rc=$(run)
ok "7 a failed pass on a RUNNING host => FAILED (no silent retry), complete shards kept" '[ $rc = 1 ] && grep -q "h2h exited" $W/C/council/p1-tera/FAILED && [ $(ls $W/C/council/p1-tera/h2h/shards/shard-*.jsonl | wc -l) = 4 ]'
# 8 writable PREREG
setup "1 0" prew; rc=$(run)
ok "8 writable PREREG => FAILED" '[ $rc = 1 ] && grep -q "no read-only council/p1-tera/PREREG.txt" $W/C/council/p1-tera/FAILED'
# 9 no ARMS_DONE within the wait
setup "1 0" nodone; rc=$(run)
ok "9 no ARMS_DONE within the wait => FAILED" '[ $rc = 1 ] && grep -q "no ARMS_DONE within" $W/C/council/p1-tera/FAILED'
echo "mock-verdict: $pass passed, $failn failed"
