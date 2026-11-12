#!/bin/bash
# fastenv parity (JS tracker/encoders vs poke-env + Python reference encoders) on the pinned and the current Showdown
# build, 100 seeded battles per format (2 CPUs: the last two of the job, nice 10). Inside the fastenv job:
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && scripts/fastenv/in_job.sh JOB scripts/fastenv/sets/parity_all.sh [N=100] [SEED=21]
# Logs: sets-diff/parity/{v1,v2}_{pinned,current}_FORMAT.log   (PIN / CUR / OUT / FORMATS override)
set -uo pipefail
# two CPUs of this job's cpuset (the last two; CPUS=a,b overrides): stay within the 2-CPU budget, away from the first
CPUS=${CPUS:-$(taskset -pc $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{e = $2 == "" ? $1 : $2; for (i = $1; i <= e; i++) print i}' | tail -2 | paste -sd,)}
C1=${CPUS%%,*}
W=/vol/bitbucket/fr1226/porygon-ws-fastenv
N=${1:-100}; SEED=${2:-21}
PIN=$(realpath "${PIN:-$W/ps-sets-pinned}") CUR=$(realpath "${CUR:-$W/ps-current}")  # absolute: we cd below
OUT=${OUT:-$W/sets-diff/parity}; mkdir -p "$OUT"
cd "$W/scripts/fastenv"
for f in ${FORMATS:-gen9randombattle gen4randombattle}; do
  for v in pinned current; do
    PS=$PIN; [ $v = current ] && PS=$CUR
    for p in v1 v2; do
      S=parity_v1.py; [ $p = v2 ] && S=sets/parity_v2_cluster.py
      t0=$(date +%s)
      PORYGON_PS=$PS PYTHONPATH=$W/src taskset -c $CPUS nice -n 10 $PY $S --record "$N" $f --seed "$SEED" > "$OUT/${p}_${v}_$f.log" 2>&1
      rc=$?
      echo "$p $v $f rc=$rc $(( $(date +%s) - t0 ))s: $(grep -v '^\s*$' "$OUT/${p}_${v}_$f.log" | tail -3 | tr '\n' ' ' | cut -c1-400)"
    done
  done
done
