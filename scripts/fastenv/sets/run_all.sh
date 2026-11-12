#!/bin/bash
# Pinned (7b02eb3) vs current Showdown random-battle sets, gen9 + gen4: static set-file diff, N teams per build per
# seed set, analysis. Runs inside the fastenv job on 2 CPUs (taskset: the last two CPUs of the job, nice 10):
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && scripts/fastenv/in_job.sh JOB scripts/fastenv/sets/run_all.sh [N=20000]
# Outputs: sets-diff/{static_gen9,static_gen4}.{json,txt}, teams/*.jsonl.gz, analysis_FORMAT.{json,txt}
# Monthly reuse: PIN=<last month's build> CUR=<new build> OUT=<new dir> scripts/fastenv/sets/run_all.sh
set -euo pipefail
# two CPUs of this job's cpuset (the last two; CPUS=a,b overrides): stay within the 2-CPU budget, away from the first
CPUS=${CPUS:-$(taskset -pc $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{e = $2 == "" ? $1 : $2; for (i = $1; i <= e; i++) print i}' | tail -2 | paste -sd,)}
C1=${CPUS%%,*}
W=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
OUT=${OUT:-$W/sets-diff}
PIN=${PIN:-$W/ps-sets-pinned} CUR=${CUR:-$W/ps-current}
N=${1:-20000}
S=$(cd "$(dirname "$0")" && pwd)
NODE="nice -n 10 $G/.node/bin/node"
PE=$(ls -d $G/.venv/lib/python3*/site-packages/poke_env/data/static)
mkdir -p "$OUT/teams"
echo "pinned $(git -C $PIN rev-parse --short HEAD), current $(git -C $CUR rev-parse --short HEAD), N=$N" | tee "$OUT/versions.txt"
for g in gen9 gen4; do taskset -c $C1 $NODE "$S/static_diff.js" "$PIN" "$CUR" $g "$OUT/static_$g.json" | tee "$OUT/static_$g.txt"; done
for f in gen9randombattle gen4randombattle; do
  for v in pinned current; do
    PS=$PIN; [ $v = current ] && PS=$CUR
    taskset -c $C1 $NODE "$S/gen_teams.js" "$PS" $f 1 "$N" "$OUT/teams/${f}_${v}_s1.jsonl.gz" &
    taskset -c ${CPUS##*,} $NODE "$S/gen_teams.js" "$PS" $f 2 "$N" "$OUT/teams/${f}_${v}_s2.jsonl.gz" &
    wait
  done
  taskset -c $C1 $NODE --max-old-space-size=4096 "$S/analyze.js" $f "$OUT/teams" "$PIN" "$CUR" "$W/src/porygon/data/vocab.json" \
    "$W/src/porygon/data/${f}_library.json.gz" "$PE" "$OUT/analysis_$f.json" > /dev/null
  cat "$OUT/analysis_$f.txt"
done
