#!/bin/bash
# Build a pokemon-showdown checkout the way gloria/pokemon-showdown was built (npm ci + node build -> dist/sim).
# Usage (inside the fastenv job): scripts/fastenv/sets/build_ps.sh DIR ; runs on 2 CPUs (taskset: the last two CPUs of the job), nice 10.
set -euo pipefail
# two CPUs of this job's cpuset (the last two; CPUS=a,b overrides): stay within the 2-CPU budget, away from the first
CPUS=${CPUS:-$(taskset -pc $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{e = $2 == "" ? $1 : $2; for (i = $1; i <= e; i++) print i}' | tail -2 | paste -sd,)}
C1=${CPUS%%,*}
D=$1
export PATH=/vol/bitbucket/fr1226/gloria/.node/bin:$PATH GOMAXPROCS=2 UV_THREADPOOL_SIZE=2
export npm_config_cache=/vol/bitbucket/fr1226/porygon-ws-fastenv/.npm-cache npm_config_audit=false npm_config_fund=false npm_config_update_notifier=false
cd "$D"
echo "== $(date) $(hostname) building $D at $(git rev-parse HEAD)"
time taskset -c $CPUS nice -n 10 npm ci --omit=optional --no-audit --no-fund
node -e "require(\"esbuild\"); console.log(\"esbuild\", require(\"esbuild\").version)"
time taskset -c $CPUS nice -n 10 node build
ls dist/sim | head -3
node -e "const {Teams}=require(\"./dist/sim\"); const t=Teams.generate(\"gen9randombattle\",{seed:[1,2,3,4]}); console.log(t.map(s=>s.species+\"@\"+s.level).join(\" \"))"
echo "== done $(date)"
