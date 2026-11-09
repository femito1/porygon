#!/bin/bash
# The monthly random-battle set refresh in one step (fastenv HANDOFF "Showdown set refresh"), inside a fastenv job
# with >= 4 CPUs (e.g. a jobq step of slurm_helper.sbatch):
#   MONTH=2026-11 [PIN=<last build>] [N=20000] [NPAR=100] scripts/fastenv/sets/month_all.sh
# Checkout first (login node, git only):
#   git -C ps-current fetch origin && git -C ps-current worktree add --detach ../ps-$MONTH origin/master
# Steps: build ps-$MONTH (build_ps.sh, skipped if dist/sim exists) -> run_all.sh (static + generated-team diff vs PIN)
# in parallel with parity_all.sh (v1 + v2, PIN + month, gen9 + gen4) -> vocab_diff.js, engine_names_check.js,
# fp_sets_check.js -> netv2's v2 fixtures replayed on the month build -> dist/ diff vs ps-current -> chmod -R a-w
# ps-$MONTH -> smoke test on the read-only build (v2 parity 20 gen4 battles, FastEnv v1 15 s). Outputs: sets-diff-$MONTH/.
set -uo pipefail
W=/vol/bitbucket/fr1226/porygon-ws-fastenv; G=/vol/bitbucket/fr1226/gloria
M=${MONTH:?set MONTH=YYYY-MM}; PS=$W/ps-$M; PIN=$(realpath "${PIN:-$W/ps-sets-pinned}")
OUT=$W/sets-diff-$M; S=$W/scripts/fastenv/sets
NETV2=${NETV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}
FP=${FP:-/vol/bitbucket/fr1226/foul-play/fp/data/pkmn_sets_cache}
ENGINE_SRC=${ENGINE_SRC:-$G/.poke-engine}
PY=${PY:-$G/.venv/bin/python}; export PY
NODE="nice -n 10 $G/.node/bin/node"
mkdir -p "$OUT/logs"
mapfile -t CPU < <(taskset -pc $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{e = $2 == "" ? $1 : $2; for (i = $1; i <= e; i++) print i}')
n=${#CPU[@]}; [ "$n" -ge 4 ] || { echo "need >= 4 CPUs, have $n"; exit 1; }
P1=${CPU[n-4]},${CPU[n-3]}; P2=${CPU[n-2]},${CPU[n-1]}
step() { echo "== $(date -u +%FT%T) $*"; }
SHA=$(git -C "$PS" rev-parse HEAD)
step "month $M: build $PS at $SHA; pinned $PIN at $(git -C "$PIN" rev-parse HEAD); node $($G/.node/bin/node --version)"
if [ ! -d "$PS/dist/sim" ]; then
  CPUS=$P2 bash "$S/build_ps.sh" "$PS" > "$OUT/logs/build.log" 2>&1 || { tail "$OUT/logs/build.log"; exit 1; }
fi
tail -3 "$OUT/logs/build.log" 2>/dev/null
step "run_all (CPUs $P1) || parity_all (CPUs $P2)"
CPUS=$P1 PIN=$PIN CUR=$PS OUT=$OUT bash "$S/run_all.sh" "${N:-20000}" > "$OUT/logs/run_all.log" 2>&1 & a=$!
CPUS=$P2 PIN=$PIN CUR=$PS OUT=$OUT/parity bash "$S/parity_all.sh" "${NPAR:-100}" 21 > "$OUT/logs/parity_all.log" 2>&1 & b=$!
wait $a; echo "run_all rc=$?"; wait $b; echo "parity_all rc=$?"
cat "$OUT/logs/parity_all.log"
step "vocab / poke-engine names / Foul Play cache"
taskset -c "$P1" $NODE "$S/vocab_diff.js" "$PIN" "$PS" "$W/src/porygon/data/vocab.json" "$OUT/vocab_regen.json" > "$OUT/logs/vocab.log" 2>&1
echo "vocab_diff rc=$?"
for f in gen9randombattle gen4randombattle; do
  taskset -c "$P1" $NODE "$S/engine_names_check.js" "$ENGINE_SRC" $f "$OUT/teams" "$OUT/engine_names_$f.json" > /dev/null 2>"$OUT/logs/engine_$f.err"
  echo "engine_names $f rc=$?"
  taskset -c "$P1" $NODE "$S/fp_sets_check.js" $f "$FP/$f.json" "$PIN" "$PS" "$OUT/teams" "$OUT/fp_sets_$f.json" > /dev/null 2>"$OUT/logs/fp_$f.err"
  echo "fp_sets $f rc=$?"
done
step "netv2 v2 fixtures replayed on the month build"
( cd "$W/scripts/fastenv" && PORYGON_PS=$PS PYTHONPATH=$W/src taskset -c "$P2" nice -n 10 $PY sets/parity_v2_cluster.py \
    --fixture "$NETV2/fixtures/gen9" --fixture "$NETV2/fixtures/gen4" > "$OUT/logs/fixtures.log" 2>&1 )
echo "fixtures rc=$?"; grep -v '^\s*$' "$OUT/logs/fixtures.log" | tail -6
step "dist/ vs ps-current ($(git -C "$W/ps-current" rev-parse --short HEAD))"
diff -rq "$W/ps-current/dist" "$PS/dist" > "$OUT/logs/dist_vs_ps-current.txt" 2>&1
echo "dist diff rc=$? ($(wc -l < "$OUT/logs/dist_vs_ps-current.txt") lines):"; head -20 "$OUT/logs/dist_vs_ps-current.txt"
step "read-only build"
chmod -R a-w "$PS" && echo "chmod a-w done; writable files left: $(find "$PS" -perm /222 -not -type l | wc -l)"
( cd "$W/scripts/fastenv" && PORYGON_PS=$PS PYTHONPATH=$W/src taskset -c "$P2" nice -n 10 $PY sets/parity_v2_cluster.py \
    --record 20 gen4randombattle --seed 5 > "$OUT/logs/smoke_v2_readonly.log" 2>&1 )
echo "smoke v2 parity rc=$?: $(grep -v '^\s*$' "$OUT/logs/smoke_v2_readonly.log" | tail -1)"
( cd "$W" && PORYGON_PS=$PS PYTHONPATH=$W/src taskset -c "$P2" nice -n 10 $PY scripts/fastenv/bench_fastenv.py \
    --format gen4randombattle --workers 2 --battles 16 --seconds 15 --encoder v1 > "$OUT/logs/smoke_fastenv_readonly.log" 2>&1 )
echo "smoke FastEnv rc=$?: $(tail -2 "$OUT/logs/smoke_fastenv_readonly.log" | tr '\n' ' ')"
echo "$M $SHA $PS" > "$OUT/BUILD.txt"
step "done: PORYGON_PS=$PS ($SHA)"
