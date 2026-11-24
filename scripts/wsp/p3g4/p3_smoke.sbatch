#!/bin/bash
# wsp P3 launcher check (outside the slot; PREREG $W/council/p3-gen4/PREREG.txt): the frozen P3 harness
# ($W/council/p3-gen4/code/fp4_iters.sbatch + fp_fixed_iters.py + fp_iters_check.py) on gen4 at the P3 budget,
# 200,000 iterations per 1000 ms state, 4 games per player mode (K 2 Foul Play instances, seed 4031125, not the slot
# seed): --player search = the incumbent's config (v1@500M prior, fitted eval, 4 x 1 s) and --player policy = a
# stand-in v2 gen4 net (R1b-F@300M, runs/g4r1bf/prior-R1bF-300M.pt, T 0.5; R1b-F-cd does not exist yet).
# PASS iff for each mode: no ITERS_MISMATCH, ITERS_OK with N-count states, the launcher line names the gen4 engine
# build of Foul Play's own venv, >= 2 complete pairs; the search mode logged the expected SearchPlayer config; and
# the two modes played the same teams on their common pairs (same seed => same team pairs whatever the player).
# Writes $W/council/p3-gen4/smoke/RESULT (first word PASS or FAIL); the slot chain (stage i) requires PASS.
#   cd $W && sbatch --partition=training council/p3-gen4/code/p3_smoke.sbatch
#SBATCH --job-name=ws-wsp-p3smoke
#SBATCH --partition=training
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-wsp/logs/slurm-%j.out
set -uo pipefail
W=/vol/bitbucket/fr1226/porygon-ws-wsp; C=/vol/bitbucket/fr1226/porygon-ws-netv2; G=/vol/bitbucket/fr1226/gloria
D=$W/council/p3-gen4; K=$D/code; S=$D/smoke; J=${SLURM_JOB_ID:?}
PY=$G/.venv/bin/python
INC=$G/runs/20261025-195826-301040/final_model.zip; EVAL=$G/src/porygon/data/eval_r1.json
STANDIN=$C/runs/g4r1bf/prior-R1bF-300M.pt
SEED=4031125; ITERS=200000
mkdir -p "$S" "$D/runs"; rm -f "$S/RESULT"
ok=1; lines=(); dirs=()
off=0
for mode in search policy; do
  if [ "$mode" = search ]; then net=$INC; extra=(--ms 1000 --samples 4 --prior-weight 0.25 --eval-config "$EVAL")
  else net=$STANDIN; extra=(--temperature 0.5); fi
  ( cd "$C" && SLURM_SUBMIT_DIR=$C bash "$K/fp4_iters.sbatch" "$net" 4 --gen 4 --player "$mode" --k 2 --seed "$SEED" \
      --fp-iters "$ITERS" --tag "smoke_$mode" --port-offset "$off" "${extra[@]}" ) > "$S/arm-$mode-$J.log" 2>&1
  off=$(( off + 1 ))
  R=$(ls -td "$D"/runs/foulplay-smoke_"$mode"-*-"$J" 2>/dev/null | head -n 1)
  if [ -z "$R" ]; then ok=0; lines+=("$mode: no run dir (arm-$mode-$J.log)"); continue; fi
  mv "$R" "$S/"; R=$S/$(basename "$R"); dirs+=("$R")
  games=$(wc -l < "$R/results.jsonl" 2>/dev/null || echo 0)
  pairs=$("$PY" -I "$K/p3_read.py" pairs "$R" 2>/dev/null) || pairs=0
  launcher=$(zcat "$R/fp0.log.gz" 2>/dev/null | grep -m1 '^fp_fixed_iters:' || true)
  counts=$(zcat "$R/fp0.log.gz" 2>/dev/null | grep -oE 'Iterations [0-9]+: [0-9]+' | awk '{print $3}' | sort -n | uniq -c |
           awk '{printf "%sx%s ", $1, $2}')
  line="$mode: $games games, $pairs complete pairs; $(cat "$R/ITERS_OK" 2>/dev/null || echo 'no ITERS_OK'); fp0.log counts: $counts; $launcher"
  if [ -f "$R/ITERS_MISMATCH" ]; then ok=0; line="$line; MISMATCH $(cat "$R/ITERS_MISMATCH")"; fi
  grep -qs "OK (.*N [1-9]" "$R/ITERS_OK" || { ok=0; line="$line; FAIL: no ITERS_OK with N-count states"; }
  [[ "$launcher" == *"fp_fixed_iters: $ITERS iterations"*"poke-engine 0.0.48 at /vol/bitbucket/fr1226/foul-play/.venv/"* ]] ||
    { ok=0; line="$line; FAIL: launcher line is not the gen4 build in Foul Play's .venv"; }
  [ "$pairs" -ge 2 ] || { ok=0; line="$line; FAIL: < 2 complete pairs"; }
  if [ "$mode" = search ]; then
    cfg=$(grep -o 'SearchPlayer([^)]*)' "$R/vs.log" | head -n 1)
    line="$line; $cfg"
    [[ "$cfg" == "SearchPlayer(ms=1000, samples=4, prior_weight=0.25, "*"eval_config=$EVAL, oracle=None)" ]] ||
      { ok=0; line="$line; FAIL: unexpected SearchPlayer config"; }
  fi
  lines+=("$line")
done
if [ "${#dirs[@]}" = 2 ]; then
  read -r both same diff < <("$PY" -I "$K/p3_read.py" samepairs "${dirs[0]}" "${dirs[1]}" 2>/dev/null || echo "0 0 0")
  lines+=("same team pairs across the two modes: $both common complete pairs, $same same teams, $diff different")
  [ "${both:-0}" -ge 1 ] && [ "${diff:-1}" = 0 ] || ok=0
else
  ok=0
fi
{ [ "$ok" = 1 ] && echo "PASS job $J $(date -u +%FT%TZ)" || echo "FAIL job $J $(date -u +%FT%TZ)"; printf '%s\n' "${lines[@]}"; } > "$S/RESULT"
cat "$S/RESULT"
[ "$ok" = 1 ]
