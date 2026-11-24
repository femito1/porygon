#!/bin/bash
# netv2 2026-11-24, council P1 Stage 0 (integrity, before any arm): runs stage0_record.js + stage0_check.py from the
# read-only tree $C/tree-<sha> (the arms' code) against today's code ($C/fastenv/js = the live trainers' JS, and the
# pre-P1 Python encoder of $C/src-8892762), November sets (fastenv's ps-2026-11 = the trainers' collector build).
#   srun --jobid=<data host> --overlap --ntasks=1 -c 2 --mem=6G -J nv2-p1t-s0 bash stage0.sh TREE_SHA [N]
# -> $C/council/p1-tera/stage0/{SUMMARY.json,RESULT.txt,*.log}; RESULT.txt line 1 = STAGE0 PASS | STAGE0 FAIL.
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2; G=/vol/bitbucket/fr1226/gloria; SHA=${1:?tree sha}; N=${2:-300}
T=$C/tree-$SHA; O=$C/council/p1-tera/stage0; mkdir -p "$O"
export PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$T/src STAGE0_JS=$T/fastenv/js
NODE=$G/.node/bin/node; PY=$G/.venv/bin/python; R=$C/scripts/netv2_p1tera
W=$(mktemp -d /tmp/nv2-p1t-s0-XXXXXX); trap 'rm -rf "$W"' EXIT
cd "$W" || exit 1
echo "$(date -u +%FT%T) stage 0 on $(hostname -s): tree $T, N $N, PS $PORYGON_PS" | tee "$O/stage0.log"
"$NODE" "$R/stage0_record.js" volcarona gen9randombattle 7 > "$O/volcarona.json" 2>> "$O/stage0.log" || { echo "STAGE0 FAIL (volcarona recorder)" > "$O/RESULT.txt"; exit 1; }
"$NODE" "$R/stage0_record.js" leak "$N" gen9randombattle 11 > "$O/leak.json" 2>> "$O/stage0.log" || { echo "STAGE0 FAIL (leak recorder)" > "$O/RESULT.txt"; exit 1; }
echo "$(date -u +%FT%T) volcarona + leak recorded; stream $N battles (seed 1) | check" | tee -a "$O/stage0.log"
"$NODE" --max-old-space-size=3072 "$R/stage0_record.js" stream "$N" gen9randombattle 1 "$C/fastenv/js" 2>> "$O/stage0.log" \
  | "$PY" -u "$R/stage0_check.py" --stream /dev/stdin --old-py "$C/src-8892762/porygon/v2/encoder.py" --leak "$O/leak.json" \
      --volcarona "$O/volcarona.json" --min-battles "$N" --out "$O/SUMMARY.json" > "$O/check.log" 2>&1
rc=${PIPESTATUS[1]}
{ grep -m1 '^STAGE0' "$O/check.log" || echo "STAGE0 FAIL (checker rc $rc, no verdict line)"; cat "$O/check.log"; } > "$O/RESULT.txt"
echo "$(date -u +%FT%T) done: $(head -n 1 "$O/RESULT.txt")" | tee -a "$O/stage0.log"
