#!/bin/bash
# Stage 1 of the network + search test (NETSEARCH-DESIGN.md §6): re-score the offline study's positions and searches
# (study-20261109: S1, S2, S2c, S3) with the gate-passing snapshot, then the Stage 2 margins by the §3.1 rule on the
# pooled R curve (S2 + S2c). Pure search does not depend on the net, so only the network's distribution changes; S2's
# PUCT searches (P) used the old net's priors and are dropped. Run inside a job (helper step), cwd = $C:
#   NET=nets/g9p-100M.pt OUT=netsearch/stage1-g9p100M bash scripts/fastenv/netsearch_stage1.sh
set -uo pipefail
G=/vol/bitbucket/fr1226/gloria; D=/vol/bitbucket/fr1226/porygon-ws-data
PY=${PY:-$G/.venv/bin/python}
export PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$PWD/src OMP_NUM_THREADS=1
I=netsearch/study-20261109
mkdir -p "$OUT"
SL="$D/datasets/searchlabels/records/gen9randombattle/part-0000[67].jsonl.gz"
HU3="$D/datasets/human/records/gen9randombattle/part-0027[01].jsonl.gz"
HU2C="$D/datasets/human/records/gen9randombattle/part-0027[23].jsonl.gz"
S=scripts/fastenv/netsearch_study.py; A=scripts/fastenv/netsearch_analyze.py
echo "$(date -u +%T) net $NET sha256 $(sha256sum "$NET" | cut -c1-64)" | tee "$OUT/net.txt"
rc=0
$PY $S rescore --records "$SL" --source 6771931 --net "$NET" --rows $I/s1-sl-AB.jsonl --out "$OUT/s1.jsonl" || rc=1
$PY $S rescore --records "$SL" --source 6771931 --net "$NET" --rows $I/s2-sl-ARP.jsonl --out "$OUT/s2-withP.jsonl" || rc=1
$PY -c 'import json, sys
for line in open(sys.argv[1]):
    if line.strip():
        row = json.loads(line); row.pop("P", None); print(json.dumps(row))' "$OUT/s2-withP.jsonl" > "$OUT/s2.jsonl" || rc=1
$PY $S rescore --records "$HU2C" --net "$NET" --rows $I/s2c-hu-AR.jsonl --out "$OUT/s2c.jsonl" || rc=1
$PY $S rescore --records "$HU3" --net "$NET" --rows $I/s3-hu-AB.jsonl --out "$OUT/s3.jsonl" || rc=1
echo "$(date -u +%T) rescored (rc $rc): $(wc -l < "$OUT/s1.jsonl") / $(wc -l < "$OUT/s2.jsonl") / $(wc -l < "$OUT/s2c.jsonl") / $(wc -l < "$OUT/s3.jsonl") rows"
for x in s1 s2 s2c s3; do $PY $A "$OUT/$x.jsonl" --json "$OUT/$x.json" > "$OUT/$x.txt" 2>&1 || rc=1; done
$PY $A "$OUT/s2.jsonl" "$OUT/s2c.jsonl" --json "$OUT/s2all.json" > "$OUT/s2all.txt" 2>&1 || rc=1
$PY scripts/fastenv/netsearch_arms.py "$OUT/s2all.json" --json "$OUT/arms.json" | tee "$OUT/arms.txt" || rc=1
echo "$(date -u +%T) done rc $rc"
exit $rc
