#!/bin/bash
# NNUE gate 1 chain (docs/netsearch/NNUE-GATE1-PREREG.md v2, md5 43f045da; main's parity conditions in
# $C/nnue/gate1/notes.log): one cluster-side chain, a step of fastenv's helper job (12 CPUs, T4), no agent.
#   checks   : PREREG.txt r-- with md5 = PREREG.md5 = 43f045da..., teacher sha256, >= 12 CPUs, a GPU, frozen netv2 tree;
#   load     : the teacher loads through the frozen tree and is gen9;
#   fparity  : FEATURE-level parity on step 3's 60 games (re-played, same input logs) -> FEATURE-PARITY.txt; not PASS stops;
#   generate : nnue_gen.py (cd@805 both sides, T 1.0) to >= 10M positions (stops at 10.5M); up to 3 attempts (resumes);
#   fit      : nnue_fit.py; up to 2 more fits only on a crash / divergence (recorded in notes.log);
#   seth     : lookahead2_study.py outcome --student on Step 2's 3,000 human positions (K 4 dets, seed 11);
#   timing   : nnue_report.py timing pinned to one core; report : nnue_report.py report -> VERDICT.txt (r--) + RESULT;
#   end      : cancels the helper unless nnue/gate1/KEEP_HELPER exists.
# Stage files in $S: STATUS, FAILED, FEATURE-PARITY.txt, VERDICT.txt, RESULT; work dir $S/chain (code, data, fit, seth).
# Launch (the code is frozen into $S/chain/code first, from a laptop rsync of the committed tree):
#   echo "NNUE_CODE=$S/chain/code bash $S/chain/code/scripts/fastenv/nnue_gate1.sh" > jobq/nnue-gate1.sh.tmp && mv ...
# SMOKE=1: the same chain at toy sizes into nnue/gate1-smoke (keeps the helper; its verdict means nothing).
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
D=/vol/bitbucket/fr1226/porygon-ws-data
PY=$G/.venv/bin/python
SMOKE=${SMOKE:-0}
GATE=$C/nnue/gate1
if [ "$SMOKE" = 1 ]; then S=$C/nnue/gate1-smoke; else S=$GATE; fi
CODE=${NNUE_CODE:?set NNUE_CODE to the frozen code dir}
W=$S/chain
NET=$C/nets/g9q-cd-805.03M.pt; NET_SHA=e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f
PREREG_MD5=43f045dac71adb39dffeb5c75d1088cb
FROZEN=$C/frozen/netv2-fba02fb-20261110
HUNEW="$D/datasets/human/records/gen9randombattle/part-0070[0-3].jsonl.gz"
STEP2=$C/lookahead2/study-20261115/outcome-human.jsonl
if [ "$SMOKE" = 1 ]; then
  TARGET=20000 CAP=30000 EPOCHS=2 PAIRS=30 TIMING_N=300 BOOT=50
else
  TARGET=10000000 CAP=10500000 EPOCHS=20 PAIRS=1500 TIMING_N=10000 BOOT=1000
fi
export PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src NETV2_SRC=$FROZEN/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1
J=${SLURM_JOB_ID:-none}
cd "$C" || exit 1
mkdir -p "$S" "$W"
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$S/STATUS"; }
note() { echo "$(now) $*" >> "$S/notes.log"; }
fail() { st "FAILED at stage $1: $2"; echo "$(now) stage $1: $2 (job $J)" > "$S/FAILED"; exit 1; }
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
md5of() { md5sum < "$1" | cut -c1-32; }
[ ! -e "$S/VERDICT.txt" ] && [ ! -e "$S/FAILED" ] || { echo "$S already holds a finished chain" >&2; exit 1; }
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT"), smoke $SMOKE"

# ---- checks
[ -f "$GATE/PREREG.txt" ] && [ ! -w "$GATE/PREREG.txt" ] || fail checks "PREREG.txt missing or writable"
[ "$(md5of "$GATE/PREREG.txt")" = "$PREREG_MD5" ] && [ "$(cut -c1-32 "$GATE/PREREG.md5")" = "$PREREG_MD5" ] \
  || fail checks "PREREG md5 != $PREREG_MD5"
[ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "teacher sha256 mismatch"
[ "${SLURM_CPUS_ON_NODE:-0}" -ge 12 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 12)"
nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
[ -d "$FROZEN/src/porygon/v2" ] || fail checks "frozen netv2 tree missing"
ls $HUNEW > /dev/null 2>&1 || fail checks "set H records missing"
st "checks OK: PREREG md5 $PREREG_MD5, teacher sha256 ok"
ld=$("$PY" -c '
import sys; sys.path.insert(0, sys.argv[2])
import netsearch_study as ST; ST.setup("gen9randombattle")
n = ST.load_net(sys.argv[1], "cpu"); print("gen", int(n.gen), "params", sum(p.numel() for p in n.parameters()))
' "$NET" "$CODE/scripts/fastenv" 2>&1 | tail -1)
case $ld in "gen 9 "*) st "load OK through fba02fb: $ld" ;; *) fail load "teacher does not load: $ld" ;; esac

# ---- feature parity (main's condition): only a PASS lets generation start
FP=$S/FEATURE-PARITY.txt
if [ ! -s "$FP" ]; then
  "$PY" "$CODE/scripts/fastenv/nnue_feature_parity.py" --play 60 --workers 6 --seed 916 --dump-dir "$W/fparity" \
    --same-as "$GATE/parity/dumps/dump-*.jsonl.gz" --out "$FP" > "$W/fparity.log" 2>&1
  [ -s "$FP" ] || fail fparity "no FEATURE-PARITY.txt (see $W/fparity.log)"
  chmod a-w "$FP" "${FP%.txt}.json"
  gzip -q "$W"/fparity/dump-*.jsonl || true
fi
[ "$(head -1 "$FP")" = PASS ] || fail fparity "$(head -1 "$FP")"
st "feature parity PASS: $(sed -n 4p "$FP")"

# ---- generate
DATA=$W/data
mkdir -p "$DATA"
for att in 1 2 3; do
  "$PY" "$CODE/scripts/fastenv/nnue_gen.py" --net "$NET" --out "$DATA" --target "$TARGET" --cap "$CAP" --workers 5 \
    --battles 32 --fifo-dir /tmp >> "$W/gen.log" 2>&1
  rc=$?
  [ $rc = 0 ] && break
  st "generate attempt $att exit $rc ($(tail -1 "$W/gen.log"))"
  note "generate attempt $att exit $rc; resuming (existing shards kept, new seeds)"
done
[ $rc = 0 ] || fail generate "nnue_gen.py exit $rc after 3 attempts (see $W/gen.log)"
n=$("$PY" -c 'import json,sys,pathlib; d=pathlib.Path(sys.argv[1]); print(sum(json.loads(p.read_text())["n"] for p in d.glob("shard-*.json") if p.with_suffix(".npz").exists()))' "$DATA")
[ "$n" -ge "$TARGET" ] && [ "$n" -le "$CAP" ] || fail generate "positions $n outside [$TARGET, $CAP]"
st "generate done: $n positions ($(tail -1 "$W/gen.log"))"

# ---- fit (one fit; up to 2 more only for a crash / divergence)
FIT=$W/fit
for att in 1 2 3; do
  "$PY" "$CODE/scripts/fastenv/nnue_fit.py" --data "$DATA" --out "$FIT" --epochs "$EPOCHS" --seed $((att - 1)) \
    > "$W/fit-$att.log" 2>&1
  rc=$?
  [ $rc = 0 ] && break
  st "fit attempt $att exit $rc ($(tail -1 "$W/fit-$att.log"))"
  note "fit attempt $att exit $rc (crash / divergence): $(tail -1 "$W/fit-$att.log"); re-fit with seed $att"
done
[ $rc = 0 ] && [ -s "$FIT/student.npz" ] || fail fit "nnue_fit.py exit $rc after 3 fits (see $W/fit-*.log)"
st "fit done: $(tail -1 "$W/fit-$att.log")"

# ---- set H: Step 2's 3,000 human positions, teacher re-scored with cd@805, student + evaluate() on the same K 4 dets
SH=$W/seth
mkdir -p "$SH"
"$PY" "$CODE/scripts/fastenv/lookahead2_study.py" outcome --records "$HUNEW" --net "$NET" --student "$FIT/student.npz" \
  --out "$SH/outcome-human-student.jsonl" --pairs "$PAIRS" --per-record 2 --dets 4 > "$SH/outcome.log" 2>&1 \
  || fail seth "outcome exit $? (see $SH/outcome.log)"
"$PY" "$CODE/scripts/fastenv/lookahead2_study.py" analyze-outcome "$SH/outcome-human-student.jsonl" \
  --json "$SH/outcome-human-student.json" > "$SH/outcome-human-student.txt" 2>&1 || true
st "set H done: $(tail -1 "$SH/outcome.log")"

# ---- timing (one core) and report
CPU1=$(taskset -pc $$ | sed 's/.*: //' | cut -d, -f1 | cut -d- -f1)
taskset -c "$CPU1" "$PY" "$CODE/scripts/fastenv/nnue_report.py" timing --data "$DATA" --student "$FIT/student.npz" \
  --out "$W/timing.json" --n "$TIMING_N" > "$W/timing.log" 2>&1 || st "timing exit $? (report goes on without it)"
"$PY" "$CODE/scripts/fastenv/nnue_report.py" report --data "$DATA" --student "$FIT/student.npz" \
  --seth "$SH/outcome-human-student.jsonl" --timing "$W/timing.json" --fit "$FIT/fit.json" \
  --parity "$GATE/PARITY.json" --feature-parity "${FP%.txt}.json" --step2 "$STEP2" --boot "$BOOT" \
  --out "$W/REPORT.txt" > "$W/report.log" 2>&1 || fail report "nnue_report.py exit $? (see $W/report.log)"
case $(head -1 "$W/REPORT.txt") in PASS:*|PARTIAL:*|FAIL:*) ;; *) fail report "bad verdict line: $(head -1 "$W/REPORT.txt")" ;; esac
cp "$W/REPORT.txt" "$S/VERDICT.txt.tmp" && mv -f "$S/VERDICT.txt.tmp" "$S/VERDICT.txt" && chmod a-w "$S/VERDICT.txt"
head -1 "$S/VERDICT.txt" > "$S/RESULT"
st "VERDICT: $(head -1 "$S/VERDICT.txt")"
trap - TERM INT
if [ "$SMOKE" != 1 ] && [ ! -e "$GATE/KEEP_HELPER" ] && [ "$J" != none ]; then
  st "chain done; cancelling the helper job $J"
  scancel "$J"
else
  st "chain done (helper kept)"
fi
exit 0
