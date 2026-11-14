#!/bin/bash
# Re-freeze #1 cluster-side chain (PLAN "Cluster-side pipelines"; REFREEZE-PREREG.md md5 ad1911d2, read-only):
#   delta  : wait for the helper's jobq/stage1q.rc (Stage 1 re-score with cand), read delta_c = arms.txt's
#            "L = 0.0005 at delta = X" (judge R, else B; cap 0.2 inside netsearch_arms.py), cancel the helper;
#   plan   : plans/refreeze1.env.in with @DELTA_C@ -> refreeze1/refreeze1.env; both nets' sha256 checked;
#   R      : netsearch_slot.sh on it (500 pairs fixed n, tag 20261112), run dir runs/netsearch-refreeze1-<date>-<job>;
#   verdict: refreeze1_verdict.py (void / ADOPT iff lower 95% > 50% / NOT ADOPTED; paired diff 2m - 1);
#   release: the ledger RELEASE line, appended under flock to $C/LEDGER-OUTBOX.md (the ledger itself is on the laptop).
# Stage dir refreeze1/: STATUS (one line per stage), RESULT (the verdict's first line), FAILED (on any failure).
# Claim the slot in .orchestra/ledger.md first.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch scripts/fastenv/refreeze1_r.sbatch HELPER_JOB_ID
#SBATCH --job-name=ws-fastenv-refreeze1
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
C=$PWD
G=/vol/bitbucket/fr1226/gloria
PY=$G/.venv/bin/python
HELPER=${1:-}
S=$C/refreeze1
CAND=$C/nets/g9q-cd-305.04M.pt; CAND_SHA=31d6f659b4bdafa1da6298b919329c2f1d470fc2a49b242aa301539764160fa0
OLD=$C/nets/g9p-cd-104.71M.pt; OLD_SHA=35cd1007bb90bdf939cedf877f5fcdd89ef640628f77007b3b6638d5add4ad1c
D1=$C/netsearch/stage1-g9qcd-305.04M
mkdir -p "$S"
[ ! -e "$S/RESULT" ] && [ ! -e "$S/FAILED" ] || { echo "$S already holds a finished chain" >&2; exit 1; }
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$S/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
fail() {
  st "FAILED at stage $1: $2"
  echo "$(now) stage $1: $2 (job $SLURM_JOB_ID)" > "$S/FAILED"
  outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $SLURM_JOB_ID | re-freeze #1 chain FAILED at stage $1 ($2); $S/FAILED; no verdict; slot FREE"
  exit 1
}
trap 'fail signal "job signalled (TERM/INT; time limit or scancel)"' TERM INT
st "chain start: job $SLURM_JOB_ID on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, helper ${HELPER:-none}"

# ---- delta
for _ in $(seq 360); do [ -e jobq/stage1q.rc ] && break; sleep 15; done
[ -e jobq/stage1q.rc ] || fail delta "jobq/stage1q.rc missing after 90 min"
rc=$(cat jobq/stage1q.rc)
[ "$rc" = 0 ] || fail delta "stage1q exit code $rc (jobq/stage1q.log)"
if [ -n "$HELPER" ]; then scancel "$HELPER" && st "helper $HELPER cancelled (its steps are done)"; fi
grep -q "sha256 $CAND_SHA" "$D1/net.txt" || fail delta "$D1/net.txt is not cand's sha256"
DELTA=$(sed -n 's/^  L = 0.0005 at delta = \([0-9.]*\) .*/\1/p' "$D1/arms.txt")
JUDGE=$(sed -n '1s/.*: judge \([RB]\),.*/\1/p' "$D1/arms.txt")
[[ "$DELTA" =~ ^[0-9]+\.[0-9]+$ ]] || fail delta "no 'L = 0.0005 at delta' line in $D1/arms.txt"
awk -v d="$DELTA" 'BEGIN { exit !(d >= 0 && d <= 0.2) }' || fail delta "delta_c $DELTA outside [0, 0.2]"
echo "$DELTA" > "$S/DELTA_C"
st "stage delta OK: delta_c = $DELTA (judge $JUDGE, level 0.0005, $D1/arms.txt sha256 $(sha256sum "$D1/arms.txt" | cut -c1-16))"
sed -n '1,8p' "$D1/arms.txt" | sed 's/^/  | /' | tee -a "$S/STATUS"

# ---- plan
sed "s/@DELTA_C@/$DELTA/" scripts/fastenv/plans/refreeze1.env.in > "$S/refreeze1.env" || fail plan "template"
grep -q "margin=$DELTA," "$S/refreeze1.env" && ! grep -q @DELTA_C@ "$S/refreeze1.env" || fail plan "substitution"
[ "$(sha256sum "$CAND" | cut -c1-64)" = "$CAND_SHA" ] || fail plan "cand sha256 mismatch"
[ "$(sha256sum "$OLD" | cut -c1-64)" = "$OLD_SHA" ] || fail plan "G9p-cd sha256 mismatch"
st "stage plan OK: $S/refreeze1.env (sha256 $(sha256sum "$S/refreeze1.env" | cut -c1-16)); cand and G9p-cd sha256 verified"

# ---- R
OUT=$C/runs/netsearch-refreeze1-$(date -u +%Y%m%d-%H%M%S)-$SLURM_JOB_ID
echo "$OUT" > "$S/RUN_DIR"
st "stage R start: $OUT (500 pairs, tag 20261112, fixed n)"
bash scripts/fastenv/netsearch_slot.sh "$S/refreeze1.env" "$OUT" &
wait $!; rc=$?
[ -s "$OUT/result.json" ] || fail R "no result.json (netsearch_slot.sh exit $rc; $OUT/run.log)"
st "stage R done: netsearch_slot.sh exit $rc; $OUT/result.txt"

# ---- verdict
$PY scripts/fastenv/refreeze1_verdict.py "$OUT" > "$S/verdict.out" 2>&1; vrc=$?
[ -s "$OUT/VERDICT.txt" ] || fail verdict "refreeze1_verdict.py exit $vrc ($S/verdict.out)"
cp "$OUT/VERDICT.txt" "$S/VERDICT.txt"
head -1 "$OUT/VERDICT.txt" > "$S/RESULT"
st "stage verdict: $(head -1 "$OUT/VERDICT.txt")"

# ---- release
L2=$(sed -n 2p "$OUT/VERDICT.txt" | sed 's/^ *//'); L4=$(sed -n 4p "$OUT/VERDICT.txt" | sed 's/^ *//')
outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $SLURM_JOB_ID | re-freeze #1 R done: $(head -1 "$OUT/VERDICT.txt"); $L2; $L4; $OUT/VERDICT.txt; slot FREE"
st "chain done (ledger RELEASE line in $C/LEDGER-OUTBOX.md)"
trap - TERM INT
exit 0
