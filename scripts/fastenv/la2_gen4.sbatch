#!/bin/bash
# LA2-gen4 stage 1, the kill test (docs/netsearch/LA2-GEN4-PREREG.md = council/la2-gen4/PREREG.txt, r--, md5 in
# PREREG.md5), ONE slot job, frozen code council/la2-gen4/code (the prep chain's):
#   prereg (r--, md5) -> checks (net md5 90bf853d + sha256, DELTA r-- and = the PREREG's machine line, the study rows md5,
#   PREP-RESULT's PASS lines, >= 24 CPUs, a GPU, the frozen netv2 tree) -> smoke (2 duplicate pairs = 4 games, tag
#   <TAG>smoke) -> main (10 blocks of 100 duplicate pairs = 2,000 games, fixed n, tag <TAG>, block j on a fresh server
#   seeded <TAG>b<j>) -> VERDICT.txt (r--; line 1 GO | STOP) + RESULT -> one RELEASE line in $C/LEDGER-OUTBOX.md.
#   A = lookahead2(R1b-F-cd@640M, δ*, K 8, T 0.5); B = the raw R1b-F-cd@640M at T 0.5 (netv2's NetPlayer); gen4.
# Stage files in council/la2-gen4/: STATUS, FAILED, SMOKE.txt, smoke-run/, main-run/, VERDICT.txt, RESULT.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch council/la2-gen4/code/scripts/fastenv/la2_gen4.sbatch
# Laptop mock (tests/test_la2_gen4.py; refused inside a Slurm job): LA2G4_MOCK=DIR runs prereg, checks of the files,
# the verdict and the RELEASE line on DIR/fastenv; the games are replaced by DIR/run (a result.json).
#SBATCH --job-name=ws-fastenv-la2g4
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
PY=$G/.venv/bin/python
MOCK=
if [ -n "${LA2G4_MOCK:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  MOCK=$LA2G4_MOCK C=$LA2G4_MOCK/fastenv PY=${LA2G4_MOCK_PY:-python3}
fi
S=$C/council/la2-gen4
CODE=$S/code
HELP=$CODE/scripts/fastenv/la2_gen4.py
NET=$C/nets/g4r1bf-cd-640.04M.pt; NET_MD5=90bf853d643ade36ad76f6ebe0dbc61f
FROZEN=$C/frozen/netv2-fba02fb-20261110
TAG=4241125 SMOKE_TAG=4241125smoke
LABEL=R1b-F-cd@640M
[ -z "$MOCK" ] || NET_MD5=${LA2G4_MOCK_NET_MD5:-$NET_MD5}  # laptop mock only: the stand-in net's md5
J=${SLURM_JOB_ID:-mock}
cd "$C" || exit 1
now() { date -u +%FT%TZ; }
st() { echo "$(now) [slot] $*" | tee -a "$S/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
release() {  # one ledger line; the free text may not hold the field separator or a newline
  local m=${1//|//}; m=${m//$'\n'/ }; m=${m:0:700}
  outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $J | LA2-gen4 $m; slot FREE"
}
fail() {
  st "FAILED at stage $1: $2"; echo "$(now) [slot] stage $1: $2 (job $J)" >> "$S/FAILED"
  release "FAILED at stage $1 ($2); $S/FAILED"
  exit 1
}
md5of() { md5sum < "$1" | cut -c1-32; }
for f in VERDICT.txt FAILED smoke-run main-run; do
  [ ! -e "$S/$f" ] || { echo "$S already holds a chain ($f)" >&2; release "refused: $S already holds a chain or a failure ($f); nothing run"; exit 1; }
done
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT" 2>/dev/null)${MOCK:+ (MOCK $MOCK)}"

# ---- prereg
[ -f "$S/PREREG.txt" ] && [ ! -w "$S/PREREG.txt" ] || fail prereg "PREREG.txt missing or writable"
[ -s "$S/PREREG.md5" ] && [ "$(md5of "$S/PREREG.txt")" = "$(cut -c1-32 "$S/PREREG.md5")" ] || fail prereg "PREREG md5 != PREREG.md5"
[ -f "$HELP" ] || fail prereg "frozen code $CODE missing"
st "prereg OK: PREREG md5 $(md5of "$S/PREREG.txt")"

# ---- checks: the prep's outputs and the PREREG's machine line "LA2G4 DELTA*=<δ*> STUDY_ROWS_MD5=<md5> NET_SHA256=<sha>"
[ -f "$S/DELTA" ] && [ ! -w "$S/DELTA" ] || fail checks "DELTA missing or writable"
MARGIN=$(head -1 "$S/DELTA")
[[ "$MARGIN" =~ ^[0-9]+\.?[0-9]*$ ]] || fail checks "δ* not a number: $MARGIN"
ROWS_MD5=$(md5of "$S/prep/set/study-rows.jsonl" 2>/dev/null)
NET_SHA=$(grep -o 'NET_SHA256=[0-9a-f]\{64\}' "$S/PREREG.txt" | head -1 | cut -d= -f2)
LINE="LA2G4 DELTA*=$MARGIN STUDY_ROWS_MD5=$ROWS_MD5 NET_SHA256=$NET_SHA"
grep -qF "$LINE" "$S/PREREG.txt" || fail checks "the PREREG does not hold the machine line '$LINE'"
for w in "PASS: gen4randombattle leaf parity" "IDENTICAL:" "gen9 identity: PASS"; do
  grep -qF "$w" "$S/PREP-RESULT.txt" || fail checks "PREP-RESULT.txt has no '$w'"
done
[ "$(md5of "$NET")" = "$NET_MD5" ] && [ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "net md5 / sha256"
if [ -z "$MOCK" ]; then
  [ "${SLURM_CPUS_ON_NODE:-0}" -ge 24 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 24)"
  nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
  [ -d "$FROZEN/src/porygon/v2" ] || fail checks "frozen netv2 tree missing"
  [ -f "$CODE/scripts/fastenv/netsearch_slot.sh" ] || fail checks "frozen code"
fi
st "checks OK: δ* $MARGIN = DELTA = the PREREG's, study rows md5 $ROWS_MD5, net md5 + sha256, prep PASS lines"
A="lookahead:$NET,margin=$MARGIN,temperature=0.5,worlds=8,workers=11,leaf_device=cuda"
B="$NET,temperature=0.5"

plan() {  # NAME TAG PAIRS BLOCKS -> $S/NAME.env (Bet 1's plan: the lookahead arm vs the same raw net, fixed n)
  cat > "$S/$1.env" <<EOF
NAME=$1
GEN=4
TAG=$2
NETV2_DIR=$FROZEN
OPP='$B'
ARMS=('$A')
BLOCK_PAIRS=$3
BLOCKS=$4
CONC=3
FUTILITY=
MIN_CPUS=24
BLOCK_TIMEOUT_MIN=240
EOF
}
if [ -n "$MOCK" ]; then
  OUT=$MOCK/run
  st "MOCK: smoke and main skipped (run $OUT)"
else
  # ---- smoke (mechanical, 4 games)
  plan la2g4smoke "$SMOKE_TAG" 2 1
  SM=$S/smoke-run
  st "smoke start: $SM (2 pairs, tag $SMOKE_TAG)"
  bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2g4smoke.env" "$SM" > "$S/smoke.log" 2>&1 &
  wait $!; rc=$?
  sm=$("$PY" "$HELP" smoke "$SM" --games 4 2>&1)
  { echo "smoke $SM (exit $rc)"; echo "$sm"; cat "$SM/result.txt" 2>/dev/null; } > "$S/SMOKE.txt"
  st "smoke: $sm"
  case $sm in PASS*) ;; *) fail smoke "${sm:0:400}" ;; esac

  # ---- main: fixed n, 10 blocks of 100 duplicate pairs
  plan la2g4 "$TAG" 100 10
  OUT=$S/main-run
  st "main start: $OUT (1,000 pairs, fixed n, tag $TAG)"
  bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2g4.env" "$OUT" > "$S/main.log" 2>&1 &
  wait $!; rc=$?
  [ -s "$OUT/result.json" ] || fail main "no result.json (exit $rc; $OUT/run.log)"
  st "main done (exit $rc)"
fi

# ---- verdict (mechanical)
"$PY" "$HELP" verdict "$OUT" --delta "$MARGIN" --label "$LABEL" --seed "$TAG" > "$S/VERDICT.txt.tmp" 2> "$S/verdict.err" ||
  fail verdict "verdict script ($S/verdict.err)"
mv -f "$S/VERDICT.txt.tmp" "$S/VERDICT.txt" && chmod a-w "$S/VERDICT.txt"
V=$(head -1 "$S/VERDICT.txt")
V2=$(sed -n 2p "$S/VERDICT.txt")
V3=$(sed -n 3p "$S/VERDICT.txt")
echo "$V" > "$S/RESULT"
st "VERDICT: $V; $V2; $V3"
case $V in VOID*) fail verdict "$V" ;; esac
release "done: $V: $V2; $V3; $S/VERDICT.txt"
trap - TERM INT
st "chain done (ledger RELEASE line in $C/LEDGER-OUTBOX.md)"
exit 0
