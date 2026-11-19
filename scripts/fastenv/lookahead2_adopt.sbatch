#!/bin/bash
# LA2-1350 (docs/netsearch/LA2-1350-PREREG.md = runs/la2-1350/PREREG.txt, r--, md5 in PREREG.md5): the LA2-H-style
# adoption test of a new gen9 deployment net, as ONE cluster-side slot job that needs no agent:
#   prereg -> gate (line 2 of netv2's runs/conf1350/RESULTS.txt CONFIRMED, else WITHDRAWN: nothing played) -> checks ->
#   delta (lookahead2_delta.sh: Bet 1's δ rule on the new net -> DELTA) -> smoke (2 pairs) -> main (500 pairs, fixed n)
#   -> verdict (lookahead2_adopt.py: line 1 ADOPTED (BETTER) / ADOPTED (NON-INFERIOR, margin 3 points) / NOT ADOPTED /
#   VOID) -> one RELEASE line in $C/LEDGER-OUTBOX.md (also after WITHDRAWN or FAILED).
#   A (candidate) = lookahead2(G9q-cd@1355.04M, δ*, K 8, T 0.5); OPP = the benchmark bot lookahead2(G9q-cd@805.03M, δ 0.005,
#   K 8, T 0.5). Seed tag 8135444: block j plays on a fresh duplicate server seeded 8135444b<j>.
# Stage dir runs/la2-1350/ prepared before submission: PREREG.txt (r--) + PREREG.md5, code/ (src + scripts + COMMIT, the
# commit's tree; the δ stage and both players run from it). Stage files: STATUS, WITHDRAWN, FAILED, DELTA,
# DELTA-VERDICT.txt, DELTA-RESULT, SMOKE.txt, VERDICT.txt (r--), RESULT; logs delta.log, smoke.log, main.log.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch --dependency=afterany:1238699 runs/la2-1350/code/scripts/fastenv/lookahead2_adopt.sbatch
# Laptop mock (tests/test_lookahead2_adopt.py; refused inside a Slurm job): LA2A_MOCK=DIR runs the prereg, gate and
# verdict stages on DIR/fastenv and DIR/netv2, the stages between replaced by DIR/DELTA_IN and DIR/run (a result.json).
#SBATCH --job-name=ws-fastenv-la2x1350
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
NV=/vol/bitbucket/fr1226/porygon-ws-netv2
PY=$G/.venv/bin/python
MOCK=
if [ -n "${LA2A_MOCK:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  MOCK=$LA2A_MOCK C=$LA2A_MOCK/fastenv NV=$LA2A_MOCK/netv2 PY=${LA2A_MOCK_PY:-python3}
fi
S=$C/runs/la2-1350
CODE=$S/code
HELP=$CODE/scripts/fastenv/lookahead2_adopt.py
GATE_FILE=$NV/runs/conf1350/RESULTS.txt
GATE_MD5=6cf40c0c  # netv2's net 01355036714.pt, named on line 1 of RESULTS.txt
NET=$C/nets/g9q-cd-1355.04M.pt; NET_SHA=23593a3138cd17a5a010e42ca9cca1604b4ee98081e64df1c4ee7f49533886cd
BOT=$C/nets/g9q-cd-805.03M.pt; BOT_SHA=e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f
BOT_DELTA=0.005
OPP="lookahead:$BOT,margin=$BOT_DELTA,temperature=0.5,worlds=8,workers=11,leaf_device=cuda"
FROZEN=$C/frozen/netv2-fba02fb-20261110
REF=$C/runs/la2-h/code  # the benchmark bot's frozen code (LA2-H, 36a2006)
BENCH=$C/lookahead2/study-cd1355-20261119
TAG=8135444 SMOKE_TAG=8135444smoke
J=${SLURM_JOB_ID:-mock}
cd "$C" || exit 1
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$S/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
release() {  # one ledger line; the free text may not hold the field separator or a newline
  local m=${1//|//}; m=${m//$'\n'/ }; m=${m:0:600}
  outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $J | LA2-1350 $m; slot FREE"
}
fail() {
  st "FAILED at stage $1: $2"; echo "$(now) stage $1: $2 (job $J)" >> "$S/FAILED"
  release "FAILED at stage $1 ($2); $S/FAILED; no verdict"
  exit 1
}
md5of() { md5sum < "$1" | cut -c1-32; }
for f in VERDICT.txt FAILED WITHDRAWN DELTA smoke-run main-run; do
  [ ! -e "$S/$f" ] || { echo "$S already holds a chain ($f)" >&2; release "refused: $S already holds a chain ($f); nothing run"; exit 1; }
done
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT" 2>/dev/null)${MOCK:+ (MOCK $MOCK)}"

# ---- prereg
[ -f "$S/PREREG.txt" ] && [ ! -w "$S/PREREG.txt" ] || fail prereg "PREREG.txt missing or writable"
[ -s "$S/PREREG.md5" ] && [ "$(md5of "$S/PREREG.txt")" = "$(cut -c1-32 "$S/PREREG.md5")" ] || fail prereg "PREREG md5 != PREREG.md5"
[ -f "$HELP" ] || fail prereg "frozen code $CODE missing"
st "prereg OK: PREREG md5 $(md5of "$S/PREREG.txt")"

# ---- gate (PREREG design 1): play only on F1350 CONFIRMED
g=$("$PY" "$HELP" gate "$GATE_FILE" --md5 "$GATE_MD5" 2>&1); grc=$?
st "gate (exit $grc): $g"
case $grc in
  0) ;;
  3)
    why=${g#WITHDRAWN: }
    { echo "WITHDRAWN: $why"; echo "  gate file $GATE_FILE; job $J; $(now); nothing played"; } > "$S/WITHDRAWN"
    echo "WITHDRAWN: $why" > "$S/RESULT"
    release "WITHDRAWN ($why); nothing played; $S/WITHDRAWN"
    trap - TERM INT
    st "chain done: WITHDRAWN (ledger RELEASE line in $C/LEDGER-OUTBOX.md)"
    exit 0 ;;
  *) fail gate "gate script exit $grc: $g" ;;
esac

if [ -n "$MOCK" ]; then
  cp "$MOCK/DELTA_IN" "$S/DELTA" && chmod a-w "$S/DELTA" || fail delta "MOCK: no DELTA_IN"
  OUT=$MOCK/run
  st "MOCK: checks, δ study, smoke and main skipped (δ* $(head -1 "$S/DELTA"), run $OUT)"
  MARGIN=$(head -1 "$S/DELTA")
  [[ "$MARGIN" =~ ^[0-9]+\.?[0-9]*$ ]] || fail delta "δ* not a number: $MARGIN"
else
  # ---- checks
  [ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "G9q-cd@1355.04M sha256"
  [ "$(sha256sum "$BOT" | cut -c1-64)" = "$BOT_SHA" ] || fail checks "G9q-cd@805.03M sha256"
  [ "${SLURM_CPUS_ON_NODE:-0}" -ge 24 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 24)"
  nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
  [ -d "$FROZEN/src/porygon/v2" ] || fail checks "frozen netv2 tree missing"
  [ -f "$CODE/scripts/fastenv/netsearch_slot.sh" ] && [ -f "$CODE/scripts/fastenv/lookahead2_delta.sh" ] || fail checks "frozen code"
  for m in lookahead2.py netsearch.py v3leaf.py; do
    cmp -s "$CODE/src/porygon/$m" "$REF/src/porygon/$m" || fail checks "src/porygon/$m differs from the benchmark bot's code $REF"
  done
  [ ! -e "$BENCH" ] || fail checks "bench dir $BENCH exists"
  st "checks OK: nets sha256, ${SLURM_CPUS_ON_NODE} CPUs, GPU, lookahead2 / netsearch / v3leaf = $REF"

  # ---- delta (PREREG design 2): Bet 1's δ rule, unchanged, on the new net; DELTA before any game
  st "delta start: lookahead2_delta.sh on G9q-cd@1355.04M, bench $BENCH"
  LA2_FROZEN=1 LA2D_STAGE=$S LA2D_NET=$NET LA2D_NET_SHA=$NET_SHA LA2D_LABEL=G9q-cd@1355.04M LA2D_OUT=$BENCH \
    LA2D_PREFIX=DELTA- LA2D_CANCEL=0 bash "$CODE/scripts/fastenv/lookahead2_delta.sh" > "$S/delta.log" 2>&1 &
  wait $!; rc=$?
  [ "$rc" = 0 ] && [ -s "$S/DELTA" ] && [ ! -e "$S/FAILED" ] || fail delta "lookahead2_delta.sh exit $rc ($S/delta.log)"
  MARGIN=$(head -1 "$S/DELTA")
  [[ "$MARGIN" =~ ^[0-9]+\.?[0-9]*$ ]] || fail delta "δ* not a number: $MARGIN"
  st "delta done: δ* $MARGIN ($(head -1 "$S/DELTA-VERDICT.txt" 2>/dev/null | cut -c1-220))"
  A="lookahead:$NET,margin=$MARGIN,temperature=0.5,worlds=8,workers=11,leaf_device=cuda"

  plan() {  # NAME TAG PAIRS BLOCKS -> $S/NAME.env (LA2-H's plan, the opponent = the benchmark bot)
    cat > "$S/$1.env" <<EOF
NAME=$1
GEN=9
TAG=$2
NETV2_DIR=$FROZEN
OPP='$OPP'
ARMS=('$A')
BLOCK_PAIRS=$3
BLOCKS=$4
CONC=3
FUTILITY=
MIN_CPUS=24
BLOCK_TIMEOUT_MIN=300
EOF
  }

  # ---- smoke (mechanical): 2 pairs = 4 games, both lookahead players in one client
  plan la2x1350smoke "$SMOKE_TAG" 2 1
  SM=$S/smoke-run
  st "smoke start: $SM (2 pairs, tag $SMOKE_TAG)"
  bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2x1350smoke.env" "$SM" > "$S/smoke.log" 2>&1 &
  wait $!; rc=$?
  sm=$("$PY" "$HELP" smoke "$SM" --games 4 2>&1)
  { echo "smoke $SM (exit $rc)"; echo "$sm"; cat "$SM/result.txt" 2>/dev/null; } > "$S/SMOKE.txt"
  st "smoke: $sm"
  case $sm in PASS*) ;; *) fail smoke "${sm:0:400}" ;; esac

  # ---- main (PREREG design 3): fixed n, 5 blocks of 100 duplicate pairs
  plan la2x1350 "$TAG" 100 5
  OUT=$S/main-run
  st "main start: $OUT (500 pairs, fixed n, tag $TAG)"
  bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2x1350.env" "$OUT" > "$S/main.log" 2>&1 &
  wait $!; rc=$?
  [ -s "$OUT/result.json" ] || fail main "no result.json (exit $rc; $OUT/run.log)"
  st "main done (exit $rc)"
fi

# ---- verdict (PREREG design 4, mechanically)
"$PY" "$HELP" verdict "$OUT" --delta "$MARGIN" --label G9q-cd@1355.04M --bot-label G9q-cd@805.03M --bot-delta "$BOT_DELTA" \
  > "$S/VERDICT.txt.tmp" 2> "$S/verdict.err" || fail verdict "verdict script ($S/verdict.err)"
mv -f "$S/VERDICT.txt.tmp" "$S/VERDICT.txt" && chmod a-w "$S/VERDICT.txt"
V1=$(head -1 "$S/VERDICT.txt")
echo "$V1" > "$S/RESULT"
st "VERDICT: $V1"
case $V1 in VOID*) fail verdict "$V1" ;; esac
release "done: $V1; $S/VERDICT.txt"
trap - TERM INT
st "chain done (ledger RELEASE line in $C/LEDGER-OUTBOX.md)"
exit 0
