#!/bin/bash
# LA2-2030 (docs/netsearch/LA2-2030-PREREG.md = runs/la2-2030/PREREG.txt, r--, md5 in PREREG.md5): LA2-1350's adoption
# test one net later (lookahead2_adopt.sbatch d44c739 with the constants changed), plus LA2-FP200, as ONE slot job:
#   prereg -> gate (line 2 of netv2's runs/conf2030/RESULTS.txt CONFIRMED and line 1 names md5 55940b3f, else
#   WITHDRAWN: nothing played) -> checks -> delta (lookahead2_delta.sh: Bet 1's δ rule on the new net -> DELTA) ->
#   smoke (2 pairs vs the benchmark bot + 2 pairs vs Foul Play at 200k iterations = 8 games) -> main (500 pairs vs the
#   benchmark bot, fixed n) -> verdict (VERDICT.txt line 1 ADOPTED (BETTER) / ADOPTED (NON-INFERIOR, margin 3 points) /
#   NOT ADOPTED / VOID) -> only if ADOPTED: fp200 (the new benchmark bot vs Foul Play gen9 at a FIXED 200,000 iterations
#   per state, 500 pairs fixed n, one automatic replay of an infrastructure VOID) -> FP200.txt -> one RELEASE line.
#   A (candidate) = lookahead2(G9q-cd@2035.04M, δ*, K 8, T 0.5); OPP = the benchmark bot lookahead2(G9q-cd@1355.04M,
#   δ 0.0075, K 8, T 0.5). Seed tags: h2h 8233571 (block j: 8233571b<j>), FP200 8233792; smokes <tag>smoke.
# Stage dir runs/la2-2030/ prepared before submission: PREREG.txt (r--) + PREREG.md5, code/ (git archive + COMMIT).
# Stage files: STATUS, WITHDRAWN, FAILED, DELTA, DELTA-VERDICT.txt, DELTA-RESULT, SMOKE.txt, VERDICT.txt (r--),
# FP200.txt (r--), VOID-<att>, RESULT; logs delta.log, smoke.log, smoke-fp.log, main.log, fp200-<att>.log.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch runs/la2-2030/code/scripts/fastenv/la2_2030.sbatch
# Laptop mock (tests/test_la2_2030.py; refused inside a Slurm job): LA22_MOCK=DIR runs the prereg, gate, verdict, the
# FP200 branch, its void replay and its reading on DIR/fastenv, DIR/netv2; the games are replaced by DIR/DELTA_IN,
# DIR/run (a result.json) and DIR/fprun-<att> (FP200 attempt att's run dir).
#SBATCH --job-name=ws-fastenv-la2x2030
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
NV=/vol/bitbucket/fr1226/porygon-ws-netv2
FQ=/vol/bitbucket/fr1226/porygon-ws-wsp/fpcal/code
PY=$G/.venv/bin/python
MOCK=
if [ -n "${LA22_MOCK:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  MOCK=$LA22_MOCK C=$LA22_MOCK/fastenv NV=$LA22_MOCK/netv2 FQ=$LA22_MOCK/wsp PY=${LA22_MOCK_PY:-python3}
fi
S=$C/runs/la2-2030
CODE=$S/code
HELP=$CODE/scripts/fastenv/lookahead2_adopt.py
GATE_FILE=$NV/runs/conf2030/RESULTS.txt
GATE_MD5=55940b3f  # netv2's net 02035041391.pt, named on line 1 of RESULTS.txt
NET=$C/nets/g9q-cd-2035.04M.pt; NET_SHA=de82e0dd75b0ca79e35938d4085599ffc37c881cb60ea8985ddea198db5df7a6
BOT=$C/nets/g9q-cd-1355.04M.pt; BOT_SHA=23593a3138cd17a5a010e42ca9cca1604b4ee98081e64df1c4ee7f49533886cd
BOT_DELTA=0.0075
OPP="lookahead:$BOT,margin=$BOT_DELTA,temperature=0.5,worlds=8,workers=11,leaf_device=cuda"
FROZEN=$C/frozen/netv2-fba02fb-20261110
REF=$C/runs/la2-1350/code  # the benchmark bot's frozen code (LA2-1350, d44c739)
BENCH=$C/lookahead2/study-cd2035-20261123
TAG=8233571 SMOKE_TAG=8233571smoke
FP_TAG=8233792 FP_SMOKE_TAG=8233792smoke
FP_ITERS=200000 FP_CONC=8
FPOPP="foulplay:ms=1000,parallelism=1,iters=$FP_ITERS"
FWRAP=$FQ/fp_fixed_iters.py FWRAP_MD5=f9c32cdf FCHK=$FQ/fp_iters_check.py FCHK_MD5=a1ff4886
FPI=$FQ/fp9_iters.sbatch FPI_MD5=82a207a6  # wsp's launcher, the reference the two files above come with
if [ -n "$MOCK" ]; then  # laptop mock only: the mock's stand-in files
  FWRAP_MD5=${LA22_MOCK_FWRAP_MD5:-$FWRAP_MD5} FCHK_MD5=${LA22_MOCK_FCHK_MD5:-$FCHK_MD5} FPI_MD5=${LA22_MOCK_FPI_MD5:-$FPI_MD5}
fi
LABEL=G9q-cd@2035.04M BOT_LABEL=G9q-cd@1355.04M
J=${SLURM_JOB_ID:-mock}
cd "$C" || exit 1
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$S/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
release() {  # one ledger line; the free text may not hold the field separator or a newline
  local m=${1//|//}; m=${m//$'\n'/ }; m=${m:0:700}
  outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $J | LA2-2030 $m; slot FREE"
}
V1=
fail() {
  st "FAILED at stage $1: $2"; echo "$(now) stage $1: $2 (job $J)" >> "$S/FAILED"
  release "FAILED at stage $1 ($2); $S/FAILED${V1:+; stage 1 stands: ${V1:0:200}}"
  exit 1
}
md5of() { md5sum < "$1" | cut -c1-32; }
for f in VERDICT.txt FP200.txt FAILED WITHDRAWN DELTA smoke-run smoke-fp-run main-run fp-run; do
  [ ! -e "$S/$f" ] || { echo "$S already holds a chain ($f)" >&2; release "refused: $S already holds a chain ($f); nothing run"; exit 1; }
done
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT" 2>/dev/null)${MOCK:+ (MOCK $MOCK)}"

# ---- prereg
[ -f "$S/PREREG.txt" ] && [ ! -w "$S/PREREG.txt" ] || fail prereg "PREREG.txt missing or writable"
[ -s "$S/PREREG.md5" ] && [ "$(md5of "$S/PREREG.txt")" = "$(cut -c1-32 "$S/PREREG.md5")" ] || fail prereg "PREREG md5 != PREREG.md5"
[ -f "$HELP" ] || fail prereg "frozen code $CODE missing"
st "prereg OK: PREREG md5 $(md5of "$S/PREREG.txt")"

# ---- gate (PREREG): play only on F2030 CONFIRMED for this net
g=$("$PY" "$HELP" gate "$GATE_FILE" --md5 "$GATE_MD5" --name F2030 2>&1); grc=$?
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

# ---- checks (the launcher files are checked in the mock too)
[ "$(md5of "$FWRAP" | cut -c1-8)" = "$FWRAP_MD5" ] && [ "$(md5of "$FCHK" | cut -c1-8)" = "$FCHK_MD5" ] &&
  [ "$(md5of "$FPI" | cut -c1-8)" = "$FPI_MD5" ] || fail checks "wsp fixed-iteration launcher md5 (fp_fixed_iters / fp_iters_check / fp9_iters)"
plan() {  # NAME TAG PAIRS BLOCKS OPP CONC -> $S/NAME.env (LA2-H's plan)
  cat > "$S/$1.env" <<EOF
NAME=$1
GEN=9
TAG=$2
NETV2_DIR=$FROZEN
OPP='$5'
ARMS=('$A')
BLOCK_PAIRS=$3
BLOCKS=$4
CONC=$6
FUTILITY=
MIN_CPUS=24
BLOCK_TIMEOUT_MIN=300
EOF
}
if [ -n "$MOCK" ]; then
  cp "$MOCK/DELTA_IN" "$S/DELTA" && chmod a-w "$S/DELTA" || fail delta "MOCK: no DELTA_IN"
  OUT=$MOCK/run
  st "MOCK: checks, δ study, smoke and main skipped (δ* $(head -1 "$S/DELTA"), run $OUT)"
  MARGIN=$(head -1 "$S/DELTA")
  [[ "$MARGIN" =~ ^[0-9]+\.?[0-9]*$ ]] || fail delta "δ* not a number: $MARGIN"
  A="lookahead:$NET,margin=$MARGIN,temperature=0.5,worlds=8,workers=11,leaf_device=cuda"
else
  [ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "$LABEL sha256"
  [ "$(sha256sum "$BOT" | cut -c1-64)" = "$BOT_SHA" ] || fail checks "$BOT_LABEL sha256"
  [ "${SLURM_CPUS_ON_NODE:-0}" -ge 24 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 24)"
  nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
  [ -d "$FROZEN/src/porygon/v2" ] || fail checks "frozen netv2 tree missing"
  [ -f "$CODE/scripts/fastenv/netsearch_slot.sh" ] && [ -f "$CODE/scripts/fastenv/lookahead2_delta.sh" ] || fail checks "frozen code"
  for m in lookahead2.py netsearch.py v3leaf.py; do
    cmp -s "$CODE/src/porygon/$m" "$REF/src/porygon/$m" || fail checks "src/porygon/$m differs from the benchmark bot's code $REF"
  done
  [ ! -e "$BENCH" ] || fail checks "bench dir $BENCH exists"
  st "checks OK: nets sha256, ${SLURM_CPUS_ON_NODE} CPUs, GPU, lookahead2 / netsearch / v3leaf = $REF, wsp launcher md5s"

  # ---- delta: Bet 1's δ rule, unchanged, on the new net; DELTA before any game
  st "delta start: lookahead2_delta.sh on $LABEL, bench $BENCH"
  LA2_FROZEN=1 LA2D_STAGE=$S LA2D_NET=$NET LA2D_NET_SHA=$NET_SHA LA2D_LABEL=$LABEL LA2D_OUT=$BENCH \
    LA2D_PREFIX=DELTA- LA2D_CANCEL=0 bash "$CODE/scripts/fastenv/lookahead2_delta.sh" > "$S/delta.log" 2>&1 &
  wait $!; rc=$?
  [ "$rc" = 0 ] && [ -s "$S/DELTA" ] && [ ! -e "$S/FAILED" ] || fail delta "lookahead2_delta.sh exit $rc ($S/delta.log)"
  MARGIN=$(head -1 "$S/DELTA")
  [[ "$MARGIN" =~ ^[0-9]+\.?[0-9]*$ ]] || fail delta "δ* not a number: $MARGIN"
  st "delta done: δ* $MARGIN ($(head -1 "$S/DELTA-VERDICT.txt" 2>/dev/null | cut -c1-220))"
  A="lookahead:$NET,margin=$MARGIN,temperature=0.5,worlds=8,workers=11,leaf_device=cuda"

  # ---- smoke (mechanical, 8 games): 2 pairs vs the benchmark bot, then 2 pairs vs Foul Play at 200k iterations
  plan la2x2030smoke "$SMOKE_TAG" 2 1 "$OPP" 3
  SM=$S/smoke-run
  st "smoke start: $SM (2 pairs vs the benchmark bot, tag $SMOKE_TAG)"
  bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2x2030smoke.env" "$SM" > "$S/smoke.log" 2>&1 &
  wait $!; rc=$?
  sm=$("$PY" "$HELP" smoke "$SM" --games 4 2>&1)
  plan la2fp200smoke "$FP_SMOKE_TAG" 2 1 "$FPOPP" "$FP_CONC"
  SMF=$S/smoke-fp-run
  st "smoke start: $SMF (2 pairs vs Foul Play at $FP_ITERS iterations, tag $FP_SMOKE_TAG)"
  bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2fp200smoke.env" "$SMF" > "$S/smoke-fp.log" 2>&1 &
  wait $!; rcf=$?
  smf=$("$PY" "$HELP" smoke "$SMF" --games 4 --fp 2>&1)
  { echo "smoke $SM (exit $rc)"; echo "$sm"; cat "$SM/result.txt" 2>/dev/null
    echo "smoke $SMF (exit $rcf)"; echo "$smf"; cat "$SMF/result.txt" 2>/dev/null
    cat "$SMF"/arm-1/block-*/ITERS_OK 2>/dev/null; } > "$S/SMOKE.txt"
  st "smoke vs the benchmark bot: $sm"
  st "smoke vs Foul Play $FP_ITERS: $smf"
  case $sm in PASS*) ;; *) fail smoke "vs the benchmark bot: ${sm:0:400}" ;; esac
  case $smf in PASS*) ;; *) fail smoke "vs Foul Play: ${smf:0:400}" ;; esac

  # ---- main: fixed n, 5 blocks of 100 duplicate pairs vs the benchmark bot
  plan la2x2030 "$TAG" 100 5 "$OPP" 3
  OUT=$S/main-run
  st "main start: $OUT (500 pairs, fixed n, tag $TAG)"
  bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2x2030.env" "$OUT" > "$S/main.log" 2>&1 &
  wait $!; rc=$?
  [ -s "$OUT/result.json" ] || fail main "no result.json (exit $rc; $OUT/run.log)"
  st "main done (exit $rc)"
fi

# ---- verdict (stage 1, mechanically)
"$PY" "$HELP" verdict "$OUT" --delta "$MARGIN" --label "$LABEL" --bot-label "$BOT_LABEL" --bot-delta "$BOT_DELTA" \
  > "$S/VERDICT.txt.tmp" 2> "$S/verdict.err" || fail verdict "verdict script ($S/verdict.err)"
mv -f "$S/VERDICT.txt.tmp" "$S/VERDICT.txt" && chmod a-w "$S/VERDICT.txt"
V=$(head -1 "$S/VERDICT.txt")
echo "$V" > "$S/RESULT"
st "VERDICT: $V"
case $V in VOID*) fail verdict "$V" ;; esac
V1=$V
case $V1 in
  "ADOPTED "*) ;;
  *)
    st "FP200 not run: stage 1 did not adopt"
    echo "FP200 not run (stage 1 NOT ADOPTED)" >> "$S/RESULT"
    release "done: $V1; FP200 not run; $S/VERDICT.txt"
    trap - TERM INT
    st "chain done (ledger RELEASE line in $C/LEDGER-OUTBOX.md)"
    exit 0 ;;
esac

# ---- fp200 (stage 2, a measurement): the new benchmark bot vs Foul Play gen9 at a FIXED 200,000 iterations per state
games_in() { cat "$1"/arm-*/block-*/results.jsonl 2>/dev/null | grep -vc '"status": "no_accept"'; }
is_void() {  # RUN: an infrastructure VOID = HARNESS_FAIL, 0 games, a Foul Play instance that did not start on handshake timeouts
  local n; n=$(games_in "$1"); n=${n:-0}
  [ -e "$1/HARNESS_FAIL" ] && [ "$n" = 0 ] && grep -qs "did not start" "$1/run.log" &&
    { grep -qs "timed out during handshake" "$1/run.log" ||
      zcat -f "$1"/arm-*/block-*/fp/*.log* 2>/dev/null | grep -q "timed out during handshake"; }
}
R=$S/fp-run voids=0 vtext=
for att in 1 2; do
  st "fp200 attempt $att: $R (500 pairs vs Foul Play gen9 at a fixed $FP_ITERS iterations per state, fixed n, tag $FP_TAG, $FP_CONC instances)"
  if [ -n "$MOCK" ]; then
    { [ -d "$MOCK/fprun-$att" ] && cp -a "$MOCK/fprun-$att" "$R"; } || fail fp200 "MOCK: no fprun-$att"
  else
    plan la2fp200 "$FP_TAG" 100 5 "$FPOPP" "$FP_CONC"
    bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2fp200.env" "$R" > "$S/fp200-$att.log" 2>&1 &
    wait $!; rc=$?
    st "fp200 attempt $att: harness exit $rc, $(games_in "$R") games"
  fi
  if is_void "$R"; then
    voids=$((voids + 1))
    echo "$(now) attempt $att (run $R): a Foul Play instance did not start (websocket handshake timeouts), 0 games: INFRASTRUCTURE VOID" > "$S/VOID-$att"
    vtext="$vtext | $(cat "$S/VOID-$att")"
    st "VOID attempt $att: $(cat "$S/VOID-$att")"
    if [ "$att" = 1 ]; then
      mv "$R" "$S/fp-run-void1" || fail fp200 "cannot move the void run aside"
      st "replaying once, same seed $FP_TAG and settings"; [ -n "$MOCK" ] || sleep 60; continue
    fi
    fail fp200 "two infrastructure voids (handshake timeouts): nothing read"
  fi
  break
done
[ -s "$R/result.json" ] || fail fp200 "no result.json ($R/run.log)"
"$PY" "$HELP" fp200 "$R" --delta "$MARGIN" --label "$LABEL" --iters "$FP_ITERS" --seed "$FP_TAG" --voids "$voids" \
  --voids-text "$vtext" > "$S/FP200.txt.tmp" 2> "$S/fp200.err" || fail fp200 "fp200 script ($S/fp200.err)"
mv -f "$S/FP200.txt.tmp" "$S/FP200.txt" && chmod a-w "$S/FP200.txt"
F1=$(head -1 "$S/FP200.txt")
echo "$F1" >> "$S/RESULT"
st "FP200: $F1"
case $F1 in VOID*) fail fp200 "$F1" ;; esac
release "done: $V1; FP200: ${F1#LA2-FP200: }; $S/VERDICT.txt, $S/FP200.txt"
trap - TERM INT
st "chain done (ledger RELEASE line in $C/LEDGER-OUTBOX.md)"
exit 0
