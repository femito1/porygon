#!/bin/bash
# Scale gate (docs/netsearch/SCALE-GATE-PREREG.md = runs/scale-gate/PREREG.txt, read-only, md5 in PREREG.md5).
# One cluster-side chain per mode, both on the frozen code runs/scale-gate/code (made once before submission):
#   A (the slot, 24 CPUs): arm 1 = D4 (deepen budget 4096) vs D1 (deepen budget 1024)          -> T1
#   B (fastenv's helper, 12 CPUs): arm 1 = lookahead2, arm 2 = the raw net, both vs D1         -> T2, T3
# Stages: checks -> smoke (2 pairs per arm, tag scalesmoke20261117, read mechanically) -> main (tag scale20261117,
# blocks of 100 pairs, Bet 1's sequential design, at most 9 blocks) -> part verdict (scale_gate_verdict.py part) ->
# combined VERDICT.txt (r--) by whichever chain finishes last (FLAT on T1 needs no B: A then stops B via B-run/STOP and
# writes it at once). A writes its ledger RELEASE line to $C/LEDGER-OUTBOX.md. Stage files in runs/scale-gate/:
# STATUS-{A,B}, SMOKE-{A,B}.txt, VERDICT-{A,B}.txt, RESULT-{A,B}, part-{A,B}.json, VERDICT.txt, RESULT, FAILED-{A,B}.
#   cd $C && sbatch --dependency=afterany:1173956 runs/scale-gate/code/scripts/fastenv/scale_gate.sbatch A
#   cd $C && sbatch --nice=500 --cpus-per-task=12 --mem=32G runs/scale-gate/code/scripts/fastenv/scale_gate.sbatch B
#SBATCH --job-name=ws-fastenv-scale
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=60:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
MODE=${1:?usage: scale_gate.sbatch A|B}
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
PY=$G/.venv/bin/python
S=$C/runs/scale-gate
CODE=$S/code
cd "$C" || exit 1
J=${SLURM_JOB_ID:-none}
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$S/STATUS-$MODE"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
fail() {
  st "FAILED at stage $1: $2"; echo "$(now) stage $1: $2 (job $J)" > "$S/FAILED-$MODE"
  [ "$MODE" != A ] || outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $J | scale gate chain A FAILED at stage $1 ($2); $S/FAILED-A; no verdict; slot FREE"
  exit 1
}
case $MODE in A|B) ;; *) echo "unknown mode $MODE" >&2; exit 2 ;; esac
[ ! -e "$S/RESULT-$MODE" ] && [ ! -e "$S/FAILED-$MODE" ] && [ ! -e "$S/$MODE-run" ] && [ ! -e "$S/smoke-$MODE-run" ] \
  || { echo "$S already holds chain $MODE" >&2; exit 1; }
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
md5of() { md5sum < "$1" | cut -c1-32; }
NET=$C/nets/g9q-cd-805.03M.pt; NET_SHA=e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f
FROZEN=$C/frozen/netv2-fba02fb-20261110
DESIGN_SRC=$CODE/scripts/fastenv/plans/seq-la2-bet1.json; DESIGN_MD5=f0911c195e850d1e11ea11359c0b0100
if [ "$MODE" = A ]; then W=10; NEED=24; else W=5; NEED=12; fi
D1="deepen:$NET,budget=1024,margin=0.005,temperature=0.5,worlds=8,workers=$W,leaf_device=cuda"
D4="deepen:$NET,budget=4096,margin=0.005,temperature=0.5,worlds=8,workers=$W,leaf_device=cuda"
L="lookahead:$NET,margin=0.005,temperature=0.5,worlds=8,workers=$W,leaf_device=cuda"
RAW="$NET,temperature=0.5"
if [ "$MODE" = A ]; then ARMS="('$D4')"; else ARMS="('$L' '$RAW')"; fi
st "chain $MODE start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT" 2>/dev/null)"

# ---- checks
[ -f "$S/PREREG.txt" ] && [ ! -w "$S/PREREG.txt" ] || fail checks "PREREG.txt missing or writable"
[ -s "$S/PREREG.md5" ] && [ "$(md5of "$S/PREREG.txt")" = "$(cut -c1-32 "$S/PREREG.md5")" ] || fail checks "PREREG md5 != PREREG.md5"
[ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "net sha256"
[ "$(md5of "$DESIGN_SRC")" = "$DESIGN_MD5" ] || fail checks "design md5"
[ "$(head -1 "$C/runs/la2-delta800/DELTA")" = 0.005 ] || fail checks "lookahead2 δ* is not 0.005"
[ "${SLURM_CPUS_ON_NODE:-0}" -ge "$NEED" ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want $NEED)"
nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
[ -d "$FROZEN/src/porygon/v2" ] && [ -f "$CODE/src/porygon/deepen.py" ] || fail checks "frozen trees missing"
st "checks OK: PREREG md5 $(md5of "$S/PREREG.txt"), deepen.py sha256 $(sha256sum "$CODE/src/porygon/deepen.py" | cut -c1-16), workers $W per side"

plan() {  # NAME TAG PAIRS BLOCKS -> $S/NAME.env
  cat > "$S/$1.env" <<EOF
NAME=$1
GEN=9
TAG=$2
NETV2_DIR=$FROZEN
OPP='$D1'
ARMS=$ARMS
BLOCK_PAIRS=$3
BLOCKS=$4
CONC=3
FUTILITY=
MIN_CPUS=$NEED
BLOCK_TIMEOUT_MIN=480
SEQ_DESIGN=$DESIGN_SRC
EOF
}

# ---- smoke (mechanical)
plan scalesmoke$MODE scalesmoke20261117 2 1
SM=$S/smoke-$MODE-run
st "smoke start: $SM (2 pairs per arm)"
bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/scalesmoke$MODE.env" "$SM" > "$S/smoke-$MODE.log" 2>&1 &
wait $!; rc=$?
sm=$("$PY" - "$SM" <<'PYEOF' 2>&1
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
r = json.loads((run / "result.json").read_text())
checks = {}
def searched(s, tag, deep):
    checks[f"{tag} searched/leaves > 0"] = s.get("searched", 0) > 0 and s.get("leaves", 0) > 0
    checks[f"{tag} failed worlds 0"] = s.get("failed_worlds", 0) == 0
    checks[f"{tag} fallbacks <= max(2, 2%)"] = s.get("fallbacks", 0) <= max(2, 0.02 * s.get("decisions", 0))
    if deep:
        checks[f"{tag} extra leaves > 0"] = s.get("extra_leaves", 0) > 0
for a in r["arms"]:
    k = a["arm"]
    ev = "\n".join(p.read_text() for p in run.glob(f"arm-{k}/block-*/evaluate.log"))
    checks[f"arm {k} 4 games"] = a["games"] == 4
    checks[f"arm {k} no Traceback"] = "Traceback" not in ev
    if a["spec"].startswith(("deepen:", "lookahead:")):
        searched(a["stats_a"], f"arm {k}", a["spec"].startswith("deepen:"))
        checks[f"arm {k} gate rows"] = a["gate"]["decisions"] > 0
    searched(a["stats_b"], f"arm {k} opponent D1", True)
    checks[f"arm {k} opponent gate rows"] = a["gate_opp"]["decisions"] > 0
checks["no HARNESS_FAIL"] = not (run / "HARNESS_FAIL").exists()
ok = all(checks.values())
print(("PASS" if ok else "FAIL"), "; ".join(f"{k} {'ok' if v else 'NO'}" for k, v in checks.items() if not v) or "all checks ok",
      "|", "; ".join(f"arm {a['arm']}: A {a['stats_a'].get('searched')} searched, {a['stats_a'].get('extra_leaves')} extra; "
                     f"D1 {a['stats_b'].get('searched')} searched, {a['stats_b'].get('extra_leaves')} extra" for a in r["arms"]))
PYEOF
)
{ echo "smoke $SM (netsearch_slot.sh exit $rc)"; echo "$sm"; cat "$SM/result.txt" 2>/dev/null; } > "$S/SMOKE-$MODE.txt"
st "smoke: $sm"
case $sm in PASS*) ;; *) fail smoke "$sm" ;; esac

# ---- main
plan scale$MODE scale20261117 100 9
OUT=$S/$MODE-run
st "main start: $OUT (blocks of 100 pairs, at most 9; tag scale20261117)"
if [ "$MODE" = A ]; then
  # T1 FUTILITY needs no chain B: stop B (after its current block) as soon as arm 1 of A decides FUTILITY
  ( while [ ! -e "$OUT/arm-1/SEQ_DECISION" ]; do sleep 300; [ -e "$S/RESULT-A" ] && exit 0; done
    if grep -q "FUTILITY" "$OUT/arm-1/SEQ_DECISION" && [ -d "$S/B-run" ]; then touch "$S/B-run/STOP"; st "T1 FUTILITY: chain B told to stop (B-run/STOP)"; fi ) &
  WATCH=$!
fi
bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/scale$MODE.env" "$OUT" > "$S/main-$MODE.log" 2>&1 &
wait $!; rc=$?
[ -s "$OUT/result.json" ] || fail main "no result.json (netsearch_slot.sh exit $rc; $OUT/run.log)"
st "main done (exit $rc)"

# ---- verdicts
"$PY" "$CODE/scripts/fastenv/scale_gate_verdict.py" part "$MODE" "$OUT" --json "$S/part-$MODE.json" > "$S/VERDICT-$MODE.txt.tmp" 2> "$S/verdict-$MODE.err" \
  || fail verdict "scale_gate_verdict.py part (exit $?; $S/verdict-$MODE.err)"
mv -f "$S/VERDICT-$MODE.txt.tmp" "$S/VERDICT-$MODE.txt" && chmod a-w "$S/VERDICT-$MODE.txt"
head -1 "$S/VERDICT-$MODE.txt" > "$S/RESULT-$MODE"
st "part verdict: $(head -1 "$S/VERDICT-$MODE.txt")"
if [ "$MODE" = A ] && grep -q "^T1 FUTILITY" "$S/RESULT-A" && [ -d "$S/B-run" ] && [ ! -e "$S/RESULT-B" ]; then touch "$S/B-run/STOP"; fi
(
  flock 8
  if [ ! -e "$S/VERDICT.txt" ]; then
    args=("$S/part-A.json"); [ -e "$S/part-B.json" ] && args+=("$S/part-B.json")
    if [ -e "$S/part-A.json" ]; then
      v=$("$PY" "$CODE/scripts/fastenv/scale_gate_verdict.py" combine "${args[@]}")
      case $v in PENDING*) st "combined: $(echo "$v" | head -1) (the other chain writes VERDICT.txt)" ;;
        *) printf '%s\n' "$v" > "$S/VERDICT.txt"; chmod a-w "$S/VERDICT.txt"; head -1 "$S/VERDICT.txt" > "$S/RESULT"
           st "VERDICT: $(head -1 "$S/VERDICT.txt")" ;; esac
    else
      st "chain A not finished: it writes VERDICT.txt"
    fi
  fi
) 8> "$S/verdict.lock"
trap - TERM INT
[ -z "${WATCH:-}" ] || kill "$WATCH" 2>/dev/null
if grep -q VOID "$S/RESULT-$MODE"; then
  if [ "$MODE" = B ] && [ -e "$OUT/STOP" ]; then st "chain B stopped by T1 FUTILITY (no T2 decision needed)"
  else fail verdict "$(head -1 "$S/RESULT-$MODE")"; fi
fi
[ "$MODE" != A ] || outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $J | scale gate chain A done: $(head -1 "$S/RESULT-A"); combined $(head -1 "$S/RESULT" 2>/dev/null || echo 'pending chain B'); slot FREE"
st "chain $MODE done"
exit 0
