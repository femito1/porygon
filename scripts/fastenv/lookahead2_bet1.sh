#!/bin/bash
# Bet 1 cluster-side chain (PLAN "Cluster-side pipelines"): lookahead2 on G9q-cd@605M vs the same raw net at T 0.5,
# pre-registered in runs/la2-bet1/PREREG.txt (read-only; its md5 = runs/la2-bet1/PREREG.md5, the signed-off value).
#   checks : PREREG md5, net sha256, design md5, >= 12 CPUs, a GPU;
#   smoke  : netsearch_slot.sh on la2smoke.env (2 pairs, own tag): read mechanically -> SMOKE.txt (FAIL stops);
#   main   : netsearch_slot.sh on la2bet1.env (blocks of 100 pairs, wsp's sequential design after every block);
#   verdict: lookahead2_verdict.py -> VERDICT.txt (r--; line 1 GO / NO-GO / VOID) + RESULT (line 1);
#   end    : cancels the helper job it runs in (unless runs/la2-bet1/KEEP_HELPER exists).
# Stage files in runs/la2-bet1/: STATUS (one line per stage), FAILED (any failure), SMOKE.txt, VERDICT.txt, RESULT.
# Run as a step of fastenv's helper job (cwd $C):
#   echo 'bash scripts/fastenv/lookahead2_bet1.sh' > jobq/la2-bet1.sh.tmp && mv jobq/la2-bet1.sh.tmp jobq/la2-bet1.sh
# The first call freezes src + scripts into runs/la2-bet1/code and re-runs itself from there, so later syncs of $C
# cannot change a running chain (every slot run also copies its own code from that frozen tree).
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
PY=$G/.venv/bin/python
S=$C/runs/la2-bet1
cd "$C" || exit 1
if [ -z "${LA2_FROZEN:-}" ]; then
  mkdir -p "$S"
  [ ! -e "$S/VERDICT.txt" ] && [ ! -e "$S/FAILED" ] && [ ! -e "$S/code" ] || { echo "$S already holds a chain" >&2; exit 1; }
  mkdir -p "$S/code" && rsync -a --exclude __pycache__ src scripts "$S/code/" && cp COMMIT "$S/code/COMMIT" || exit 1
  LA2_FROZEN=1 exec bash "$S/code/scripts/fastenv/lookahead2_bet1.sh"
fi
CODE=$S/code
NET=$C/nets/g9q-cd-605.04M.pt; NET_SHA=b7957be4c74dd37fc41892706c250d47f6a110558621e7f813e0ad9a9124001b
DESIGN_SRC=$CODE/scripts/fastenv/plans/seq-la2-bet1.json; DESIGN_MD5=f0911c195e850d1e11ea11359c0b0100
FROZEN=$C/frozen/netv2-fba02fb-20261110
MARGIN=@MARGIN@
WORLDS=8
A="lookahead:$NET,margin=$MARGIN,temperature=0.5,worlds=$WORLDS,workers=10,leaf_device=cuda"
B="$NET,temperature=0.5"
J=${SLURM_JOB_ID:-none}
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$S/STATUS"; }
fail() { st "FAILED at stage $1: $2"; echo "$(now) stage $1: $2 (job $J)" > "$S/FAILED"; exit 1; }
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
md5of() { md5sum < "$1" | cut -c1-32; }
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT")"

# ---- checks
[ -f "$S/PREREG.txt" ] && [ ! -w "$S/PREREG.txt" ] || fail checks "PREREG.txt missing or writable"
[ -s "$S/PREREG.md5" ] && [ "$(md5of "$S/PREREG.txt")" = "$(cut -c1-32 "$S/PREREG.md5")" ] || fail checks "PREREG md5 != PREREG.md5"
[ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "net sha256 mismatch"
[ "$(md5of "$DESIGN_SRC")" = "$DESIGN_MD5" ] || fail checks "design md5 mismatch"
[[ "$MARGIN" =~ ^[0-9]+\.?[0-9]*$ ]] || fail checks "margin not filled in ($MARGIN)"
[ "${SLURM_CPUS_ON_NODE:-0}" -ge 12 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 12)"
nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
[ -d "$FROZEN/src/porygon/v2" ] || fail checks "frozen netv2 tree missing"
cp "$DESIGN_SRC" "$S/seq-design.json" && chmod a-w "$S/seq-design.json"
st "checks OK: PREREG md5 $(md5of "$S/PREREG.txt"), net sha256 ok, design md5 ok, margin $MARGIN, worlds $WORLDS"

plan() {  # NAME TAG PAIRS BLOCKS -> $S/NAME.env
  cat > "$S/$1.env" <<EOF
NAME=$1
GEN=9
TAG=$2
NETV2_DIR=$FROZEN
OPP='$B'
ARMS=('$A')
BLOCK_PAIRS=$3
BLOCKS=$4
CONC=3
FUTILITY=
MIN_CPUS=12
BLOCK_TIMEOUT_MIN=240
SEQ_DESIGN=$S/seq-design.json
EOF
}

# ---- smoke
plan la2smoke la2smoke20261115 2 1
SM=$S/smoke-run
st "smoke start: $SM (2 pairs, tag la2smoke20261115)"
bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2smoke.env" "$SM" > "$S/smoke.log" 2>&1 &
wait $!; rc=$?
sm=$("$PY" - "$SM" <<'PYEOF' 2>&1
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
r = json.loads((run / "result.json").read_text())
a = r["arms"][0]
sa = a["stats_a"]
ev = "\n".join(p.read_text() for p in run.glob("arm-1/block-*/evaluate.log"))
look = run / "arm-1" / "seq-look-1.txt"
checks = {"4 games": a["games"] == 4, "searched > 0": sa.get("searched", 0) > 0, "leaves > 0": sa.get("leaves", 0) > 0,
          "gate rows": a["gate"]["decisions"] > 0, "fallbacks <= max(2, 2%)": sa.get("fallbacks", 0) <= max(2, 0.02 * sa.get("decisions", 0)),
          "no Traceback": "Traceback" not in ev, "seq look ran": look.exists() and "seq monitor" in look.read_text(),
          "no HARNESS_FAIL": not (run / "HARNESS_FAIL").exists()}
ok = all(checks.values())
print(("PASS" if ok else "FAIL"), "; ".join(f"{k} {'ok' if v else 'NO'}" for k, v in checks.items()),
      f"| A decisions {sa.get('decisions')} searched {sa.get('searched')} overrides {sa.get('overrides')} fallbacks "
      f"{sa.get('fallbacks')} leaves {sa.get('leaves')} la_seconds {sa.get('la_seconds', 0):.1f}")
PYEOF
)
{ echo "smoke $SM (netsearch_slot.sh exit $rc)"; echo "$sm"; cat "$SM/result.txt" 2>/dev/null; } > "$S/SMOKE.txt"
st "smoke: $sm"
case $sm in PASS*) ;; *) fail smoke "$sm" ;; esac

# ---- main
plan la2bet1 la2b120261115 100 9
OUT=$S/main-run
st "main start: $OUT (blocks of 100 pairs, at most 9; sequential design n_max 800 pairs; tag la2b120261115)"
bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/la2bet1.env" "$OUT" > "$S/main.log" 2>&1 &
wait $!; rc=$?
[ -s "$OUT/result.json" ] || fail main "no result.json (netsearch_slot.sh exit $rc; $OUT/run.log)"
st "main done: netsearch_slot.sh exit $rc; $(head -2 "$OUT/result.txt" | tail -1 | sed 's/^ *//')"

# ---- verdict
"$PY" "$CODE/scripts/fastenv/lookahead2_verdict.py" "$OUT" --json "$S/verdict.json" > "$S/VERDICT.txt.tmp" 2> "$S/verdict.err" \
  || fail verdict "lookahead2_verdict.py exit $? ($S/verdict.err)"
mv -f "$S/VERDICT.txt.tmp" "$S/VERDICT.txt" && chmod a-w "$S/VERDICT.txt"
head -1 "$S/VERDICT.txt" > "$S/RESULT"
st "VERDICT: $(head -1 "$S/VERDICT.txt")"
case $(head -1 "$S/VERDICT.txt") in VOID*) fail verdict "$(head -1 "$S/VERDICT.txt")" ;; esac
trap - TERM INT
if [ ! -e "$S/KEEP_HELPER" ] && [ "$J" != none ]; then
  st "chain done; cancelling the helper job $J"
  scancel "$J"
else
  st "chain done (helper kept)"
fi
exit 0
