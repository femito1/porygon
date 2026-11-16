#!/bin/bash
# Bet 1 GO follow-ups on the benchmark slot (docs/netsearch/LA2-FP-PREREG.md, LA2-H-PREREG.md), on G9q-cd@805.03M
# with δ* from runs/la2-delta800/DELTA (LA2-DELTA800-PREREG.md; the chain waits up to 3 h for it, then FAILED).
# One cluster-side job: checks -> smoke -> main (fixed n) -> verdict -> ledger RELEASE line.
#   MODE fp : arm 1 = lookahead2(δ*, K 8), arm 2 = the raw G9q-cd@805.03M, both vs Foul Play gen9 (Stage 3b harness),
#             500 pairs each on tag la2fp20261116; Q = arm 1's rate (headline) and the paired arm 1 - arm 2. The smoke
#             (2 pairs per arm = 4 games over the 3 Foul Play instances) exercises the shared lookahead pool.
#   MODE h  : lookahead2(δ*, K 8) vs the benchmark bot H(0.0212) + G9p-cd@104.71M, 500 pairs on tag la2h20261116;
#             ADOPTED iff the lower 95% bound > 50%.
# Stage dir runs/la2-<MODE>/ prepared before submission: PREREG.txt r-- + PREREG.md5, and code/ (src + scripts +
# COMMIT frozen at submission, so later syncs of $C cannot change the run). Stage files: STATUS, SMOKE.txt,
# VERDICT.txt (r--), RESULT, FAILED; the RELEASE line goes to $C/LEDGER-OUTBOX.md.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch --dependency=afterany:PREV scripts/fastenv/lookahead2_slot.sbatch fp
#SBATCH --job-name=ws-fastenv-la2slot
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
PY=$G/.venv/bin/python
MODE=${1:?usage: lookahead2_slot.sbatch fp|h}
S=$C/runs/la2-$MODE
cd "$C" || exit 1
J=${SLURM_JOB_ID:-none}
now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$S/STATUS"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
fail() {
  st "FAILED at stage $1: $2"; echo "$(now) stage $1: $2 (job $J)" > "$S/FAILED"
  outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $J | Bet 1 follow-up $MODE FAILED at stage $1 ($2); $S/FAILED; no verdict; slot FREE"
  exit 1
}
case $MODE in fp|h) ;; *) echo "unknown mode $MODE" >&2; exit 2 ;; esac
mkdir -p "$S"
[ ! -e "$S/VERDICT.txt" ] && [ ! -e "$S/FAILED" ] && [ ! -e "$S/main-run" ] && [ ! -e "$S/smoke-run" ] || { echo "$S already holds a chain" >&2; exit 1; }
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
NET=$C/nets/g9q-cd-805.03M.pt; NET_SHA=e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f
DSTAGE=$C/runs/la2-delta800
OLD=$C/nets/g9p-cd-104.71M.pt; OLD_SHA=35cd1007bb90bdf939cedf877f5fcdd89ef640628f77007b3b6638d5add4ad1c
FROZEN=$C/frozen/netv2-fba02fb-20261110
RAW="$NET,temperature=0.5"
H="netsearch:$OLD,margin=0.0212,temperature=0.5,ms=1000,samples=4,workers=11"
md5of() { md5sum < "$1" | cut -c1-32; }
st "chain $MODE start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}"
[ -f "$S/PREREG.txt" ] && [ ! -w "$S/PREREG.txt" ] || fail checks "PREREG.txt missing or writable"
[ -s "$S/PREREG.md5" ] && [ "$(md5of "$S/PREREG.txt")" = "$(cut -c1-32 "$S/PREREG.md5")" ] || fail checks "PREREG md5 != PREREG.md5"
[ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "G9q-cd@805.03M sha256"
[ "$MODE" != h ] || [ "$(sha256sum "$OLD" | cut -c1-64)" = "$OLD_SHA" ] || fail checks "G9p-cd sha256"
[ "${SLURM_CPUS_ON_NODE:-0}" -ge 24 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 24)"
nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
[ -s "$S/code/COMMIT" ] && [ -f "$S/code/scripts/fastenv/netsearch_slot.sh" ] || fail checks "frozen code $S/code missing"
CODE=$S/code
for ((w = 0; w < 36; w++)); do [ -s "$DSTAGE/DELTA" ] || [ -e "$DSTAGE/FAILED" ] && break; st "waiting for $DSTAGE/DELTA"; sleep 300; done
[ -s "$DSTAGE/DELTA" ] && [ ! -e "$DSTAGE/FAILED" ] || fail checks "no δ* ($DSTAGE/DELTA missing or the δ chain FAILED)"
MARGIN=$(head -1 "$DSTAGE/DELTA")
[[ "$MARGIN" =~ ^[0-9]+\.?[0-9]*$ ]] || fail checks "δ* not a number: $MARGIN"
A="lookahead:$NET,margin=$MARGIN,temperature=0.5,worlds=8,workers=11,leaf_device=cuda"
st "checks OK (PREREG md5 $(md5of "$S/PREREG.txt"), code $(cat "$CODE/COMMIT"), δ* $MARGIN from $DSTAGE/DELTA)"

plan() {  # NAME TAG PAIRS BLOCKS
  if [ "$MODE" = fp ]; then opp='foulplay:ms=1000,parallelism=1'; arms="('$A' '$RAW')"; else opp=$H; arms="('$A')"; fi
  cat > "$S/$1.env" <<EOF
NAME=$1
GEN=9
TAG=$2
NETV2_DIR=$FROZEN
OPP='$opp'
ARMS=$arms
BLOCK_PAIRS=$3
BLOCKS=$4
CONC=3
FUTILITY=
MIN_CPUS=24
BLOCK_TIMEOUT_MIN=300
EOF
}

# ---- smoke (mechanical): 2 pairs per arm (fp: 4 games over the 3 Foul Play instances, the shared lookahead pool) / 2 pairs (h)
if [ "$MODE" = fp ]; then plan la2fpsmoke la2fpsmoke20261116 2 1; SMP=la2fpsmoke; NG=4; else plan la2hsmoke la2hsmoke20261116 2 1; SMP=la2hsmoke; NG=4; fi
SM=$S/smoke-run
bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/$SMP.env" "$SM" > "$S/smoke.log" 2>&1 &
wait $!; rc=$?
sm=$("$PY" - "$SM" "$NG" <<'PYEOF' 2>&1
import json, sys
from pathlib import Path
run, ng = Path(sys.argv[1]), int(sys.argv[2])
r = json.loads((run / "result.json").read_text())
checks = {}
for a in r["arms"]:
    k = a["arm"]
    sa = a["stats_a"]
    ev = "\n".join(p.read_text() for p in run.glob(f"arm-{k}/block-*/evaluate.log"))
    checks[f"arm {k} {ng} games"] = a["games"] == ng
    checks[f"arm {k} no Traceback"] = "Traceback" not in ev
    if a["spec"].startswith("lookahead:"):
        checks[f"arm {k} searched/leaves > 0"] = sa.get("searched", 0) > 0 and sa.get("leaves", 0) > 0
        checks[f"arm {k} gate rows"] = a["gate"]["decisions"] > 0
        checks[f"arm {k} failed worlds 0"] = sa.get("failed_worlds", 0) == 0
        checks[f"arm {k} fallbacks <= max(2, 2%)"] = sa.get("fallbacks", 0) <= max(2, 0.02 * sa.get("decisions", 0))
checks["no HARNESS_FAIL"] = not (run / "HARNESS_FAIL").exists()
ok = all(checks.values())
print(("PASS" if ok else "FAIL"), "; ".join(f"{k} {'ok' if v else 'NO'}" for k, v in checks.items()))
PYEOF
)
{ echo "smoke $SM (exit $rc)"; echo "$sm"; cat "$SM/result.txt" 2>/dev/null; } > "$S/SMOKE.txt"
st "smoke: $sm"
case $sm in PASS*) ;; *) fail smoke "$sm" ;; esac

# ---- main (fixed n: 5 blocks of 100 pairs per arm)
if [ "$MODE" = fp ]; then plan la2fp la2fp20261116 100 5; MP=la2fp; else plan la2h la2h20261116 100 5; MP=la2h; fi
OUT=$S/main-run
st "main start: $OUT ($MP, 500 pairs per arm, fixed n)"
bash "$CODE/scripts/fastenv/netsearch_slot.sh" "$S/$MP.env" "$OUT" > "$S/main.log" 2>&1 &
wait $!; rc=$?
[ -s "$OUT/result.json" ] || fail main "no result.json (exit $rc; $OUT/run.log)"
st "main done (exit $rc)"

# ---- verdict (the PREREG's rule, mechanically)
if [ "$MODE" = fp ]; then (cd "$OUT/code/scripts/fastenv" && "$PY" netsearch_slot.py paired "$OUT" 1 2 --json "$S/paired.json") > "$S/paired.txt" 2>&1 || fail verdict "paired"; fi
"$PY" - "$OUT" "$MODE" "$S" > "$S/VERDICT.txt.tmp" 2> "$S/verdict.err" <<'PYEOF' || fail verdict "verdict script ($S/verdict.err)"
import json, sys
from pathlib import Path
run, mode, s = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
r = json.loads((run / "result.json").read_text())
arms = r["arms"]
void = (run / "HARNESS_FAIL").exists() or any(a["void"] for a in arms) or any(a["pairs"] < 450 for a in arms)
def line(a):
    return (f"arm {a['arm']}: {a['mean']:.2%} ± {a['hw']:.2%} [{a['lo']:.2%}, {a['hi']:.2%}] ({a['pairs']} pairs; won both "
            f"{a['won_both']}, split {a['split']}, lost both {a['lost_both']}); overrides {a['override_rate']}; fallbacks "
            f"{a['fallback_rate_a']}/{a['fallback_rate_b']}; {a['seconds'] / 3600:.2f} h | {a['spec']}")
if void:
    head = "VOID: harness failure, > 2% fallbacks or < 450 pairs on an arm; replay with the same seeds after the fix"
elif mode == "h":
    a = arms[0]
    head = ("ADOPTED: lookahead2 beats the benchmark bot (lower 95% > 50%)" if a["lo"] > 0.5
            else "NOT ADOPTED: lookahead2 does not beat the benchmark bot by the rule (lower 95% <= 50%)")
else:
    p = json.loads((s / "paired.json").read_text())
    tag = "CARRIES OVER" if p["lo"] > 0 else ("REVERSED" if p["hi"] < 0 else "NOT SHOWN")
    head = (f"{tag}: lookahead2 - raw net vs Foul Play, paired {p['diff']:+.2%} ± {p['hw']:.2%} [{p['lo']:+.2%}, "
            f"{p['hi']:+.2%}] on {p['pairs']} team pairs; lookahead2 vs Foul Play {arms[0]['mean']:.2%} ± {arms[0]['hw']:.2%}")
print(head)
for a in arms:
    print("  " + line(a))
print(f"  opponent {r['opponent']}; run {run}")
PYEOF
mv -f "$S/VERDICT.txt.tmp" "$S/VERDICT.txt" && chmod a-w "$S/VERDICT.txt"
head -1 "$S/VERDICT.txt" > "$S/RESULT"
st "VERDICT: $(head -1 "$S/VERDICT.txt")"
case $(head -1 "$S/VERDICT.txt") in VOID*) fail verdict "$(head -1 "$S/VERDICT.txt")" ;; esac
outbox "$(date -u +%FT%H:%MZ) | fastenv | RELEASE | $J | Bet 1 follow-up $MODE done: $(head -1 "$S/VERDICT.txt"); $S/VERDICT.txt; slot FREE"
trap - TERM INT
st "chain done (ledger RELEASE line in $C/LEDGER-OUTBOX.md)"
exit 0
