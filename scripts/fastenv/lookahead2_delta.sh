#!/bin/bash
# lookahead2 δ* for G9q-cd@800M (docs/netsearch/LA2-DELTA800-PREREG.md): Bet 1's pre-registered δ rule, unchanged, on
# a new net. One cluster-side chain, run as a step of fastenv's helper job (cwd $C):
#   echo 'bash scripts/fastenv/lookahead2_delta.sh' > jobq/la2-delta800.sh.tmp && mv jobq/la2-delta800.sh.tmp jobq/la2-delta800.sh
#   checks : PREREG.txt r-- with md5 = PREREG.md5, net sha256, >= 12 CPUs, a GPU, the frozen netv2 tree;
#   load   : the net loads through the frozen tree (v3_gates.load_net) and is gen9;
#   bench  : the 1,403 study positions, A / B = K 8, judge R = K 32 (Bet 1's la2-bench2 arguments and seed);
#   rule   : analyze-bench on the pooled file -> margin_rule.delta -> DELTA (one number) + VERDICT.txt (r--) + RESULT;
#   end    : cancels the helper job unless runs/la2-delta800/KEEP_HELPER exists.
# Stage files in runs/la2-delta800/: STATUS, FAILED, DELTA, RESULT, VERDICT.txt. Code frozen into runs/la2-delta800/code.
# Another net (the same rule; environment, the defaults are the la2-delta800 run): LA2D_STAGE (stage dir, holding
# PREREG.txt / PREREG.md5 and code/), LA2D_NET + LA2D_NET_SHA, LA2D_LABEL (the net's name in VERDICT), LA2D_OUT (bench
# dir), LA2D_PREFIX (prefix of the VERDICT.txt / RESULT names and of the STATUS lines, e.g. DELTA-), LA2D_CANCEL (1:
# cancel the job at the end; 0 inside a slot chain, lookahead2_adopt.sbatch, which runs this with LA2_FROZEN=1).
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
D=/vol/bitbucket/fr1226/porygon-ws-data
PY=$G/.venv/bin/python
S=${LA2D_STAGE:-$C/runs/la2-delta800}
cd "$C" || exit 1
if [ -z "${LA2_FROZEN:-}" ]; then
  mkdir -p "$S"
  [ ! -e "$S/DELTA" ] && [ ! -e "$S/FAILED" ] && [ ! -e "$S/code" ] || { echo "$S already holds a chain" >&2; exit 1; }
  mkdir -p "$S/code" && rsync -a --exclude __pycache__ src scripts "$S/code/" && cp COMMIT "$S/code/COMMIT" || exit 1
  LA2_FROZEN=1 exec bash "$S/code/scripts/fastenv/lookahead2_delta.sh"
fi
CODE=$S/code
NET=${LA2D_NET:-$C/nets/g9q-cd-805.03M.pt}; NET_SHA=${LA2D_NET_SHA:-e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f}
LABEL=${LA2D_LABEL:-G9q-cd@805.03M}
PFX=${LA2D_PREFIX:-}
FROZEN=$C/frozen/netv2-fba02fb-20261110
OUT=${LA2D_OUT:-$C/lookahead2/study-cd805-20261116}
I=$C/netsearch/study-20261109
ST=$CODE/scripts/fastenv/lookahead2_study.py
SL="$D/datasets/searchlabels/records/gen9randombattle/part-0000[67].jsonl.gz"
HU3="$D/datasets/human/records/gen9randombattle/part-0027[01].jsonl.gz"
HU2C="$D/datasets/human/records/gen9randombattle/part-0027[23].jsonl.gz"
export PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src NETV2_SRC=$FROZEN/src OMP_NUM_THREADS=1
J=${SLURM_JOB_ID:-none}
now() { date -u +%FT%TZ; }
st() { echo "$(now) ${PFX:+${PFX%-} }$*" | tee -a "$S/STATUS"; }
fail() { st "FAILED at stage $1: $2"; echo "$(now) stage $1: $2 (job $J)" > "$S/FAILED"; exit 1; }
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
md5of() { md5sum < "$1" | cut -c1-32; }
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT")"

# ---- checks
[ -f "$S/PREREG.txt" ] && [ ! -w "$S/PREREG.txt" ] || fail checks "PREREG.txt missing or writable"
[ -s "$S/PREREG.md5" ] && [ "$(md5of "$S/PREREG.txt")" = "$(cut -c1-32 "$S/PREREG.md5")" ] || fail checks "PREREG md5 != PREREG.md5"
[ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "net sha256 mismatch"
[ "${SLURM_CPUS_ON_NODE:-0}" -ge 12 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 12)"
nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
[ -d "$FROZEN/src/porygon/v2" ] || fail checks "frozen netv2 tree missing"
st "checks OK: PREREG md5 $(md5of "$S/PREREG.txt"), net sha256 ok"

# ---- load check through the frozen tree
ld=$(PYTHONPATH=$FROZEN/src:$G/.engine-gen9 "$PY" -c '
import sys; sys.path.insert(0, sys.argv[2])
from v3_gates import load_net
n = load_net(sys.argv[1], "cpu"); print("gen", int(n.gen), "params", sum(p.numel() for p in n.parameters()))
' "$NET" "$CODE/scripts/fastenv" 2>&1 | tail -1)
case $ld in "gen 9 "*) st "load OK through fba02fb: $ld" ;; *) fail load "net does not load through fba02fb: $ld" ;; esac

# ---- bench (Bet 1's arguments; default seed 11)
mkdir -p "$OUT"
echo "$(now) node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?}; net sha256 $(sha256sum "$NET" | cut -c1-64); code $(cat "$CODE/COMMIT")" > "$OUT/manifest.txt"
B=(--worlds 8 --retest --ref 32 --workers 10 --concurrency 3)
bench() {  # NAME ROWS SOURCE(- = none) RECORD_GLOB
  local src=(); [ "$3" = - ] || src=(--source "$3")
  rm -f "$OUT/bench-$1.jsonl"
  "$PY" "$ST" bench --records "$4" "${src[@]}" --rows "$I/$2.jsonl" --net "$NET" --out "$OUT/bench-$1.jsonl" "${B[@]}" \
    > "$OUT/bench-$1.log" 2>&1 || fail bench "bench $1 ($OUT/bench-$1.log)"
  "$PY" "$ST" analyze-bench "$OUT/bench-$1.jsonl" --json "$OUT/bench-$1.json" > "$OUT/bench-$1.txt" 2>&1 || fail bench "analyze $1"
  st "bench $1 done: $(head -1 "$OUT/bench-$1.txt")"
}
bench s2 s2-sl-ARP 6771931 "$SL"
bench s2c s2c-hu-AR - "$HU2C"
bench s1 s1-sl-AB 6771931 "$SL"
bench s3 s3-hu-AB - "$HU3"
cat "$OUT"/bench-s1.jsonl "$OUT"/bench-s2.jsonl "$OUT"/bench-s2c.jsonl "$OUT"/bench-s3.jsonl > "$OUT/bench-all.jsonl"
"$PY" "$ST" analyze-bench "$OUT/bench-all.jsonl" --json "$OUT/bench-all.json" > "$OUT/bench-all.txt" 2>&1 || fail rule "analyze all"
cat "$OUT"/bench-s2.jsonl "$OUT"/bench-s2c.jsonl > "$OUT/bench-s2all.jsonl"
"$PY" "$ST" analyze-bench "$OUT/bench-s2all.jsonl" --json "$OUT/bench-s2all.json" > "$OUT/bench-s2all.txt" 2>&1 || true
sha256sum "$OUT"/bench-*.jsonl > "$OUT/bench-sha256.txt"

# ---- the rule (pooled)
line=$("$PY" - "$OUT/bench-all.json" <<'PYEOF' 2>&1
import json, sys
d = json.load(open(sys.argv[1]))
m = d["margin_rule"]
if d.get("usable", 0) < 1403 * 0.98 or m["delta"] is None:
    print(f"NONE usable {d.get('usable')} delta {m['delta']}")
else:
    print(f"{m['delta']} override {m['override']:.4f} judgeR_gain_vs_raw {m['judgeR_gain_vs_raw']:+.5f} usable {d['usable']}")
PYEOF
)
case $line in NONE*|*Error*|*Traceback*|"") fail rule "no δ* by the rule: $line" ;; esac
DELTA=${line%% *}
[[ "$DELTA" =~ ^[0-9]+\.?[0-9]*$ ]] || fail rule "δ* not a number: $line"
{ echo "DELTA* = $DELTA for $LABEL by Bet 1's δ rule (pooled 1,403 study positions): $line"
  echo "  net $NET sha256 $NET_SHA; bench $OUT; code $(cat "$CODE/COMMIT")"
  cat "$OUT/bench-all.txt"; } > "$S/${PFX}VERDICT.txt.tmp"
mv -f "$S/${PFX}VERDICT.txt.tmp" "$S/${PFX}VERDICT.txt" && chmod a-w "$S/${PFX}VERDICT.txt"
echo "$DELTA" > "$S/DELTA.tmp" && mv -f "$S/DELTA.tmp" "$S/DELTA" && chmod a-w "$S/DELTA"
head -1 "$S/${PFX}VERDICT.txt" > "$S/${PFX}RESULT"
st "DELTA* = $DELTA ($line)"
trap - TERM INT
if [ "${LA2D_CANCEL:-1}" != 1 ]; then
  st "chain done (job $J kept: LA2D_CANCEL=${LA2D_CANCEL})"
elif [ ! -e "$S/KEEP_HELPER" ] && [ "$J" != none ]; then
  st "chain done; cancelling the helper job $J"
  scancel "$J"
else
  st "chain done (helper kept)"
fi
exit 0
