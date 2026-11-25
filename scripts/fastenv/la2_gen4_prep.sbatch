#!/bin/bash
# LA2-gen4 stage 1, prep (council P3 "if GO"; docs/netsearch/LA2-GEN4-PREREG.md): fastenv's helper job, frozen code
# council/la2-gen4/code (git archive + COMMIT). Everything before the slot games, on R1b-F-cd@640M (the pinned gen4 net):
#   checks -> gen (fastenv gen4 self-play of the net vs itself at T 0.5, exit_la2.py gen --all-decisions, ps-sets-pinned
#   = the benchmark server's October sets) -> split (la2_gen4.py split: 300 parity battles + 1,403 study positions, one
#   per game) -> checks in parallel: (a) gen4 leaf parity (v3_gates.py p1, the ns-10 method, on the 300 battles),
#   (b, c) gen9 P1 (ns-10's records, seed and net) with the new code and with the benchmark bot's frozen code: p1-rows
#   byte-identical, (d) gen9 byte-identity of the lookahead2 leaves (la2_gen4.py ident) -> bench (the 1,403 positions,
#   A K 8 / B K 8 / judge R K 32, Bet 1's arguments and seed 11) -> δ* by Bet 1's rule (pooled, >= 98% usable) ->
#   DELTA (r--) + PREP-RESULT.txt (r--). Any failure writes FAILED (+ STATUS) and stops; nothing here plays a game.
# The slot chain (la2_gen4.sbatch) starts only after the agent has written δ* and the set md5 into the PREREG.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch --nice=500 council/la2-gen4/code/scripts/fastenv/la2_gen4_prep.sbatch
# Stages resume (gen continues from its finished parts; finished outputs are kept), so a resubmission after a fix
# continues. Files in council/la2-gen4/: STATUS, FAILED, DELTA, PREP-RESULT.txt; work files in council/la2-gen4/prep/.
#SBATCH --job-name=ws-fastenv-la2g4prep
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
D=/vol/bitbucket/fr1226/porygon-ws-data
PY=$G/.venv/bin/python
S=$C/council/la2-gen4
P=$S/prep
CODE=$S/code
FROZEN=$C/frozen/netv2-fba02fb-20261110
OLD=$C/runs/la2-2030/code  # the gen9 benchmark bot's frozen code (LA2-2030, 8093ece): v3leaf / lookahead2 as played
NET=$C/nets/g4r1bf-cd-640.04M.pt; NET_MD5=90bf853d643ade36ad76f6ebe0dbc61f
G9NET=$C/nets/g9p-cd-104.71M.pt; G9NET_SHA=35cd1007bb90bdf939cedf877f5fcdd89ef640628f77007b3b6638d5add4ad1c
SL="$D/datasets/searchlabels/records/gen9randombattle/part-0000[67].jsonl.gz"
I9=$C/netsearch/study-20261109/s1-sl-AB.jsonl
GSEED=4241124 GAMES=1800 NPAR=300 NSTUDY=1403
X=$CODE/scripts/fastenv
J=${SLURM_JOB_ID:-none}
cd "$C" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NETV2_SRC=$FROZEN/src PORYGON_NODE=$G/.node/bin/node PATH=$G/.node/bin:$PATH
G4ENV=(env PORYGON_GEN=4 PYTHONPATH=$CODE/src)
G9NEW=(env PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src)
G9OLD=(env PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$OLD/src)
now() { date -u +%FT%TZ; }
st() { echo "$(now) [prep] $*" | tee -a "$S/STATUS"; }
fail() { st "FAILED at stage $1: $2"; echo "$(now) [prep] stage $1: $2 (job $J)" >> "$S/FAILED"; exit 1; }
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
md5of() { md5sum < "$1" | cut -c1-32; }
mkdir -p "$P"
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT" 2>/dev/null)"
[ ! -e "$S/FAILED" ] || fail start "$S/FAILED exists (read it, fix, move it aside)"
[ ! -e "$S/DELTA" ] || fail start "$S/DELTA exists (prep already done)"

# ---- checks
[ "${SLURM_CPUS_ON_NODE:-0}" -ge 12 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 12)"
nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
[ "$(md5of "$NET")" = "$NET_MD5" ] || fail checks "gen4 net md5 != $NET_MD5"
[ "$(sha256sum "$G9NET" | cut -c1-64)" = "$G9NET_SHA" ] || fail checks "gen9 P1 net sha256"
[ -d "$FROZEN/src/porygon/v2" ] || fail checks "frozen netv2 tree missing"
for f in src/porygon/v3leaf.py src/porygon/lookahead2.py scripts/fastenv/v3_gates.py; do
  [ -f "$OLD/$f" ] || fail checks "old code $OLD/$f missing"
done
[ -d "$C/ps-sets-pinned/dist/sim" ] || fail checks "no built simulator in ps-sets-pinned"
[ "$(md5of "$C/ps-sets-pinned/dist/data/random-battles/gen4/sets.json")" = 77a82cf623550fbb12d177150f13b231 ] ||
  fail checks "ps-sets-pinned gen4 sets.json is not the benchmark server's (77a82cf6)"
ld=$("${G4ENV[@]}" "$PY" - "$NET" "$G9NET" <<'PYEOF' 2>&1 | tail -1
import sys
sys.path.insert(0, "/vol/bitbucket/fr1226/porygon-ws-fastenv/council/la2-gen4/code/scripts/fastenv")
import netsearch_study as ST
ST.setup("gen4randombattle")
from porygon.v2.model import load
n4, n9 = load(sys.argv[1], device="cpu"), load(sys.argv[2], device="cpu")
print("gen4 net gen", int(n4.gen), "params", sum(p.numel() for p in n4.parameters()), "| gen9 net gen", int(n9.gen))
PYEOF
)
case $ld in "gen4 net gen 4 params "*"gen9 net gen 9") st "checks OK: nets load through fba02fb ($ld), ${SLURM_CPUS_ON_NODE} CPUs, GPU, old code $OLD" ;;
  *) fail checks "net load: $ld" ;; esac

# ---- gen: fastenv gen4 self-play of R1b-F-cd vs itself at T 0.5 (resumes from finished parts)
if [ ! -s "$P/set/split.json" ]; then
  st "gen start: $GAMES games, seed $GSEED, T 0.5, PORYGON_PS ps-sets-pinned"
  "${G4ENV[@]}" PORYGON_PS=$C/ps-sets-pinned "$PY" -u "$X/exit_la2.py" gen --format gen4randombattle --net "$NET" \
    --out "$P/gen" --games "$GAMES" --seed "$GSEED" --workers 5 --battles 32 --temperature 0.5 --all-decisions \
    --fifo-dir "${TMPDIR:-/tmp}" --max-hours 2 >> "$P/gen.log" 2>&1 || fail gen "exit_la2.py gen ($P/gen.log)"
  st "gen done: $(tail -1 "$P/gen.log")"
  sp=$("$PY" "$X/la2_gen4.py" split --gen "$P/gen" --out "$P/set" --seed "$GSEED" --parity "$NPAR" --study "$NSTUDY" 2>&1)
  st "split: $sp"
  case $sp in OK*) ;; *) fail split "$sp" ;; esac
fi

# ---- checks in parallel: gen4 leaf parity, gen9 P1 new vs old, gen9 leaf identity
pids=()
if [ ! -s "$P/parity-g4/p1.json" ]; then
  "${G4ENV[@]}" "$PY" -u "$X/v3_gates.py" p1 --format gen4randombattle --records "$P/set/parity-records.jsonl.gz" \
    --net "$NET" --out "$P/parity-g4" --records-max "$NPAR" --pairs 100000 --per-record 3 --dets 2 --seed 7311 \
    --no-roots > "$P/parity-g4.log" 2>&1 & pids+=($!)
fi
if [ ! -s "$P/p1-g9-new/p1.json" ]; then
  "${G9NEW[@]}" "$PY" -u "$X/v3_gates.py" p1 --records "$SL" --source 6771931 --net "$G9NET" --out "$P/p1-g9-new" \
    --no-roots > "$P/p1-g9-new.log" 2>&1 & pids+=($!)
fi
if [ ! -s "$P/p1-g9-old/p1.json" ]; then
  "${G9OLD[@]}" "$PY" -u "$OLD/scripts/fastenv/v3_gates.py" p1 --records "$SL" --source 6771931 --net "$G9NET" \
    --out "$P/p1-g9-old" > "$P/p1-g9-old.log" 2>&1 & pids+=($!)
fi
if [ ! -s "$P/ident-g9.json" ]; then
  "${G9NEW[@]}" "$PY" -u "$X/la2_gen4.py" ident --rows "$I9" --records "$SL" --source 6771931 \
    --old "$OLD/src/porygon/v3leaf.py" --positions 300 --dets 2 --json "$P/ident-g9.json" > "$P/ident-g9.log" 2>&1 & pids+=($!)
fi
st "checks running (${#pids[@]} processes): gen4 parity, gen9 P1 new / old, gen9 identity"
for p in "${pids[@]}"; do wait "$p"; done
rm -f "$P/p1-g9-old/roots.pkl"
par=$("$PY" "$X/la2_gen4.py" parity "$P/parity-g4/p1.json" 2>&1); prc=$?
st "gen4 $par"
g9=$("$PY" "$X/la2_gen4.py" parity "$P/p1-g9-new/p1.json" 2>&1); g9rc=$?
st "gen9 P1 (new code) $g9"
rdf=$("$PY" "$X/la2_gen4.py" rows-diff "$P/p1-g9-old/p1-rows.jsonl" "$P/p1-g9-new/p1-rows.jsonl" 2>&1); rdrc=$?
st "gen9 P1 rows old vs new: $rdf"
idl=$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["line"])' "$P/ident-g9.json" 2>&1); idrc=$?
grep -q '"pass": true' "$P/ident-g9.json" 2>/dev/null || idrc=1
st "gen9 identity: $idl"
[ "$prc" = 0 ] || fail parity "gen4 leaf parity not PASS: ${par:0:300}"
[ "$g9rc" = 0 ] && [ "$rdrc" = 0 ] && [ "$idrc" = 0 ] || fail gen9 "gen9 not byte-identical or P1 not PASS (P1 $g9rc, rows $rdrc, identity $idrc)"

# ---- bench + δ* (Bet 1's rule, pooled over the 1,403 positions)
B=$P/bench
mkdir -p "$B"
if [ ! -s "$B/bench-all.json" ]; then
  st "bench start: $NSTUDY positions, A K 8 / B K 8 / R K 32, seed 11, workers 10, concurrency 3"
  rm -f "$B/bench-all.jsonl"
  "${G4ENV[@]}" "$PY" -u "$X/lookahead2_study.py" bench --format gen4randombattle --records "$P/set/study-records.jsonl.gz" \
    --rows "$P/set/study-rows.jsonl" --net "$NET" --out "$B/bench-all.jsonl" --worlds 8 --retest --ref 32 --workers 10 \
    --concurrency 3 --seed 11 > "$B/bench-all.log" 2>&1 || fail bench "bench ($B/bench-all.log: $(tail -1 "$B/bench-all.log"))"
  "${G4ENV[@]}" "$PY" "$X/lookahead2_study.py" analyze-bench "$B/bench-all.jsonl" --json "$B/bench-all.json.tmp" \
    > "$B/bench-all.txt" 2>&1 || fail bench "analyze-bench"
  mv -f "$B/bench-all.json.tmp" "$B/bench-all.json"
  sha256sum "$B/bench-all.jsonl" > "$B/bench-sha256.txt"
fi
st "bench: $(head -1 "$B/bench-all.txt")"
dl=$("$PY" "$X/la2_gen4.py" delta "$B/bench-all.json" --positions "$NSTUDY" 2>&1) || fail rule "no δ* by the rule: $dl"
DELTA=${dl%% *}
[[ "$DELTA" =~ ^[0-9]+\.?[0-9]*$ ]] || fail rule "δ* not a number: $dl"
pp=$("$PY" "$X/la2_gen4.py" probs "$B/bench-all.jsonl" "$P/set/study-records.jsonl.gz" 2>&1)
{ echo "DELTA* = $DELTA for R1b-F-cd@640M (gen4) by Bet 1's δ rule (pooled $NSTUDY study positions): $dl"
  echo "  net $NET md5 $NET_MD5 sha256 $(sha256sum "$NET" | cut -c1-64); code $(cat "$CODE/COMMIT")"
  echo "  study set $P/set: $("$PY" -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps(d["md5"]), "| games", d["games_with_record"], "| sides", d["study_sides"], "| turn median", d["study_turn_median"])' "$P/set/split.json")"
  echo "  self-play: $(tail -1 "$P/gen.log")"
  echo "  gen4 $par"
  echo "  gen9 P1 (new code) $g9"
  echo "  gen9 P1 rows old vs new: $rdf"
  echo "  gen9 identity: $idl"
  echo "  replayed vs self-play T-1 probabilities (informational): $pp"
  echo "  bench $B/bench-all.jsonl $(cut -c1-64 "$B/bench-sha256.txt")"
  cat "$B/bench-all.txt"; } > "$S/PREP-RESULT.txt.tmp"
mv -f "$S/PREP-RESULT.txt.tmp" "$S/PREP-RESULT.txt" && chmod a-w "$S/PREP-RESULT.txt"
echo "$DELTA" > "$S/DELTA.tmp" && mv -f "$S/DELTA.tmp" "$S/DELTA" && chmod a-w "$S/DELTA"
st "DELTA* = $DELTA ($dl); PREP-RESULT.txt written; prep done (the slot chain needs the PREREG with δ* and the set md5)"
trap - TERM INT
exit 0
