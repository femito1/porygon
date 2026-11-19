#!/bin/bash
# ExIt-LA2 pilot, stage 1 (docs/netsearch/EXIT-LA2-PREREG.md = runs/exit-la2/PREREG.txt, r--, md5 in PREREG.md5).
# fastenv's helper job (12 CPUs, 1 GPU, training partition, nice 500), frozen code runs/exit-la2/code:
#   cd $C && sbatch --nice=500 runs/exit-la2/code/scripts/fastenv/exit_pilot.sbatch smoke   # the smoke alone
#   cd $C && sbatch --nice=500 runs/exit-la2/code/scripts/fastenv/exit_pilot.sbatch main    # checks -> ... -> VERDICT
# main: checks -> smoke (skipped if smoke/SMOKE.txt says PASS for this frozen code) -> gen (60k games) -> label
# (120k positions) -> shards (E, C) -> fine-tunes E, C -> load checks -> manipulation check (+ the one dose retry at
# lr 3e-5 if r_E < 0.25) -> T1 E vs C, T2 E vs S, T3 C vs S -> VERDICT.txt (r--) + RESULT. Every stage is resumable
# (a resubmission continues from the finished parts); a failed stage writes FAILED and stops.
# Stage files in runs/exit-la2/: STATUS, FAILED, SMOKE (smoke/SMOKE.txt), gen/, labels/, shards/{E,C}/, ft-{E,C}/,
# check.json, T{1,2,3}.json, VERDICT.txt, RESULT, NOTES.txt (records, by hand).
#SBATCH --job-name=ws-fastenv-exit
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16000
#SBATCH --time=30:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
MODE=${1:?usage: exit_pilot.sbatch smoke|main}
C=/vol/bitbucket/fr1226/porygon-ws-fastenv
G=/vol/bitbucket/fr1226/gloria
PY=$G/.venv/bin/python
S=$C/runs/exit-la2
CODE=$S/code
FROZEN=$C/frozen/netv2-fba02fb-20261110
NET=$C/nets/g9q-cd-805.03M.pt; NET_SHA=e93ac3f8a73590d905b150976287b261243f7861ce347f0c94baaf445b34b11f
DELTA_FILE=$C/runs/la2-delta800/DELTA
X=$CODE/scripts/fastenv/exit_la2.py
PS_SELF=$C/ps-2026-11      # G9q's training sets (self-play positions)
PS_EVAL=$C/ps-sets-pinned  # October (the benchmark / netv2 step-matched in-process tests)
cd "$C" || exit 1
J=${SLURM_JOB_ID:-none}
export PORYGON_GEN=9 PYTHONPATH=$G/.engine-gen9:$CODE/src NETV2_SRC=$FROZEN/src OMP_NUM_THREADS=1
export PORYGON_NODE=$G/.node/bin/node PATH=$G/.node/bin:$PATH
now() { date -u +%FT%TZ; }
st() { echo "$(now) [$MODE] $*" | tee -a "$S/STATUS"; }
fail() { st "FAILED at stage $1: $2"; echo "$(now) [$MODE] stage $1: $2 (job $J)" >> "$S/FAILED"; exit 1; }
trap 'fail signal "TERM/INT (time limit or scancel)"' TERM INT
md5of() { md5sum < "$1" | cut -c1-32; }
st "chain start: job $J on $(hostname), SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?}, code $(cat "$CODE/COMMIT")"
[ ! -e "$S/FAILED" ] || fail start "$S/FAILED exists (read it, fix, move it aside)"
[ ! -e "$S/VERDICT.txt" ] || fail start "VERDICT.txt exists"

# ---- checks (both modes)
[ -f "$S/PREREG.txt" ] && [ ! -w "$S/PREREG.txt" ] || fail checks "PREREG.txt missing or writable"
[ "$(md5of "$S/PREREG.txt")" = "$(cut -c1-32 "$S/PREREG.md5")" ] || fail checks "PREREG md5 != PREREG.md5"
[ "$(sha256sum "$NET" | cut -c1-64)" = "$NET_SHA" ] || fail checks "net sha256 mismatch"
[ "${SLURM_CPUS_ON_NODE:-0}" -ge 12 ] || fail checks "only ${SLURM_CPUS_ON_NODE:-?} CPUs (want 12)"
nvidia-smi -L > /dev/null 2>&1 || fail checks "no GPU"
[ "$(cat "$DELTA_FILE")" = 0.005 ] || fail checks "DELTA is $(cat "$DELTA_FILE"), not 0.005"
for d in "$PS_SELF" "$PS_EVAL"; do [ -d "$d/dist/sim" ] || fail checks "no built simulator in $d"; done
lr=$(PYTHONPATH=$FROZEN/src "$PY" -c '
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["args"]
a = a if isinstance(a, dict) else vars(a)
last = a["lr"] * a["final_lr_frac"]
print("lr", a["lr"], "final_lr_frac", a["final_lr_frac"], "last %.3g" % last, "OK" if 5e-6 <= last <= 1.5e-5 else "OUT")
' "$NET" 2>&1 | tail -1)
case $lr in *OK) st "checks OK: PREREG md5 $(md5of "$S/PREREG.txt"), net sha256 ok, delta 0.005, S $lr" ;;
  *) fail checks "S's last lr not 1e-5 +- 50% (stop for main): $lr" ;; esac

ft() {  # OUT SHARD_DIR LR EXTRA... : the PREREG fine-tune (netv2 pretrain from the frozen fba02fb tree)
  local out=$1 sh=$2 lr=$3; shift 3
  [ -s "$out/final.pt" ] && return 0
  rm -rf "$out"
  PYTHONPATH=$FROZEN/src OMP_NUM_THREADS=4 "$PY" -u -m porygon.v2.pretrain --shards "$sh/shard-*.npz" --gen 9 \
    --out "$out" --init "$NET" --epochs 2 --batch-size 512 --lr "$lr" --warmup 50 --weight-decay 0.01 \
    --value-coef 0.5 --aux-coef 0 --val-frac 0.05 --seed 0 --no-bf16 "$@" > "$out.log" 2>&1 || return 1
  [ -s "$out/final.pt" ]
}
loadcheck() {  # NET: loads through fba02fb, gen 9, finite logits on one zero observation
  "$PY" - "$1" <<'PYEOF' 2>&1 | tail -1
import sys
sys.path.insert(0, "/vol/bitbucket/fr1226/porygon-ws-fastenv/runs/exit-la2/code/scripts/fastenv")
import netsearch_study as ST
ST.setup("gen9randombattle")
import torch
from porygon.v2.model import load, obs_to_tensors
from porygon.v2.encoder import empty
n = load(sys.argv[1], "cpu")
o = n(obs_to_tensors([empty(9)], "cpu"), aux=False)
print("LOAD", "OK" if int(n.gen) == 9 and torch.isfinite(o["logits"]).all() else "BAD", int(n.gen))
PYEOF
}

# ---- one pass of the pipeline on DIR with sizes; the smoke and the main run differ only in sizes and seeds
pipeline() {  # DIR GAMES GEN_SEED FT_EXTRA EVAL_GAMES(T1 T2 T3) EVAL_W EVAL_B
  local D=$1 games=$2 gseed=$3 ftx=$4 e1=$5 e2=$6 e3=$7 ew=$8 eb=$9
  mkdir -p "$D"
  PORYGON_PS=$PS_SELF "$PY" -u "$X" gen --net "$NET" --out "$D/gen" --games "$games" --seed "$gseed" --workers 5 \
    --battles 32 --fifo-dir "${TMPDIR:-/tmp}" >> "$D/gen.log" 2>&1 || { fail gen "gen ($D/gen.log)"; }
  st "gen done: $(tail -1 "$D/gen.log")"
  "$PY" -u "$X" label --records "$D/gen/rec-*.jsonl.gz" --net "$NET" --out "$D/labels" --delta-file "$DELTA_FILE" \
    --workers 11 --concurrency 4 --seed 7191102 >> "$D/label.log" 2>&1 || fail label "label ($D/label.log)"
  st "label done: $(tail -1 "$D/label.log")"
  rm -rf "$D/shards"
  "$PY" -u "$X" shards --labels "$D/labels" --out "$D/shards" > "$D/shards.log" 2>&1 || fail shards "shards"
  st "shards: $(tail -1 "$D/shards.log")"
  local lr=1e-5 sfx=""
  # shellcheck disable=SC2086
  ft "$D/ft-E" "$D/shards/E" $lr $ftx || fail ft "fine-tune E ($D/ft-E.log)"
  # shellcheck disable=SC2086
  ft "$D/ft-C" "$D/shards/C" $lr $ftx || fail ft "fine-tune C ($D/ft-C.log)"
  for a in E C; do
    l=$(loadcheck "$D/ft-$a/final.pt"); case $l in "LOAD OK"*) ;; *) fail load "ft-$a: $l" ;; esac
  done
  st "fine-tunes done (lr $lr); load OK"
  "$PY" -u "$X" check --labels "$D/labels" --start "$NET" --net E="$D/ft-E/final.pt" --net C="$D/ft-C/final.pt" \
    --json "$D/check.json" > "$D/check.log" 2>&1 || fail check "check ($D/check.log)"
  rE=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["E"]["r"])' "$D/check.json")
  st "manipulation r_E $rE"
  DOSE=""
  if "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 0.25 else 1)' "$rE"; then
    DOSE=--dose-retry; lr=3e-5; sfx=-lr3e-5
    # shellcheck disable=SC2086
    ft "$D/ft-E$sfx" "$D/shards/E" $lr $ftx || fail ft "dose retry E"
    # shellcheck disable=SC2086
    ft "$D/ft-C$sfx" "$D/shards/C" $lr $ftx || fail ft "dose retry C"
    for a in E C; do
      l=$(loadcheck "$D/ft-$a$sfx/final.pt"); case $l in "LOAD OK"*) ;; *) fail load "ft-$a$sfx: $l" ;; esac
    done
    "$PY" -u "$X" check --labels "$D/labels" --start "$NET" --net E="$D/ft-E$sfx/final.pt" \
      --net C="$D/ft-C$sfx/final.pt" --json "$D/check$sfx.json" > "$D/check$sfx.log" 2>&1 || fail check "check retry"
    st "dose retry ran (r_E $rE < 0.25): lr 3e-5, $(cat "$D/check$sfx.json" | tr -d '\n ' | cut -c1-200)"
  fi
  NE=$D/ft-E$sfx/final.pt; NC=$D/ft-C$sfx/final.pt; CK=$D/check$sfx.json
  ev() {  # NAME A B GAMES SEED
    [ -s "$D/$1.json" ] && return 0
    PORYGON_PS=$PS_EVAL "$PY" -u "$X" eval --a "$2" --b "$3" --games "$4" --seed "$5" --workers "$ew" --battles "$eb" \
      --json "$D/$1.json.tmp" > "$D/$1.log" 2>&1 && mv "$D/$1.json.tmp" "$D/$1.json"
  }
  ev T1 "$NE" "$NC" "$e1" 7191111 || fail eval "T1 ($D/T1.log)"; st "T1 E vs C: $(tail -1 "$D/T1.log")"
  ev T2 "$NE" "$NET" "$e2" 7191112 || fail eval "T2 ($D/T2.log)"; st "T2 E vs S: $(tail -1 "$D/T2.log")"
  ev T3 "$NC" "$NET" "$e3" 7191113 || fail eval "T3 ($D/T3.log)"; st "T3 C vs S: $(tail -1 "$D/T3.log")"
}

if [ "$MODE" = smoke ] || ! grep -q "^PASS $(cat "$CODE/COMMIT")" "$S/smoke/SMOKE.txt" 2>/dev/null; then
  # ---- smoke: 200 games (own seed), every position, a ~24-step fine-tune (batch 32), 400-game h2h per test
  SM=$S/smoke
  [ ! -e "$SM/SMOKE.txt" ] || mv "$SM" "$S/smoke.old-$(date -u +%Y%m%dT%H%M%S)"
  pipeline "$SM" 200 7191199 "--batch-size 32 --warmup 5" 400 400 400 4 16
  "$PY" -u "$X" verdict --labels "$SM/labels" --check "$SM/check.json" --t1 "$SM/T1.json" --t2 "$SM/T2.json" \
    --t3 "$SM/T3.json" --out "$SM/verdict-smoke.txt" --min-rows 380 > /dev/null 2>&1 || fail smoke "verdict"
  res=$("$PY" - "$SM" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
bad = []
recs = sum(json.loads(p.read_text())["records"] for p in (d / "gen").glob("rec-*.json"))
rows = sum(len(p.read_text().splitlines()) for p in (d / "labels").glob("label-*.meta.jsonl"))
drops = {}
for p in (d / "labels").glob("drops-*.jsonl"):
    for line in p.read_text().splitlines():
        r = json.loads(line)["reason"]; drops[r] = drops.get(r, 0) + 1
par = drops.get("parity-legal", 0) + drops.get("parity-probs", 0)
if recs < 380: bad.append(f"records {recs} < 380")
if rows < 0.97 * recs: bad.append(f"labelled {rows} of {recs}")
if par > max(2, 0.005 * recs): bad.append(f"parity drops {par}")
if drops.get("lookahead", 0) > max(2, 0.02 * recs): bad.append(f"lookahead failures {drops.get('lookahead')}")
for k in ("error", "not-replayed"):
    if drops.get(k, 0) > max(2, 0.01 * recs): bad.append(f"{k} {drops[k]}")
for t in ("T1", "T2", "T3"):
    r = json.loads((d / f"{t}.json").read_text())
    if r["pairs"] < 150: bad.append(f"{t} pairs {r['pairs']}")
ck = json.loads((d / "check.json").read_text())
if not ck["E"]["kl"] < ck["S"]["kl"]: bad.append(f"E did not move toward pi_LA (kl E {ck['E']['kl']} S {ck['S']['kl']})")
print(("FAIL " + "; ".join(bad)) if bad else "PASS", f"records {recs}, labelled {rows}, drops {drops}, r_E {ck['E']['r']:.3f}")
PYEOF
)
  case $res in
    PASS*) echo "PASS $(cat "$CODE/COMMIT") $(now) job $J: $res" > "$SM/SMOKE.txt"; cat "$SM/verdict-smoke.txt" >> "$SM/SMOKE.txt"
           st "SMOKE PASS: $res" ;;
    *) echo "FAIL $(now) job $J: $res" > "$SM/SMOKE.txt"; fail smoke "$res" ;;
  esac
  [ "$MODE" = smoke ] && { st "smoke-only run done"; exit 0; }
else
  st "smoke: PASS on record ($(head -1 "$S/smoke/SMOKE.txt" | cut -c1-120))"
fi

# ---- main (PREREG sizes)
pipeline "$S" 60000 7191101 "" 16000 16000 8000 10 32
"$PY" -u "$X" verdict --labels "$S/labels" --check "$CK" --t1 "$S/T1.json" --t2 "$S/T2.json" --t3 "$S/T3.json" \
  --out "$S/VERDICT.txt.tmp" --min-rows 110000 $DOSE > "$S/verdict.log" 2>&1 || fail verdict "verdict ($S/verdict.log)"
mv -f "$S/VERDICT.txt.tmp" "$S/VERDICT.txt" && chmod a-w "$S/VERDICT.txt"
head -1 "$S/VERDICT.txt" > "$S/RESULT"
st "VERDICT: $(head -1 "$S/VERDICT.txt")"
trap - TERM INT
exit 0
