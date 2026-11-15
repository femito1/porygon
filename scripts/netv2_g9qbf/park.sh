#!/bin/bash
# netv2 2026-11-15: PARK a live g9 lineage whole (runs/g9qbf/PREREG.txt steps 2 and 6), adapted from
# scripts/netv2_g9t/park-g9t.sh (cmp-verified copy, net counts, MD5SUMS, final_model.pt loads with optimizer + counters at
# the saved step, PARKED.txt, published by mv -T). Runs only inside an INLINE_CMD of the g9 job (nv2_pause: trainer
# stopped, LOCAL exported, cwd $NV2):
#   PARK_LIN=G9q2 PARK_STATE=runs/g9q2 PARK_DEST=runs/g9q2-parked PARK_WHY='...' PARK_QUEUED=<queued INLINE file> bash park.sh
# netv2.env v17 then never resumes the lineage while PARK_DEST/PARKED.txt exists (MODE g9q2 / g9qbf => HOLD rounds).
# No runs/g9/PARKED here: the g9 run itself stays alive for the next lineage. Any failure before the publish changes
# nothing (a partial .tmp may be left) and exits 1.
set -u
C=${NV2:-$PWD}; SRC=$C/src-24c633f; GPY=${NV2_PARK_PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}
LIN=${PARK_LIN:?}; T9=$C/${PARK_STATE:?}; P=$C/${PARK_DEST:?}; WHY=${PARK_WHY:?}; QUEUED=${PARK_QUEUED:?}
say() { echo "$(date -u +%FT%T) park $LIN: $*"; echo "$(date -u +%FT%T) g9 ${SLURM_JOB_ID:-?} park $LIN $*" >> "$T9/park.log" 2>/dev/null || true; }
fail() { say "FAIL ($LIN not parked): $*"; exit 1; }
[ -n "${LOCAL:-}" ] || fail "LOCAL unset (run me only through nv2_pause)"
L=$LOCAL/run/$LIN; mkdir -p "$T9"
say "start: $L -> $P ($WHY)"
[ ! -e "$P" ] || fail "$P exists already"
if pgrep -u "$(id -u)" -f "porygon[.]v2[.]train .*--run-dir $LOCAL/run(/| |$)" > /dev/null 2>&1; then fail "a trainer with --run-dir under $LOCAL/run is still running"; fi
[ -f "$L/final_model.pt" ] || fail "no $L/final_model.pt"
[ "$L/final_model.pt" -nt "$QUEUED" ] || fail "$L/final_model.pt is not newer than the queued INLINE ($QUEUED): stale"
age=$(( $(date +%s) - $(stat -c %Y "$L/final_model.pt") ))
[ "$age" -le "${PARK_MAX_AGE_S:-1800}" ] || fail "$L/final_model.pt is ${age} s old (> ${PARK_MAX_AGE_S:-1800})"
ls0=$(grep -n "v2 trainer:" "$L/train.log" 2>/dev/null | tail -n 1 | cut -d: -f1)
sv=$(tail -n +"${ls0:-1}" "$L/train.log" 2>/dev/null | grep "saved final_model.pt at" | tail -n 1)
[ -n "$sv" ] || fail "no 'saved final_model.pt at' after $LIN's last start in $L/train.log: stale"
ss=$(grep -oE 'saved final_model.pt at [0-9,]+ steps' <<< "$sv" | tr -dc 0-9)
us=$(grep -oE ' upd [0-9]+ steps [0-9,]+' "$L/train.log" 2>/dev/null | tail -n 1 | sed 's/.* steps //' | tr -dc 0-9)
cs=$(ls "$L/checkpoints" 2>/dev/null | grep -oE '^ckpt_[0-9]+\.pt$' | tr -dc '0-9\n' | sort -n | tail -n 1)
[ -n "$ss" ] && [ "$ss" -ge "${us:-0}" ] && [ "$ss" -ge "$((10#${cs:-0}))" ] || fail "the last save (${ss:-?}) is not the newest step (update ${us:-none}, checkpoint ${cs:-none}): stale"
[ -d "$L/pool" ] || fail "no $L/pool"
need=$(du -sk "$L" 2>/dev/null | cut -f1); free=$(df -Pk "$C/runs" 2>/dev/null | awk 'NR == 2 {print $4}')
[ -n "$need" ] && [ -n "$free" ] && [ "$free" -gt $(( need * 2 + 1048576 )) ] || fail "disk: need ~$(( ${need:-0} / 1024 )) MiB x 2 + 1 GiB, $(( ${free:-0} / 1024 )) MiB free"
rm -rf "$P.tmp"; mkdir -p "$P.tmp/state" || fail "cannot create $P.tmp"
cp -a "$L/." "$P.tmp/" || fail "copy $L -> $P.tmp failed"
for f in PREREG.txt G9Q2_STARTED G9QBF_STARTED FORK_POINT FORK_POINT.src TRIPPED FU.started FU.reason fu-start-best.pt notes.log first-argv.txt first-argv-diff.txt; do
  if [ -f "$T9/$f" ]; then cp -p "$T9/$f" "$P.tmp/state/" || fail "copy $PARK_STATE/$f failed"; fi; done
n=0; while IFS= read -r -d '' f; do
  r=${f#"$L"/}; cmp -s "$f" "$P.tmp/$r" || fail "copy mismatch: $r"; n=$((n + 1))
done < <(find "$L" -type f -print0)
for d in pool pool/frozen checkpoints; do
  a=$(ls "$L/$d" 2>/dev/null | grep -c '\.pt$'); b=$(ls "$P.tmp/$d" 2>/dev/null | grep -c '\.pt$')
  [ "$a" = "$b" ] || fail "$d: $a nets here, $b copied"
done
( cd "$P.tmp" && find . -type f ! -name MD5SUMS -print0 | sort -z | xargs -0 md5sum > MD5SUMS ) || fail "md5sums failed"
ck=$(PYTHONPATH=$SRC:$C/src timeout 600 "$GPY" -c "
import torch; ck = torch.load('$P.tmp/final_model.pt', map_location='cpu', weights_only=False)
c = ck.get('counters', {}); assert 'optimizer' in ck and c.get('steps', 0) > 0, (sorted(ck), c)
print('counters:', c); print('keys:', sorted(ck))" 2>&1) || fail "final_model.pt does not load with optimizer + counters: $(tail -n 2 <<< "$ck")"
ks=$(grep -oE "'steps': [0-9]+" <<< "$ck" | head -n 1 | tr -dc 0-9)
[ "$ks" = "$ss" ] || fail "final_model.pt's counters say steps ${ks:-?}, its save line $ss"
md5s() { (cd "$P.tmp" && for f in final_model.pt best_model.pt ema_model.pt; do [ -f "$f" ] && echo "$(md5sum < "$f" | cut -c1-8) $f"; done | tr '\n' ' '); }
{ echo "$LIN PARKED $(date -u +%FT%T) in job ${SLURM_JOB_ID:-?} ($(hostname -s 2>/dev/null)) at $ss steps: $WHY"
  echo "files: $n ($(du -sh "$P.tmp" | cut -f1)), pool $(ls "$P.tmp/pool" | grep -c '\.pt$') nets, frozen $(ls "$P.tmp/pool/frozen" 2>/dev/null | grep -c '\.pt$'), checkpoints $(ls "$P.tmp/checkpoints" 2>/dev/null | grep -c '\.pt$'); md5: $(md5s); all files: MD5SUMS"
  echo "blocks: netv2.env v17 HOLDs the g9 job round after round while this PARKED.txt exists and runs/g9/MODE names this lineage"
  echo "resume: a $LIN lineage dir seeded from this dir (final_model.pt = trainer state with optimizer + counters, pool/ incl. frozen/, best / ema) through a mocked netv2.env mode; an orchestrator / user decision. Never resume it while it also trains elsewhere."
} > "$P.tmp/PARKED.txt" || fail "cannot write PARKED.txt"
mv -T "$P.tmp" "$P" || fail "publish (mv -T $P.tmp $P) failed"
say "DONE: $LIN parked in $P at $ss steps ($(cd "$P" && for f in final_model.pt best_model.pt ema_model.pt; do [ -f "$f" ] && echo "$(md5sum < "$f" | cut -c1-8) $f"; done | tr '\n' ' '))"
echo "$ss"
