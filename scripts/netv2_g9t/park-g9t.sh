# netv2 2026-11-14: PARK the live G9t lineage whole in runs/g9t-parked/ (a STOP by a G9t PREREG rule; orchestrator
# 2026-11-14 ~15:30: a stop must be persistent, not a 4-h HOLD). Adapted from inline-a16-park-g9r-start-g9t.sh (32464c15).
# Runs only inside an INLINE_CMD of the g9 job (nv2_pause: trainer stopped, LOCAL exported, cwd $NV2), as
#   PARK_RULE=<n> PARK_QUEUED_FILE=<the queued INLINE file> [PARK_MAX_AGE_S=<s>] bash $NV2/scripts/netv2_g9t/park-g9t.sh
# Copies $LOCAL/run/G9t (G9t's save after the SIGINT: final_model.pt with optimizer + counters, checkpoints, pool incl.
# frozen, best / ema, logs) + its guard state runs/g9t/{FU.*} to runs/g9t-parked.tmp, verifies (cmp of every file, net
# counts, md5sums, final_model.pt loads with optimizer + counters), writes PARKED.txt, publishes by mv -T. ONLY THEN:
#  - runs/g9t/G9T_STARTED's run field g9 -> "parked" (original kept as G9T_STARTED.orig-parked): netv2.env v14's
#    nv2_g9t_recipe then refuses G9t in run g9 ("G9t started in run parked, not g9") and, with runs/g9r-parked/PARKED.txt
#    present, the job HOLDs round after round (allocation kept, never G9r, never G9t) -- env mock 53 (run-field case);
#  - runs/g9/PARKED (the v12 watchdog's deliberate stop: a fresh g9 job, e.g. the successor 1137377, exits 0 at once,
#    no training, no further successor) -- env mock 45.
# Any failure before the publish: nothing changes (a partial .tmp may be left), FAIL, exit 1: G9t relaunches from its
# own save (the caller records FAILED). A failure after it: the park dir is renamed .void-<t>, G9T_STARTED restored.
set -u
C=${NV2:-$PWD}; SHA=ff32b71; SRC=$C/src-$SHA; T9=$C/runs/g9t; P=$C/runs/g9t-parked
GPY=${NV2_PARK_PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}
QUEUED=${PARK_QUEUED_FILE:?set PARK_QUEUED_FILE}; RULE=${PARK_RULE:?set PARK_RULE}
say() { echo "$(date -u +%FT%T) $*"; echo "$(date -u +%FT%T) g9 ${SLURM_JOB_ID:-?} park-g9t $*" >> "$T9/park.log" 2>/dev/null || true; }
fail() { say "FAIL (G9t not parked; MODE $(cat "$C/runs/g9/MODE" 2>/dev/null)): $*"; exit 1; }
[ -n "${LOCAL:-}" ] || fail "LOCAL unset (run me only through nv2_pause)"
M0=$(cat "$C/runs/g9/MODE" 2>/dev/null || true)
[ "$M0" = g9t ] || fail "runs/g9/MODE is '$M0' (expected g9t)"
LIN=G9t; L=$LOCAL/run/$LIN
say "start (rule $RULE): lineage $LIN ($L) -> $P"
read -r st_t st_c st_l st_r st_j < "$T9/G9T_STARTED" 2>/dev/null || fail "no runs/g9t/G9T_STARTED"
[ "$st_c $st_l $st_r" = "g9q 0.5 g9" ] || fail "runs/g9t/G9T_STARTED is '$st_t $st_c $st_l $st_r $st_j' (expected 'g9q 0.5 g9')"
[ -f "$C/runs/g9r-parked/PARKED.txt" ] || fail "no runs/g9r-parked/PARKED.txt (without it v14 would fall back to G9r)"
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
for f in G9T_STARTED RECIPE PREREG.txt FU.started FU.reason fu-start-best.pt; do if [ -f "$T9/$f" ]; then cp -p "$T9/$f" "$P.tmp/state/" || fail "copy runs/g9t/$f failed"; fi; done
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
{ echo "$LIN PARKED $(date -u +%FT%T) in job ${SLURM_JOB_ID:-?} ($(hostname -s 2>/dev/null)): STOPPED by runs/g9t/PREREG.txt rule $RULE (see runs/g9t/chain/STATUS); A16 placement = orchestrator / user"
  echo "steps: $ss (final_model.pt saved $(date -u -d "@$(stat -c %Y "$L/final_model.pt")" +%FT%T))"
  echo "source: $L (the trainer's save after the SIGINT); guard state runs/g9t (copied to state/)"
  echo "final_model.pt $ck"
  echo "last update: $(grep -E ' upd [0-9]+ steps' "$L/train.log" | tail -n 1 | cut -c1-200)"
  echo "last save: $sv"
  echo "last start: $(grep 'v2 trainer:' "$L/train.log" | tail -n 1 | cut -c1-300)"
  grep '\[eval @' "$L/train.log" | tail -n 3
  echo "files: $n ($(du -sh "$P.tmp" | cut -f1)), pool $(ls "$P.tmp/pool" | grep -c '\.pt$') nets, frozen $(ls "$P.tmp/pool/frozen" 2>/dev/null | grep -c '\.pt$'), checkpoints $(ls "$P.tmp/checkpoints" 2>/dev/null | grep -c '\.pt$'); md5: $(md5s); all files: MD5SUMS"
  echo "blocks: runs/g9t/G9T_STARTED run field -> parked (original: runs/g9t/G9T_STARTED.orig-parked; v14 then HOLDs the g9 job), runs/g9/PARKED (fresh g9 jobs exit at once)"
  echo "resume: a G9t lineage dir seeded from this dir (final_model.pt = trainer state with optimizer + counters, pool/ incl. frozen/, best / ema) under MODE g9t (G9t = runs/g9t/RECIPE 'g9q 0.5', src-$SHA): restore G9T_STARTED.orig-parked, rm runs/g9/PARKED + runs/g9/HOLD; an orchestrator / user decision. Never resume it while it also trains elsewhere."
} > "$P.tmp/PARKED.txt" || fail "cannot write PARKED.txt"
[ ! -e "$P" ] && mv -T "$P.tmp" "$P" || fail "could not publish $P"
cat "$P/PARKED.txt"
void() { local t; t=$(date -u +%m%d-%H%M%S); mv -T "$P" "$P.void-$t" 2>/dev/null || true
  cp -p "$T9/G9T_STARTED.orig-parked" "$T9/G9T_STARTED" 2>/dev/null || true; rm -f "$C/runs/g9/PARKED"; fail "$1: park dir renamed $P.void-$t, G9T_STARTED restored, G9t continues"; }
cp -p "$T9/G9T_STARTED" "$T9/G9T_STARTED.orig-parked" || void "cannot back up G9T_STARTED"
{ echo "$st_t $st_c $st_l parked ${st_j:-}" > "$T9/G9T_STARTED.tmp" && mv -f "$T9/G9T_STARTED.tmp" "$T9/G9T_STARTED"; } || void "G9T_STARTED rewrite failed"
read -r _ a b r _ < "$T9/G9T_STARTED"; [ "$a $b $r" = "g9q 0.5 parked" ] || void "G9T_STARTED reads '$a $b $r' after the rewrite"
echo "G9t parked $(date -u +%FT%T) by rule $RULE: runs/g9t-parked/PARKED.txt" > "$C/runs/g9/PARKED" || void "cannot write runs/g9/PARKED"
say "DONE: $LIN parked in $P at $ss steps ($(md5s)); G9T_STARTED run -> parked; runs/g9/PARKED written"
