# netv2 G9t start on the A16 (job 810367 or a later g9 job, run g9, MODE g9r; netv2.env >= v14): PARK the live G9r
# lineage whole in runs/g9r-parked/, then MODE g9t (G9t = G9q's recipe + --gae-lambda 0.5, runs/g9t/RECIPE "g9q 0.5").
# Install as runs/g9/INLINE_CMD (cp to INLINE_CMD.tmp, then mv), touch runs/g9/restart, SIGINT the G9r trainer. nv2_pause
# runs it once, synchronously, with the trainer stopped and LOCAL exported (cwd = $NV2; log runs/g9/inline-<time>.log;
# its keeper feeds the 45-min watchdog); right after it, the same sourcing of netv2.env (nv2_swap, then the MODE read)
# starts G9t from P3 (steps 0) and writes runs/g9t/G9T_STARTED. While this file is queued as runs/g9/INLINE_CMD the v12
# watchdog blocks fresh g9 job starts (it names runs/g9r-parked: nv2_wd_blocked's park pattern), as intended.
# The park marker is runs/g9r-parked/PARKED.txt, never runs/g9/PARKED (the watchdog's "never start run g9"); with it,
# netv2.env v14 never falls back to G9r when MODE g9t is refused (HOLD instead).
# G9r's just-saved state ($LOCAL/run/G9r: final_model.pt with optimizer + counters, checkpoints, pool incl. frozen,
# best / ema, train.log, tb, eval, GUARD_TRIPPED*, everything in the lineage dir; guard state runs/g9r -> state/) is
# copied to runs/g9r-parked.tmp, verified (cmp of every file, net counts, md5sums, final_model.pt loads with optimizer +
# counters under gloria's python + src-ff32b71:src), PARKED.txt written, published by mv -T, and ONLY THEN runs/g9/MODE =
# g9t (temp + mv). Any failure: nothing changes (MODE g9r, no runs/g9r-parked; a partial .tmp may be left), a FAIL line
# here and in runs/g9t/park.log, and G9r relaunches from its own save.
# Refuses (FAIL, nothing done) unless: MODE g9 is g9r and G9r started (runs/g9r/G9R_STARTED); runs/g9t/RECIPE has
# exactly one non-comment line, "g9q 0.5"; runs/g9t/PREREG.txt exists and is read-only; runs/g9t/G9T_STARTED is absent;
# runs/g9r-parked is absent; netv2.env has v14's g9t mode (the A16 call site + the parked-HOLD guard); the ff32b71
# wrapper + its read-only src (train.py md5 below) and P3 exist; no trainer of this job's run dir is alive; G9r's
# final_model.pt is fresh (newer than this queued INLINE_CMD, i.e. saved after the SIGINT, and <= PARK_MAX_AGE_S old),
# its train.log has a "saved final_model.pt at N steps" after the trainer's last start, N >= the last update's steps
# and the newest checkpoint's, and the load's counters say steps N.
set -u
C=${NV2:-$PWD}; SHA=ff32b71; TRAIN_MD5=3a667fb3f851f75694e03150140f193a
WRAP=$C/bin/py-src-$SHA; SRC=$C/src-$SHA; P3=$C/runs/pretrain/P3-g9-s2.pt; T9=$C/runs/g9t; P=$C/runs/g9r-parked
GPY=${NV2_PARK_PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}  # NV2_PARK_PY: the laptop mocks only
QUEUED=${PARK_QUEUED_FILE:-$0}  # the queued INLINE_CMD (nv2_pause runs runs/g9/INLINE_CMD.running; mv keeps its mtime)
mkdir -p "$T9" || { echo "FAIL: cannot create $T9"; exit 1; }
say() { echo "$(date -u +%FT%T) $*"; echo "$(date -u +%FT%T) g9 ${SLURM_JOB_ID:-?} park-g9r $*" >> "$T9/park.log" 2>/dev/null || true; }
fail() { say "FAIL (MODE untouched: $(cat "$C/runs/g9/MODE" 2>/dev/null)): $*"; exit 1; }
[ -n "${LOCAL:-}" ] || fail "LOCAL unset (run me only through nv2_pause)"
M0=$(cat "$C/runs/g9/MODE" 2>/dev/null || true)
[ "$M0" = g9r ] || fail "runs/g9/MODE is '$M0' (expected g9r)"
LIN=G9r; L=$LOCAL/run/$LIN
say "start: MODE $M0, lineage $LIN ($L) -> $P, then MODE g9t"
[ -f "$C/runs/g9r/G9R_STARTED" ] || fail "no runs/g9r/G9R_STARTED: G9r never started"
[ -f "$T9/RECIPE" ] || fail "no runs/g9t/RECIPE (want exactly 'g9q 0.5')"
n=$(grep -cvE '^[[:space:]]*(#|$)' "$T9/RECIPE" 2>/dev/null || true)
line=$(grep -vE '^[[:space:]]*(#|$)' "$T9/RECIPE" 2>/dev/null | head -n 1 | tr -d '\r' || true)
[ "$n" = 1 ] && [ "$line" = "g9q 0.5" ] || fail "runs/g9t/RECIPE is not exactly 'g9q 0.5' ($n line(s), first: '$(printf '%s' "$line" | head -c 80)')"
[ -f "$T9/PREREG.txt" ] || fail "no runs/g9t/PREREG.txt (pre-register G9t before its first step)"
pm=$(stat -c %a "$T9/PREREG.txt" 2>/dev/null || echo 777)
[ $(( 8#$pm & 8#222 )) = 0 ] && [ ! -w "$T9/PREREG.txt" ] || fail "runs/g9t/PREREG.txt is not read-only (mode $pm): chmod a-w it"
[ ! -e "$T9/G9T_STARTED" ] || fail "runs/g9t/G9T_STARTED exists ('$(head -c 80 "$T9/G9T_STARTED")'): G9t started already"
[ ! -e "$P" ] || fail "$P exists already"
grep -q 'nv2_cfg_g9t g9 8' "$C/netv2.env" 2>/dev/null && grep -q 'runs/g9r-parked/PARKED.txt' "$C/netv2.env" 2>/dev/null || fail "netv2.env has no v14 g9t mode (A16 call site + the parked-HOLD guard): upload v14 first"
[ -x "$WRAP" ] && grep -q "$SRC" "$WRAP" || fail "no executable $WRAP naming $SRC"
[ "$(md5sum < "$SRC/porygon/v2/train.py" 2>/dev/null | cut -d' ' -f1)" = "$TRAIN_MD5" ] || fail "$SRC/porygon/v2/train.py is not ws/netv2 $SHA's (md5 $TRAIN_MD5)"
[ -f "$P3" ] || fail "no $P3"
if pgrep -u "$(id -u)" -f "porygon[.]v2[.]train .*--run-dir $LOCAL/run(/| |$)" > /dev/null 2>&1; then fail "a trainer with --run-dir under $LOCAL/run is still running"; fi
[ -f "$L/final_model.pt" ] || fail "no $L/final_model.pt"
[ "$L/final_model.pt" -nt "$QUEUED" ] || fail "$L/final_model.pt is not newer than the queued INLINE_CMD ($QUEUED): stale (not saved after the SIGINT)"
age=$(( $(date +%s) - $(stat -c %Y "$L/final_model.pt") ))
[ "$age" -le "${PARK_MAX_AGE_S:-1800}" ] || fail "$L/final_model.pt is ${age} s old (the trainer did not just save it)"
ls0=$(grep -n "v2 trainer:" "$L/train.log" 2>/dev/null | tail -n 1 | cut -d: -f1)
sv=$(tail -n +"${ls0:-1}" "$L/train.log" 2>/dev/null | grep "saved final_model.pt at" | tail -n 1)
[ -n "$sv" ] || fail "no 'saved final_model.pt at' after $LIN's last start in $L/train.log: stale"
ss=$(grep -oE 'saved final_model.pt at [0-9,]+ steps' <<< "$sv" | tr -dc 0-9)
us=$(grep -oE ' upd [0-9]+ steps [0-9,]+' "$L/train.log" 2>/dev/null | tail -n 1 | sed 's/.* steps //' | tr -dc 0-9)
cs=$(ls "$L/checkpoints" 2>/dev/null | grep -oE '^ckpt_[0-9]+\.pt$' | tr -dc '0-9\n' | sort -n | tail -n 1)
[ -n "$ss" ] && [ "$ss" -ge "${us:-0}" ] && [ "$ss" -ge "$((10#${cs:-0}))" ] || fail "the last save (${ss:-?} steps) is not the newest step (last update ${us:-none}, newest checkpoint ${cs:-none}): stale"
[ -d "$L/pool" ] || fail "no $L/pool"
need=$(du -sk "$L" 2>/dev/null | cut -f1); free=$(df -Pk "$C/runs" 2>/dev/null | awk 'NR == 2 {print $4}')
[ -n "$need" ] && [ -n "$free" ] && [ "$free" -gt $(( need * 2 + 1048576 )) ] || fail "disk: need ~$(( ${need:-0} / 1024 )) MiB x 2 + 1 GiB, $(( ${free:-0} / 1024 )) MiB free"
rm -rf "$P.tmp"; mkdir -p "$P.tmp/state" || fail "cannot create $P.tmp"
cp -a "$L/." "$P.tmp/" || fail "copy $L -> $P.tmp failed (partial copy left in $P.tmp)"
for f in G9R_STARTED FU.started FU.reason fu-start-best.pt; do if [ -f "$C/runs/g9r/$f" ]; then cp -p "$C/runs/g9r/$f" "$P.tmp/state/" || fail "copy runs/g9r/$f failed"; fi; done
n=0; while IFS= read -r -d '' f; do  # every file of the lineage dir, byte for byte
  r=${f#"$L"/}; cmp -s "$f" "$P.tmp/$r" || fail "copy mismatch: $r (partial copy left in $P.tmp)"; n=$((n + 1))
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
[ "$ks" = "$ss" ] || fail "final_model.pt's counters say steps ${ks:-?}, its train.log save line $ss: stale or not this save"
md5s() { (cd "$P.tmp" && for f in final_model.pt best_model.pt ema_model.pt; do [ -f "$f" ] && echo "$(md5sum < "$f" | cut -c1-8) $f"; done | tr '\n' ' '); }
{ echo "$LIN PARKED $(date -u +%FT%T) in job ${SLURM_JOB_ID:-?} ($(hostname -s 2>/dev/null)) for G9t (MODE g9r -> g9t; runs/g9t/RECIPE 'g9q 0.5', runs/g9t/PREREG.txt)"
  echo "steps: $ss (final_model.pt saved $(date -u -d "@$(stat -c %Y "$L/final_model.pt")" +%FT%T), ${age} s before the check)"
  echo "source: $L (the trainer's save after the SIGINT, restart in place); guard state runs/g9r (copied to state/)"
  echo "final_model.pt $ck"
  echo "last update: $(grep -E ' upd [0-9]+ steps' "$L/train.log" | tail -n 1 | cut -c1-200)"
  echo "last save: $sv"
  echo "last start: $(grep 'v2 trainer:' "$L/train.log" | tail -n 1 | cut -c1-300)"
  echo "job's last relaunch: $(grep 'restart in place from' "$LOCAL/run/train.log" 2>/dev/null | tail -n 1)"
  grep '\[eval @' "$L/train.log" | tail -n 3
  echo "guard / fallback: FU.started '$(cat "$C/runs/g9r/FU.started" 2>/dev/null)', FU.reason '$(head -c 200 "$C/runs/g9r/FU.reason" 2>/dev/null)'; trip files: $(cd "$P.tmp" && ls GUARD_TRIPPED* 2>/dev/null | tr '\n' ' ')"
  echo "files: $n ($(du -sh "$P.tmp" | cut -f1)), pool $(ls "$P.tmp/pool" | grep -c '\.pt$') nets, frozen $(ls "$P.tmp/pool/frozen" 2>/dev/null | grep -c '\.pt$'), checkpoints $(ls "$P.tmp/checkpoints" 2>/dev/null | grep -c '\.pt$'); md5: $(md5s); all files: MD5SUMS"
  echo "resume: a G9r lineage dir seeded from this dir (final_model.pt = trainer state with optimizer + counters, pool/ incl. frozen/, best / ema) under MODE g9r's recipe (guard state runs/g9r, src-$SHA); a netv2.env change + user decision. Never resume it while it is also training elsewhere."
} > "$P.tmp/PARKED.txt" || fail "cannot write PARKED.txt"
[ ! -e "$P" ] && mv -T "$P.tmp" "$P" || fail "could not publish $P"
cat "$P/PARKED.txt"
if ! { echo g9t > "$C/runs/g9/MODE.tmp" && mv -f "$C/runs/g9/MODE.tmp" "$C/runs/g9/MODE" && [ "$(cat "$C/runs/g9/MODE")" = g9t ]; }; then
  t=$(date -u +%m%d-%H%M%S); mv -T "$P" "$P.void-$t" 2>/dev/null || true
  echo "$M0" > "$C/runs/g9/MODE" 2>/dev/null || true
  fail "MODE write failed: park dir renamed $P.void-$t, $LIN continues"
fi
for f in "$L"/GUARD_TRIPPED; do if [ -f "$f" ]; then mv -f "$f" "$f.parked-$(date -u +%H%M%S)" || true; fi; done  # it travelled with the park
say "DONE: $LIN parked in $P at $ss steps ($(md5s)); MODE g9r -> g9t"
