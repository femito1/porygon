# netv2 G9r start on the A16 (job 741849 or its successor 810367, run g9; netv2.env >= v10): PARK the lineage the A16
# trains now, then MODE g9r. Install as runs/g9/INLINE_CMD (cp to INLINE_CMD.tmp, then mv), touch runs/g9/restart,
# SIGINT the trainer. nv2_pause runs it once, synchronously, with the trainer stopped and LOCAL exported (cwd = $NV2;
# log runs/g9/inline-<time>.log; its keeper feeds the 45-min watchdog); right after it, the same sourcing of netv2.env
# (nv2_swap, then the MODE read) starts G9r from P3.
#   runs/g9/MODE g9q  -> lineage G9q ($LOCAL/run/G9q, guard state runs/g9q)  -> runs/g9q-parked/
#   runs/g9/MODE g9p2 -> lineage G9p ($LOCAL/run/G9p, guard state runs/g9a30) -> runs/g9p-parked-a16/
# The lineage's just-saved state (final_model.pt with optimizer + counters, best / ema, train.log, tb, eval, GUARD_TRIPPED*,
# pool incl. frozen, checkpoints, everything in the lineage dir) is copied to <park>.tmp, verified (cmp of every file,
# md5sums, the checkpoint loads with optimizer + counters), PARKED.txt written, published by mv -T, and ONLY THEN
# runs/g9/MODE = g9r (temp + mv). Any failure: nothing changes (MODE untouched, no park dir; a partial <park>.tmp may be
# left), a FAIL line here and in runs/g9r/park.log, and the same lineage relaunches from its own save.
# Refuses (FAIL, nothing done) unless: MODE is g9q / g9p2 (g9p2: G9P_A16_STARTED), G9r never started, the 50M swap is
# resolved for this side (no runs/swap-50M, or a DECISION and, for COMMIT, runs/g9/SWAP_APPLIED + MODE g9p2; ABORT:
# MODE g9q), netv2.env has the g9r mode, the G9r wrapper + read-only src-ff32b71 (train.py md5 below) exist, P3 is
# there, runs/g9r/PREREG.txt exists, the lineage's final_model.pt is fresh and saved after its last start, and no
# trainer of this run dir is alive.
set -u
C=${NV2:-$PWD}; SHA=ff32b71; TRAIN_MD5=3a667fb3f851f75694e03150140f193a
WRAP=$C/bin/py-src-$SHA; SRC=$C/src-$SHA; P3=$C/runs/pretrain/P3-g9-s2.pt; S=$C/runs/swap-50M
GPY=${NV2_PARK_PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}  # NV2_PARK_PY: the laptop mocks only
mkdir -p "$C/runs/g9r" || { echo "FAIL: cannot create $C/runs/g9r"; exit 1; }
say() { echo "$(date -u +%FT%T) $*"; echo "$(date -u +%FT%T) g9 ${SLURM_JOB_ID:-?} park $*" >> "$C/runs/g9r/park.log" 2>/dev/null || true; }
fail() { say "FAIL (MODE untouched: $(cat "$C/runs/g9/MODE" 2>/dev/null)): $*"; exit 1; }
[ -n "${LOCAL:-}" ] || fail "LOCAL unset (run me only through nv2_pause)"
M0=$(cat "$C/runs/g9/MODE" 2>/dev/null || true)
case $M0 in
  g9q)  LIN=G9q; STATE=g9q;   P=$C/runs/g9q-parked ;;
  g9p2) LIN=G9p; STATE=g9a30; P=$C/runs/g9p-parked-a16
        [ -f "$C/runs/g9/G9P_A16_STARTED" ] || fail "MODE g9p2 but no runs/g9/G9P_A16_STARTED: G9p never started on this A16 (G9q may still run here)" ;;
  *)    fail "runs/g9/MODE is '$M0' (expected g9q or g9p2)" ;;
esac
L=$LOCAL/run/$LIN
say "start: MODE $M0, lineage $LIN ($L) -> $P"
[ ! -e "$C/runs/g9r/G9R_STARTED" ] || fail "runs/g9r/G9R_STARTED exists: G9r started already"
if [ -d "$S" ]; then  # the 50M swap must be resolved for this side before the A16 changes lineage
  d=$(head -n 1 "$S/DECISION" 2>/dev/null | cut -d' ' -f1 || true)
  case $d in
    COMMIT) [ -f "$C/runs/g9/SWAP_APPLIED" ] && [ "$M0" = g9p2 ] || fail "swap DECISION COMMIT but this side has not applied it (SWAP_APPLIED / MODE g9p2)" ;;
    ABORT)  [ "$M0" = g9q ] || fail "swap DECISION ABORT but MODE $M0" ;;
    *)      fail "runs/swap-50M exists without a DECISION (the swap is unresolved)" ;;
  esac
fi
grep -q '"$G9_MODE" = g9r' "$C/netv2.env" 2>/dev/null && grep -q "bin/py-src-$SHA" "$C/netv2.env" || fail "netv2.env has no g9r mode with py-src-$SHA (upload v10 first)"
[ -x "$WRAP" ] && grep -q "$SRC" "$WRAP" || fail "no executable $WRAP naming $SRC"
[ "$(md5sum < "$SRC/porygon/v2/train.py" 2>/dev/null | cut -d' ' -f1)" = "$TRAIN_MD5" ] || fail "$SRC/porygon/v2/train.py is not ws/netv2 $SHA's (md5 $TRAIN_MD5)"
[ -f "$P3" ] || fail "no $P3"
[ -f "$C/runs/g9r/PREREG.txt" ] || fail "no runs/g9r/PREREG.txt (pre-register G9r before its first step)"
[ -f "$L/final_model.pt" ] || fail "no $L/final_model.pt"
age=$(( $(date +%s) - $(stat -c %Y "$L/final_model.pt") ))
[ "$age" -le "${PARK_MAX_AGE_S:-1800}" ] || fail "$L/final_model.pt is ${age} s old (the trainer did not just save it)"
ls0=$(grep -n "v2 trainer:" "$L/train.log" 2>/dev/null | tail -n 1 | cut -d: -f1)
tail -n +"${ls0:-1}" "$L/train.log" 2>/dev/null | grep -q "saved final_model.pt at" || fail "no 'saved final_model.pt at' after $LIN's last start in $L/train.log"
if pgrep -u "$(id -u)" -f "porygon.v2.train .*--run-dir $L( |$)" > /dev/null 2>&1; then fail "a trainer with --run-dir $L is still running"; fi
[ -d "$L/pool" ] || fail "no $L/pool"
[ ! -e "$P" ] || fail "$P exists already"
need=$(du -sk "$L" 2>/dev/null | cut -f1); free=$(df -Pk "$C/runs" 2>/dev/null | awk 'NR == 2 {print $4}')
[ -n "$need" ] && [ -n "$free" ] && [ "$free" -gt $(( need * 2 + 1048576 )) ] || fail "disk: need ~$(( ${need:-0} / 1024 )) MiB x 2 + 1 GiB, $(( ${free:-0} / 1024 )) MiB free"
rm -rf "$P.tmp"; mkdir -p "$P.tmp/state" || fail "cannot create $P.tmp"
cp -a "$L/." "$P.tmp/" || fail "copy $L -> $P.tmp failed (partial copy left in $P.tmp)"
for f in FU.started FU.reason fu-start-best.pt; do if [ -f "$C/runs/$STATE/$f" ]; then cp -p "$C/runs/$STATE/$f" "$P.tmp/state/" || fail "copy runs/$STATE/$f failed"; fi; done
n=0; while IFS= read -r -d '' f; do  # every file of the lineage dir, byte for byte
  r=${f#"$L"/}; cmp -s "$f" "$P.tmp/$r" || fail "copy mismatch: $r (partial copy left in $P.tmp)"; n=$((n + 1))
done < <(find "$L" -type f -print0)
for d in pool pool/frozen checkpoints; do
  a=$(ls "$L/$d" 2>/dev/null | grep -c '\.pt$'); b=$(ls "$P.tmp/$d" 2>/dev/null | grep -c '\.pt$')
  [ "$a" = "$b" ] || fail "$d: $a nets here, $b copied"
done
( cd "$P.tmp" && find . -type f ! -name MD5SUMS -print0 | sort -z | xargs -0 md5sum > MD5SUMS ) || fail "md5sums failed"
ck=$(PYTHONPATH=$C/src timeout 600 "$GPY" -c "
import torch; ck = torch.load('$P.tmp/final_model.pt', map_location='cpu', weights_only=False)
c = ck.get('counters', {}); assert 'optimizer' in ck and c.get('steps', 0) > 0, (sorted(ck), c)
print('counters:', c); print('keys:', sorted(ck))" 2>&1) || fail "final_model.pt does not load with optimizer + counters: $(tail -n 2 <<< "$ck")"
md5s() { (cd "$P.tmp" && for f in final_model.pt best_model.pt ema_model.pt; do [ -f "$f" ] && echo "$(md5sum < "$f" | cut -c1-8) $f"; done | tr '\n' ' '); }
{ echo "$LIN PARKED $(date -u +%FT%T) in job ${SLURM_JOB_ID:-?} ($(hostname -s 2>/dev/null)) for G9r (MODE $M0 -> g9r; .orchestra/netv2/G9R-PROPOSAL.md, runs/g9r/PREREG.txt)"
  echo "source: $L (the trainer's save at the restart in place); guard state runs/$STATE (copied to state/)"
  echo "final_model.pt $ck"
  echo "last update: $(grep -E ' upd [0-9]+ steps' "$L/train.log" | tail -n 1 | cut -c1-200)"
  echo "last save: $(grep 'saved final_model.pt at' "$L/train.log" | tail -n 1)"
  echo "last start: $(grep 'v2 trainer:' "$L/train.log" | tail -n 1 | cut -c1-300)"
  echo "job's last relaunch: $(grep 'restart in place from' "$LOCAL/run/train.log" 2>/dev/null | tail -n 1)"
  grep '\[eval @' "$L/train.log" | tail -n 3
  echo "guard / fallback: FU.started '$(cat "$C/runs/$STATE/FU.started" 2>/dev/null)', FU.reason '$(head -c 200 "$C/runs/$STATE/FU.reason" 2>/dev/null)'; trip files: $(cd "$P.tmp" && ls GUARD_TRIPPED* 2>/dev/null | tr '\n' ' ')"
  echo "files: $n ($(du -sh "$P.tmp" | cut -f1)), pool $(ls "$P.tmp/pool" | grep -c '\.pt$') nets, frozen $(ls "$P.tmp/pool/frozen" 2>/dev/null | grep -c '\.pt$'), checkpoints $(ls "$P.tmp/checkpoints" 2>/dev/null | grep -c '\.pt$'); md5: $(md5s); all files: MD5SUMS"
  echo "resume: a $LIN lineage dir seeded from this dir (final_model.pt = trainer state with optimizer + counters, pool/ incl. frozen/, best / ema) under MODE $M0's recipe (guard state runs/$STATE); a netv2.env change + user decision. Never resume it while it is also training elsewhere."
} > "$P.tmp/PARKED.txt" || fail "cannot write PARKED.txt"
[ ! -e "$P" ] && mv -T "$P.tmp" "$P" || fail "could not publish $P"
cat "$P/PARKED.txt"
if ! { echo g9r > "$C/runs/g9/MODE.tmp" && mv -f "$C/runs/g9/MODE.tmp" "$C/runs/g9/MODE" && [ "$(cat "$C/runs/g9/MODE")" = g9r ]; }; then
  t=$(date -u +%m%d-%H%M%S); mv -T "$P" "$P.void-$t" 2>/dev/null || true
  echo "$M0" > "$C/runs/g9/MODE" 2>/dev/null || true
  fail "MODE write failed: park dir renamed $P.void-$t, $LIN continues"
fi
for f in "$L"/GUARD_TRIPPED; do if [ -f "$f" ]; then mv -f "$f" "$f.parked-$(date -u +%H%M%S)" || true; fi; done  # it travelled with the park
say "DONE: $LIN parked in $P ($(md5s)); MODE $M0 -> g9r"
