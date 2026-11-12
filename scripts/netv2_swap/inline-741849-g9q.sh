# netv2 G9q <-> G9p GPU swap, PHASE 1 on the A16 side (job 741849, run g9; netv2.env >= v9): install as
# runs/g9/INLINE_CMD (cp to INLINE_CMD.tmp, then mv), touch runs/g9/restart, SIGINT the trainer. nv2_pause runs it once,
# synchronously, with the trainer stopped and LOCAL exported (cwd = $NV2; log runs/g9/inline-<time>.log; its keeper
# feeds the 45-min watchdog). It only COPIES: G9q's just-saved state ($LOCAL/run/G9q) -> runs/g9q/handover-a30/, published
# whole (handover-a30.tmp + mv, DONE inside) or, on any failure, runs/swap-50M/FAIL-g9 and nothing else. It never touches
# MODE: phase 2 (nv2_swap in netv2.env, right after this) decides COMMIT / ABORT with the other side and applies it.
set -u
C=${NV2:-$PWD}; S=$C/runs/swap-50M; SIDE=g9
L=$LOCAL/run/G9q; H=$C/runs/g9q/handover-a30
GPY=/vol/bitbucket/fr1226/gloria/.venv/bin/python
st() { echo "$(date -u +%FT%T) $SIDE ${SLURM_JOB_ID:-741849} phase1 $*" >> "$S/STATE"; echo "phase1 $*"; }
fail() { st "FAIL: $*"; echo "$(date -u +%FT%T) $*" >> "$S/FAIL-$SIDE"; exit 1; }
mkdir -p "$S" || { echo "cannot create $S"; exit 1; }
if [ -e "$S/DECISION" ]; then st "REFUSED (nothing done): a decision exists already: $(head -n 1 "$S/DECISION")"; exit 0; fi
st "start: $L -> $H"
[ -f "$L/final_model.pt" ] || fail "no $L/final_model.pt"
age=$(( $(date +%s) - $(stat -c %Y "$L/final_model.pt") ))
[ "$age" -le "${SWAP_MAX_AGE_S:-1800}" ] || fail "$L/final_model.pt is ${age} s old (the trainer did not just save it)"
[ -d "$L/pool" ] || fail "no $L/pool"
[ ! -e "$H" ] || fail "$H exists already (an earlier attempt? see PROCEDURE.txt)"
ok=1
rm -rf "$H.tmp"; mkdir -p "$H.tmp" || ok=0
for f in best_model.pt ema_model.pt train.log GUARD_TRIPPED; do if [ -f "$L/$f" ]; then cp -p "$L/$f" "$H.tmp/" || ok=0; fi; done
cp -rp "$L/pool" "$H.tmp/pool" || ok=0
cp -p "$L/final_model.pt" "$H.tmp/final_model.pt" || ok=0
cmp -s "$L/final_model.pt" "$H.tmp/final_model.pt" || ok=0
for d in pool pool/frozen; do
  a=$(ls "$L/$d" 2>/dev/null | grep -c '\.pt$'); b=$(ls "$H.tmp/$d" 2>/dev/null | grep -c '\.pt$')
  [ "$a" = "$b" ] || { echo "$d: $a nets here, $b copied"; ok=0; }
done
[ "$ok" = 1 ] || fail "copy into $H.tmp failed (partial copy left in $H.tmp)"
PYTHONPATH=$C/src timeout 600 "$GPY" -c "import torch; ck = torch.load('$H.tmp/final_model.pt', map_location='cpu', weights_only=False); print('handover counters:', ck.get('counters'))" > "$H.tmp/HANDOVER.txt" 2>&1
grep -E " upd [0-9]+ steps" "$L/train.log" | tail -n 1 >> "$H.tmp/HANDOVER.txt"
{ echo "G9q handover from job ${SLURM_JOB_ID:-741849} ($(hostname -s 2>/dev/null)) at $(date -u +%FT%T): $(ls "$H.tmp/pool" | grep -c '\.pt$') pool nets, $(ls "$H.tmp/pool/frozen" 2>/dev/null | grep -c '\.pt$') frozen"
  md5sum "$H.tmp/final_model.pt" | cut -d' ' -f1; } > "$H.tmp/DONE" || fail "cannot write $H.tmp/DONE"
if [ ! -e "$H" ] && mv -T "$H.tmp" "$H"; then
  st "DONE: $H ($(head -n 1 "$H/DONE"); $(tail -n 1 "$H/HANDOVER.txt"))"
else
  rm -f "$H.tmp/DONE"; fail "could not publish $H"
fi
