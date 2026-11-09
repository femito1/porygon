# Generic self-resubmitting tracker: bash v2diag_watch_any.sh CONF (CONF sets D GEN POS REF REFNAME WANT START_FROM
# POOLS[] SEGS[] TRACK_ARGS[] JOBRE; see v2diag/conf/*.conf). Each new pool snapshot (mtime after START_FROM, parameter
# count == WANT, one row per step) gets decay vs REF + switching slopes on POS -> one row in $D/table.md, flags in
# $D/flags.txt. Resubmits itself 20 min later while a job matching JOBRE is queued/running. Stop: touch $D/STOP.
CONF=$1; source $CONF
mkdir -p $D/ck $D/out
[ -e $D/STOP ] && { echo "$CONF stopped"; exit 0; }
START=$(date -d "$START_FROM" +%s)
# queue the next pass first (after this one ends, at least 20 min from now), so a pass that hits the time limit
# cannot break the chain
if squeue -h -u $USER -o %j | grep -q -E "$JOBRE"; then
  sbatch --begin=now+20minutes --dependency=afterany:$SLURM_JOB_ID --partition=long scripts/wsp/v2diag_run.sbatch \
    scripts/wsp/v2diag_watch_any.sh $CONF | tail -1
else
  echo "no job matching $JOBRE: not resubmitting $CONF"
fi
cp --update=none $REF $D/ck/$REFNAME
for f in $(ls ${POOLS[@]} 2>/dev/null); do
  [ $(stat -c %Y $f) -le $START ] && continue
  b=$(basename $f .pt); [[ $b =~ ^[0-9]+$ ]] || continue; st=$((10#$b)); key=$st
  [ -e $D/out/$key/done ] || [ -e $D/out/$key/skip ] && continue
  sleep 30
  mkdir -p $D/out/$key
  n=$(python scripts/wsp/v2diag_nparams.py $f 2>/dev/null | tail -1)
  if [ "$n" != "$WANT" ]; then echo "skip $f: $n params (want $WANT)"; echo "$n" > $D/out/$key/skip; continue; fi
  cp $f $D/ck/s_$key.pt
  python scripts/wsp/v2diag_decay.py --gen $GEN --ref $D/ck/$REFNAME --pos "$POS" --out $D/out/$key --n 8000 $D/ck/s_$key.pt 2>&1 | grep -v Warn | tail -2
  python scripts/wsp/v2diag_switch.py --gen $GEN --pos "$POS" --out $D/out/$key --n 2000 $D/ck/s_$key.pt 2>&1 | grep -v Warn
  python scripts/wsp/v2diag_track.py --decay $D/out/$key/decay.json --switch $D/out/$key/switch_sensitivity.json \
    --ckpt s_$key.pt --step $st --table $D/table.md --flags $D/flags.txt "${SEGS[@]}" --ref $REFNAME "${TRACK_ARGS[@]}" \
    && touch $D/out/$key/done && rm -f $D/ck/s_$key.pt
done
