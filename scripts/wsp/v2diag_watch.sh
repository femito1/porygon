# Self-resubmitting tracker for the restarted R1 (from 7.5M): each new pool snapshot in runs/r1/pool with step > 7.5M
# gets decay (vs 7.5M) + switching slopes, one row appended to v2diag/r1b/table.md, flags to v2diag/r1b/flags.txt.
# Resubmits itself 20 min later while netv2's r1 job is queued or running (stop: touch v2diag/r1b/STOP).
R=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/r1
D=v2diag/r1b; mkdir -p $D/ck $D/out
REF=v2diag/r1/ck/pool_7500005.pt
[ -e $D/STOP ] && { echo stopped; exit 0; }
# START holds the epoch time of the verified relaunch (lr 2e-4, ent_coef 0, zero_avoid 0); nothing older is counted
START=$(cat $D/START 2>/dev/null || echo 9999999999)
for f in $(ls $R/pool/*.pt 2>/dev/null | sort); do
  [ $(stat -c %Y $f) -le $START ] && continue
  b=$(basename $f .pt); st=$((10#$b))
  [ $st -le 7500005 ] && continue
  [ -e $D/out/$st/done ] && continue
  sleep 30  # let the mirror finish writing
  mkdir -p $D/out/$st; cp $f $D/ck/s_$st.pt
  python scripts/wsp/v2diag_decay.py --ref $REF --pos v2diag/pos --out $D/out/$st --n 8000 $D/ck/s_$st.pt 2>&1 | grep -v Warn | tail -2
  python scripts/wsp/v2diag_switch.py --pos v2diag/pos --out $D/out/$st --n 2000 $D/ck/s_$st.pt 2>&1 | grep -v Warn
  python scripts/wsp/v2diag_track.py --decay $D/out/$st/decay.json --switch $D/out/$st/switch_sensitivity.json \
    --ckpt s_$st.pt --step $st --table $D/table.md --flags $D/flags.txt \
    --trainlog $R/20261108-220401-718701/train.log --since "$(cat $D/SINCE)" && touch $D/out/$st/done
done
if squeue -h -u $USER -o %j | grep -q -E "^ws-netv2-r1"; then
  sbatch --begin=now+20minutes --partition=long scripts/wsp/v2diag_run.sbatch scripts/wsp/v2diag_watch.sh | tail -1
else
  echo "no r1 job queued/running: not resubmitting"
fi
