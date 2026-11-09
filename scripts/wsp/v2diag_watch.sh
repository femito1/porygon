# Self-resubmitting tracker for R1b (gen4 lr test, relaunched 01:12:12 from restart-best-7.5M.pt). Since the GPU swap
# at ~03:11-03:15 R1b trains in job 741849 (logs in runs/g9/20261109-021316-741849/) and G9 in 718701 (logs in
# runs/r1/...718701/): both pool dirs are scanned, and a snapshot is analysed only if its parameter count is gen4's
# (5,702,969), so no gen9 net is ever analysed on gen4 positions. One row per snapshot in v2diag/r1b/table.md.
# Stop: touch v2diag/r1b/STOP. Resubmits itself every 20 min while any ws-netv2-r1*/g9* job is queued or running.
NV2=/vol/bitbucket/fr1226/porygon-ws-netv2/runs
D=v2diag/r1b; mkdir -p $D/ck $D/out
REF=v2diag/r1/ck/pool_7500005.pt
WANT=5702969
[ -e $D/STOP ] && { echo stopped; exit 0; }
START=$(cat $D/START 2>/dev/null || echo 9999999999)
SEGS=(--trainlog "$NV2/r1/20261108-220401-718701/train.log|2026-11-09 01:12:12|2026-11-09 03:14:57"
      --trainlog "$NV2/g9/20261109-021316-741849/train.log|2026-11-09 03:11:58|9999")
for f in $(ls $NV2/r1/pool/*.pt $NV2/g9/pool/*.pt 2>/dev/null); do
  [ $(stat -c %Y $f) -le $START ] && continue
  b=$(basename $f .pt); st=$((10#$b)); key=$st; [[ $f == */g9/pool/* ]] && key=g9p_$st
  [ -e $D/out/$key/done ] || [ -e $D/out/$key/skip ] && continue
  sleep 30
  mkdir -p $D/out/$key
  n=$(python scripts/wsp/v2diag_nparams.py $f 2>/dev/null | tail -1)
  if [ "$n" != "$WANT" ]; then echo "skip $f: $n params (want $WANT)"; echo "$n" > $D/out/$key/skip; continue; fi
  cp $f $D/ck/s_$key.pt
  python scripts/wsp/v2diag_decay.py --gen 4 --ref $REF --pos v2diag/pos --out $D/out/$key --n 8000 $D/ck/s_$key.pt 2>&1 | grep -v Warn | tail -2
  python scripts/wsp/v2diag_switch.py --pos v2diag/pos --out $D/out/$key --n 2000 $D/ck/s_$key.pt 2>&1 | grep -v Warn
  python scripts/wsp/v2diag_track.py --decay $D/out/$key/decay.json --switch $D/out/$key/switch_sensitivity.json \
    --ckpt s_$key.pt --step $st --table $D/table.md --flags $D/flags.txt "${SEGS[@]}" --ref pool_7500005.pt \
    --boundary "17000000:R1b-F — from 17.0M the pool mix is active (fastenv collector, job 741849)" && touch $D/out/$key/done
done
if squeue -h -u $USER -o %j | grep -q -E "^ws-netv2-(r1|g9)"; then
  sbatch --begin=now+20minutes --partition=long scripts/wsp/v2diag_run.sbatch scripts/wsp/v2diag_watch.sh | tail -1
else
  echo "no netv2 r1/g9 job queued/running: not resubmitting"
fi
