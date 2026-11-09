# Self-resubmitting tracker for G9 (gen9 main run from P1-g9h-dmg-s2 + fresh value head; counting from 02:13:02). Since
# the GPU swap at ~03:11-03:15 G9 trains in job 718701 (logs in runs/r1/...718701/) and R1b in 741849 (logs in
# runs/g9/...741849/): both pool dirs are scanned, and a snapshot is analysed only if its parameter count is gen9's
# (5,809,977). One row per snapshot in v2diag/g9b/table.md, flags in flags.txt. Stop: touch v2diag/g9b/STOP.
NV2=/vol/bitbucket/fr1226/porygon-ws-netv2/runs
R=$NV2/g9
D=v2diag/g9b; mkdir -p $D/ck $D/out
POS='/vol/bitbucket/fr1226/porygon-ws-data/datasets/v2pre/gen9-searchlabels/shard-*.npz'
WANT=5809977
[ -e $D/STOP ] && { echo stopped; exit 0; }
if [ ! -s $D/REF ] && [ -s $R/init.txt ]; then
  head -n 1 $R/init.txt > $D/REF; stat -c %Y $R/init.txt > $D/START; date -d @$(cat $D/START) "+%Y-%m-%d %H:%M:%S" > $D/SINCE
fi
REF=$(cat $D/REF 2>/dev/null)
START=$(cat $D/START 2>/dev/null || echo 9999999999)
SEGS=(--trainlog "$NV2/g9/20261109-021316-741849/train.log|2026-11-09 02:13:02|2026-11-09 03:11:58"
      --trainlog "$NV2/r1/20261108-220401-718701/train.log|2026-11-09 03:14:57|9999")
if [ -n "$REF" ]; then
  cp --update=none $REF $D/ck/ref.pt
  for f in $(ls $NV2/g9/pool/*.pt $NV2/r1/pool/*.pt 2>/dev/null); do
    [ $(stat -c %Y $f) -le $START ] && continue
    b=$(basename $f .pt); st=$((10#$b)); key=$st; [[ $f == */r1/pool/* ]] && key=r1p_$st
    [ -e $D/out/$key/done ] || [ -e $D/out/$key/skip ] && continue
    sleep 30
    mkdir -p $D/out/$key
    n=$(python scripts/wsp/v2diag_nparams.py $f 2>/dev/null | tail -1)
    if [ "$n" != "$WANT" ]; then echo "skip $f: $n params (want $WANT)"; echo "$n" > $D/out/$key/skip; continue; fi
    cp $f $D/ck/s_$key.pt
    python scripts/wsp/v2diag_decay.py --gen 9 --ref $D/ck/ref.pt --pos "$POS" --out $D/out/$key --n 8000 $D/ck/s_$key.pt 2>&1 | grep -v Warn | tail -2
    python scripts/wsp/v2diag_switch.py --gen 9 --pos "$POS" --out $D/out/$key --n 2000 $D/ck/s_$key.pt 2>&1 | grep -v Warn
    python scripts/wsp/v2diag_track.py --decay $D/out/$key/decay.json --switch $D/out/$key/switch_sensitivity.json \
      --ckpt s_$key.pt --step $st --table $D/table.md --flags $D/flags.txt "${SEGS[@]}" \
      --ref ref.pt --relative-top09 --collapse --auc-min-step 5e6 && touch $D/out/$key/done
  done
fi
if squeue -h -u $USER -o %j | grep -q -E "^ws-netv2-(g9|r1)" || [ -e $D/WAIT ]; then
  sbatch --begin=now+20minutes --partition=long scripts/wsp/v2diag_run.sbatch scripts/wsp/v2diag_watch_g9.sh | tail -1
else
  echo "no netv2 g9/r1 job queued/running and no WAIT: not resubmitting"
fi
