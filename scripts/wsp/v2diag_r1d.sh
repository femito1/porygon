R=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/r1
D=v2diag/r1
for f in $R/pool/*.pt; do b=$(basename $f .pt); cp --update=none $f $D/ck/pool_$((10#$b)).pt; done
CK="$D/ck/pool_5040725.pt $D/ck/pool_10000256.pt $D/ck/pool_12500326.pt $D/ck/pool_15000399.pt $D/ck/pool_17500521.pt"
ls $D/ck
echo "#### decay vs 7.5M"
python scripts/wsp/v2diag_decay.py --ref $D/ck/pool_7500005.pt --pos v2diag/pos --out $D/out $D/ck/seg706657_final_4835831.pt $CK 2>&1 | grep -v Warn
echo "#### train.log 718701"
python scripts/wsp/v2diag_log.py $R/20261108-220401-718701/train.log --window 1e6
echo "#### switching"
python scripts/wsp/v2diag_switch.py --pos v2diag/pos --out $D/out --n 3000 $D/ck/pool_7500005.pt $CK 2>&1 | grep -v Warn
echo "#### attention / embeddings / value"
python scripts/wsp/v2diag.py --init $D/ck/init_model.pt --pos v2diag/pos --out $D/out --n 8000 $D/ck/pool_7500005.pt $D/ck/pool_10000256.pt $D/ck/pool_17500521.pt 2>&1 | grep -v -E "Warn|group rel drift"
