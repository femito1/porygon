R=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/r1
D=v2diag/r1; mkdir -p $D/ck $D/out
cp --update=none $R/20261108-204642-710000/init_model.pt $D/ck/init_model.pt
cp --update=none $R/pool/00002540836.pt $D/ck/pool_2540836.pt
cp --update=none $R/20261108-211009-706657/final_model.pt $D/ck/seg706657_final_4835831.pt
for f in $R/20261108-220401-718701/checkpoints/ckpt_*.pt; do [ -e "$f" ] && cp --update=none $f $D/ck/; done
ls $D/ck
for f in $R/*/train.log; do echo "#### $f"; python scripts/wsp/v2diag_log.py $f --window 1e6; done
python scripts/wsp/v2diag.py --init $D/ck/init_model.pt --pos v2diag/pos --out $D/out --n 12000 $(ls $D/ck/*.pt) 2>&1 | grep -v Warn
