R=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/r1
D=v2diag/r1
for f in $R/*/checkpoints/ckpt_*.pt; do cp --update=none $f $D/ck/; done
NEW=$(ls $D/ck/ckpt_*.pt | sort)
echo "checkpoints: $NEW"
for f in $(ls $R/*/train.log | sort | tail -1); do echo "#### $f (decoupled PPO from 4.84M)"; python scripts/wsp/v2diag_log.py $f --window 1e6; done
python scripts/wsp/v2diag.py --init $D/ck/init_model.pt --pos v2diag/pos --out $D/out --n 12000 $NEW 2>&1 | grep -v -E "Warn|group rel drift"
python scripts/wsp/v2diag_switch.py --pos v2diag/pos --out $D/out --n 3000 $D/ck/seg706657_final_4835831.pt $NEW 2>&1 | grep -v Warn
