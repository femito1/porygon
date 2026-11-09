POS='/vol/bitbucket/fr1226/porygon-ws-data/datasets/v2pre/gen9-searchlabels/shard-*.npz'
mkdir -p v2diag/g9/test
python scripts/wsp/v2diag_decay.py --gen 9 --ref v2diag/g9/ck/p1_best.pt --pos "$POS" --out v2diag/g9/test --n 8000 v2diag/g9/ck/p1_last.pt 2>&1 | grep -v Warn
echo "#### gen4 regression (10M vs 7.5M; earlier: H 1.177, vol status 0.262)"
python scripts/wsp/v2diag_decay.py --gen 4 --ref v2diag/r1/ck/pool_7500005.pt --pos v2diag/pos --out v2diag/r1/test --n 12000 v2diag/r1/ck/pool_10000256.pt 2>&1 | grep -v Warn
