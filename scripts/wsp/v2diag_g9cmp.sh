NV2=/vol/bitbucket/fr1226/porygon-ws-netv2/runs
D=v2diag/g9cmp; mkdir -p $D/ck $D/out
POS='/vol/bitbucket/fr1226/porygon-ws-data/datasets/v2pre/gen9-searchlabels/shard-*.npz'
for t in $(seq 1 40); do f=$(ls $NV2/r1/20261108-220401-718701/G9s/pool/0000005*.pt $NV2/r1/20261108-220401-718701/G9s/pool/0000004*.pt 2>/dev/null | head -1); [ -n "$f" ] && break; sleep 30; done
echo "G9s 5M: $f"; sleep 30
cp --update=none $NV2/pretrain/P1-g9h-dmg-s2.pt $D/ck/P1.pt
cp --update=none $NV2/g9s/snap-00002507561.pt $D/ck/G9s_2.5M.pt
[ -n "$f" ] && cp --update=none $f $D/ck/G9s_5M.pt
cp --update=none $NV2/r1/handover-g9p/final_model.pt $D/ck/G9p_13M.pt
L=$(ls $D/ck/G9s_2.5M.pt $D/ck/G9s_5M.pt $D/ck/G9p_13M.pt 2>/dev/null)
echo "#### vs P1"; python scripts/wsp/v2diag_decay.py --gen 9 --ref $D/ck/P1.pt --pos "$POS" --out $D/out/vsP1 --n 8000 $L 2>&1 | grep -v Warn
echo "#### vs G9p@13M"; python scripts/wsp/v2diag_decay.py --gen 9 --ref $D/ck/G9p_13M.pt --pos "$POS" --out $D/out/vsG9p --n 8000 $(ls $D/ck/G9s_*.pt) 2>&1 | grep -v Warn
echo "#### switching"; python scripts/wsp/v2diag_switch.py --gen 9 --pos "$POS" --out $D/out --n 3000 $D/ck/P1.pt $L 2>&1 | grep -v Warn
echo "#### search agreement / attention"; python scripts/wsp/v2diag.py --init $NV2/g9s/init-g9s.pt --pos "$POS" --out $D/out/diag --n 8000 $D/ck/P1.pt $L 2>&1 | grep -E "^==|policy:|att max|FLAGS"
