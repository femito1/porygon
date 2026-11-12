D=v2diag/g9; mkdir -p $D/ck $D/out
P=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/g9h-dmg-s0
cp --update=none $P/best.pt $D/ck/p1_best.pt; cp --update=none $P/last.pt $D/ck/p1_last.pt
python - <<'PY'
import torch, porygon.v2.model as M
ck = torch.load("v2diag/g9/ck/p1_last.pt", map_location="cpu", weights_only=False)
torch.manual_seed(0)
M.save(M.Net(9, **ck["cfg"]), "v2diag/g9/ck/randinit_model.pt")
print("random init saved (true P1 init not saved: drift is vs a fresh random init)")
PY
POS='/vol/bitbucket/fr1226/porygon-ws-data/datasets/v2pre/gen9-searchlabels/shard-*.npz'
python scripts/wsp/v2diag.py --init $D/ck/randinit_model.pt --pos "$POS" --out $D/out --n 12000 $D/ck/randinit_model.pt $D/ck/p1_best.pt 2>&1 | grep -v Warn
python scripts/wsp/v2diag_switch.py --gen 9 --pos "$POS" --out $D/out --n 3000 $D/ck/randinit_model.pt $D/ck/p1_best.pt 2>&1 | grep -v Warn
