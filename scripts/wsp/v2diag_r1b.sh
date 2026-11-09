D=v2diag/r1
python scripts/wsp/v2diag_switch.py --pos v2diag/pos --out $D/out --n 3000 $D/ck/init_model.pt $D/ck/pool_2540836.pt $D/ck/seg706657_final_4835831.pt 2>&1 | grep -v Warn
echo "#### pretrain checkpoint + gen9 data layout"
P=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/g9h-dmg-s0
ls -la $P | head; ls /vol/bitbucket/fr1226/porygon-ws-data/datasets/v2pre/gen9-searchlabels/ 2>/dev/null | head
python - <<'PY'
import torch, glob, numpy as np
P="/vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/g9h-dmg-s0"
ck=torch.load(f"{P}/last.pt", map_location="cpu", weights_only=False)
print("pretrain keys:", list(ck)[:12], {k: ck[k] for k in ck if k in ("gen","step","steps","epoch")})
import porygon.v2.model as M
try:
    n=M.load(f"{P}/last.pt"); print("loads with porygon.v2.model.load OK, layers", n.cfg["n_layers"])
except Exception as e: print("load failed:", repr(e)[:300])
fs=sorted(glob.glob("/vol/bitbucket/fr1226/porygon-ws-data/datasets/v2pre/gen9-searchlabels/shard-*.npz"))
print(len(fs), "gen9 shards")
if fs:
    z=np.load(fs[0]); print({k: z[k].shape for k in list(z.keys())[:40]})
PY
