mkdir -p v2diag/test
python - <<'PY'
import torch
import porygon.v2.model as M
for s in (0, 1):
    torch.manual_seed(s)
    M.save(M.Net(4), f"v2diag/test/rand{s}_model.pt")
print("saved")
PY
python scripts/wsp/v2diag.py --init v2diag/test/rand0_model.pt --pos v2diag/pos --out v2diag/test --n 4000 v2diag/test/rand0_model.pt v2diag/test/rand1_model.pt 2>&1 | grep -v Warn
