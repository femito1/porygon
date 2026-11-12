#!/bin/bash
# eds: comparison variant B, final 500M edit (saved + verified), then the 600M edit. Run inside the box job.
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=2
set -x
O=out/eds
CK=/vol/bitbucket/fr1226/porygon-ws-wsp/ckpt/edits
mkdir -p $CK
python scripts/wsp/eds_gate2.py 500M 2 "4,30,0.5" --final 4,30,0.5 --recenter > $O/gate2_c2_M30.log 2>&1
python scripts/wsp/eds_edit.py 500M 2 393 -4 B1 --why "briefed variant: same unit/gate design with threshold counter >= 2" > $O/editB1.log 2>&1
python scripts/wsp/eds_edit.py 500M 4 393 -4 A4 --save $CK/500M_sleeptalk.pt --why "final 500M edit = A3 config; saved" > $O/editA4.log 2>&1
python scripts/wsp/eds_verify.py $CK/500M_sleeptalk.pt 500M A4 > $O/verifyA4.log 2>&1
# 600M: cache (done before), unit choice, gate fit, edit, verify
until [ -f /tmp/wsp-674320/eds/600M_value.npy ]; do sleep 10; done
python scripts/wsp/eds_units.py 600M > $O/units600.log 2>&1
U6=$(python -c "import numpy as np; print(int(np.load('$O/best_unit_600M.npy')))")
python scripts/wsp/eds_gate2.py 600M 4 "4,30,0.5" --final 4,30,0.5 --recenter > $O/gate2_600M_c4.log 2>&1
python scripts/wsp/eds_edit.py 600M 4 $U6 -4 A600 --save $CK/600M_sleeptalk.pt --why "600M: same program, directions/unit re-read on 600M" > $O/editA600.log 2>&1
python scripts/wsp/eds_verify.py $CK/600M_sleeptalk.pt 600M A600 > $O/verifyA600.log 2>&1
echo ALLDONE
