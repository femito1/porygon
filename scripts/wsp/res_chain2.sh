#!/bin/bash
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=2
python scripts/wsp/res_mlp.py /vol/bitbucket/fr1226/porygon-ws-wsp/ckpt/ppo_2499840_steps.zip > out/res/mlp_2.5M.log 2>&1
python scripts/wsp/res_lens.py /vol/bitbucket/fr1226/porygon-ws-wsp/ckpt/ppo_2499840_steps.zip > out/res/lens_2.5M.log 2>&1
