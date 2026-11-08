#!/bin/bash
# queued emb jobs; run detached from the login node:
#   nohup srun --jobid=674320 --overlap --ntasks=1 bash scripts/wsp/emb_run3.sh > out/emb/srun3.log 2>&1 < /dev/null &
# (never edit this file while it runs: bash reads it lazily over NFS)
source /vol/bitbucket/fr1226/porygon-ws-wsp/scripts/wsp/env.sh
export OMP_NUM_THREADS=2
python -u scripts/wsp/emb_probe.py c > out/emb/probe_c.txt 2>&1
for n in s2.5M s21M s136M x30_640; do
  for try in 1 2; do
    timeout 1150 python -u scripts/wsp/emb_func.py $n >> out/emb/func_$n.txt 2>&1
    [ -f out/emb/func_$n.json ] && break
  done
done
python -u scripts/wsp/emb_grad.py 600M 12 > out/emb/grad_600M.txt 2>&1
echo ALLDONE >> out/emb/run3.done
