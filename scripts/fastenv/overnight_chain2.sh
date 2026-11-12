#!/bin/bash
# After overnight_chain.sh: the clean per-sample reference S2 = Showdown collector WITH working pool workers
# (combo3 = netv2 7d775ed + patches/netv2-pool-worker-fix.patch), same recipe and seed as F / S, to 2.1M; then
# F vs S2 at 1M / 2M (1000 games, T 0.5), S2 vs bots, and the collector-level comparison.
#   in_job.sh JOB 'scripts/fastenv/overnight_chain2.sh'
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
G=/vol/bitbucket/fr1226/gloria
until grep -q "chain done" logs/overnight-chain.out 2>/dev/null; do sleep 60; done
echo "$(date -u +%T) chain 1 done; starting S2"
COMBO=$ROOT/combo3 scripts/fastenv/run_v2_showdown.sh 16 5 runs/g9smoke-sd2-onpol --format gen9randombattle \
  --n-battles 22 --pool-battles 11 --no-overlap --no-bf16 --n-epochs 3 --batch-size 4096 --n-layers 4 --ffn-mult 1 \
  --lr 7e-5 --eval-battles 0 --total-steps 2100000 --snapshot-every 1000000 --checkpoint-every 1000000 --seed 11
echo "$(date -u +%T) S2 exit $?"; tail -2 runs/g9smoke-sd2-onpol/train.log | cut -c1-160
scripts/fastenv/eval_g9_persample.sh runs/g9smoke-fe-onpol runs/g9smoke-sd2-onpol 16
PYTHONPATH=$ROOT/combo/src $G/.venv/bin/python scripts/fastenv/compare_runs.py runs/g9smoke-fe-onpol runs/g9smoke-sd2-onpol --at 0.5e6 1e6 2e6
echo "$(date -u +%T) chain 2 done"
