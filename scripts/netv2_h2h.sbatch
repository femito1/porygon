#!/bin/bash
# Raw network vs raw network (v1 .zip / pool .pt, v2 .pt, or any porygon.v2.evaluate spec), duplicate pairs:
# SHARDS duplicate-battle servers in parallel, each with its own seeds and one porygon.v2.evaluate process.
#   sbatch scripts/netv2_h2h.sbatch A B N [--gen 4|9] [--shards S] [--concurrency C] [--seed TAG] [evaluate args]
#   e.g. sbatch scripts/netv2_h2h.sbatch runs/v2/net_100M.pt /vol/bitbucket/fr1226/gloria/runs/resume/pool/00100000000.pt 1000
# A's win rate (pair-mean 95% interval) in runs/h2h-<date>-<job>/result.txt; per-shard results.jsonl, battles.jsonl
# (both true teams), summary.json, evaluate.log. Nets sample their policy (deterministic=1 in a spec: argmax).
# Defaults: --gen 4, 8 shards x 12 concurrent battles, --seed = job id (the same TAG replays the same team pairs:
# common random numbers when two arms face the same opponent).
#SBATCH --job-name=ws-netv2-h2h
#SBATCH --partition=long
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
source scripts/netv2_common.sh
ALL="$*"
A=$1; B=$2; N=$3; shift 3
GEN=4; SHARDS=8; CONC=12; SEED=$SLURM_JOB_ID
while [ $# -gt 0 ]; do
  case "$1" in
    --gen) GEN=$2; shift 2 ;; --shards) SHARDS=$2; shift 2 ;; --concurrency) CONC=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;; *) break ;;
  esac
done
netv2_gen "$GEN"
netv2_out h2h
netv2_manifest h2h "$ALL" "$A" "$B"
trap netv2_stop_servers EXIT
netv2_preread "$GLORIA"/.venv/lib/python3.12/site-packages/torch
netv2_shards "$A" "$B" "$N" "$SHARDS" "$CONC" "$@"
