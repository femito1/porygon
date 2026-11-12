#!/bin/bash
# The benchmark-slot job of the network + search test (NETSEARCH-DESIGN.md §6): one 24-CPU job runs a plan with
# netsearch_slot.sh (fastenv's own duplicate-battle server copy, code snapshot in the run dir). Claim the slot in
# .orchestra/ledger.md first.
#   cd /vol/bitbucket/fr1226/porygon-ws-fastenv && sbatch scripts/fastenv/netsearch_slot.sbatch scripts/fastenv/plans/P.env
# Output: runs/netsearch-<NAME>-<date>-<job>/ (result.txt, manifest.txt, looks.jsonl, run.log, arm-K/block-J/).
#SBATCH --job-name=ws-fastenv-netsearch
#SBATCH --partition=training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
echo "$(date -u +%FT%TZ) node $(hostname) SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?} plan $1"
exec bash scripts/fastenv/netsearch_slot.sh "$1"
