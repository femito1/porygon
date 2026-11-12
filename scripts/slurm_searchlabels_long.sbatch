#!/bin/bash
# Search labels on the shared long benchmark slot (claim it in .orchestra/ledger.md first). The long partition's
# submit plugin takes the CPU count from these #SBATCH lines, not from `sbatch -c`, so this wrapper exists.
#   sbatch [--dependency=afterany:J] [--time=H:MM:SS] scripts/slurm_searchlabels_long.sbatch GEN CHUNKS N [--model CKPT] [args]
#SBATCH --job-name=ws-data-searchlabels4
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/vol/bitbucket/fr1226/porygon-ws-data/logs/slurm-%j.out
exec bash "$SLURM_SUBMIT_DIR/scripts/slurm_searchlabels.sbatch" "$@"
