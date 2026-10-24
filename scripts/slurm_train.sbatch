#!/bin/bash
# Self-play + opponent-pool PPO on a Slurm GPU node (Imperial DoC cluster: 32 cores/user, 72 h).
# Setup (Node, Showdown, venv) happens inside the job and is reused by later jobs; results are
# pushed to SYNC_REMOTE every 20 min and when training ends (scripts/sync_results.sh).
#
# Settings go in slurm.env next to the repo (sbatch --export=... gets jobs held with
# user_env_retrieval_failed on this cluster):
#   SHA=<showdown commit> RESUME=<zip> POOL=<pool dir> TOTAL=<total steps> CHAIN=<follow-up jobs>
#   ANCHOR=<zip or .pt evaluated against every eval; default RESUME>
#   sbatch scripts/slurm_train.sbatch [extra porygon.train args]
# Each job resumes from the newest checkpoint under runs/ (so a requeue after a node failure, or
# the next job in a chain, continues where the last one stopped) and, while CHAIN > 0, submits
# its own continuation when it finishes.
#SBATCH --job-name=porygon
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --requeue
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
[ -f slurm.env ] && source slurm.env
NODE=v22.22.0
export PATH=$ROOT/.node/bin:$PATH

# --- one-time setup (idempotent) ---
if ! [ -x .node/bin/node ]; then
  mkdir -p .node && curl -fsSL "https://nodejs.org/dist/$NODE/node-$NODE-linux-x64.tar.xz" | tar -xJ --strip-components=1 -C .node
fi
if ! [ -f pokemon-showdown/package.json ]; then
  mkdir -p pokemon-showdown && (cd pokemon-showdown && git init -q && \
    git fetch -q --depth 1 https://github.com/smogon/pokemon-showdown "${SHA:?Showdown commit}" && git checkout -q FETCH_HEAD)
fi
[ -d pokemon-showdown/node_modules ] || (cd pokemon-showdown && npm install --no-audit --no-fund --silent)
if ! [ -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu126
  .venv/bin/pip install -q -e .
fi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name())"

# --- servers: one per ~3 cores, ports derived from the job id (nodes are shared) ---
CORES=$(nproc)  # the cluster's job-submit plugin reshapes -c 32 (SLURM_CPUS_PER_TASK=6, 30 usable)
N_SERVERS=$(( CORES / 3 )); SELFPLAY=$(( CORES * 2 )); POOL_BATTLES=$CORES
BASE=$(( 20000 + (SLURM_JOB_ID % 2000) * 16 ))
rm -f pokemon-showdown/config/config.js  # start_servers.sh writes the no-disk-writes config
scripts/start_servers.sh "$N_SERVERS" "$BASE"
trap 'pkill -u "$USER" -f "pokemon-showdown start --skip-build --no-security" || true' EXIT

LATEST=$(ls -t runs/2*/checkpoints/*.zip runs/2*/final_model.zip "${RESUME:-/nonexistent}" 2>/dev/null | head -1 || true)
RUN=runs/$(date +%Y%m%d-%H%M%S)-$SLURM_JOB_ID; mkdir -p "$RUN"
echo "resuming from ${LATEST:-scratch}"
SYNC_REMOTE=${SYNC_REMOTE:-git@github.com:femito1/porygon-runs.git}
export SYNC_KEY=${SYNC_KEY:-$HOME/.ssh/porygon_runs_deploy}
POOL_NFS=${POOL:-$RUN/pool}; mkdir -p "$POOL_NFS"
ANCHOR=${ANCHOR:-${RESUME:-}}

# The shared disk can stall for minutes, so the run and pool live on the node (tmpfs /tmp, counts
# toward --mem) and are mirrored to $RUN / $POOL_NFS every 10 min and at the end; resuming reads
# the mirror. Locally only the two newest checkpoints are kept once mirrored.
LOCAL=${TMPDIR:-/tmp}/porygon-$SLURM_JOB_ID; mkdir -p "$LOCAL/run" "$LOCAL/pool"
# Copy the pool in the background, newest first (cp -p keeps mtimes, which PoolOpponent's "recent"
# picks use): copying ~1.3 GB took over an hour on a slow day. Until snapshots arrive, pool
# battles fall back to the baseline bots; PoolOpponent re-lists the directory every battle.
( ls -t "$POOL_NFS"/*.pt 2>/dev/null | while read -r f; do cp -p "$f" "$LOCAL/pool/.tmp" && mv "$LOCAL/pool/.tmp" "$LOCAL/pool/${f##*/}"; done ) &
export POOL_DIR=$LOCAL/pool
mirror() {  # status = the rsyncs'; pruning (no checkpoints yet is fine) must not fail it
  rsync -a "$LOCAL/run/" "$RUN/" && rsync -a --exclude .tmp "$LOCAL/pool/" "$POOL_NFS/" || return 1
  { ls -t "$LOCAL"/run/checkpoints/*.zip 2>/dev/null || true; } | tail -n +3 | xargs -r rm -f
}
( while sleep 600; do mirror || echo "$(date -u +%T) mirror FAILED" >&2; done ) &
MIRROR=$!
cleanup() {  # also on scancel (SIGTERM): Slurm allows ~30 s before SIGKILL, enough for an incremental rsync
  kill $MIRROR 2>/dev/null || true
  mirror && rm -rf "$LOCAL"  # keep the local copy if the mirror failed
  pkill -u "$USER" -f "pokemon-showdown start --skip-build --no-security" || true
}
trap cleanup EXIT
trap 'exit 143' TERM
scripts/sync_results.sh "$LOCAL/run" "$SYNC_REMOTE" >> "$RUN/sync.log" 2>&1 &

.venv/bin/python -m porygon.train --ports "$BASE-$((BASE + N_SERVERS - 1))" --n-battles "$SELFPLAY" \
  --pool-battles "$POOL_BATTLES" --pool-dir "$LOCAL/pool" --run-dir "$LOCAL/run" \
  ${LATEST:+--resume "$LATEST"} ${ANCHOR:+--anchor "$ANCHOR"} --total-steps "${TOTAL:-400000000}" --max-hours 70.5 --checkpoint-every 5000000 \
  "$@" 2>&1 | tee "$LOCAL/run/train.log" || true
scripts/sync_results.sh "$LOCAL/run" "$SYNC_REMOTE" once >> "$RUN/sync.log" 2>&1
kill $MIRROR 2>/dev/null || true; mirror || true

# --- continue in a new job (walltime is 72 h) unless the chain is used up or training is done ---
STEPS=$(grep -oE "total_timesteps +\| +[0-9]+" "$RUN/train.log" | tail -1 | grep -oE "[0-9]+$" || echo 0)
if [ "${CHAIN:-0}" -gt 0 ] && [ -f "$RUN/final_model.zip" ] && [ "$STEPS" -lt "${TOTAL:-400000000}" ]; then
  sed -i "s/^CHAIN=.*/CHAIN=$(( CHAIN - 1 ))/" slurm.env
  sbatch scripts/slurm_train.sbatch "$@"
fi
