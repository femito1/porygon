#!/bin/bash
# A pre-registered vs-search check at a training step (STATUS.md): at job start, copy the lineage's FIRST pool
# snapshot >= STEP found in the given (mirrored, NFS) pool dirs to DST, then run the unchanged harness
# scripts/netv2_vs_search.sbatch on "DST,temperature=0.5" (from a node-local copy, so a later re-sync of scripts/
# cannot break the running job). If no such snapshot exists yet it waits up to WAIT_MIN minutes (default 90), then
# fails without playing. DST is never overwritten (a rerun plays the same net).
#   sbatch --dependency=afterany:J [--begin=T] --job-name=ws-netv2-vs9-... scripts/netv2_vs_search_at.sbatch \
#     STEP DST "DIR1 DIR2 ..." N [netv2_vs_search.sbatch args, e.g. --gen 9 --seed 738164]
#SBATCH --job-name=ws-netv2-vs-search
#SBATCH --partition=long
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
set -euo pipefail
ROOT=$SLURM_SUBMIT_DIR; cd "$ROOT"
STEP=$1 DST=$2 DIRS=$3 N=$4; shift 4
first_snap() {  # first snapshot (numeric file name = steps) >= STEP in DIRS
  local d f n
  for d in $DIRS; do
    for f in "$d"/*.pt; do
      [ -e "$f" ] || continue
      n=$(basename "$f" .pt); case $n in *[!0-9]*|"") continue ;; esac
      if [ "$((10#$n))" -ge "$STEP" ]; then echo "$((10#$n)) $f"; fi
    done
  done | sort -n | head -n 1 | cut -d' ' -f2
}
if [ ! -f "$DST" ]; then
  s=""
  for ((t = 0; t < ${WAIT_MIN:-90}; t++)); do s=$(first_snap); [ -n "$s" ] && break; sleep 60; done
  [ -n "$s" ] || { echo "$(date -u +%T) no snapshot >= $STEP in: $DIRS (waited ${WAIT_MIN:-90} min): not playing" >&2; exit 1; }
  sleep 30; s=$(first_snap)  # the mirror renames complete files into place; re-pick in case a smaller one just landed
  mkdir -p "$(dirname "$DST")"
  cp -p "$s" "$DST.tmp-$SLURM_JOB_ID" && mv -f "$DST.tmp-$SLURM_JOB_ID" "$DST"
  echo "$s" > "$DST.src"
fi
echo "$(date -u +%T) net for the check at $STEP: $DST (from $(cat "$DST.src" 2>/dev/null || echo '?'))"
H=${TMPDIR:-/tmp}/netv2-vs-search-$SLURM_JOB_ID.sbatch
cp scripts/netv2_vs_search.sbatch "$H"
exec bash "$H" "$DST,temperature=0.5" "$N" "$@"
