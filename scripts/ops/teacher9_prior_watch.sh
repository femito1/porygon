#!/bin/bash
# Keep logs/teacher9-prior.txt pointing at G9p's newest pool snapshot (copied, so pruning cannot remove it) until
# the teacher test 770666 starts (it reads the pointer at start). Runs as a light step inside 677193.
set -u
D=/vol/bitbucket/fr1226/porygon-ws-data; P=$D/logs/teacher9-prior.txt
POOL=/vol/bitbucket/fr1226/porygon-ws-netv2/runs/g9/20261109-021316-741849/G9/pool
mkdir -p $D/nets
while [ "$(squeue -j 770666 -h -o %T 2>/dev/null)" = PENDING ]; do
  f=$(ls "$POOL" 2>/dev/null | grep -E '^[0-9]+\.pt$' | sort -n | tail -1)
  if [ -n "$f" ] && [ ! -f "$D/nets/g9p-$f" ]; then
    cp "$POOL/$f" "$D/nets/.g9p-$f.tmp" && mv "$D/nets/.g9p-$f.tmp" "$D/nets/g9p-$f" &&
      printf '%s\n# G9p newest pool snapshot %s (auto, %s); source %s\n' "$D/nets/g9p-$f" "$f" "$(date -Is)" "$POOL" > "$P.tmp" &&
      mv "$P.tmp" "$P" && echo "$(date -Is) pointer -> g9p-$f"
  fi
  sleep 120
done
echo "$(date -Is) 770666 is $(squeue -j 770666 -h -o %T); final pointer: $(head -1 $P)"
