#!/bin/bash
# Vendor what scripts/wsp/spd_h2h.py runs on, pinned to commits, from git to the cluster (other streams' worktrees and
# cluster dirs are never touched): fastenv's JS (simulator wrapper, first-person tracker, v1/v2 encoders, tables, bots)
# plus the vocab its aux targets read -> $WSP/spd/fastenv-<commit>/ (SPD_FASTENV_JS = .../fastenv/js); netv2's src
# (porygon.v2, for v2: nets) -> $WSP/spd/netv2-<commit>/src (spd_h2h.py --v2-src). An existing vendor dir is kept.
#   scripts/wsp/spd_vendor.sh [FASTENV_COMMIT=b9b8a85] [NETV2_COMMIT=ws/netv2]    (laptop, in the ws-wsp worktree)
set -euo pipefail
FE=$(git rev-parse --short "${1:-b9b8a85}"); NV=$(git rev-parse --short "${2:-ws/netv2}")
WSP=/vol/bitbucket/fr1226/porygon-ws-wsp
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git archive --format=tar "$FE" fastenv/js src/porygon/data/vocab.json > "$tmp/fastenv.tar"
git archive --format=tar "$NV" src/porygon > "$tmp/netv2.tar"
for spec in "fastenv-$FE:fastenv.tar:ws/fastenv $FE" "netv2-$NV:netv2.tar:ws/netv2 $NV"; do
  IFS=: read -r dir file version <<< "$spec"
  for try in 1 2 3 4 5 6; do  # the laptop-cluster link drops: retry while ssh itself fails (255)
    rc=0
    ssh -o BatchMode=yes gpucluster2 "set -e; d=$WSP/spd/$dir; if [ -f \$d/VERSION ]; then echo \"\$d exists (\$(cat \$d/VERSION))\"; cat > /dev/null; exit 0; fi
      rm -rf \$d.tmp; mkdir -p \$d.tmp; tar -x -C \$d.tmp; echo '$version' > \$d.tmp/VERSION; mv \$d.tmp \$d; echo \"vendored \$d\"" \
      < "$tmp/$file" 2>&1 | grep -v -E "WARNING|prosecuted|may be monitored|Authorized usage|Use of IT|imperial.ac.uk|DO NOT RUN|^$" \
      || rc=${PIPESTATUS[0]}
    [ "$rc" = 255 ] || break
    sleep 15
  done
  [ "$rc" = 0 ] || { echo "vendoring $dir failed ($rc)" >&2; exit 1; }
done
