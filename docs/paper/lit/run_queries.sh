#!/bin/bash
# Runs every query block of protocol.md through litsearch.py (cached: reruns are free). Output: lit/runs/queries.log
cd "$(dirname "$0")/../../.."
mkdir -p docs/paper/lit/runs
while IFS='|' read -r block q; do
  [ -z "$q" ] && continue
  echo "=== [$block] $q"
  .venv/bin/python docs/paper/tools/litsearch.py search "$q" --limit 40 2>&1
done < docs/paper/lit/queries.txt
