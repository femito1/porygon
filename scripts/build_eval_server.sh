#!/usr/bin/env bash
# Showdown checkout for benchmarks in .showdown-eval/: the training server's commit plus
# patches/showdown-duplicate-battles.patch (seeded duplicate battles, true teams logged; off unless
# PORYGON_DUPLICATE is set). Kept apart so rebuilding it never touches a running training server.
#   scripts/build_eval_server.sh && PS_DIR=$PWD/.showdown-eval PORYGON_DUPLICATE=seed scripts/start_servers.sh 1 8001
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EVAL=$ROOT/.showdown-eval
SHA=$(git -C "$ROOT/pokemon-showdown" rev-parse HEAD)
[ -d "$EVAL/.git" ] || git clone -q "$ROOT/pokemon-showdown" "$EVAL"
git -C "$EVAL" fetch -q "$ROOT/pokemon-showdown" "$SHA"
git -C "$EVAL" checkout -q -f "$SHA"
git -C "$EVAL" apply "$ROOT/patches/showdown-duplicate-battles.patch"
[ -e "$EVAL/node_modules" ] || ln -s ../pokemon-showdown/node_modules "$EVAL/node_modules"
(cd "$EVAL" && node build)
echo "eval Showdown built at $EVAL"
