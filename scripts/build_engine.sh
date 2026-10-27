#!/usr/bin/env bash
# Build poke-engine 0.0.48 (gen4) with patches/poke-engine-0.0.48-gen4-durations.patch into .venv:
# gen4 taunt / encore / confusion end after their random durations instead of lasting forever.
# Needs Rust (cargo). The checkout lives in .poke-engine/ and is reset to the tag on every build.
#   scripts/build_engine.sh
#   STOCK=1 VENV=.venv-stock scripts/build_engine.sh   # unpatched build in its own venv (engine A/B tests)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/.poke-engine
VENV=${VENV:-$ROOT/.venv}
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
[ -d "$SRC/.git" ] || git clone -q https://github.com/pmariglia/poke-engine.git "$SRC"
git -C "$SRC" fetch -q --tags
git -C "$SRC" checkout -q -f v0.0.48
git -C "$SRC" clean -qfd
[ -n "${STOCK:-}" ] || git -C "$SRC" apply "$ROOT/patches/poke-engine-0.0.48-gen4-durations.patch"
"$VENV/bin/python" -m pip install -q --no-cache-dir --force-reinstall --no-deps "$SRC/poke-engine-py" \
  --config-settings="build-args=--features poke-engine/gen4 --no-default-features"
"$VENV/bin/python" -c "import poke_engine; print('poke-engine installed in $VENV')"
