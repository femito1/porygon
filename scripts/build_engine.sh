#!/usr/bin/env bash
# Build poke-engine 0.0.48 (gen4) with patches/poke-engine-0.0.48-gen4-durations.patch into .venv:
# gen4 taunt / encore / confusion end after their random durations instead of lasting forever.
# Needs Rust (cargo). The checkout lives in .poke-engine/ and is reset to the tag on every build.
#   scripts/build_engine.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/.poke-engine
[ -d "$SRC/.git" ] || git clone -q https://github.com/pmariglia/poke-engine.git "$SRC"
git -C "$SRC" fetch -q --tags
git -C "$SRC" checkout -q -f v0.0.48
git -C "$SRC" clean -qfd
git -C "$SRC" apply "$ROOT/patches/poke-engine-0.0.48-gen4-durations.patch"
"$ROOT/.venv/bin/python" -m pip install -q --no-cache-dir --force-reinstall --no-deps "$SRC/poke-engine-py" \
  --config-settings="build-args=--features poke-engine/gen4 --no-default-features"
"$ROOT/.venv/bin/python" -c "import poke_engine; print('poke-engine (patched) installed')"
