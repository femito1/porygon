#!/usr/bin/env bash
# Build poke-engine 0.0.48 (gen4) with patches/poke-engine-0.0.48-*.patch into .venv:
#   gen4-durations: gen4 taunt / encore / confusion end after their random durations instead of lasting forever;
#   search-options: monte_carlo_tree_search_priors (side one's root options chosen by PUCT on given priors) and
#   monte_carlo_tree_search_joint (also the root's joint statistics; rm_gamma: regret matching at every node).
# Needs Rust (cargo). The checkout lives in .poke-engine/ and is reset to the tag on every build.
#   scripts/build_engine.sh
#   STOCK=1 VENV=.venv-stock scripts/build_engine.sh   # unpatched build in its own venv (engine A/B tests)
#   GEN=9 VENV=.venv-gen9 scripts/build_engine.sh     # gen9 (terastallization) build; the gen4-durations patch is
#                                                      # gated to gen4 and compiles out. Run with PORYGON_GEN=9.
#   WHEEL_DIR=dir scripts/build_engine.sh               # build a wheel into dir instead of installing
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/.poke-engine
VENV=${VENV:-$ROOT/.venv}
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
[ -d "$SRC/.git" ] || git clone -q https://github.com/pmariglia/poke-engine.git "$SRC"
git -C "$SRC" fetch -q --tags
git -C "$SRC" checkout -q -f v0.0.48
git -C "$SRC" clean -qfd
if [ -z "${STOCK:-}" ]; then
  for p in "$ROOT"/patches/poke-engine-0.0.48-*.patch; do git -C "$SRC" apply "$p"; done  # absolute paths
fi
case "${GEN:-4}" in
  4) FEATURES=poke-engine/gen4 ;;
  9) FEATURES=poke-engine/terastallization ;;
  *) echo "GEN must be 4 or 9" >&2; exit 1 ;;
esac
if [ -n "${WHEEL_DIR:-}" ]; then
  "$VENV/bin/python" -m pip wheel -q --no-deps "$SRC/poke-engine-py" -w "$WHEEL_DIR" \
    --config-settings="build-args=--features $FEATURES --no-default-features"
  exit
fi
"$VENV/bin/python" -m pip install -q --no-cache-dir --force-reinstall --no-deps "$SRC/poke-engine-py" \
  --config-settings="build-args=--features $FEATURES --no-default-features"
"$VENV/bin/python" -c "import poke_engine; print('poke-engine (gen${GEN:-4}) installed in $VENV')"
