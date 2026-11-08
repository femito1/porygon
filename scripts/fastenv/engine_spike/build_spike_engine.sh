#!/usr/bin/env bash
# Build the spike's poke-engine (Porygon's patched 0.0.48 + spike_bindings.rs: spike_options, spike_random_games) for
# gen4 and gen9 into $W/eng4 and $W/eng9 (pip --target dirs; put one ahead on PYTHONPATH). Runs inside a cluster
# job: the toolchain is read from gloria's .rustup, cargo's registry is copied into $W (gloria is never written).
# $W/pe-src: a copy of the main checkout's patched .poke-engine/ (rsync it there first, without target/ and .git/).
set -euo pipefail
W=${W:-/vol/bitbucket/fr1226/porygon-ws-fastenv/engine_spike}
G=/vol/bitbucket/fr1226/gloria
HERE=$(cd "$(dirname "$0")" && pwd)
TC=$(ls -d "$G"/.rustup/toolchains/stable-*| head -1)
export PATH=$TC/bin:$PATH CARGO_HOME=$W/cargo-home CARGO_TARGET_DIR=$W/build/target
mkdir -p "$CARGO_HOME" "$W/build"
[ -d "$CARGO_HOME/registry" ] || cp -r "$G/.cargo/registry" "$CARGO_HOME/"
rm -rf "$W/build/pe" && cp -r "$W/pe-src" "$W/build/pe"
LIB=$W/build/pe/poke-engine-py/src/lib.rs
cat "$HERE/spike_bindings.rs" >> "$LIB"
sed -i 's/^\(\s*\)m.add_function(wrap_pyfunction!(calculate_damage, m)?)?;/&\n\1m.add_function(wrap_pyfunction!(spike_options, m)?)?;\n\1m.add_function(wrap_pyfunction!(spike_random_games, m)?)?;/' "$LIB"
grep -q 'wrap_pyfunction!(spike_random_games' "$LIB"
cargo --version
for g in 4 9; do
  case $g in 4) F=poke-engine/gen4 ;; 9) F=poke-engine/terastallization ;; esac
  rm -rf "$W/wheels$g" "$W/eng$g"
  "$G/.venv/bin/python" -m pip wheel -q --no-deps "$W/build/pe/poke-engine-py" -w "$W/wheels$g" \
    --config-settings="build-args=--features $F --no-default-features"
  "$G/.venv/bin/python" -m pip install -q --no-deps --target "$W/eng$g" "$W"/wheels$g/*.whl
  PYTHONPATH=$W/eng$g "$G/.venv/bin/python" -c "import poke_engine as pe; assert pe.spike_random_games; print('eng$g ok')"
done
