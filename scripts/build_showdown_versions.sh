#!/usr/bin/env bash
# Build Showdown's simulator at every commit a replay corpus needs, for scripts/resim_inputlog.js.
#
#   scripts/build_showdown_versions.sh [-j N] SHA...                 # these commits
#   scripts/build_showdown_versions.sh [-j N] --corpus FILE...       # every `>version` in these (jsonl[.gz]) files
#   scripts/build_showdown_versions.sh --count FILE...               # only count battles per commit (no build)
#
# Layout under ROOT (default /vol/bitbucket/fr1226/porygon-ws-data/showdown; env SHOWDOWN_ROOT):
#   ps.git/                 bare clone of smogon/pokemon-showdown (cloned / fetched on demand)
#   builds/<sha>/dist/      sim, lib, data, config transpiled with esbuild (sources removed), node_modules -> shared
#   builds/index.json       sha -> commit time (resim_inputlog.js picks the newest older build for unbuilt commits)
#   versions.tsv            battles per commit (with --corpus / --count)
# Run inside a job (esbuild uses every core it gets; -j N parallel builds), never on the login node.
# NODE_MODULES: node_modules with esbuild + ts-chacha20 (default: gloria's pokemon-showdown, used read-only).
# FULL=1 keeps data/text (move / ability descriptions; the simulator does not read them).
set -euo pipefail
ROOT=${SHOWDOWN_ROOT:-/vol/bitbucket/fr1226/porygon-ws-data/showdown}
NODE_MODULES=${NODE_MODULES:-/vol/bitbucket/fr1226/gloria/pokemon-showdown/node_modules}
NODE=${NODE:-node}
FULL=${FULL:-0}
JOBS=4
MODE=shas
ARGS=()
while (($#)); do
  case $1 in
    -j) JOBS=$2; shift 2 ;;
    --corpus) MODE=corpus; shift ;;
    --count) MODE=count; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
mkdir -p "$ROOT/builds"

count_versions() {  # battles per `>version` sha in jsonl[.gz] files (JSON-escaped input logs)
  for f in "$@"; do
    case $f in *.gz) zcat -- "$f" ;; *) cat -- "$f" ;; esac
  done | grep -o '>version [0-9a-f]\{40\}' | awk '{print $2}' | sort | uniq -c | sort -rn | awk '{print $2"\t"$1}'
}

if [[ $MODE != shas ]]; then
  count_versions "${ARGS[@]}" > "$ROOT/versions.tsv"
  echo "$(wc -l < "$ROOT/versions.tsv") commits, $(awk '{s+=$2} END {print s}' "$ROOT/versions.tsv") battles: $ROOT/versions.tsv"
  [[ $MODE == count ]] && exit 0
  mapfile -t ARGS < <(cut -f1 "$ROOT/versions.tsv")
fi

if [[ ! -d $ROOT/ps.git ]]; then
  git clone --bare --quiet https://github.com/smogon/pokemon-showdown.git "$ROOT/ps.git"
fi
missing=()
for sha in "${ARGS[@]}"; do
  [[ -f $ROOT/builds/$sha/dist/sim/battle-stream.js ]] && continue
  git --git-dir="$ROOT/ps.git" cat-file -e "$sha^{commit}" 2>/dev/null || missing+=("$sha")
done
if ((${#missing[@]})); then
  git --git-dir="$ROOT/ps.git" fetch --quiet origin '+refs/heads/*:refs/heads/*' || true
fi

build_one() {
  local sha=$1 out="$ROOT/builds/$1" tmp="$ROOT/builds/.tmp-$1"
  [[ -f $out/dist/sim/battle-stream.js ]] && return 0
  if ! git --git-dir="$ROOT/ps.git" cat-file -e "$sha^{commit}" 2>/dev/null; then
    echo "missing commit $sha" >&2; return 0
  fi
  rm -rf "$tmp" && mkdir -p "$tmp"
  git --git-dir="$ROOT/ps.git" archive "$sha" sim lib data config tsconfig.json package.json 2>/dev/null \
    | tar -x -C "$tmp" || { echo "archive failed $sha" >&2; rm -rf "$tmp"; return 0; }
  ln -sfn "$NODE_MODULES" "$tmp/node_modules"
  (cd "$tmp" && "$NODE" -e '
    const esbuild = require("esbuild"), fs = require("fs"), path = require("path");
    const full = process.env.FULL === "1";
    const skip = p => !full && (p === "./data/text" || p === "data/text");  // formats need every mod
    const walk = (d, out) => { for (const f of fs.readdirSync(d)) { const p = d + "/" + f;
      if (skip(p)) continue;
      if (fs.statSync(p).isDirectory()) walk(p, out);
      else if ((p.endsWith(".ts") && !p.endsWith(".d.ts")) || p.endsWith(".tsx")) out.push(p); } return out; };
    const files = []; for (const d of ["sim", "lib", "data", "config"]) if (fs.existsSync(d)) walk("./" + d, files);
    esbuild.buildSync({entryPoints: files, outdir: "./dist", outbase: ".", format: "cjs", platform: "node",
      tsconfig: "./tsconfig.json", logLevel: "error"});
    const cp = d => { for (const f of fs.readdirSync(d)) { const p = d + "/" + f;
      if (skip(p)) continue;
      if (fs.statSync(p).isDirectory()) cp(p);
      else if (p.endsWith(".json") || p.endsWith(".js")) { fs.mkdirSync(path.dirname("dist/" + p), {recursive: true});
        if (!fs.existsSync("dist/" + p)) fs.copyFileSync(p, "dist/" + p); } } };
    for (const d of ["data", "config", "lib", "sim"]) if (fs.existsSync(d)) cp(d);
  ') || { echo "build failed $sha" >&2; rm -rf "$tmp"; return 0; }
  rm -rf "$tmp/sim" "$tmp/lib" "$tmp/data" "$tmp/config"
  rm -rf "$out" && mv "$tmp" "$out"
  echo "built $sha"
}
export -f build_one
export ROOT NODE_MODULES NODE FULL
printf '%s\n' "${ARGS[@]}" | xargs -r -P "$JOBS" -I{} bash -c 'build_one {}'

# sha -> commit time for every build
"$NODE" -e '
  const fs = require("fs"), cp = require("child_process"), root = process.argv[1];
  const idx = {};
  for (const sha of fs.readdirSync(root + "/builds")) {
    if (!fs.existsSync(`${root}/builds/${sha}/dist/sim/battle-stream.js`)) continue;
    try { idx[sha] = parseInt(cp.execFileSync("git", ["--git-dir", root + "/ps.git", "log", "-1", "--format=%ct", sha]).toString()); }
    catch (e) {}
  }
  fs.writeFileSync(root + "/builds/index.json", JSON.stringify(idx, null, 1));
  console.log(Object.keys(idx).length + " builds indexed");
' "$ROOT"
