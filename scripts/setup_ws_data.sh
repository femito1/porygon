#!/usr/bin/env bash
# Data-stream cluster dir: make /vol/bitbucket/fr1226/porygon-ws-data/code a working sbatch submit dir that uses
# gloria's environments READ-ONLY (gloria is the shared main copy: nothing here may write into it).
#
#   laptop (worktree root):  scripts/setup_ws_data.sh sync   # rsync code + COMMIT, then run the setup over ssh
#   cluster:                 bash /vol/bitbucket/fr1226/porygon-ws-data/code/scripts/setup_ws_data.sh
#
# What it creates in code/ (idempotent, light: symlinks plus a ~50 MB copy, fine on the login node):
#   .venv .node .engine-gen9  -> symlinks to gloria's (python / node only read them; our sbatch scripts export
#                                PYTHONDONTWRITEBYTECODE=1 so no .pyc lands in gloria's site-packages)
#   .showdown-eval            -> a private COPY of gloria's duplicate-battle Showdown (dist, config, data, server
#                                sources; .git and logs left out), node_modules symlinked to gloria's. Needed because
#                                scripts/start_servers.sh writes into PS_DIR (config.js if missing, logs/, `node
#                                build`, which transpiles into dist/) and the servers write logs/ and databases/.
#   pokemon-showdown          -> symlink to our .showdown-eval copy (the duplicate patch is inert unless
#                                PORYGON_DUPLICATE is set), so no script can fall back to gloria's checkout.
#   runs, logs                -> ../runs, ../logs (outputs stay in porygon-ws-data/)
# Python path: our sbatch scripts put code/src first (gloria's .venv has an editable install of gloria/src, which
# comes after PYTHONPATH). Foul Play (/vol/bitbucket/fr1226/foul-play) is used in place: only run.py from its dir
# with PYTHONDONTWRITEBYTECODE=1; its login patch must already be there (checked, never applied here).
set -euo pipefail
G=/vol/bitbucket/fr1226/gloria
W=/vol/bitbucket/fr1226/porygon-ws-data
C=$W/code

if [ "${1:-}" = sync ]; then  # laptop side
  cd "$(dirname "$0")/.."
  printf '%s\n' "$(git describe --always --dirty) ($(git rev-parse --abbrev-ref HEAD))" > /tmp/porygon-ws-data-COMMIT
  ssh -o BatchMode=yes gpucluster2 "mkdir -p $C"
  rsync -a --exclude __pycache__ --exclude '*.egg-info' src scripts tests patches pyproject.toml gpucluster2:$C/
  rsync -a /tmp/porygon-ws-data-COMMIT gpucluster2:$C/COMMIT
  ssh -o BatchMode=yes gpucluster2 "bash $C/scripts/setup_ws_data.sh" 2>&1 | grep -v -E \
    'WARNING|prosecuted|monitored|Authorized|College|imperial.ac.uk|DO NOT RUN|Use of IT|^\s*$' || true
  exit 0
fi

mkdir -p "$W/runs" "$W/logs" "$C"
cd "$C"
link() {  # link NAME TARGET (replace a stale link, refuse to clobber a real dir)
  if [ -L "$1" ]; then ln -sfn "$2" "$1"; elif [ -e "$1" ]; then echo "$C/$1 exists and is not a link" >&2; exit 1
  else ln -s "$2" "$1"; fi
}
link .venv "$G/.venv"
link .node "$G/.node"
link .engine-gen9 "$G/.engine-gen9"
link runs ../runs
link logs ../logs

# private duplicate-battle Showdown
rsync -a --exclude .git --exclude logs --exclude node_modules --exclude 'databases/*.db*' \
  "$G/.showdown-eval/" .showdown-eval/
link .showdown-eval/node_modules "$G/pokemon-showdown/node_modules"
mkdir -p .showdown-eval/logs/repl
grep -q nofswriting .showdown-eval/config/config.js || {
  cp .showdown-eval/config/config-example.js .showdown-eval/config/config.js
  printf '\nexports.nofswriting = true;\nexports.repl = false;\n' >> .showdown-eval/config/config.js
}
link pokemon-showdown .showdown-eval

# checks (read-only)
ok=1
[ -x .venv/bin/python ] || { echo "missing gloria .venv" >&2; ok=0; }
[ -f .engine-gen9/poke_engine/__init__.py ] || { echo "missing gloria .engine-gen9" >&2; ok=0; }
[ -f .showdown-eval/dist/server/index.js ] || { echo "no built eval Showdown in the copy" >&2; ok=0; }
grep -q "patched for the Porygon benchmark" /vol/bitbucket/fr1226/foul-play/fp/websocket_client.py ||
  { echo "Foul Play login patch missing (do not patch here; ask the orchestrator)" >&2; ok=0; }
[ -x /vol/bitbucket/fr1226/foul-play/.venv-gen9/bin/python ] || { echo "no Foul Play gen9 venv" >&2; ok=0; }
[ -d src/porygon ] || { echo "no code synced yet (scripts/setup_ws_data.sh sync from the laptop)" >&2; ok=0; }
echo "code dir $C ready=$ok; commit $(cat COMMIT 2>/dev/null || echo '?')"
ls -la "$C" | sed -n '2,$p'
