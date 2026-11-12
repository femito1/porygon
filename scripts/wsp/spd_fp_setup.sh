#!/bin/bash
# Test root for the Foul Play harness (scripts/slurm_foulplay.sbatch + scripts/vs_foulplay.py) that touches nothing
# shared: a private Foul Play copy ($WSP/spd/foul-play, gen4 venv; the sbatch patches this copy, never the shared
# clone) and $WSP/spd/fproot = symlinks to gloria's .venv/.node/.showdown-eval (read-only) and to $WSP/scripts, with a
# start_servers.sh that does not run `node build` in the shared .showdown-eval. Run on the cluster (light: one copy).
#   scripts/wsp/spd_fp_setup.sh
#   cd $WSP/spd/fproot && sbatch --partition=long -c 8 --mem=32G --time=1:00:00 --job-name=ws-wsp-spdfp \
#     scripts/slurm_foulplay.sbatch $GLORIA/runs/20261025-195826-301040/final_model.zip 20 5 --player policy \
#     [--pre-game-wait sleep --start-poll 0.5]      # (the old behaviour, for the before/after comparison)
set -euo pipefail
W=/vol/bitbucket/fr1226/porygon-ws-wsp; G=/vol/bitbucket/fr1226/gloria; R=$W/spd/fproot
[ -x "$W/spd/foul-play/.venv/bin/python" ] || rsync -a --exclude .git --exclude .venv-gen9 --exclude tests \
  /vol/bitbucket/fr1226/foul-play/ "$W/spd/foul-play/"
mkdir -p "$R/scripts" "$R/logs" "$R/runs"
for x in .venv .node .cargo .rustup .showdown-eval; do ln -sfn "$G/$x" "$R/$x"; done
ln -sfn "$W/src" "$R/src"
for f in vs_foulplay.py analyze_benchmark.py stop_servers.sh build_eval_server.sh slurm_foulplay.sbatch; do
  ln -sfn "$W/scripts/$f" "$R/scripts/$f"
done
sed 's#^if .*(cd "$PS" \&\& node build); fi$#: \# spd test copy: no build in the shared .showdown-eval#' \
  "$W/scripts/start_servers.sh" > "$R/scripts/start_servers.sh"
chmod +x "$R/scripts/start_servers.sh"
grep -q "spd test copy" "$R/scripts/start_servers.sh" || { echo "start_servers.sh changed: no build line removed" >&2; exit 1; }
echo "test root $R ready (Foul Play copy $W/spd/foul-play)"
