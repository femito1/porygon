#!/bin/bash
# VTG probe driver (netv2, 2026-11-20; measurement only, no training): a step in data's host job, e.g.
#   setsid nohup srun --jobid=1172609 --overlap --ntasks=1 -c 6 --gres=gpu:1 --mem=20G -J nv2-vtg \
#     bash -c 'cp $C/scripts/netv2_vtg/run-probe.sh /tmp/nv2-vtg-$$.sh && bash /tmp/nv2-vtg-$$.sh' > LOG 2>&1 < /dev/null &
# Probes (a) G9q-cd@1350 final and (b) G9q's newest mirrored 5M checkpoint, both with G9q's recipe (the args stored in
# (b)), fresh on-policy self-play rollouts of 40,960 (10 minibatches of 4096 each): 10 for (a), 24 for (b) (the arm would
# branch from G9q; more rollouts tighten the secondary signal estimate), seeds 7311001 / 7311002, 5 fastenv workers.
# Outputs runs/vtg-probe/{STATUS, cd1350.{txt,json}, g9q.{txt,json}, NETS.txt, RESULT.txt} or FAILED.
# Env knobs (smoke): VTG_OUT, VTG_ROLLOUT, VTG_ROLLOUTS.
set -uo pipefail
C=/vol/bitbucket/fr1226/porygon-ws-netv2
OUT=${VTG_OUT:-$C/runs/vtg-probe}
PY=$C/bin/py-src-24c633f
export PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
export PATH=/vol/bitbucket/fr1226/gloria/.node/bin:$PATH
export OMP_NUM_THREADS=1
CD=$C/runs/g9q-cd1350/final/01355036714.pt
G9Q_DIR=$C/runs/r1/20261120-114114-1174225/G9q/checkpoints
SCR=/tmp/ws-netv2-vtg-$$
mkdir -p "$OUT" "$SCR"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/STATUS"; }
fail() { say "FAILED: $*"; echo "$(date -u +%FT%TZ) $*" > "$OUT/FAILED"; rm -rf "$SCR"; exit 1; }
trap 'rm -rf "$SCR"' EXIT
rm -f "$OUT/FAILED"
say "start on $(hostname) job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}; CPUs $(nproc); $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null)"
G9Q=$(ls "$G9Q_DIR"/ckpt_*.pt 2>/dev/null | sort | tail -n 1)
[ -n "$G9Q" ] || fail "no G9q checkpoint in $G9Q_DIR"
cp "$CD" "$SCR/cd1350.pt" || fail "copy $CD"
cp "$G9Q" "$SCR/g9q.pt" || fail "copy $G9Q"
m_cd=$(md5sum < "$SCR/cd1350.pt" | cut -c1-32); m_g=$(md5sum < "$SCR/g9q.pt" | cut -c1-32)
case $m_cd in 6cf40c0c*) ;; *) fail "cd@1350 md5 $m_cd != 6cf40c0c..." ;; esac
printf 'cd1350 %s md5 %s\ng9q %s md5 %s\nrecipe = args stored in the g9q checkpoint\n' "$CD" "$m_cd" "$G9Q" "$m_g" > "$OUT/NETS.txt"
say "nets: cd1350 $m_cd, g9q $(basename "$G9Q") $m_g"
EXTRA=()
[ -n "${VTG_ROLLOUT:-}" ] && EXTRA+=(--rollout "$VTG_ROLLOUT")
for spec in "cd1350 $SCR/cd1350.pt 7311001 10" "g9q $SCR/g9q.pt 7311002 24"; do
  set -- $spec
  say "probe $1 start"
  $PY -u "$C/scripts/netv2_vtg/vtg_probe.py" --ckpt "$2" --recipe "$SCR/g9q.pt" --tag "$1" --out "$OUT" \
    --rollouts "${VTG_ROLLOUTS:-$4}" --workers 5 --seed "$3" "${EXTRA[@]}" > "$OUT/$1.log" 2>&1 \
    || fail "probe $1 rc $? (tail: $(tail -n 3 "$OUT/$1.log" | tr '\n' ' '))"
  say "probe $1 done"
done
{
  echo "VTG probe RESULT (netv2, $(date -u +%FT%TZ)): trunk-gradient norms of G9q's PPO loss terms at fixed weights,"
  echo "fresh on-policy self-play rollouts (fastenv, PORYGON_PS ps-2026-11), G9q's recipe = args stored in its checkpoint."
  echo "NOTE: G9q's stored args give vf_coef 0.5 / max_grad_norm 0.5 (v2 trainer defaults; docs/thesis/methods.md 4.2). The"
  echo "brief's 0.4375 / 0.543 are the v1 SB3 trainer's (methods.md 4.1). Ratios scale linearly with vf_coef (at 0.4375: x 0.875);"
  echo "the max grad norm only sets the clip coefficient, which scales every term alike (ratios identical before/after clipping)."
  echo "value_trunk_grad v scales the value gradient into the trunk exactly by v (model.grad_scale on the value head's input)."
  echo "Script scripts/netv2_vtg/vtg_probe.py md5 $(md5sum < "$C/scripts/netv2_vtg/vtg_probe.py" | cut -c1-8); driver md5 $(md5sum < "$0" | cut -c1-8)"
  cat "$OUT/NETS.txt"
  echo
  cat "$OUT/cd1350.txt"
  echo
  cat "$OUT/g9q.txt"
} > "$OUT/RESULT.txt.tmp" && mv "$OUT/RESULT.txt.tmp" "$OUT/RESULT.txt"
chmod a-w "$OUT/RESULT.txt" "$OUT"/*.json "$OUT"/*.txt 2>/dev/null || true
say "DONE: $OUT/RESULT.txt"
