#!/bin/bash
# netv2 2026-11-15: entropy-schedule arm INLINE #E1 (runs/g9qent/PREREG.txt START), run inside an INLINE_CMD of the g9 job
# while it HOLDs on a parked lineage (netv2.env v18 park guard: nv2_pause, LOCAL exported, cwd $NV2), from a node-local
# md5-checked copy. Fork point = G9q's newest mirrored checkpoint -> runs/g9qent/fork-point-<S0>.pt + FORK_POINT
# ("PATH S0 MD5"); its pool (<= S0) + best / ema staged in $LOCAL/run/G9qent (.staged-S0); runs/g9/MODE = g9qent; then
# runs/g9/HOLD removed so this round's nv2_pause returns and the park guard re-reads MODE (=> the arm's first start).
# A failure before the MODE write changes nothing but leaves FAILED (the park HOLD goes on).
set -u
C=${NV2:-$PWD}; E=$C/runs/g9qent; K=$C/runs/g9qent-chain
mkdir -p "$E" "$K"
say() { echo "$(date -u +%FT%T) E1 $*" | tee -a "$K/STATUS"; }
fail() { say "FAILED: $*"; echo "$(date -u +%FT%T) E1 $*" >> "$K/FAILED"; exit 1; }
[ -n "${LOCAL:-}" ] || fail "LOCAL unset (run me only through nv2_pause)"
m=$(cat "$C/runs/g9/MODE" 2>/dev/null)
case $m in
  g9q2) [ -f "$C/runs/g9q2-parked/PARKED.txt" ] || fail "MODE g9q2 but G9q2 is not parked" ;;
  g9qbf) [ -f "$C/runs/g9qbf-parked/PARKED.txt" ] || fail "MODE g9qbf but G9qbf is not parked" ;;
  *) fail "runs/g9/MODE is '$m' (want a parked g9q2 / g9qbf)" ;;
esac
[ ! -e "$E/G9QENT_STARTED" ] || fail "runs/g9qent/G9QENT_STARTED exists (the arm started before)"
say "start in job ${SLURM_JOB_ID:-?} on $(hostname -s) (MODE $m, parked)"
GQ=$(ls -td "$C"/runs/r1/*/G9q 2>/dev/null | head -n 1); [ -d "$GQ/pool" ] || fail "no mirrored G9q lineage dir"
CK=$(ls -t "$GQ"/checkpoints/ckpt_*.pt 2>/dev/null | head -n 1); [ -n "$CK" ] || fail "no mirrored G9q checkpoint in $GQ"
S0=$(basename "$CK" .pt); S0=$((10#${S0#ckpt_}))
FP=$E/fork-point-$(printf %011d "$S0").pt
cp -p "$CK" "$FP.tmp" && mv -f "$FP.tmp" "$FP" && chmod a-w "$FP" || fail "could not copy the fork point"
FM=$(md5sum < "$FP" | cut -c1-8); [ "$FM" = "$(md5sum < "$CK" | cut -c1-8)" ] || fail "fork point copy md5 mismatch"
echo "$GQ $CK" > "$E/FORK_POINT.src"
say "fork point: $CK -> $FP (S0 $S0, md5 $FM)"
D=$LOCAL/run/G9qent; rm -rf "$D"; mkdir -p "$D/pool/frozen" || fail "cannot create $D"
np=0; for f in "$GQ"/pool/*.pt; do [ -e "$f" ] || continue; n=$(basename "$f" .pt); case $n in *[!0-9]*|"") continue ;; esac
  if [ "$((10#$n))" -le "$S0" ]; then cp -p "$f" "$D/pool/.$n.tmp" && mv -f "$D/pool/.$n.tmp" "$D/pool/$n.pt" || fail "pool copy $f"; np=$((np + 1)); fi; done
nf=0; for f in "$GQ"/pool/frozen/*.pt; do [ -e "$f" ] || continue; b=$(basename "$f"); n=${b%.pt}; n=${n#best-}
  case $n in *[!0-9]*|"") ;; *) if [ "$((10#$n))" -gt "$S0" ]; then continue; fi ;; esac
  cp -p "$f" "$D/pool/frozen/$b" || fail "frozen copy $f"; nf=$((nf + 1)); done
for f in best_model.pt ema_model.pt; do if [ -f "$GQ/$f" ]; then cp -p "$GQ/$f" "$D/$f" || fail "copy $f"; fi; done
[ "$np" -gt 0 ] && [ "$nf" -gt 0 ] || fail "staged pool empty ($np snapshots, $nf frozen)"
touch "$D/.staged-$S0"; say "pool staged in $D: $np snapshots <= S0 + $nf frozen ($(du -sh "$D" | cut -f1)); best / ema copied"
echo "$FP $S0 $FM" > "$E/FORK_POINT.tmp" && mv -f "$E/FORK_POINT.tmp" "$E/FORK_POINT" || fail "could not write FORK_POINT"
cp -p "$C/runs/g9/MODE" "$C/runs/g9/MODE.$m-before-g9qent"
echo g9qent > "$C/runs/g9/MODE.tmp" && mv -f "$C/runs/g9/MODE.tmp" "$C/runs/g9/MODE" || fail "could not write MODE g9qent"
rm -f "$C/runs/g9/HOLD"
say "MODE g9qent written, runs/g9/HOLD removed: the arm starts at this round (fork point $FP, S0 $S0)"
