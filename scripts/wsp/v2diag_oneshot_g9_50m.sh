# Tracker one-shot (deploy as $WSP/v2diag/oneshot/g9_50m.sh; queued 2026-11-09 ~13:00 at the orchestrator's request):
# the G9s@50M vs G9p@50M fixed-position profile next to netv2's step-matched 50M check, with exactly the 25M profile's
# measurements (v2diag/manual/g9_25m_b.sh) on the frozen positions v2diag/pos9. Each net = its run's first pool snapshot
# >= 50,000,000 steps in a mirror dir. Until G9p's exists, is >= 2 min old and loads as a full gen9 net, the script
# re-queues itself for the next pass and exits at once, so it never holds up the tracker. Outputs:
# v2diag/g9cmp/out/{vsP1_50M, vsG9p50_50M, sw50M} and this pass's slurm log.
NV2=/vol/bitbucket/fr1226/porygon-ws-netv2/runs
D=v2diag/g9cmp; POS=v2diag/pos9
first_ge() {  # the file with the smallest step >= 50M among the given pool files (basename = zero-padded step)
  ls "$@" 2>/dev/null | awk -F/ '{print $NF" "$0}' | sort | while read -r b f; do
    s=${b%.pt}; [[ $s =~ ^[0-9]+$ ]] && [ $((10#$s)) -ge 50000000 ] && { echo "$f"; break; }; done; }
ok_net() { [ -n "$1" ] && [ "$(python scripts/wsp/v2diag_nparams.py "$1" 2>/dev/null | tail -1)" = "5809977" ]; }
GP=$(first_ge $NV2/r1/20261108-220401-718701/G9p/pool/*.pt $NV2/g9/20261109-021316-741849/G9/pool/*.pt)
GS=$(first_ge $NV2/g9/20261109-021316-741849/G9s/pool/*.pt $NV2/r1/20261108-220401-718701/G9s/pool/*.pt)
if [ -z "$GP" ] || [ $(( $(date +%s) - $(stat -c %Y "$GP") )) -lt 120 ] || ! ok_net "$GP" || ! ok_net "$GS"; then
  echo "G9p >= 50M not ready in a mirror pool (G9p: ${GP:-none}; G9s: ${GS:-none}): re-queued for the next pass"
  cp "$0" v2diag/oneshot/.g9_50m.sh.tmp && mv v2diag/oneshot/.g9_50m.sh.tmp v2diag/oneshot/g9_50m.sh
  exit 0
fi
echo "G9s@50M: $GS"; echo "G9p@50M: $GP"
cp "$GS" $D/ck/G9s_50M.pt; cp "$GP" $D/ck/G9p_50M.pt
echo "## vs P1"; python scripts/wsp/v2diag_decay.py --gen 9 --ref $D/ck/P1.pt --pos $POS --out $D/out/vsP1_50M --n 8000 $D/ck/G9s_50M.pt $D/ck/G9p_50M.pt 2>&1 | grep -v Warn
echo "## G9s@50M vs G9p@50M"; python scripts/wsp/v2diag_decay.py --gen 9 --ref $D/ck/G9p_50M.pt --pos $POS --out $D/out/vsG9p50_50M --n 8000 $D/ck/G9s_50M.pt 2>&1 | grep -v Warn
echo "## switching"; python scripts/wsp/v2diag_switch.py --gen 9 --pos $POS --out $D/out/sw50M --n 3000 $D/ck/P1.pt $D/ck/G9s_50M.pt $D/ck/G9p_50M.pt 2>&1 | grep -v Warn
echo "## done $(date -u +%T)"
