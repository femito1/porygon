#!/bin/bash
# netv2 2026-11-16: the G9-P4 BRIDGE INLINE (runs/p4/PREREG.txt, START), run inside an INLINE_CMD of the g9 job while it
# HOLDs on the parked G9qent (netv2.env v18 in the job's memory; nv2_pause: LOCAL exported, cwd $NV2), from a node-local
# md5-checked copy. Why: the job script reads netv2.env only when a trainer exits with runs/g9/restart; the A40 sits in
# v18's park HOLD with v18 in memory, which has no MODE g9p4. The bridge = ONE short trainer launch on a v18 path that
# touches no live or parked lineage: v18's MODE g9qbf first start (the never-started bf16 fork lineage G9qbf, from its
# existing runs/g9qbf/FORK_POINT; env mocks 60c / 62h) with an empty staged pool in its own dir $LOCAL/run/G9qbf. The
# chain then deploys v19 in that trainer window, writes MODE g9p4 and stops the bridge with ONE SIGINT: the restart
# re-reads netv2.env (v19) => G9-P4's first start. The bridge trains nothing that is ever read (G9qbf stays unused;
# runs/g9qbf/BRIDGE.txt says so). Any refusal before the MODE write changes nothing but leaves FAILED (the HOLD goes on).
set -u
C=${NV2:-$PWD}; K=$C/runs/p4; Q=$C/runs/g9qbf
mkdir -p "$K"
say() { echo "$(date -u +%FT%T) BRIDGE $*" | tee -a "$K/STATUS"; }
fail() { say "FAILED: $*"; echo "$(date -u +%FT%T) BRIDGE $*" >> "$K/FAILED"; exit 1; }
[ -n "${LOCAL:-}" ] || fail "LOCAL unset (run me only through nv2_pause)"
m=$(cat "$C/runs/g9/MODE" 2>/dev/null)
[ "$m" = g9qent ] && [ -f "$C/runs/g9qent-parked/PARKED.txt" ] || fail "runs/g9/MODE is '$m' (want g9qent with runs/g9qent-parked/PARKED.txt)"
[ ! -e "$Q/G9QBF_STARTED" ] || fail "runs/g9qbf/G9QBF_STARTED exists (G9qbf started before)"
[ ! -e "$C/runs/g9qbf-parked" ] || fail "runs/g9qbf-parked exists"
FP="" S="" FM=""; read -r FP S FM < "$Q/FORK_POINT" || fail "no runs/g9qbf/FORK_POINT"
case $S in ""|*[!0-9]*) fail "FORK_POINT step '$S'" ;; esac
[ -f "$FP" ] && [ "$(md5sum < "$FP" | cut -c1-8)" = "$FM" ] || fail "FORK_POINT file $FP missing or md5 not $FM"
D=$LOCAL/run/G9qbf; [ ! -e "$D" ] || fail "$D exists"
mkdir -p "$D/pool/frozen" && touch "$D/.staged-$S" || fail "cannot stage $D"
{ echo "G9qbf BRIDGE $(date -u +%FT%T) in job ${SLURM_JOB_ID:-?}: v18's MODE g9qbf first start used ONLY as the one trainer launch that lets"
  echo "the g9 job re-read netv2.env (v19, MODE g9p4) for G9-P4 (runs/p4/PREREG.txt). Not an experiment: the bf16 A/B ended SKIP"
  echo "(runs/g9qbf/SKIP.txt); G9qbf (from $FP, empty pool) is stopped by ONE SIGINT within minutes and never read or resumed."; } > "$Q/BRIDGE.txt"
cp -p "$C/runs/g9/MODE" "$C/runs/g9/MODE.g9qent-before-p4bridge" || fail "cannot back up MODE"
echo g9qbf > "$C/runs/g9/MODE.tmp" && mv -f "$C/runs/g9/MODE.tmp" "$C/runs/g9/MODE" || fail "could not write MODE g9qbf"
rm -f "$C/runs/g9/HOLD"
say "MODE g9qbf written (bridge; staged $D, fork point $FP S $S), runs/g9/HOLD removed: the bridge trainer starts at this round"
