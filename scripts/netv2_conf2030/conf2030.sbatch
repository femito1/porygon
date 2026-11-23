#!/bin/bash
# netv2 2026-11-23: F2030 = 1000-game confirmation of raw G9q-cd@2030 vs Foul Play gen9 at a FIXED 200,000 iterations per
# state, ONE slot job (runs/g9q-cd2030/PREREG.txt) = conf1350.sbatch (b9995dc6) with the net lookup (the one file in
# runs/g9q-cd2030/final/, md5 = the branch driver's STATUS line), the R2030 gate (runs/g9q-cd2030/READING.txt line 2 must
# start "R2030 IMPROVING: SUCCESS"), the seed 8203222, and Foul Play's budget: wsp's fixed-iteration launcher
# fp9_iters.sbatch (82a207a6, + fp_fixed_iters.py f9c32cdf, iteration check fp_iters_check.py a1ff4886) invoked as wsp's
# fpcal_rerunL.sbatch does, --fp-iters 200000. An infrastructure VOID (a Foul Play instance "did not start" after
# websocket "timed out during handshake", 0 games) is replayed ONCE in this job with the same seed -> RESULTS.txt -> one
# ledger RELEASE line. Submitted HELD; r2030-chain.sh releases it on R2030 SUCCESS (or cancels it):
#   cd $C && sbatch --hold scripts/netv2_conf2030/conf2030.sbatch
#SBATCH --job-name=ws-netv2-conf2030
#SBATCH --partition=training
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j.out
set -uo pipefail
C=${MOCK_C:-/vol/bitbucket/fr1226/porygon-ws-netv2}; cd "$C" || exit 1
D=$C/runs/conf2030; mkdir -p "$D"; J=${SLURM_JOB_ID:?}
PRE_MD5=7a68bf3e
ND=$C/runs/g9q-cd2030; PRE=$ND/PREREG.txt
FQ=${MOCK_FQ:-/vol/bitbucket/fr1226/porygon-ws-wsp/fpcal}; FPI=$FQ/code/fp9_iters.sbatch; FPI_MD5=82a207a6
FCHK=$FQ/code/fp_iters_check.py; FCHK_MD5=a1ff4886; FWRAP=$FQ/code/fp_fixed_iters.py; FWRAP_MD5=f9c32cdf
ITERS=200000; SEED_II=8203222
if [ -n "${MOCK_C:-}" ]; then PRE_MD5=$MOCK_PRE_MD5; FPI_MD5=$MOCK_FPI_MD5; FCHK_MD5=$MOCK_FCHK_MD5; FWRAP_MD5=$MOCK_FWRAP_MD5; sleep() { :; }; fi  # laptop mock only
st() { echo "$(date -u +%FT%T) $*" | tee -a "$D/STATUS"; }
md5of() { md5sum < "$1" | cut -c1-8; }
failstage() { st "FAILED-$1: $2"; echo "$(date -u +%FT%T) $2" >> "$D/FAILED-$1"; }
outbox() { ( flock 9; printf '%s\n' "$1" >> "$C/LEDGER-OUTBOX.md" ) 9> "$C/LEDGER-OUTBOX.lock"; }
newest() { ls -td "$FQ"/runs/foulplay-net-*-"$J"-"$ITERS" 2>/dev/null | head -n 1; }
# reading: CONFIRMED iff the lower 95% bound (point - half-width of the pair-mean interval) > 50
reading() {
  local p h
  p=$(grep -oE ': [0-9.]+% ± [0-9.]+%' <<< "$1" | head -n 1 | sed -E 's/^: ([0-9.]+)% ± ([0-9.]+)%/\1/')
  h=$(grep -oE ': [0-9.]+% ± [0-9.]+%' <<< "$1" | head -n 1 | sed -E 's/^: ([0-9.]+)% ± ([0-9.]+)%/\2/')
  if [ -z "$p" ] || [ -z "$h" ]; then echo "UNREADABLE result line"; return 0; fi
  awk -v p="$p" -v h="$h" 'BEGIN { lo = p - h; hi = p + h; printf "[%.1f, %.1f] => %s", lo, hi, (lo > 50 ? "CONFIRMED (lower 95% bound > 50%)" : "NOT CONFIRMED (lower 95% bound <= 50%)") }'
}
short() { grep -oE ': [0-9.]+% ± [0-9.]+%' <<< "$1" | head -n 1 | sed 's/^: //' || true; }
norun() { st "$1: nothing run"; echo "$(date -u +%FT%T) $1" > "$D/FAILED-$2"; outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | F2030 (G9q-cd@2030 vs FP gen9 200k) did not run ($1); slot FREE"; exit 1; }
st "job $J on $(hostname -s) (${SLURM_CPUS_ON_NODE:-?} CPUs) start"
if [ "$(md5of "$PRE")" != "$PRE_MD5" ] || [ -w "$PRE" ]; then norun "PREREG md5 / mode wrong" prereg; fi
# the R2030 gate (PREREG): play only after a SUCCESS reading
if ! sed -n 2p "$ND/READING.txt" 2>/dev/null | grep -q '^R2030 IMPROVING: SUCCESS'; then norun "runs/g9q-cd2030/READING.txt is missing or not SUCCESS" gate; fi
# the net: the one file in runs/g9q-cd2030/final/, md5 = the driver's STATUS line
n=$(ls "$ND"/final/*.pt 2>/dev/null | wc -l)
if [ "$n" != 1 ]; then norun "$n files in runs/g9q-cd2030/final (want 1)" net; fi
NET=$(ls "$ND"/final/*.pt)
want=$(grep "G9q-cd2030 final at" "$ND/STATUS" 2>/dev/null | tail -n 1 | sed -n 's/.*(md5 \([0-9a-f]*\)).*/\1/p')
NET_MD5=$(md5of "$NET")
if [ -z "$want" ] || [ "$NET_MD5" != "$want" ]; then norun "net md5 $NET_MD5 != the driver's STATUS md5 '${want:-none}'" net; fi
grep -q "(md5 $NET_MD5)" "$ND/READING.txt" || norun "net md5 $NET_MD5 is not the one in READING.txt" net
st "net G9q-cd@2030 = $NET (md5 $NET_MD5); R2030 SUCCESS"
res_ii="not run"; rd_ii=""; R=""; voids=0
if [ "$(md5of "$FPI")" != "$FPI_MD5" ] || [ "$(md5of "$FCHK")" != "$FCHK_MD5" ] || [ "$(md5of "$FWRAP")" != "$FWRAP_MD5" ]; then
  failstage ii "fixed-iteration launcher md5 (fp9_iters / fp_iters_check / fp_fixed_iters)"; res_ii="FAILED (launcher md5)"
else
  for att in 1 2; do
    prev=$(newest)
    st "F2030 attempt $att: G9q-cd@2030 vs Foul Play gen9 at a fixed $ITERS iterations per state, 1000 games, seed $SEED_II"
    bash -c "cd $C && SLURM_SUBMIT_DIR=$C bash $FPI $NET 1000 --gen 9 --seed $SEED_II --fp-iters $ITERS --temperature 0.5" > "$D/ii-$att.log" 2>&1; hrc=$?
    R=$(newest); [ "$R" = "$prev" ] && R=""
    games=0; [ -n "$R" ] && [ -f "$R/results.jsonl" ] && games=$(grep -c . "$R/results.jsonl")
    if [ -n "$R" ] && [ -f "$R/result.txt" ] && grep -q '%' "$R/result.txt"; then
      if [ -f "$R/ITERS_MISMATCH" ]; then failstage ii "iteration check: $(cat "$R/ITERS_MISMATCH") (run $R; the run is void, not replayed)"; res_ii="FAILED (iteration check)"
      else res_ii="$(head -n 1 "$R/result.txt")"; rd_ii="$(reading "$res_ii")"; st "F2030 $res_ii => $rd_ii"
           [ -f "$R/ITERS_OK" ] && st "iteration check: $(cat "$R/ITERS_OK")"; fi
      break
    fi
    if [ "$games" = 0 ] && grep -q "Foul Play instance [0-9]* did not start" "$D/ii-$att.log" && grep -q "timed out during handshake" "$D/ii-$att.log"; then
      voids=$((voids + 1))
      echo "$(date -u +%FT%T) attempt $att (run ${R:-none}, harness rc $hrc): a Foul Play instance did not start (websocket handshake timeouts), 0 games: INFRASTRUCTURE VOID" > "$D/VOID-$att"
      st "VOID attempt $att: $(cat "$D/VOID-$att")"
      if [ "$att" = 1 ]; then st "replaying once, same seed $SEED_II and settings"; sleep 60; continue; fi
      failstage ii "two infrastructure voids (handshake timeouts): nothing read"; res_ii="FAILED (two infrastructure voids)"; break
    fi
    failstage ii "harness rc $hrc / no result.txt (run ${R:-none}, $games games; log $D/ii-$att.log)"; res_ii="FAILED"; break
  done
fi
{ echo "F2030: G9q-cd@2030 1000-game confirmation vs Foul Play gen9 at a FIXED $ITERS iterations per state (runs/g9q-cd2030/PREREG.txt $PRE_MD5), job $J; net $(basename "$NET") md5 $NET_MD5; measurement, reading per the PREREG"
  echo "vs Foul Play gen9, fixed $ITERS iterations per 1000 ms state [wsp fp9_iters $FPI_MD5, seed $SEED_II (new)]: $res_ii${rd_ii:+ => $rd_ii}"
  echo "infrastructure voids replayed: $voids$(for f in "$D"/VOID-*; do [ -e "$f" ] && printf ' | %s' "$(cat "$f")"; done)"
  echo "context only (not paired tests): G9q-cd@1350 vs Foul Play gen9 59.1% ± 2.9% [56.2, 62.0] at time-based 1 s per state (runs/conf1350/RESULTS.txt); fpcal cd@805 61.3 / 57.8 / 54.9% at 50k / ~210k time-based / 800k"
  for f in "$D"/FAILED-*; do if [ -e "$f" ]; then echo "failed: $f"; fi; done
  true; } > "$D/RESULTS.txt.tmp"
mv -f "$D/RESULTS.txt.tmp" "$D/RESULTS.txt" && chmod a-w "$D/RESULTS.txt"
st "RESULTS written"
fl=""; for f in "$D"/FAILED-*; do if [ -e "$f" ]; then fl="$fl $(basename "$f")"; fi; done
outbox "$(date -u +%FT%H:%MZ) | netv2 | RELEASE | $J | F2030 G9q-cd@2030 1000-g confirmation done: vs FP gen9 (fixed 200k iterations) $(short "$res_ii") ${rd_ii:-$res_ii}; voids replayed $voids; failed stages:${fl:- none}; runs/conf2030/RESULTS.txt; slot FREE"
st "chain done (RELEASE line in $C/LEDGER-OUTBOX.md)"
