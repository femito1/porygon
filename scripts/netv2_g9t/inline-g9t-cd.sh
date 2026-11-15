# netv2 2026-11-14, G9t chain, the INLINE body (orchestrator-approved chain 15:2x; runs/g9t/PREREG.txt e57b8ab4 rule 2):
# called from runs/g9/INLINE_CMD (queued by scripts/netv2_g9t/chain-s2.sh) inside nv2_pause of G9t's job (trainer
# stopped, LOCAL exported, cwd $NV2, the GPU free):
#  1 G9t-cd = G9t's argv (runs/g9t/chain/g9t-trainer-argv.bin, captured from the live trainer) + --resume G9t's first
#    checkpoint >= 45M + --cooldown-start S --total-steps S+5M --final-lr-frac 0.1 --guard-entropy-min 0.35, own run /
#    pool dirs ($LOCAL/run/G9t-cd; pool snapshots + frozen bests after S moved aside; no best_model.pt),
#    `timeout -s INT -k 300 21600`, through G9t's wrapper (src-ff32b71), November sets: exactly the G9q-cd@50M branch
#    (runs/g9r/PREREG-CD50.txt, logs/nv2agent-cd50-20261112.sh) with G9t's argv;
#  2 final -> runs/g9t-cd/final/<steps>.pt (an early guard / timeout stop: its final, reported with its steps);
#  3 rule 2 h2h, seed 9850 (scripts/netv2_cmp_step.sh a80c7da2, A "G9s" = G9t-cd, B "G9p" = runs/g9q-cd/final/
#    00050038148.pt 7c9e56b3), here on the freed A16 (accepted deviation: the data host could not be pre-loaded);
#  4 rule 2 read mechanically: "G9q-cd > G9t-cd" SUCCESS => park G9t (scripts/netv2_g9t/park-g9t.sh: persistent stop);
#    "G9t-cd > G9q-cd" SUCCESS => adopted, G9t relaunches; else undecided, G9t relaunches. Placement = orchestrator.
# Any failure: runs/g9t/chain/FAILED + a STATUS line; G9t relaunches from its own save (nothing parked).
set -u
C=${NV2:-/vol/bitbucket/fr1226/porygon-ws-netv2}; G=$C/runs/g9t; K=$G/chain; L=$LOCAL/run/G9t
S=45003414; CK=$C/runs/g9/20261112-015035-810367/G9t/checkpoints/ckpt_00045003414.pt; CKMD5=1cafed09
QF=$C/runs/g9q-cd/final/00050038148.pt; QMD5=7c9e56b3; W=$C/bin/py-src-ff32b71; D=$G/cmp/cd50; F=$C/runs/g9t-cd
st() { echo "$(date -u +%FT%T) INLINE $*" | tee -a "$K/STATUS"; }
failc() { st "FAILED: $* (G9t relaunches from its own save)"; echo "$(date -u +%FT%T) INLINE $*" >> "$K/FAILED"; exit 0; }
[ -n "${LOCAL:-}" ] || failc "LOCAL unset"
[ "$(md5sum < "$CK" | cut -c1-8)" = "$CKMD5" ] || failc "branch point $CK md5 is not $CKMD5"
[ "$(md5sum < "$QF" | cut -c1-8)" = "$QMD5" ] || failc "control $QF md5 is not $QMD5"
[ -s "$K/g9t-trainer-argv.bin" ] || failc "no $K/g9t-trainer-argv.bin"
[ -x "$W" ] || failc "no wrapper $W"
export PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
mapfile -d '' -t A < "$K/g9t-trainer-argv.bin"
cut_pool() {  # $1 = pool dir, $2 = S: snapshots and frozen bests after S moved aside
  mkdir -p "$1-after-S"
  for f in "$1"/*.pt "$1"/frozen/best-*.pt; do [ -e "$f" ] || continue; n=$(basename "$f" .pt); n=${n#best-}; case $n in *[!0-9]*|"") continue ;; esac
    if [ "$((10#$n))" -gt "$2" ]; then mv -f "$f" "$1-after-S/"; fi; done
}
# --- 1. the branch ---
B=$LOCAL/run/G9t-cd; mkdir -p "$B" "$F"
if [ ! -d "$B/pool" ]; then cp -rp "$L/pool" "$B/pool-copy" && mv "$B/pool-copy" "$B/pool" || failc "pool copy"; fi
cut_pool "$B/pool" "$S"
cp -p "$CK" "$F/branch-point-$(printf %011d "$S").pt.tmp" && mv -f "$F/branch-point-$(printf %011d "$S").pt.tmp" "$F/branch-point-$(printf %011d "$S").pt"
st "G9t-cd start: resume $CK (S $S, md5 $CKMD5), cosine $S -> $((S + 5000000)) x0.1, floor 0.35, pool $(ls "$B"/pool/*.pt | wc -l) + frozen $(ls "$B"/pool/frozen/*.pt 2>/dev/null | wc -l)" | tee -a "$F/notes.log"
timeout -s INT -k 300 21600 "$W" "${A[@]:1}" --run-dir "$B" --pool-dir "$B/pool" --resume "$CK" --cooldown-start "$S" --total-steps "$((S + 5000000))" --final-lr-frac 0.1 --guard-entropy-min 0.35 >> "$B/stdout.log" 2>&1
rc=$?
sv=$(grep "saved final_model.pt at" "$B/train.log" 2>/dev/null | tail -n 1)
st "G9t-cd exit $rc; ${sv:-no save line}" | tee -a "$F/notes.log"
[ -f "$B/final_model.pt" ] && [ -n "$sv" ] || failc "no G9t-cd final_model.pt / save line"
stp=$(grep -oE "saved final_model.pt at [0-9,]+ steps" <<< "$sv" | tr -dc 0-9)
[ -n "$stp" ] || failc "cannot parse the G9t-cd save line"
mkdir -p "$F/final"; f=$F/final/$(printf %011d "$stp").pt
cp -p "$B/final_model.pt" "$f.tmp" && mv -f "$f.tmp" "$f" || failc "final copy"
cp -p "$B/train.log" "$F/branch-train.log"
lr=$(grep -oE " lr [0-9.e-]+" "$B/train.log" | tail -n 1)
ent=$(grep -oE " ent [0-9.]+" "$B/train.log" | tail -n 1)
trip=""; [ -f "$B/GUARD_TRIPPED" ] && trip="; GUARD_TRIPPED: $(head -c 200 "$B/GUARD_TRIPPED")"
st "G9t-cd final at $stp steps -> $f (md5 $(md5sum < "$f" | cut -c1-8));$lr;$ent at the end$trip" | tee -a "$F/notes.log"
[ "$stp" -ge $((S + 5000000)) ] || st "NOTE: G9t-cd stopped early at $stp (< $((S + 5000000))): its final is the net (PREREG rule 2)"
# --- 2. rule 2 h2h (seed 9850) ---
mkdir -p "$D"
if [ ! -f "$D/g9s.pt" ] || [ ! -f "$D/g9p.pt" ]; then
  cp -p "$f" "$D/g9s.pt.tmp" && mv "$D/g9s.pt.tmp" "$D/g9s.pt" && cp -p "$QF" "$D/g9p.pt.tmp" && mv "$D/g9p.pt.tmp" "$D/g9p.pt" || failc "rule 2 net copy"
  printf 'G9s %s (G9t-cd)\nG9p %s (G9q-cd)\nseed %s\nmd5 g9s %s g9p %s\n' "$f" "$QF" 9850 "$(md5sum < "$D/g9s.pt" | cut -c1-32)" "$(md5sum < "$D/g9p.pt" | cut -c1-32)" > "$D/NETS.txt"
fi
cp "$C/scripts/netv2_cmp_step.sh" "$LOCAL/nv2-g9t-cmp.sh"; [ "$(md5sum < "$LOCAL/nv2-g9t-cmp.sh" | cut -c1-8)" = a80c7da2 ] || failc "netv2_cmp_step.sh md5"
st "rule 2 h2h: G9t-cd@$stp vs G9q-cd@50,038,148, seed 9850, on the A16 ($(nproc) CPUs visible, SLURM_CPUS_ON_NODE ${SLURM_CPUS_ON_NODE:-?})"
CMP_WORKERS=${G9T_CMP_WORKERS:-8} timeout 7200 bash "$LOCAL/nv2-g9t-cmp.sh" 50000000 9850 "$D" "$F/final" "$C/runs/g9q-cd/final" > "$D/run.log" 2>&1 || failc "rule 2 h2h exited $? (log $D/run.log)"
read -r x y <<< "$(sed -n 's/^DECISION OF RECORD.*test G9s > G9p exit \([0-9]*\) at look.*test G9p > G9s exit \([0-9]*\) at look.*/\1 \2/p' "$D/RESULT.txt")"
[ -n "${y:-}" ] || failc "rule 2: no DECISION OF RECORD line in $D/RESULT.txt"
if [ "$y" = 10 ]; then v=STOP; why="\"G9q-cd > G9t-cd\" SUCCESS (exit 10) => stop G9t (park)"
elif [ "$x" = 10 ]; then v=ADOPTED; why="\"G9t-cd > G9q-cd\" SUCCESS (exit 10) => lambda 0.5 adopted; vs-search milestone (400 g, Showdown slot); G9t continues as a lineage candidate (placement = orchestrator / user)"
else v=UNDECIDED; why="\"G9t-cd > G9q-cd\" exit $x, \"G9q-cd > G9t-cd\" exit $y => lambda not adopted; G9t continues only on a GPU nobody else needs, else parked (placement = orchestrator / user)"; fi
{ echo "G9t PREREG rule 2 (runs/g9t/PREREG.txt e57b8ab4), read mechanically by scripts/netv2_g9t/inline-g9t-cd.sh"
  echo "VERDICT $v"; echo "$why"; echo "G9t-cd final $f at $stp steps; RESULT.txt md5 $(md5sum < "$D/RESULT.txt" | cut -c1-8)"
  echo "deviation (accepted by the orchestrator 2026-11-14): played on the A16 in G9t's INLINE, not in the data host"
  grep -A1 '^DECISION OF RECORD' "$D/RESULT.txt"; grep -A3 '^--- all pairs played' "$D/RESULT.txt"; } > "$D/READING.txt.tmp" && mv -f "$D/READING.txt.tmp" "$D/READING.txt" && chmod a-w "$D/READING.txt"
cp -p "$D/READING.txt" "$K/VERDICT-rule2.txt"
st "rule 2 VERDICT $v: $why"
if [ "$v" = STOP ]; then
  if PARK_RULE=2 PARK_QUEUED_FILE="$C/runs/g9/INLINE_CMD.running" PARK_MAX_AGE_S=43200 bash "$C/scripts/netv2_g9t/park-g9t.sh"; then
    st "FINAL: G9t stopped by rule 2; parked in runs/g9t-parked (PARKED.txt + MD5SUMS); the A16 job HOLDs; A16 placement = main"
  else failc "rule 2 STOP but the park FAILED (runs/g9t/park.log)"; fi
else
  st "FINAL: rule 2 $v; G9t relaunches from its own save; NEXT = main: $( [ "$v" = ADOPTED ] && echo 'vs-search 400 g of G9t-cd@50M (slot) + G9t placement' || echo 'G9t placement')"
fi
