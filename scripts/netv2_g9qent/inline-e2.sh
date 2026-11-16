#!/bin/bash
# netv2 2026-11-15: entropy-schedule arm INLINE #E2 (runs/g9qent/PREREG.txt READING), run inside an INLINE_CMD of the g9
# job after the arm's trainer stopped at its first checkpoint Sa >= S0 + 50M (nv2_pause, LOCAL exported, cwd $NV2), from
# a node-local md5-checked copy:  QE_QUEUED=<queued INLINE file> QE_PARK=<park.sh copy> bash inline-e2.sh
# 1. park the arm (park.sh -> runs/g9qent-parked; v18 then HOLDs); 2. arm-cd from the arm's checkpoint Sa; 3. G9q-cd from
# G9q's first mirrored checkpoint Sq >= S0 + 50M; both: 5M cosine to x0.1, --guard-entropy-min 0.35, own dirs, pool
# snapshots / frozen bests > S moved out, through py-src-8892762 with the arm's recorded argv (G9q-cd: minus the eight
# --ent-* tokens). Finals runs/g9qent-cd/final/, runs/g9qent-g9qcd/final/ (r--, md5 in runs/g9qent-chain/STATUS).
set -u
C=${NV2:-$PWD}; E=$C/runs/g9qent; K=$C/runs/g9qent-chain; W=$C/bin/py-src-8892762
mkdir -p "$K"
say() { echo "$(date -u +%FT%T) E2 $*" | tee -a "$K/STATUS"; }
fail() { say "FAILED: $*"; echo "$(date -u +%FT%T) E2 $*" >> "$K/FAILED"; exit 1; }
[ -n "${LOCAL:-}" ] || fail "LOCAL unset (run me only through nv2_pause)"
QUEUED=${QE_QUEUED:?}; PARK=${QE_PARK:?}
[ "$(cat "$C/runs/g9/MODE" 2>/dev/null)" = g9qent ] || fail "runs/g9/MODE is not g9qent"
[ -s "$K/arm-argv.bin" ] || fail "no runs/g9qent-chain/arm-argv.bin (the chain records the arm's argv before the SIGINT)"
[ -x "$W" ] || fail "no $W"
read -r FP S0 FM < "$E/FORK_POINT" || fail "no runs/g9qent/FORK_POINT"
T=$((S0 + 50000000)); L=$LOCAL/run/G9qent
export PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
first_ge() {  # first_ge THRESHOLD FILES...: the file of the smallest ckpt step >= THRESHOLD
  local th=$1 f n; shift
  for f in "$@"; do [ -e "$f" ] || continue; n=$(basename "$f" .pt); n=${n#ckpt_}; case $n in *[!0-9]*|"") continue ;; esac
    if [ "$((10#$n))" -ge "$th" ]; then echo "$((10#$n)) $f"; fi; done | sort -n | head -n 1 | cut -d' ' -f2; }
say "start in job ${SLURM_JOB_ID:-?}: S0 $S0, reading point $T${QE_PARK_ONLY:+ (PARK ONLY: a guard trip, no reading)}"
# ---- 1. park the arm (its state secured before the branches) ----
ss=$(PARK_LIN=G9qent PARK_STATE=runs/g9qent PARK_DEST=runs/g9qent-parked PARK_QUEUED="$QUEUED" \
  PARK_WHY="entropy-schedule arm ended at its reading point (runs/g9qent/PREREG.txt); A40 placement = orchestrator / user" \
  bash "$PARK" 2>&1 | tee -a "$K/STATUS" | tail -n 1)
if [ -f "$C/runs/g9qent-parked/PARKED.txt" ]; then say "arm parked at $ss steps (runs/g9qent-parked/PARKED.txt)"
else say "ALERT: the arm park failed (runs/g9qent/park.log); the branches still run; MODE g9qent stays (the arm would continue)"; mkdir -p "$K/park"; echo "$(date -u +%FT%T) arm park failed" >> "$K/park/FAILED"; fi
if [ -n "${QE_PARK_ONLY:-}" ]; then say "park only (a guard trip of the arm: no reading)"; exit 0; fi
CKa=$(first_ge "$T" "$L"/checkpoints/ckpt_*.pt); [ -n "$CKa" ] || fail "no arm checkpoint >= $T in $L/checkpoints"
CKq=$(first_ge "$T" "$C"/runs/r1/*/G9q/checkpoints/ckpt_*.pt); [ -n "$CKq" ] || fail "no mirrored G9q checkpoint >= $T"
Sa=$(basename "$CKa" .pt); Sa=$((10#${Sa#ckpt_})); Sq=$(basename "$CKq" .pt); Sq=$((10#${Sq#ckpt_}))
GQd=$(dirname "$(dirname "$CKq")")
say "branch points: arm checkpoint $CKa (Sa $Sa), G9q checkpoint $CKq (Sq $Sq)"
mapfile -d '' -t A0 < "$K/arm-argv.bin"
branch() {  # branch NAME CK S POOLSRC LINK(1 = hard-link a local pool) STRIP_ENT(1)
  local nm=$1 ck=$2 s=$3 src=$4 link=$5 strip=$6 B=$LOCAL/run/$1 O=$C/runs/$1 a=() i x skip=0 f n b st fn
  mkdir -p "$O"; rm -rf "$B"; mkdir -p "$B/pool-after-S"
  if [ "$link" = 1 ]; then cp -al "$src" "$B/pool" || return 1
  else mkdir -p "$B/pool/frozen"
    for f in "$src"/*.pt; do [ -e "$f" ] || continue; n=$(basename "$f" .pt); case $n in *[!0-9]*|"") continue ;; esac
      if [ "$((10#$n))" -le "$s" ]; then cp -p "$f" "$B/pool/.$n.tmp" && mv -f "$B/pool/.$n.tmp" "$B/pool/$n.pt" || return 1; fi; done
    for f in "$src"/frozen/*.pt; do [ -e "$f" ] || continue; cp -p "$f" "$B/pool/frozen/" || return 1; done
  fi
  for f in "$B"/pool/*.pt "$B"/pool/frozen/best-*.pt; do [ -e "$f" ] || continue; n=$(basename "$f" .pt); n=${n#best-}; case $n in *[!0-9]*|"") continue ;; esac
    if [ "$((10#$n))" -gt "$s" ]; then mv -f "$f" "$B/pool-after-S/"; fi; done
  for ((i = 1; i < ${#A0[@]}; i++)); do
    x=${A0[$i]}
    if [ "$skip" -gt 0 ]; then skip=$((skip - 1)); continue; fi
    if [ "$strip" = 1 ]; then case $x in --ent-schedule|--ent-final|--ent-start|--ent-horizon) skip=1; continue ;; esac; fi
    a+=("$x")
  done
  b=$O/branch-point-$(printf %011d "$s").pt; cp -p "$ck" "$b.tmp" && mv -f "$b.tmp" "$b" || return 1
  printf '%s ' "$W" "${a[@]}" --run-dir "$B" --pool-dir "$B/pool" --resume "$b" --cooldown-start "$s" --total-steps "$((s + 5000000))" --final-lr-frac 0.1 --guard-entropy-min 0.35 > "$O/argv.txt"; echo >> "$O/argv.txt"
  say "$nm: branch point $b (S $s, md5 $(md5sum < "$b" | cut -c1-8)), pool $(ls "$B"/pool/*.pt 2>/dev/null | wc -l) + frozen $(ls "$B"/pool/frozen/*.pt 2>/dev/null | wc -l), $(grep -c -- '--ent-schedule' "$O/argv.txt") x --ent-schedule"
  timeout -s INT -k 300 7200 "$W" "${a[@]}" --run-dir "$B" --pool-dir "$B/pool" --resume "$b" --cooldown-start "$s" --total-steps "$((s + 5000000))" --final-lr-frac 0.1 --guard-entropy-min 0.35 >> "$B/stdout.log" 2>&1
  say "$nm: exit $?; $(grep "saved final_model" "$B/train.log" 2>/dev/null | tail -n 1)"
  [ -f "$B/final_model.pt" ] || { say "$nm: no final_model.pt"; return 1; }
  st=$(grep "saved final_model.pt at" "$B/train.log" | tail -n 1 | grep -oE "[0-9,]+ steps" | tr -dc 0-9)
  [ -n "$st" ] && [ "$st" -ge "$((s + 5000000))" ] || { say "$nm: ended at '${st:-?}' < S + 5M (cut short)"; return 1; }
  mkdir -p "$O/final"; fn=$O/final/$(printf %011d "$st").pt
  cp -p "$B/final_model.pt" "$fn.tmp" && mv -f "$fn.tmp" "$fn" && chmod a-w "$fn" || return 1
  cp -p "$B/train.log" "$O/branch-train.log"
  say "$nm final at $st steps -> $fn (md5 $(md5sum < "$fn" | cut -c1-8)); lr at the end: $(grep -oE "lr [0-9.e-]+" "$B/train.log" | tail -n 1); ent at the end: $(grep " upd " "$B/train.log" | tail -n 1 | grep -oE "ent [0-9.]+")"
  rm -rf "$B/pool" "$B/pool-after-S"; return 0; }
branch g9qent-cd "$CKa" "$Sa" "$L/pool" 1 0 || fail "arm-cd branch failed"
branch g9qent-g9qcd "$CKq" "$Sq" "$GQd/pool" 0 1 || fail "G9q-cd branch failed"
say "INLINE #E2 done"
