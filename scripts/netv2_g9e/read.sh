#!/bin/bash
# netv2 2026-11-20: the G9E readings at K = 100 | 250 (runs/g9e/PREREG.txt): run by netv2.env v21 (MODE g9e,
# nv2_g9e_hold) inside the g9 job once the cooled exploiter runs/g9e/cdK/final/*.pt exists and no trainer runs.
# In-process duplicate pairs (wsp's spd_h2h via scripts/netv2_spd.sh, v2 src src-p2, 8 workers, cuda), 2000 pairs each
# in 20 shards of 100 (pair ids 0..1999; a cut-off reading resumes at the first missing shard; same pairs as one run):
#   E1 (decides):  G9E-cd@K vs cd@1350, both T 0.5, seed 9150101
#   info:          E1 at T 1.0 (seed 9150102); vs G9q's newest checkpoint at the reading (T 0.5, seed 9150103);
#                  vs cd@805 (T 0.5, seed 9150104); G9E's in-training eval every 25M (train.log '[eval @' lines)
# Rule (pre-registered, fixed): lower / upper = pair-mean 95% interval of A's win rate (spd_h2h merge).
#   K = 100: lower > 55 => EXPLOITABLE; upper < 55 => ROBUST; else CONTINUE (train G9E to 250M, cooldown, read again)
#   K = 250: lower > 55 => EXPLOITABLE; else ROBUST
# Writes runs/g9e/rK/{STATUS, NETS.txt, READING.txt (r--, as soon as E1 is in), INFO.txt, DONE}, the top-level
# runs/g9e/READING.txt (r--, rebuilt from rK) and, on a final verdict, runs/g9e/DONE. A failure: runs/g9e/FAILED, exit 1.
set -uo pipefail; export LC_ALL=C  # decimal points in printf / awk whatever the node locale
K=${1:?usage: read.sh 100|250}; case $K in 100|250) ;; *) echo "bad K $K" >&2; exit 2 ;; esac
C=${NV2:?}; E=$C/runs/g9e; R=$E/r$K; mkdir -p "$R"
B1350=$C/runs/g9q-cd1350/final/01355036714.pt; B1350_MD5=6cf40c0c; B805=$C/runs/g9q-cd800/final/00805029873.pt; B805_MD5=d0c5b6cf
SEED_E1=9150101; SEED_T1=9150102; SEED_Q=9150103; SEED_805=9150104; NSH=20; SHP=100; S0=1355036714
SPD=${G9E_SPD:-$C/scripts/netv2_spd.sh}; SPD_MD5=${G9E_SPD_MD5:-afc36cbe}
MERGE=${G9E_MERGE:-/vol/bitbucket/fr1226/gloria/.venv/bin/python $C/vendor/wsp-ee76994/spd_h2h.py merge}
JPY=${G9E_JPY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}; WORKERS=${G9E_WORKERS:-8}
st() { echo "$(date -u +%FT%T) ${SLURM_JOB_ID:-?} r$K: $*" | tee -a "$R/STATUS" >> "$E/STATUS"; }
fail() { st "FAILED: $*"; echo "$(date -u +%FT%T) ${SLURM_JOB_ID:-?} read.sh $K: $*" >> "$E/FAILED"; exit 1; }
md8() { md5sum < "$1" 2> /dev/null | cut -c1-8; }
[ -s "$E/PREREG.txt" ] && [ ! -w "$E/PREREG.txt" ] || fail "no read-only runs/g9e/PREREG.txt"
[ -f "$SPD" ] && [ "$(md8 "$SPD")" = "$SPD_MD5" ] || fail "$SPD missing or md5 not $SPD_MD5"
n=$(ls "$E/cd$K/final/"*.pt 2> /dev/null | wc -l); [ "$n" = 1 ] || fail "runs/g9e/cd$K/final holds $n nets (want 1)"
A=$(ls "$E/cd$K/final/"*.pt); read -r fa ma < "$E/cd$K/FINAL.md5" || true
[ "$(basename "$A")" = "${fa:-}" ] && [ "$(md8 "$A")" = "${ma:-}" ] || fail "$A does not match runs/g9e/cd$K/FINAL.md5 ('${fa:-} ${ma:-}')"
[ "$(md8 "$B1350")" = "$B1350_MD5" ] || fail "$B1350 md5 is not $B1350_MD5"
[ "$(md8 "$B805")" = "$B805_MD5" ] || fail "$B805 md5 is not $B805_MD5"
if [ ! -s "$R/NETS.txt" ]; then  # chosen once per reading (a resumed reading keeps the same G9q checkpoint)
  Q=$(ls "$C"/runs/r1/*/G9q/checkpoints/ckpt_*.pt 2> /dev/null | awk -F/ '{print $NF " " $0}' | sort | tail -n 1 | cut -d' ' -f2)
  [ -n "$Q" ] || fail "no G9q checkpoint under runs/r1/*/G9q/checkpoints/"
  { echo "A $A $(md8 "$A")"; echo "B1350 $B1350 $B1350_MD5"; echo "Q $Q $(md8 "$Q")"; echo "B805 $B805 $B805_MD5"; } > "$R/NETS.txt.tmp" && mv -f "$R/NETS.txt.tmp" "$R/NETS.txt"
fi
grep -q "^A $A $(md8 "$A")$" "$R/NETS.txt" || fail "NETS.txt names another A: $(head -n 1 "$R/NETS.txt")"
read -r _ Q QM <<< "$(grep '^Q ' "$R/NETS.txt")"
[ -f "$Q" ] && [ "$(md8 "$Q")" = "$QM" ] || fail "the G9q checkpoint $Q (NETS.txt) is gone or changed"
st "start on $(hostname -s 2> /dev/null): A = $A; nets in NETS.txt"
play() {  # play NAME SPEC_A SPEC_B SEED: 20 shards of 100 pairs (existing shards are kept), then the merged summary
  local d=$R/$1 j f
  mkdir -p "$d"
  for ((j = 0; j < NSH; j++)); do
    f=$(printf '%s/shard-%04d.jsonl' "$d" "$j"); [ -f "$f" ] && continue
    bash "$SPD" run --a "$2" --b "$3" --pairs "$SHP" --shard "$j" --seed "$4" --out "$d" --format gen9randombattle \
      --workers "$WORKERS" --v2-src "$C/src-p2" --device cuda --progress 0 --overwrite >> "$R/$1.log" 2>&1 || fail "$1 shard $j exited $? ($R/$1.log)"
    [ -f "$f" ] || fail "$1 shard $j wrote no $f"
  done
  $MERGE "$d" --out "$d/SUMMARY.json" > "$d/MERGE.txt" 2>&1 || fail "$1: merge failed ($d/MERGE.txt)"
  "$JPY" -c "
import json, sys; s = json.load(open('$d/SUMMARY.json'))
assert s['pairs'] == $((NSH * SHP)) and s['missing_in_range'] == 0 and s['pair_range'] == [0, $((NSH * SHP))], (s['pairs'], s['pair_range'])
w, h = 100 * s['win_rate'], 100 * s['ci95']; print('%.4f %.4f %.4f %d %d' % (w, w - h, w + h, s['pairs'], s['void_games']))" > "$d/EST.txt" 2>> "$d/MERGE.txt" || fail "$1: summary check failed ($d/MERGE.txt)"
  st "$1 done: $(cat "$d/EST.txt") (estimate lower upper pairs void) $(grep -m1 'A win rate' "$d/MERGE.txt" | cut -c1-140)"
}
est() { read -r w lo hi np nv < "$R/$1/EST.txt"; printf '%.2f%% [%.2f, %.2f] (pair-mean 95%%, %d pairs = %d games, %d void)' "$w" "$lo" "$hi" "$np" "$((2 * np))" "$nv"; }
# --- E1 (decides)
play E1 "v2:$A,temperature=0.5" "v2:$B1350,temperature=0.5" "$SEED_E1"
if [ ! -f "$R/READING.txt" ]; then
  read -r w lo hi _ < "$R/E1/EST.txt"
  v=$(awk -v lo="$lo" -v hi="$hi" -v k="$K" 'BEGIN { if (lo > 55) print "EXPLOITABLE"; else if (k == 100 && !(hi < 55)) print "CONTINUE"; else print "ROBUST" }')
  case $v in
    EXPLOITABLE) why="lower 95% $lo > 55" ;;
    ROBUST) if [ "$K" = 100 ]; then why="upper 95% $hi < 55"; else why="lower 95% $lo <= 55 (the 250M reading is final)"; fi ;;
    CONTINUE) why="55 inside [$lo, $hi]: train G9E to 250M (+ cooldown) and read once more (same pairs and seed)" ;;
  esac
  { echo "G9E E1@${K}M reading (runs/g9e/PREREG.txt $(md8 "$E/PREREG.txt")), read mechanically by scripts/netv2_g9e/read.sh $(md8 "$0")"
    echo "E1@${K}M VERDICT $v ($why)"
    echo "A = G9E-cd@${K}M $(basename "$A") ($(md8 "$A")) vs B = cd@1350 01355036714.pt ($B1350_MD5); both T 0.5; seed $SEED_E1; 2000 pairs (ids 0..1999)"
    echo "E1@${K}M $(est E1)"
    grep -m1 'A win rate' "$R/E1/MERGE.txt" || true; } > "$R/READING.txt.tmp" && mv -f "$R/READING.txt.tmp" "$R/READING.txt" && chmod a-w "$R/READING.txt" || fail "could not write READING.txt"
  st "READING: $(sed -n 2p "$R/READING.txt")"
fi
# --- information (no decision)
play T1 "v2:$A,temperature=1.0" "v2:$B1350,temperature=1.0" "$SEED_T1"
play Q "v2:$A,temperature=0.5" "v2:$Q,temperature=0.5" "$SEED_Q"
play B805 "v2:$A,temperature=0.5" "v2:$B805,temperature=0.5" "$SEED_805"
curve=$( { ls "$C"/runs/g9/*/G9E/train.log ${LOCAL:+"$LOCAL/run/G9E/train.log"} 2> /dev/null || true; } | xargs -r grep -h '\[eval @ ' 2> /dev/null \
  | sed -n 's/.*\[eval @ \([0-9,]*\)\]\(.*\)/\1 \2/p' | tr -d ',' | sort -n -k1,1 -u \
  | awk -v s0="$S0" -v k="$K" '{ for (m = 0; m <= k; m += 25) if (!(m in got) && $1 >= s0 + m * 1000000) { got[m] = 1; printf "  G9E +%dM: [eval @ %d]%s\n", m, $1, substr($0, index($0, " ")) } }' )
{ echo "G9E information @${K}M (no decision; runs/g9e/PREREG.txt)"
  echo "E1 at T 1.0 (training condition) vs cd@1350, seed $SEED_T1: $(est T1)"
  echo "vs G9q's newest checkpoint $(basename "$Q") ($QM; uncooled), T 0.5, seed $SEED_Q: $(est Q)"
  echo "vs cd@805 00805029873.pt ($B805_MD5), T 0.5, seed $SEED_805: $(est B805)"
  echo "specificity: E1 $(read -r w _ < "$R/E1/EST.txt"; echo "$w") vs G9q-now $(read -r w _ < "$R/Q/EST.txt"; echo "$w") vs cd@805 $(read -r w _ < "$R/B805/EST.txt"; echo "$w") (a cd@1350-specific gain is large in E1 and small in the other two)"
  echo "G9E in-training eval (vs anchor P1 / baselines) every 25M after S0 = $S0:"; echo "${curve:-  (no eval lines found)}"
} > "$R/INFO.txt.tmp" && mv -f "$R/INFO.txt.tmp" "$R/INFO.txt" || fail "could not write INFO.txt"
v=$(sed -n 's/^E1@[0-9]*M VERDICT \([A-Z]*\).*/\1/p' "$R/READING.txt")
if [ "$v" = CONTINUE ]; then top="INTERIM: CONTINUE — G9E trains on to 250M (+ cooldown); E1@250M will decide (runs/g9e/r250/)"
else top="FINAL: $v (E1@${K}M; runs/g9e/r$K/READING.txt)"; fi
{ echo "G9E READING (runs/g9e/PREREG.txt $(md8 "$E/PREREG.txt")), written mechanically by scripts/netv2_g9e/read.sh $(md8 "$0")"
  echo "$top"
  for k in 100 250; do [ -f "$E/r$k/READING.txt" ] || continue; echo "--- E1@${k}M:"; cat "$E/r$k/READING.txt"; [ -f "$E/r$k/INFO.txt" ] && { echo "--- information @${k}M:"; cat "$E/r$k/INFO.txt"; }; done
} > "$E/READING.txt.tmp" || fail "could not build READING.txt"
chmod u+w "$E/READING.txt" 2> /dev/null; mv -f "$E/READING.txt.tmp" "$E/READING.txt" && chmod a-w "$E/READING.txt" || fail "could not publish READING.txt"
date -u +%FT%T > "$R/DONE"
if [ "$v" != CONTINUE ]; then echo "$(date -u +%FT%T) $top" > "$E/DONE"; fi
st "done: $top"
