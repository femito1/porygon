#!/bin/bash
# caps-check OUTDIR: the field-wise check of argsdiff-sweep.sh's kept outputs (netv2.env v21 -> v22). Per case: old and
# new list the same job/GPU lines in the same order; per line every field but TOTAL is byte-identical; TOTAL differs
# exactly where v22 raises a cap and nowhere else:
#   r1 lines training lineage G9q (--run-dir .../run/G9q; v21 TOTAL 3e9)   3000000000 -> 5000000000
#   g9 lines training lineage G9P4 (--run-dir .../run/G9P4)                1000000000 -> 2000000000
#   g4r1bf lines (R1b-F, --run-dir .../run/R1bF)                           1000000000 -> 2000000000
# (an empty line = a source that exited before printing, e.g. a HOLD round with the mock cap: must be empty in both).
# PASS needs 0 failures and each of the three changes exercised at least once.
set -u
OUT=${1:?OUTDIR}; fails=0; cases=0; nq=0; np4=0; nbf=0; nsame=0; nempty=0
tot() { grep -oE '\|TOTAL=[^|]*\|' <<< "$1" | head -n 1 | tr -d '|' | cut -d= -f2; }
for d in "$OUT"/*/; do
  d=${d%/}; c=${d##*/}
  if [ ! -f "$d/old.txt" ] || [ ! -f "$d/new.txt" ]; then echo "FAIL $c: no kept old.txt / new.txt"; fails=$((fails + 1)); continue; fi
  cases=$((cases + 1)); f=0; q=0; p4=0; bf=0; same=0; empty=0
  if [ "$(wc -l < "$d/old.txt")" != "$(wc -l < "$d/new.txt")" ] || [ ! -s "$d/old.txt" ]; then echo "FAIL $c: line counts differ or empty output"; fails=$((fails + 1)); continue; fi
  while IFS= read -r a <&3 && IFS= read -r b <&4; do
    ka=${a%%: *}; kb=${b%%: *}; run=$(cut -d' ' -f2 <<< "$ka")
    if [ "$ka" != "$kb" ]; then echo "FAIL $c: job lines out of order: '$ka' / '$kb'"; f=1; continue; fi
    case $a in *"|TOTAL="*) ;; *) if [ "$a" = "$b" ]; then empty=$((empty + 1)); else echo "FAIL $c [$ka]: empty in one version only"; f=1; fi; continue ;; esac
    if [ "$(sed -E 's/\|TOTAL=[^|]*\|/|TOTAL=*|/' <<< "$a")" != "$(sed -E 's/\|TOTAL=[^|]*\|/|TOTAL=*|/' <<< "$b")" ]; then
      echo "FAIL $c [$ka]: a field other than TOTAL differs"; diff <(tr '|' '\n' <<< "$a") <(tr '|' '\n' <<< "$b") | sed 's/^/    /' | head -n 8; f=1; continue
    fi
    ta=$(tot "$a"); tb=$(tot "$b"); want=""
    case "$run:$b" in
      r1:*"/run/G9q "*) want="3000000000 5000000000"; q=$((q + 1)) ;;
      g9:*"/run/G9P4 "*) want="1000000000 2000000000"; p4=$((p4 + 1)) ;;
      g4r1bf:*"/run/R1bF "*) want="1000000000 2000000000"; bf=$((bf + 1)) ;;
      g4r1bf:*) echo "FAIL $c [$ka]: an R1b-F line without --run-dir .../run/R1bF"; f=1; continue ;;
      *) want="$ta $ta"; same=$((same + 1)) ;;
    esac
    if [ "$ta $tb" != "$want" ]; then echo "FAIL $c [$ka]: TOTAL $ta -> $tb, expected ${want/ / -> }"; f=1; fi
  done 3< "$d/old.txt" 4< "$d/new.txt"
  if [ "$f" = 0 ]; then echo "ok   $c: $(wc -l < "$d/new.txt") lines: G9q $q, G9-P4 $p4, R1b-F $bf raised as intended; $same unchanged; $empty empty in both"; else fails=$((fails + 1)); fi
  nq=$((nq + q)); np4=$((np4 + p4)); nbf=$((nbf + bf)); nsame=$((nsame + same)); nempty=$((nempty + empty))
done
if [ "$nq" = 0 ] || [ "$np4" = 0 ] || [ "$nbf" = 0 ]; then echo "FAIL: a change was never exercised (G9q $nq, G9-P4 $np4, R1b-F $nbf lines)"; fails=$((fails + 1)); fi
echo "cases $cases, failed $fails; lines: G9q $nq / G9-P4 $np4 / R1b-F $nbf raised as intended, $nsame unchanged, $nempty empty in both"
[ "$fails" = 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
