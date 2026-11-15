#!/bin/bash
# netv2 2026-11-15: bf16 A/B INLINE #1 (runs/g9qbf/PREREG.txt steps 2-4), run inside an INLINE_CMD of the g9 job (A40,
# nv2_pause: G9q2's trainer stopped, LOCAL exported, cwd $NV2), from a node-local md5-checked copy:
#   QBF_QUEUED=<queued INLINE file> QBF_PARK=<park.sh copy> bash inline1.sh
# 1. park G9q2 (park.sh) -> runs/g9q2-parked; 2. fork point = G9q's newest mirrored checkpoint -> runs/g9qbf/fork-point-
# <S0>.pt + FORK_POINT ("PATH S0 MD5"); 3. stage its pool (<= S0) + best / ema in $LOCAL/run/G9qbf (.staged-S0);
# 4. microbench bf16 then fp32 (G9q2's live argv minus --seed, minus --no-bf16 for bf16; own dirs, hard-linked pool,
# --total-steps S0 + 1.1M, no evals / snapshots / checkpoints, 0.75 h) -> runs/g9qbf-bench/RESULTS.txt; GO (median sps
# bf16 >= 1.15 x fp32, >= 20 measured updates each) => runs/g9/MODE = g9qbf; SKIP => runs/g9qbf/SKIP.txt (MODE stays
# g9q2: parked => v17 HOLDs). A failure before the park changes nothing (G9q2 relaunches); after it, MODE stays g9q2.
set -u
C=${NV2:-$PWD}; B=$C/runs/g9qbf-bench; F=$C/runs/g9qbf; Q2=$C/runs/g9q2; W=$C/bin/py-src-24c633f
GPY=${QBF_PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}  # QBF_PY: the laptop mock only
mkdir -p "$B" "$F"
say() { echo "$(date -u +%FT%T) $*" | tee -a "$B/STATUS"; }
fail() { say "FAILED: $*"; echo "$(date -u +%FT%T) $*" >> "$B/FAILED"; exit 1; }
[ -n "${LOCAL:-}" ] || fail "LOCAL unset (run me only through nv2_pause)"
QUEUED=${QBF_QUEUED:?}; PARK=${QBF_PARK:?}
[ "$(cat "$C/runs/g9/MODE" 2>/dev/null)" = g9q2 ] || fail "runs/g9/MODE is '$(cat "$C/runs/g9/MODE" 2>/dev/null)', not g9q2"
[ -s "$B/g9q2-argv.bin" ] || fail "no runs/g9qbf-bench/g9q2-argv.bin (the chain records G9q2's live argv before the SIGINT)"
export PORYGON_PS=/vol/bitbucket/fr1226/porygon-ws-fastenv/ps-2026-11
say "INLINE #1 start in job ${SLURM_JOB_ID:-?} on $(hostname -s) ($(timeout 20 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1))"
# ---- 1. park G9q2 (orchestrator placement decision) ----
ss=$(PARK_LIN=G9q2 PARK_STATE=runs/g9q2 PARK_DEST=runs/g9q2-parked PARK_QUEUED="$QUEUED" \
  PARK_WHY="placement decision (orchestrator 2026-11-15 ~20:05, item 3; not a rule deviation): after G9q2's R50 / OVERALL readings; the A40 runs the bf16 A/B (runs/g9qbf/PREREG.txt)" \
  bash "$PARK" 2>&1 | tee -a "$B/STATUS" | tail -n 1)
[ -f "$C/runs/g9q2-parked/PARKED.txt" ] || fail "G9q2 park failed (G9q2 relaunches from its own save; see runs/g9q2/park.log)"
echo "$(date -u +%FT%T) PLACEMENT DECISION (orchestrator 2026-11-15 ~20:05, item 3; not a rule deviation): G9q2 parked at $ss steps in runs/g9q2-parked/ (PARKED.txt + MD5SUMS) after R50 / OVERALL; the A40 job runs the bf16 A/B on G9q's recipe (runs/g9qbf/PREREG.txt)" >> "$Q2/notes.log"
say "G9q2 parked at $ss steps (runs/g9q2-parked/PARKED.txt); from here MODE g9q2 => HOLD (v17)"
# ---- 2. fork point ----
GQ=$(ls -td "$C"/runs/r1/*/G9q 2>/dev/null | head -n 1); [ -d "$GQ/pool" ] || fail "no mirrored G9q lineage dir"
CK=$(ls -t "$GQ"/checkpoints/ckpt_*.pt 2>/dev/null | head -n 1); [ -n "$CK" ] || fail "no mirrored G9q checkpoint in $GQ"
S0=$(basename "$CK" .pt); S0=$((10#${S0#ckpt_}))
FP=$F/fork-point-$(printf %011d "$S0").pt
cp -p "$CK" "$FP.tmp" && mv -f "$FP.tmp" "$FP" && chmod a-w "$FP" || fail "could not copy the fork point"
FM=$(md5sum < "$FP" | cut -c1-8); [ "$FM" = "$(md5sum < "$CK" | cut -c1-8)" ] || fail "fork point copy md5 mismatch"
echo "$GQ $CK" > "$F/FORK_POINT.src"; echo "$FP $S0 $FM" > "$F/FORK_POINT.tmp" && mv -f "$F/FORK_POINT.tmp" "$F/FORK_POINT" || fail "could not write FORK_POINT"
say "fork point: $CK -> $FP (S0 $S0, md5 $FM)"
# ---- 3. stage the fork's pool once (snapshots and frozen bests <= S0, P3 frozen; best / ema) ----
D=$LOCAL/run/G9qbf; rm -rf "$D"; mkdir -p "$D/pool/frozen" || fail "cannot create $D"
np=0; for f in "$GQ"/pool/*.pt; do [ -e "$f" ] || continue; n=$(basename "$f" .pt); case $n in *[!0-9]*|"") continue ;; esac
  if [ "$((10#$n))" -le "$S0" ]; then cp -p "$f" "$D/pool/.$n.tmp" && mv -f "$D/pool/.$n.tmp" "$D/pool/$n.pt" || fail "pool copy $f"; np=$((np + 1)); fi; done
nf=0; for f in "$GQ"/pool/frozen/*.pt; do [ -e "$f" ] || continue; b=$(basename "$f"); n=${b%.pt}; n=${n#best-}
  case $n in *[!0-9]*|"") ;; *) if [ "$((10#$n))" -gt "$S0" ]; then continue; fi ;; esac
  cp -p "$f" "$D/pool/frozen/$b" || fail "frozen copy $f"; nf=$((nf + 1)); done
for f in best_model.pt ema_model.pt; do if [ -f "$GQ/$f" ]; then cp -p "$GQ/$f" "$D/$f" || fail "copy $f"; fi; done
[ "$np" -gt 0 ] && [ "$nf" -gt 0 ] || fail "staged pool empty ($np snapshots, $nf frozen)"
touch "$D/.staged-$S0"; say "pool staged in $D: $np snapshots <= S0 + $nf frozen ($(du -sh "$D" | cut -f1)); best / ema copied"
# ---- 4. microbench: bf16 first, then fp32 ----
mapfile -d '' -t A0 < "$B/g9q2-argv.bin"
arm() {  # arm NAME: one trainer run; leaves $B/NAME-train.log and $B/NAME-argv.txt
  local nm=$1 a=() i x skip="" R=$LOCAL/bench-$1
  for ((i = 1; i < ${#A0[@]}; i++)); do
    x=${A0[$i]}
    if [ -n "$skip" ]; then skip=""; continue; fi
    if [ "$x" = --seed ]; then skip=1; continue; fi
    if [ "$x" = --no-bf16 ] && [ "$nm" = bf16 ]; then continue; fi
    a+=("$x")
  done
  a+=(--run-dir "$R/run" --pool-dir "$R/pool" --resume "$FP" --total-steps "$((S0 + 1100000))" --eval-every 1000000000 --snapshot-every 1000000000 --checkpoint-every 1000000000 --max-hours 0.75)
  rm -rf "$R"; mkdir -p "$R/run"; cp -al "$D/pool" "$R/pool" || return 1
  printf '%s ' "$W" "${a[@]}" > "$B/$nm-argv.txt"; echo >> "$B/$nm-argv.txt"
  say "arm $nm: start ($(grep -c -- '--no-bf16' <<< "$(tr ' ' '\n' < "$B/$nm-argv.txt")") x --no-bf16 in its argv)"
  timeout -s INT -k 300 3300 "$W" "${a[@]}" >> "$R/stdout.log" 2>&1; local rc=$?
  cp -p "$R/run/train.log" "$B/$nm-train.log" 2>/dev/null; tail -n 30 "$R/stdout.log" > "$B/$nm-stdout-tail.log" 2>/dev/null
  say "arm $nm: exit $rc; $(grep -c ' upd ' "$B/$nm-train.log" 2>/dev/null) updates"
  rm -rf "$R"; return 0
}
arm bf16 || fail "arm bf16 could not start"
arm fp32 || fail "arm fp32 could not start"
res=$("$GPY" - "$B/bf16-train.log" "$B/fp32-train.log" <<'PY' 2>&1
import re, statistics as S, sys
out = {}
for name, path in (("bf16", sys.argv[1]), ("fp32", sys.argv[2])):
    rows = []
    for ln in open(path, errors="replace"):
        if " upd " not in ln: continue
        m = {k: re.search(rf"(?:^| ){k} ([0-9.e+-]+)", ln) for k in ("sps", "collect_sps", "upd_s")}
        n = re.search(r"\(\+([0-9]+)\)", ln)
        if all(m.values()) and n:
            rows.append((float(m["sps"].group(1)), float(m["collect_sps"].group(1)), float(m["upd_s"].group(1)), int(n.group(1))))
    meas = rows[3:]
    out[name] = meas
    if len(meas) < 20:
        print(f"{name}: only {len(meas)} measured updates (of {len(rows)}): FAIL"); continue
    sps = [r[0] for r in meas]; col = [r[3] / r[1] for r in meas]; upd = [r[2] for r in meas]
    print(f"{name}: {len(meas)} measured updates (of {len(rows)}; first 3 = warm-up): sps median {S.median(sps):.0f} mean {S.mean(sps):.0f}; "
          f"collect s/update median {S.median(col):.1f}; update s median {S.median(upd):.1f}; steps/update median {S.median([r[3] for r in meas]):.0f}")
ok = all(len(out[k]) >= 20 for k in ("bf16", "fp32"))
if ok:
    rb, rf = S.median([r[0] for r in out["bf16"]]), S.median([r[0] for r in out["fp32"]])
    print(f"RATIO median sps bf16 / fp32 = {rb:.0f} / {rf:.0f} = {rb / rf:.3f} (threshold 1.15)")
    print("DECISION " + ("GO" if rb >= 1.15 * rf else "SKIP"))
else:
    print("DECISION FAIL")
PY
)
say "bench: $(tr '\n' ' ' <<< "$res")"
dec=$(sed -n 's/^DECISION //p' <<< "$res" | tail -n 1)
{ echo "bf16 speed microbench (runs/g9qbf/PREREG.txt step 3), job ${SLURM_JOB_ID:-?} on $(hostname -s); fork point $FP (S0 $S0, md5 $FM); G9q's recipe on this GPU (G9q2's argv minus --seed); arms bf16 then fp32"
  echo "$res"; } > "$B/RESULTS.txt.tmp" && mv -f "$B/RESULTS.txt.tmp" "$B/RESULTS.txt" && chmod a-w "$B/RESULTS.txt"
case $dec in
  GO) cp -p "$C/runs/g9/MODE" "$C/runs/g9/MODE.g9q2-before-g9qbf"; echo g9qbf > "$C/runs/g9/MODE.tmp" && mv -f "$C/runs/g9/MODE.tmp" "$C/runs/g9/MODE" || fail "could not write MODE g9qbf"
      say "GO: runs/g9/MODE = g9qbf (the fork starts at this relaunch from $FP with the staged pool)" ;;
  SKIP) { echo "bf16 A/B SKIP (runs/g9qbf/PREREG.txt step 3): bf16 is not >= 1.15x faster on this A40; G9q stays fp32; G9q2 parked (MODE g9q2 => HOLD, allocation kept); netv2 proposes an entropy-schedule arm for the A40"
          echo "$res"; } > "$F/SKIP.txt" && chmod a-w "$F/SKIP.txt"
        rm -rf "$D"; say "SKIP: runs/g9qbf/SKIP.txt; staged pool removed; MODE stays g9q2 (parked => HOLD)" ;;
  *) fail "bench did not decide: $(tr '\n' ' ' <<< "$res")" ;;
esac
say "INLINE #1 done"
