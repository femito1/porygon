#!/bin/bash
# netv2 2026-11-20: POINTER PARK of G9-P4 (lineage G9P4) for the G9E arm (runs/g9e/PREREG.txt). Called by netv2.env v21
# (MODE g9e, nv2_g9e_hold) once, at G9E's first start in the g9 job that follows G9-P4's last job, before any G9E trainer.
# Unlike scripts/netv2_g9qbf/park.sh it copies nothing (the cluster disk is ~full): G9-P4's state stays where its last
# job mirrored it (runs/g9/<job>/G9P4/: final_model.pt or, if that save was lost, its newest 5M checkpoint; pool/ incl.
# frozen/, best / ema, train.log, tb/); this writes runs/g9p4-parked/{PARKED.txt, POINTER, MD5SUMS} (published by mv -T).
# netv2.env v19+ then HOLDs MODE g9p4 while runs/g9p4-parked/PARKED.txt exists. Resume (main / user decision): move
# runs/g9p4-parked aside, write MODE g9p4 (nv2_lineage_dirs G9P4 resumes the newest mirrored G9P4 file = the POINTER).
# Any failure: runs/g9e/FAILED (netv2.env v21 then HOLDs the g9 job round after round) and exit 1; nothing published.
set -u
C=${NV2:?}; P=$C/runs/g9p4-parked; T=$C/runs/g9p4; E=$C/runs/g9e
GPY=${NV2_PARK_PY:-/vol/bitbucket/fr1226/gloria/.venv/bin/python}; SRC=$C/src-24c633f
mkdir -p "$T" "$E"
say() { echo "$(date -u +%FT%T) ${SLURM_JOB_ID:-?} park-p4: $*" | tee -a "$T/park.log" >> "$E/STATUS"; }
fail() { say "FAIL (G9-P4 not parked): $*"; echo "$(date -u +%FT%T) ${SLURM_JOB_ID:-?} park-p4: $*" >> "$E/FAILED"; exit 1; }
if [ -e "$P" ]; then [ -f "$P/PARKED.txt" ] && { say "already parked ($P/PARKED.txt)"; exit 0; }; fail "$P exists without PARKED.txt"; fi
others=$(squeue -h -u "${USER:-$(id -un)}" -n ws-netv2-g9 -t RUNNING -o %i 2> /dev/null | grep -vx "${SLURM_JOB_ID:-none}" || true)
[ -z "$others" ] || fail "another ws-netv2-g9 job is RUNNING ($others): G9-P4 may still train"
F=$(ls -t "$C"/runs/g9/*/G9P4/final_model.pt "$C"/runs/g9/*/G9P4/checkpoints/ckpt_*.pt 2> /dev/null | head -n 1 || true)
[ -n "$F" ] || fail "no mirrored G9P4 state under runs/g9/*/G9P4/"
case $F in
  */G9P4/final_model.pt) D=${F%/final_model.pt}; KIND=final
    l0=$(grep -n "v2 trainer:" "$D/train.log" 2> /dev/null | tail -n 1 | cut -d: -f1)
    ss=$(tail -n +"${l0:-1}" "$D/train.log" 2> /dev/null | grep -oE 'saved final_model.pt at [0-9,]+ steps' | tail -n 1 | tr -dc 0-9)
    us=$(grep -oE ' upd [0-9]+ steps [0-9,]+' "$D/train.log" 2> /dev/null | tail -n 1 | sed 's/.* steps //' | tr -dc 0-9)
    cs=$(ls "$D/checkpoints" 2> /dev/null | grep -oE '^ckpt_[0-9]+\.pt$' | tr -dc '0-9\n' | sort -n | tail -n 1)
    [ -n "$ss" ] && [ "$ss" -ge "${us:-0}" ] && [ "$ss" -ge "$((10#${cs:-0}))" ] || fail "$F: its last save (${ss:-none}) is not the newest step (update ${us:-none}, checkpoint ${cs:-none})" ;;
  */G9P4/checkpoints/ckpt_*.pt) D=${F%/checkpoints/*}; KIND=checkpoint; ss=${F##*/ckpt_}; ss=$((10#${ss%.pt})) ;;
  *) fail "unexpected newest file $F" ;;
esac
ck=$(PYTHONPATH=$SRC timeout 900 "$GPY" -c "
import torch; ck = torch.load('$F', map_location='cpu', weights_only=False)
c = ck.get('counters', {}); assert 'optimizer' in ck and c.get('steps', 0) > 0, (sorted(ck), c)
print('steps', c['steps'])" 2>&1) || fail "$F does not load with optimizer + counters: $(tail -n 2 <<< "$ck" | tr '\n' ' ')"
ks=$(grep -oE '^steps [0-9]+' <<< "$ck" | tail -n 1 | tr -dc 0-9)
[ "$ks" = "$ss" ] || fail "$F: counters say steps ${ks:-?}, expected $ss"
rm -rf "$P.tmp"; mkdir -p "$P.tmp" || fail "cannot create $P.tmp"
( cd "$D" && for f in final_model.pt best_model.pt ema_model.pt "${F#"$D"/}"; do [ -f "$f" ] && md5sum "$f"; done | sort -u -k2 ) > "$P.tmp/MD5SUMS" || fail "md5sums failed"
echo "$F $ss $(md5sum < "$F" | cut -c1-8)" > "$P.tmp/POINTER"
{ echo "G9P4 PARKED $(date -u +%FT%T) in job ${SLURM_JOB_ID:-?} ($(hostname -s 2> /dev/null)) at $ss steps ($KIND): the A40 runs G9E (runs/g9e/PREREG.txt; orchestrator 11-20)"
  echo "pointer park (no copy, disk ~full): the state is $D/ (resume file $F, md5 $(cut -d' ' -f3 "$P.tmp/POINTER"); pool $(ls "$D/pool" 2> /dev/null | grep -c '\.pt$') nets, frozen $(ls "$D/pool/frozen" 2> /dev/null | grep -c '\.pt$'); md5s in MD5SUMS). Do not delete $D."
  echo "blocks: netv2.env v19+ HOLDs the g9 job round after round while this PARKED.txt exists and runs/g9/MODE is g9p4"
  echo "resume: main / user decision: mv runs/g9p4-parked aside, MODE g9p4 (nv2_lineage_dirs G9P4 resumes the newest mirrored G9P4 file, this POINTER). Never while it trains elsewhere."
} > "$P.tmp/PARKED.txt" || fail "cannot write PARKED.txt"
mv -T "$P.tmp" "$P" || fail "publish (mv -T $P.tmp $P) failed"
say "DONE: G9P4 parked at $ss steps ($KIND $F, md5 $(cut -d' ' -f3 "$P/POINTER"))"
