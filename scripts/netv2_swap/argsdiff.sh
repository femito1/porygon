#!/bin/bash
# argsdiff: source two netv2.env versions as the job script does (set -euo pipefail, restart in place for the running
# jobs, fresh first source for the successors) in a fake cluster tree and print FORMAT|ANCHOR|PY|PORYGON_PS|TRAIN_ARGS
# for 718701 r1, 810366 r1 (a30, a40), 741849 g9, 810367 g9 (a16, a40) and 829660 g4r1bf, then diff the two.
#   bash /tmp/nv2s/argsdiff.sh OLD NEW            current MODE files (r1 g9p, g9 g9q): must print "identical"
#   bash /tmp/nv2s/argsdiff.sh OLD NEW swapped    after a committed swap (r1 g9q, g9 g9p2): first starts + successors
set -u
OLD=$(readlink -f "$1"); NEW=$(readlink -f "$2"); WHAT=${3:-current}
R=${ARGSDIFF_ROOT:-/tmp/nv2s/.argsdiff-tree}
tree() {  # tree JOB: a fresh fake tree for one job
  rm -rf "$R"; N=$R/nv2; mkdir -p "$N/bin" "$N/runs/r1" "$N/runs/g9" "$N/runs/g9q" "$N/runs/g9a30" "$N/runs/g4r1bf" "$N/runs/pretrain" "$N/runs/r1b-parked/pool"
  for w in py-src-24c633f py-src-4f18108; do printf '#!/bin/bash\nexit 0\n' > "$N/bin/$w"; chmod +x "$N/bin/$w"; done
  echo p1 > "$N/runs/pretrain/P1-g9h-dmg-s2.pt"; echo p3 > "$N/runs/pretrain/P3-g9-s2.pt"
  echo g9p > "$N/runs/r1/MODE"; echo g9q > "$N/runs/g9/MODE"
  touch "$N/runs/r1/G9P_A30_STARTED" "$N/runs/g9q/G9Q_STARTED" "$N/runs/g9/G9S_A16_STARTED" "$N/runs/g4r1bf/R1BF_STARTED" "$N/runs/r1/G9_STARTED" "$N/runs/r1/LINEAGE_DIRS" "$N/runs/g9/LINEAGE_DIRS"
  echo x > "$N/runs/r1b-parked/final_model.pt"; echo b75 > "$N/runs/r1/restart-best-7.5M.pt"
  J=$N/runs/r1/20261108-220401-718701; K=$N/runs/g9/20261109-021316-741849
  for d in "$J/G9p" "$K/G9q" "$K/G9"; do mkdir -p "$d/pool/frozen" "$d/checkpoints"; echo m > "$d/final_model.pt"; echo b > "$d/best_model.pt"; done
  touch -d '2026-11-09 05:00' "$K/G9/final_model.pt"
  L=$R/local-$1; case $1 in 8103*) L=$R/tmp/ws-netv2-$1 ;; esac  # a fresh job's LOCAL (from TMPDIR + SLURM_JOB_ID)
  mkdir -p "$L/run"; touch "$L/run/train.log"
  case $1 in
    718701) mkdir -p "$L/run/G9p/pool" "$L/run/G9"; echo top > "$L/run/final_model.pt"; echo f > "$L/run/G9p/final_model.pt"; echo s > "$L/run/G9/final_model.pt" ;;
    741849) mkdir -p "$L/run/G9q/pool" "$L/run/G9" "$L/run/G9s"; echo top > "$L/run/final_model.pt"; echo f > "$L/run/G9q/final_model.pt"; echo s > "$L/run/G9/final_model.pt" ;;
    829660) mkdir -p "$L/run/R1bF/pool"; echo f > "$L/run/R1bF/final_model.pt" ;;
  esac
  touch "$L/.toucher-G9p" "$L/.toucher-G9q" "$L/.toucher-R1bF"
  if [ "$WHAT" = swapped ]; then  # a committed swap: handovers DONE, DECISION COMMIT, MODE flipped (as nv2_swap leaves it)
    mkdir -p "$N/runs/swap-50M" "$N/runs/g9q/handover-a30/pool/frozen" "$N/runs/r1/handover-g9p-a16/pool/frozen"
    echo "COMMIT by g9 741849 2026-11-10T16:00:00" > "$N/runs/swap-50M/DECISION"
    for h in "$N/runs/g9q/handover-a30" "$N/runs/r1/handover-g9p-a16"; do echo h > "$h/final_model.pt"; echo d > "$h/DONE"; done
    echo g9q > "$N/runs/r1/MODE"; echo g9p2 > "$N/runs/g9/MODE"; touch "$N/runs/r1/SWAP_APPLIED" "$N/runs/g9/SWAP_APPLIED"
    case $1 in 8103*)  # successors: both first starts happened; the swapped lineages' post-swap mirrors are the newest
      touch "$N/runs/r1/G9Q_A30_STARTED" "$N/runs/g9/G9P_A16_STARTED"
      for d in "$J/G9q" "$K/G9p"; do mkdir -p "$d/pool/frozen" "$d/checkpoints"; echo m > "$d/final_model.pt"; done
      touch -d '2026-11-10 15:00' "$J/G9p/final_model.pt" "$K/G9q/final_model.pt"; touch -d '2026-11-11 21:00' "$J/G9q/final_model.pt" "$K/G9p/final_model.pt" ;;
    esac
  fi
}
one() {  # one ENVFILE RUN JOB GPU_MIB
  tree "$3"
  ( set -euo pipefail
    NV2=$R/nv2; LOCAL=$R/local-$3; RUN_NAME=$2; SLURM_JOB_ID=$3; SLURM_JOB_NAME=ws-netv2-$2; NV2_GPU_MIB=$4; NV2_HOLD_ITERS=0; NV2_SWAP_ITERS=0
    case $3 in 8103*) unset LOCAL ;; esac  # a fresh job: the first source of the job script (LOCAL from SLURM_JOB_ID)
    TMPDIR=$R/tmp
    source "$1" > /dev/null 2>&1
    echo "FORMAT=$FORMAT|ANCHOR=$ANCHOR|PY=${PY:-unset}|PS=${PORYGON_PS:-unset}|TRAIN_ARGS=$TRAIN_ARGS"
    sleep 0.3; kill $(jobs -p) 2> /dev/null || true ) > "$R.one" 2>&1 < /dev/null
  sed "s#$R#R#g" "$R.one"
}
run() {  # run ENVFILE
  for x in "r1 718701 24576" "r1 810366 24576" "r1 810366 46068" "g9 741849 15356" "g9 810367 15356" "g9 810367 46068" "g4r1bf 829660 15360"; do
    set -- "$1" $x; echo "$3 $2 gpu=$4: $(one "$1" "$2" "$3" "$4")"; set -- "$1"
  done
}
run "$OLD" > "$R.old"; run "$NEW" > "$R.new"
if [ "$WHAT" = swapped ]; then echo "== $NEW (swapped)"; cat "$R.new"; echo "== $OLD (swapped)"; cat "$R.old"; fi
if diff "$R.old" "$R.new" > "$R.diff"; then echo "identical ($WHAT MODE files): $(wc -l < "$R.new") job/GPU cases"; cat "$R.new"; else echo "DIFFERENT ($WHAT MODE files):"; cat "$R.diff"; fi
rm -rf "$R" "$R.old" "$R.new" "$R.diff" "$R.one"
