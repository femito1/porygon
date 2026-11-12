#!/bin/bash
# gen9 BC net (v1 encoder) on netv2's frozen P1 data with netv2's held-out games (v1-vs-v2 comparison).
# srun --jobid=677193 --overlap --ntasks=1 -c 2 --gres=gpu:1 --mem=8G bash logs/bc_gen9_p1.sh
set -uo pipefail
D=/vol/bitbucket/fr1226/porygon-ws-data
cd $D/code
export PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORYGON_COMMIT="$(cat COMMIT)+bc@8675a7f" OMP_NUM_THREADS=2
PY=.venv/bin/python
F=$D/datasets/frozen/p1-gen9-human
OUT=runs/bc-gen9-p1-v1-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-x}; mkdir -p $OUT
L=/tmp/bcp1-$$; mkdir -p $L; trap 'rm -rf $L' EXIT
{ echo "data: $F/v1 (frozen P1 snapshot, hard links)"; echo "held-out: netv2 is_val(v2pre game, 0.06) via scripts/paired_val_mask.py";
  echo "commit: $PORYGON_COMMIT"; echo "host: $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"; } > $OUT/manifest.txt
echo "=== masks"; $PY scripts/paired_val_mask.py $F/v1 $F/v2pre $OUT/valmask --frac 0.06 --expect 55196 --games /vol/bitbucket/fr1226/porygon-ws-netv2/runs/pretrain/g9h-heldout-games.txt || { echo "MASK CHECK FAILED"; exit 3; }
t=$(date +%s); cp -r $F/v1 $L/v1; echo "copied v1 ($(du -sh $L/v1 | cut -f1), $(( $(date +%s)-t ))s)"
COMMON="--sources human --skip-forced --val-mask $OUT/valmask --val-max 100000 --load-workers 2"
TRAIN="$COMMON --device cuda --loader stream --buffer-rows 200000 --batch-size 1024"
echo "=== probe"; $PY -m porygon.bc train $L/v1 $TRAIN --max-steps 160 --log-every 40 --out $L/probe 2>&1 | grep -v -i warn > $L/probe.log
grep -E "train /|samples/s|Error|Traceback" $L/probe.log | tail -3
RATE=$(grep -o '[0-9,]* samples/s' $L/probe.log | tail -1 | tr -d ', samples/')
NTRAIN=$(grep -o -E '[0-9,]+ train' $L/probe.log | head -1 | tr -d ', train')
EP=$($PY -c "import sys; r=float(sys.argv[1] or 3000); n=float(sys.argv[2] or 868000); print(max(2, min(12, int(0.67*3600*r/n))))" "$RATE" "$NTRAIN")
echo "probe rate ${RATE:-?} samples/s, train ${NTRAIN:-?} -> --epochs $EP"
echo "train args: $TRAIN --epochs $EP --max-hours 0.75" >> $OUT/manifest.txt
echo "=== train ($EP epochs) -> $OUT"
t=$(date +%s); $PY -m porygon.bc train $L/v1 $TRAIN --epochs $EP --max-hours 0.75 --log-every 200 --out $OUT 2>&1 | grep -v -i warn | tee $OUT/train.log | grep -E "train /|samples/s|val ce|done:|Error|Traceback" | tail -60
echo "(train $(( $(date +%s)-t )) s)"
for ck in best_model best_value_model final_model; do
  echo "=== eval $ck on held-out"; $PY -m porygon.bc eval $OUT/$ck.zip $L/v1 $COMMON --device cuda 2>&1 | tail -1 | tee $OUT/eval_heldout_$ck.json | cut -c1-600
done
for band in "0 1999" "2000 2199" "2200 9999"; do set -- $band
  echo "=== band $1-$2"; $PY -m porygon.bc eval $OUT/best_model.zip $L/v1 $COMMON --device cuda --drop-unrated --min-rating $1 --max-rating $2 2>&1 | tail -1 | tee $OUT/eval_band_$1_$2.json | cut -c1-400
done
echo "=== search9 smoke"; export SLURM_SUBMIT_DIR=$D/code PS_SKIP_BUILD=1
bash scripts/slurm_eval_net.sbatch search9 $OUT/best_model.zip 4 --workers 2 --concurrency 1 --ms 300 --samples 2 2>&1 | grep -v -i -E "warn|^\s*$" | tail -4
echo "=== netv2 load_player check"
PYTHONPATH=/vol/bitbucket/fr1226/porygon-ws-netv2/src:$PYTHONPATH PORYGON_GEN=9 PYTHONPATH=/vol/bitbucket/fr1226/porygon-ws-netv2/src:$D/code/.engine-gen9:$PWD/src $PY -c "import sys; from porygon.v2.evaluate import load_player; p = load_player(sys.argv[1] + ,temperature=0.5, battle_format=gen9randombattle, start_listening=False); print(load_player OK:, type(p).__name__)" $OUT/best_model.zip 2>&1 | grep -v -i warn | tail -3
echo "OUT $OUT"
