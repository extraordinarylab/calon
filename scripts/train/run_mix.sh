#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mix
#SBATCH --output=logs/%j-mix.out
#SBATCH --partition=workq
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Job B of the round-1 mixture (after run_mix_warmup.sh): 8 forks from the
# warm-up's best checkpoint (2 waves x 4 GPUs), pooled-dev calibration and
# zero-shot OOD eval of every checkpoint, then a compact summary file.
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:-42}
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
RUNS=$ROOT/runs/mix-s$SEED
INIT=$RUNS/warmup/best.pt

ARMS=("bce 0.2" "bce_scaled 0.2" "direct 0.2" "pg 0.2" "direct 1.0" "pg 1.0" "direct 2.0" "pg 2.0")
for wave in 0 4; do
  for g in 0 1 2 3; do
    read MODE SIGMA <<< "${ARMS[$((wave + g))]}"
    NAME=$MODE-s$SIGMA
    CUDA_VISIBLE_DEVICES=$g python -m calon.train.train --data $MIX --out $RUNS/$NAME --init $INIT \
      --mode $MODE --sigma $SIGMA --updates 2000 --eval_every 500 --workers 14 --seed $SEED \
      > $RUNS/$NAME.log 2>&1 &
  done
  wait
done

EVALS=("warmup best.pt")
for arm in "${ARMS[@]}"; do read MODE SIGMA <<< "$arm"; EVALS+=("$MODE-s$SIGMA final.pt"); done
i=0
for c in "${EVALS[@]}"; do
  read NAME FILE <<< "$c"
  ( CUDA_VISIBLE_DEVICES=$((i % 4)) python -m calon.train.calibrate --data $MIX --ckpt $RUNS/$NAME/$FILE \
      --out $RUNS/$NAME/calibration.json > $RUNS/$NAME-calib.log 2>&1 &&
    CUDA_VISIBLE_DEVICES=$((i % 4)) python -m calon.eval.ood_eval --ood $ROOT/data/ood \
      --ckpt $RUNS/$NAME/$FILE --calibration $RUNS/$NAME/calibration.json \
      --out $RUNS/$NAME/ood.json > $RUNS/$NAME-ood.log 2>&1 ) &
  i=$((i + 1))
  if (( i % 4 == 0 )); then wait; fi
done
wait
python -m calon.eval.summarize --runs $RUNS --baseline $ROOT/runs/fw-s42 > $RUNS/summary.txt 2>&1
echo "===== MIX DONE ====="
