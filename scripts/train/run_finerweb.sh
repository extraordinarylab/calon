#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-fw
#SBATCH --output=logs/%j-fw.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# FiNERWeb-only: 10k-update BCE warm-up on one GPU, then 8 forks from its best
# checkpoint (2 waves x 4 GPUs), then calibration of every final checkpoint.
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1
SEED=${SEED:-42}
DATA=/scratch/u6sn/yangw.u6sn/ie-rlcd/data/finerweb
RUNS=/scratch/u6sn/yangw.u6sn/ie-rlcd/runs/fw-s$SEED
mkdir -p $RUNS
git rev-parse HEAD > $RUNS/git_commit.txt

CUDA_VISIBLE_DEVICES=0 python -m ierlcd.train.train --data $DATA --out $RUNS/warmup --mode bce \
  --updates 10000 --micro 32 --eval_every 500 --seed $SEED > $RUNS/warmup.log 2>&1
INIT=$RUNS/warmup/best.pt

ARMS=("bce 0.2" "bce_scaled 0.2" "direct 0.2" "pg 0.2" "direct 1.0" "pg 1.0" "direct 2.0" "pg 2.0")
for wave in 0 4; do
  for g in 0 1 2 3; do
    read MODE SIGMA <<< "${ARMS[$((wave + g))]}"
    NAME=$MODE-s$SIGMA
    CUDA_VISIBLE_DEVICES=$g python -m ierlcd.train.train --data $DATA --out $RUNS/$NAME --init $INIT \
      --mode $MODE --sigma $SIGMA --updates 2000 --micro 32 --eval_every 250 --seed $SEED \
      > $RUNS/$NAME.log 2>&1 &
  done
  wait
done

CUDA_VISIBLE_DEVICES=0 python -m ierlcd.train.calibrate --data $DATA --ckpt $INIT \
  --out $RUNS/warmup/calibration.json > $RUNS/warmup-calib.log 2>&1 &
i=1
for arm in "${ARMS[@]}"; do
  read MODE SIGMA <<< "$arm"
  NAME=$MODE-s$SIGMA
  CUDA_VISIBLE_DEVICES=$((i % 4)) python -m ierlcd.train.calibrate --data $DATA \
    --ckpt $RUNS/$NAME/final.pt --out $RUNS/$NAME/calibration.json > $RUNS/$NAME-calib.log 2>&1 &
  i=$((i + 1))
  (( i % 4 == 0 )) && wait
done
wait
echo "===== FW DONE ====="
