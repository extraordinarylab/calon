#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mix-warmup
#SBATCH --output=logs/%j-mix-warmup.out
#SBATCH --partition=workq
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# BCE warm-up for one seed with DDP over the node's 4 GPUs (global batch 32),
# then calibration and OOD eval of its best checkpoint. SEED from the environment.
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}
UPDATES=${UPDATES:-10000}
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
RUNS=$ROOT/runs/mix-s$SEED
mkdir -p $RUNS
git rev-parse HEAD > $RUNS/git_commit.txt
torchrun --standalone --nproc_per_node=4 -m ierlcd.train --data $MIX --out $RUNS/warmup \
  --mode bce --updates $UPDATES --eval_every 1000 --workers 12 --seed $SEED \
  > $RUNS/warmup.log 2>&1
grep -E "EVAL|done" $RUNS/warmup.log | cut -c1-300
python -m ierlcd.calibrate --data $MIX --ckpt $RUNS/warmup/best.pt \
  --out $RUNS/warmup/calibration.json > $RUNS/warmup-calib.log 2>&1
python -m ierlcd.ood_eval --ood $ROOT/data/ood --ckpt $RUNS/warmup/best.pt \
  --calibration $RUNS/warmup/calibration.json --out $RUNS/warmup/ood.json \
  > $RUNS/warmup-ood.log 2>&1
echo "===== WARMUP s$SEED DONE ====="
