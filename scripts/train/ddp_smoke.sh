#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-ddp-smoke
#SBATCH --output=logs/%j-ddp-smoke.out
#SBATCH --partition=workq
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# 100-update DDP check on 4 GPUs against a single-GPU run of the same schedule.
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
OUT=$ROOT/runs/ddp-smoke
rm -rf $OUT
torchrun --standalone --nproc_per_node=4 -m calon.train.train --data $ROOT/data/mix --out $OUT/ddp4 \
  --updates 100 --eval_every 100 --eval_limit 100 --workers 12 2>&1 | grep -E "world|EVAL|done|Error" | cut -c1-300
python -m calon.train.train --data $ROOT/data/mix --out $OUT/gpu1 --updates 100 --eval_every 100 \
  --eval_limit 100 --workers 12 2>&1 | grep -E "world|EVAL|done|Error" | cut -c1-300
for d in ddp4 gpu1; do echo "$d $(grep '"step": 100' $OUT/$d/log.jsonl | cut -c1-220)"; done
