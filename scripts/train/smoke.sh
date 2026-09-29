#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-smoke
#SBATCH --output=logs/%j-smoke.out
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# Tests, FiNERWeb preparation, tokenizer coverage and a short timed training run
# for each loss mode.
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
DATA=$ROOT/data/finerweb
FINERWEB=$HOME/.cache/huggingface/hub/datasets--mneb--finerweb-english/snapshots/46a29532929a2cd439bf5c3df2281d7760d28731/data/train.jsonl

python -m pytest -q tests
python -m ierlcd.data.prepare_data --finerweb "$FINERWEB" --out "$DATA" --seed 42
python -m tools.coverage_report --data "$DATA" --W 32
nvidia-smi --query-gpu=name,memory.total --format=csv

for MICRO in 4 32; do
  python -m ierlcd.train.train --data "$DATA" --out $ROOT/runs/smoke-bce-m$MICRO --mode bce \
    --updates 100 --micro $MICRO --eval_every 100 --eval_limit 500
done
for MODE in direct pg; do
  python -m ierlcd.train.train --data "$DATA" --out $ROOT/runs/smoke-$MODE --mode $MODE \
    --init $ROOT/runs/smoke-bce-m4/final.pt --updates 100 --micro 4 --eval_every 100 \
    --eval_limit 500 --lam_ramp 20
done
python -m ierlcd.train.calibrate --data "$DATA" --ckpt $ROOT/runs/smoke-pg/final.pt \
  --out $ROOT/runs/smoke-pg/calibration.json
nvidia-smi --query-gpu=memory.used --format=csv
echo "===== SMOKE DONE ====="
