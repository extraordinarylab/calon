#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mn-smoke
#SBATCH --output=logs/%j-mn-smoke.out
#SBATCH --partition=workq
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Two-node DDP check with Qwen3-0.6B, RLCD head and description dropout on the
# mix2 public sources: 60 updates at global batch 128, then speed.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
OUT=$ROOT/runs2/mn-smoke; rm -rf $OUT
MASTER=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
srun --kill-on-bad-exit=1 bash -c "export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p \$TMPDIR; \
  torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER:29500 -m ierlcd.train --data $ROOT/data/mix2 --out $OUT \
  --mix finerweb:0.25,fewnerd:0.30,pile:0.30,massive:0.15 \
  --encoder /scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B --head rlcd --fewnerd_fine_only \
  --desc_dropout 0.3 --batch 128 --micro 8 --updates 60 --enc_lr 4e-5 --head_lr 2e-4 --eval_every 60 \
  --eval_limit 100 --workers 12" 2>&1 | grep -E "^world|filter_long|EVAL|done|Error|error" | sort -u | cut -c1-250
grep '"step": 50' $OUT/log.jsonl | cut -c1-250
