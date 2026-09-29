#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-flag
#SBATCH --output=logs/%A_%a-flag.out
#SBATCH --partition=workq
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Round-5 refinement step 2: RLCD flags on the synthetic train rows of IE_MIX.
# One node, four shards (4*i .. 4*i+3 of NSHARDS), i = SLURM_ARRAY_TASK_ID.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
NSHARDS=${NSHARDS:?}
ENC=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B
CKPT=${CKPT:-/scratch/u6sn/yangw.u6sn/ie-rlcd/runs2/p2-z4-s42/rlcd-w0.1-f1/best.pt}
for g in 0 1 2 3; do
  S=$((4 * SLURM_ARRAY_TASK_ID + g))
  CUDA_VISIBLE_DEVICES=$g python -m ierlcd.flag_rlcd --encoder $ENC --ckpt $CKPT --shard $S --num-shards $NSHARDS \
    > $TMPDIR/flag-$S.log 2>&1 &
done
wait
for g in 0 1 2 3; do grep -hE "^shard|Error" $TMPDIR/flag-$((4 * SLURM_ARRAY_TASK_ID + g)).log | tail -2 | cut -c1-200; done
