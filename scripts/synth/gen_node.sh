#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-gen-node
#SBATCH --output=logs/%A_%a-gen-node.out
#SBATCH --partition=workq
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# One node, four Qwen3.8-27B workers (one per GPU): shards 4*i .. 4*i+3 of NSHARDS
# for TASK, where i = SLURM_ARRAY_TASK_ID. Uses one job slot per 4 GPUs.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd-vllm
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TASK=${TASK:?}; NSHARDS=${NSHARDS:?}
EXTRA=""; case $TASK in r2|n|fill|ra) EXTRA="--max-tokens 768";; jd) EXTRA="--max-tokens 1024";; g|g2|g3|c2|c3) EXTRA="--max-tokens 2048";; esac
for g in 0 1 2 3; do
  S=$((4 * SLURM_ARRAY_TASK_ID + g))
  TRITON_CACHE_DIR=$TMPDIR/triton-$g CUDA_VISIBLE_DEVICES=$g python -m ierlcd.synth.llm_gen2 run --task $TASK --shard $S --num-shards $NSHARDS $EXTRA ${MODEL:+--model $MODEL} \
    > $TMPDIR/shard-$S.log 2>&1 &
done
wait
for g in 0 1 2 3; do grep -hE "^shard|Error" $TMPDIR/shard-$((4 * SLURM_ARRAY_TASK_ID + g)).log | tail -2 | cut -c1-200; done
