#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-gen2
#SBATCH --output=logs/%A_%a-gen2.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# One Qwen3.8-27B worker per array task: shard SLURM_ARRAY_TASK_ID of NSHARDS of
# TASK (g | c2 | r1 | r2 | n). Resumes from what the shard already wrote.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd-vllm
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TASK=${TASK:?}; NSHARDS=${NSHARDS:?}
EXTRA=""; case $TASK in r2|n) EXTRA="--max-tokens 768";; g|g2) EXTRA="--max-tokens 2048";; esac
python -m ierlcd.llm_gen2 run --task $TASK --shard $SLURM_ARRAY_TASK_ID --num-shards $NSHARDS \
  $EXTRA 2>&1 | grep -E "^shard|^chunk|Error|error" | cut -c1-200
