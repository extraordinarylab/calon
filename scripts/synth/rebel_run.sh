#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-rebel
#SBATCH --output=logs/%A_%a-rebel.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# One Qwen3.8-27B worker per array task: REBEL request shard SLURM_ARRAY_TASK_ID
# (of the 256 written by llm_rebel prepare). Resumes from its own output file.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd-vllm
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m ierlcd.synth.llm_rebel run --shard $SLURM_ARRAY_TASK_ID 2>&1 | grep -E "^shard|^chunk|Error|error" | cut -c1-200
