#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-synth
#SBATCH --output=logs/%A_%a-synth.out
#SBATCH --partition=workq
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# One Qwen3.8-27B worker per array task: shard SLURM_ARRAY_TASK_ID of NSHARDS of
# the TASK (a1 | a2 | c) request file. Resumes from what the shard already wrote.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd-vllm
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
TASK=${TASK:?}; NSHARDS=${NSHARDS:?}
EXTRA=""; [ "$TASK" = a2 ] && EXTRA="--max-tokens 768"
python -m ierlcd.synth.llm_annotate run --task $TASK --shard $SLURM_ARRAY_TASK_ID --num-shards $NSHARDS \
  --chunk 2048 $EXTRA 2>&1 | grep -E "^shard|^chunk|Error|error" | cut -c1-200
