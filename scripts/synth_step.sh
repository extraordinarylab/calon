#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-synth-step
#SBATCH --output=logs/%j-synth-step.out
#SBATCH --partition=workq
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# CPU-side step of the Qwen data pipeline: ARGS is passed to llm_annotate
# (e.g. "prepare-a2 --min-type-count 3" or "assemble").
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1
python -m ierlcd.llm_annotate $ARGS 2>&1 | tail -60 | cut -c1-300
