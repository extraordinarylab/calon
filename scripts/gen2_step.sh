#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-gen2-step
#SBATCH --output=logs/%j-gen2-step.out
#SBATCH --partition=workq
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# CPU-side step of the round-2 data pipeline: ARGS is passed to llm_gen2
# (e.g. "prepare-r2" or "assemble").
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m ierlcd.llm_gen2 $ARGS 2>&1 | tail -80 | cut -c1-300
