#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-crossre
#SBATCH --output=logs/%j-crossre.out
#SBATCH --partition=workq
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# Convert CrossRE to mneb-style jsonl and report overlap with the CrossNER test sets.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1
python -m ierlcd.prepare_crossre 2>&1 | tail -8 | cut -c1-400
