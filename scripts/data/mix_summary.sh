#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mix-summary
#SBATCH --output=logs/%j-mix-summary.out
#SBATCH --partition=workq
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# Per-seed tables plus the cross-seed mean/std table, written to runs/mix-summary.txt.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src
R=/scratch/u6sn/yangw.u6sn/ie-rlcd/runs
{
  python -m calon.eval.summarize --seeds $R/mix-s42 $R/mix-s43 $R/mix-s44 --baseline $R/fw-s42
  python -m calon.eval.summarize --runs $R/mix-s42 --baseline $R/fw-s42
} > $R/mix-summary.txt 2>&1
wc -l $R/mix-summary.txt
