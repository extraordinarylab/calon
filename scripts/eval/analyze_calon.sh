#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-analyze
#SBATCH --output=logs/%j-analyze.out
#SBATCH --partition=workq
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# RLCD vs BCE error analysis on the round-2 zero-shot Qwen3 runs (reads saved results).
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src
python -m tools.analyze_calon > /scratch/u6sn/yangw.u6sn/ie-rlcd/runs2/rlcd-analysis.txt 2>&1
wc -l /scratch/u6sn/yangw.u6sn/ie-rlcd/runs2/rlcd-analysis.txt
