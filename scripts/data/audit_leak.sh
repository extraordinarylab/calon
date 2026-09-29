#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-audit
#SBATCH --output=logs/%j-audit.out
#SBATCH --partition=workq
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Punctuation-insensitive leakage audit of every training source against the OOD tests,
# then the fixed PII test-set build.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1
python -m ierlcd.data.audit_leak 2>&1 | cut -c1-300
