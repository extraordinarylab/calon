#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-summary-all
#SBATCH --output=logs/%j-summary-all.out
#SBATCH --partition=workq
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# Cross-experiment table (mean ± std over seeds) and JSON for the report.
# EXPS from the environment, space-separated.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src
R=/scratch/u6sn/yangw.u6sn/ie-rlcd/${ROOT_RUNS:-runs}
python -m ierlcd.summarize --root $R --exps ${EXPS:?} --json_out $R/all-results.json > $R/all-summary.txt 2>&1
wc -l $R/all-summary.txt
