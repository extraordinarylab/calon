#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mit
#SBATCH --output=logs/%j-mit.out
#SBATCH --partition=workq
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# MIT Movie / Restaurant diagnosis: per-label baselines (gliformer-large names only,
# GLiNER2-large with descriptions), then the comparison table against our round-4 runs.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1 HF_HOME=/scratch/u6sn/yangw.u6sn/hf_cache
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
OUT=$ROOT/runs2/mit-analysis; mkdir -p $OUT
conda activate $CONDA/envs/gliformer
python -m calon.baselines.gliner2_eval --backend gliformer --ood $ROOT/data/ood --model knowledgator/gliformer-large-v1 \
  --name_only --sets mit_movie mit_restaurant --out $OUT/gliformer-large-v1-name_only.json > $OUT/glif.log 2>&1
conda activate $CONDA/envs/gliner2
python -m calon.baselines.gliner2_eval --ood $ROOT/data/ood --model fastino/gliner2-large-v1 \
  --sets mit_movie mit_restaurant --out $OUT/gliner2-large-v1-desc.json > $OUT/gliner2.log 2>&1
conda activate ie-rlcd
python -m tools.analyze_mit > $OUT/summary.txt 2>&1
wc -l $OUT/summary.txt
