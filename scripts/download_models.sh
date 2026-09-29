#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-dl
#SBATCH --output=logs/%j-dl.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
# Download extra non-Qwen annotator candidates (REPOS env, space-separated) into
# huggingface_models/<repo>, weights and tokenizer files only.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
H=/scratch/u6sn/yangw.u6sn/huggingface_models
for R in ${REPOS:?}; do
  HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download as d; \
d('$R', local_dir='$H/$R', max_workers=4, allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','tekken.json','*.py'])" \
    > $TMPDIR/dl.log 2>&1 && echo "ok $R $(du -sh $H/$R | cut -f1)" || { echo "FAILED $R"; tail -3 $TMPDIR/dl.log | cut -c1-300; }
done
