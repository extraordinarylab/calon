#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-fetch
#SBATCH --output=logs/%j-fetch.out
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# Download Few-NERD, Pile-NER-type/definition and MASSIVE at pinned revisions and
# write a format summary (file list, row counts, first records).
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1
RAW=/scratch/u6sn/yangw.u6sn/ie-rlcd/raw
mkdir -p $RAW
python -m calon.data.fetch_sources --out $RAW
du -sh $RAW/* > $RAW/sizes.txt
echo "===== FETCH DONE ====="
