#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-qwen-bidir
#SBATCH --output=logs/%j-qwen-bidir.out
#SBATCH --partition=workq
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Download Qwen2.5-0.5B (base) to scratch at a pinned revision and run the
# bidirectionality / padding / backend-parity tests plus the whole test suite.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
DEST=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen2.5-0.5B
if [ ! -s $DEST/config.json ]; then
  python - <<PY
from huggingface_hub import HfApi, snapshot_download
sha = HfApi().model_info("Qwen/Qwen2.5-0.5B").sha
snapshot_download("Qwen/Qwen2.5-0.5B", revision=sha, local_dir="$DEST", max_workers=2)
open("$DEST/REVISION", "w").write(sha + "\n")
print("downloaded", sha)
PY
fi
cat $DEST/REVISION
python -m pytest -q tests 2>&1 | tail -4
