#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-rebel-prep
#SBATCH --output=logs/%j-rebel-prep.out
#SBATCH --partition=workq
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# Download mneb/rebel, count rows, write 256 request shards, then time 512
# annotations on this GPU so the full run can be sized.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1
conda activate ie-rlcd
python -m ierlcd.llm_rebel prepare --num-shards 256 2>&1 | tail -2 | cut -c1-600
conda activate ie-rlcd-vllm
HF_HUB_OFFLINE=1 python -m ierlcd.llm_rebel run --shard 0 --limit 512 --chunk 512 2>&1 | grep -E "^shard|^chunk|Error" | cut -c1-200
head -c 1500 /scratch/u6sn/yangw.u6sn/ie-rlcd/synth_rebel/out/out-00000.jsonl; echo
