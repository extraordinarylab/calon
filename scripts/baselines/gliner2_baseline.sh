#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-gliner2
#SBATCH --output=logs/%j-gliner2.out
#SBATCH --partition=workq
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Zero-shot GLiNER2 baselines on the OOD test sets, with descriptions and names only.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
conda activate $CONDA/envs/gliner2
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1 HF_HOME=/scratch/u6sn/yangw.u6sn/hf_cache
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
OUT=$ROOT/runs2/gliner2; mkdir -p $OUT
# gliner2.5 tokenizers need protobuf and sentencepiece
python -c "import google.protobuf, sentencepiece" 2>/dev/null || \
  $CONDA/bin/uv pip install --python $CONDA/envs/gliner2/bin/python -q protobuf sentencepiece
for M in ${MODELS:-fastino/gliner2-base-v1 fastino/gliner2-large-v1 fastino/gliner2.5-base-v1}; do
  for V in desc name_only; do
    FLAG=""; [ $V = name_only ] && FLAG=--name_only
    N=$(basename $M)-$V
    python -m ierlcd.baselines.gliner2_eval --ood $ROOT/data/ood --model $M $FLAG --out $OUT/$N.json \
      > $OUT/$N.log 2>&1 || { echo "FAILED $N"; tail -5 $OUT/$N.log | cut -c1-200; continue; }
    echo "$N $(tail -1 $OUT/$N.log | cut -c1-200)"
  done
done
