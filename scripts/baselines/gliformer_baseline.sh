#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-gliformer
#SBATCH --output=logs/%j-gliformer.out
#SBATCH --partition=workq
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Zero-shot GLiFormer baselines (knowledgator/gliformer-{base,large}-v1) on the OOD
# test sets, names only and, if the package accepts them, names + descriptions.
# Builds its own env with uv on first use.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
E=$CONDA/envs/gliformer
export UV_CACHE_DIR=/scratch/u6sn/yangw.u6sn/uv-cache HF_HUB_DISABLE_XET=1 HF_HOME=/scratch/u6sn/yangw.u6sn/hf_cache
if [ ! -x $E/bin/python ]; then
  conda create -y -q -p $E python=3.11 > /dev/null
  $CONDA/bin/uv pip install --python $E/bin/python --torch-backend=cu129 -q gliformer
fi
ls -d $E/lib/python3.11/site-packages/{gliformer,torch,transformers}-*.dist-info 2>/dev/null | xargs -n1 basename
conda activate $E
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
OUT=$ROOT/runs2/gliner2; mkdir -p $OUT            # same folder: the summary lists every baseline
for M in knowledgator/gliformer-base-v1 knowledgator/gliformer-large-v1; do
  for V in name_only desc; do
    FLAG=""; [ $V = name_only ] && FLAG=--name_only
    N=$(basename $M)-$V
    python -m ierlcd.baselines.gliner2_eval --backend gliformer --ood $ROOT/data/ood --model $M $FLAG \
      --out $OUT/$N.json > $OUT/$N.log 2>&1 || { echo "FAILED $N: $(tail -2 $OUT/$N.log | cut -c1-200)"; continue; }
    echo "$N $(tail -1 $OUT/$N.log | cut -c1-200)"
  done
done
