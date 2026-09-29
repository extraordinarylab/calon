#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-pii
#SBATCH --output=logs/%j-pii.out
#SBATCH --partition=workq
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Build the ai4privacy PII evaluation set (3,000 English validation rows) with
# Qwen-authored label descriptions, then run the GLiNER2 / GLiFormer baselines on it.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HOME=/scratch/u6sn/yangw.u6sn/hf_cache
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
P=$ROOT/data/ood_pii
conda activate ie-rlcd
python -m ierlcd.data.prepare_pii build 2>&1 | tail -1 | cut -c1-600
conda activate ie-rlcd-vllm
HF_HUB_OFFLINE=1 python -m ierlcd.synth.label_bank --inventory $P/inventory.json \
  --model /scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.8-27B --out $P/label_bank.json > $P/label_bank.log 2>&1
tail -4 $P/label_bank.log | cut -c1-250
conda activate ie-rlcd
python -m ierlcd.data.prepare_pii finish 2>&1 | tail -1 | cut -c1-600
OUT=$ROOT/runs2/gliner2-pii; mkdir -p $OUT
conda activate $CONDA/envs/gliner2
for M in fastino/gliner2-base-v1 fastino/gliner2-large-v1 fastino/gliner2.5-base-v1; do
  for V in desc name_only; do
    FLAG=""; [ $V = name_only ] && FLAG=--name_only
    python -m ierlcd.baselines.gliner2_eval --ood $P --model $M $FLAG --out $OUT/$(basename $M)-$V.json \
      > $OUT/$(basename $M)-$V.log 2>&1 && echo "$(basename $M)-$V $(tail -1 $OUT/$(basename $M)-$V.log | cut -c1-200)"
  done
done
conda activate $CONDA/envs/gliformer
for M in knowledgator/gliformer-base-v1 knowledgator/gliformer-large-v1; do
  python -m ierlcd.baselines.gliner2_eval --backend gliformer --name_only --ood $P --model $M \
    --out $OUT/$(basename $M)-name_only.json > $OUT/$(basename $M)-name_only.log 2>&1 && \
    echo "$(basename $M)-name_only $(tail -1 $OUT/$(basename $M)-name_only.log | cut -c1-200)"
done
