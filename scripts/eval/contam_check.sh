#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-contam
#SBATCH --output=logs/%j-contam.out
#SBATCH --partition=workq
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Contamination check: every model on the benchmarks' OTHER splits (CrossNER train and
# validation, MIT train; first 2,000 rows per set), scored like the test sets.
# A model that trained on a split scores clearly higher on it than on test; a model
# that never saw the benchmark scores about the same. InstructUIE, whose training
# list includes mit-movie and mit-restaurant, is the positive control.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false HF_HOME=/scratch/u6sn/yangw.u6sn/hf_cache HF_HUB_OFFLINE=1
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
HFM=/scratch/u6sn/yangw.u6sn/huggingface_models
conda activate ie-rlcd
for SP in train validation; do
  python -m ierlcd.data.prepare_ood --train $ROOT/data/mix/finerweb/train.jsonl --out $ROOT/data/ood_split_$SP \
    --split $SP --max_rows 2000 > $TMPDIR/prep-$SP.log 2>&1
  grep -E '"rows"' $TMPDIR/prep-$SP.log | tr -s ' ' | tr '\n' ' '; echo " <- $SP"
done
OUT=$ROOT/runs2/contam; mkdir -p $OUT
(   # GPU 0: GLiNER2
  conda activate $CONDA/envs/gliner2
  for SP in train validation; do for M in fastino/gliner2-base-v1 fastino/gliner2-large-v1; do for V in desc name_only; do
    FLAG=""; [ $V = name_only ] && FLAG=--name_only
    CUDA_VISIBLE_DEVICES=0 python -m ierlcd.baselines.gliner2_eval --ood $ROOT/data/ood_split_$SP --model $M $FLAG \
      --out $OUT/$SP-$(basename $M)-$V.json > $OUT/$SP-$(basename $M)-$V.log 2>&1 || echo "FAILED $SP $M $V"
  done; done; done ) &
(   # GPU 1: gliformer, then GNER
  conda activate $CONDA/envs/gliformer
  for SP in train validation; do for M in knowledgator/gliformer-base-v1 knowledgator/gliformer-large-v1; do
    CUDA_VISIBLE_DEVICES=1 python -m ierlcd.baselines.gliner2_eval --backend gliformer --name_only --ood $ROOT/data/ood_split_$SP \
      --model $M --out $OUT/$SP-$(basename $M)-name_only.json > $OUT/$SP-$(basename $M)-name_only.log 2>&1 || echo "FAILED $SP $M"
  done; done
  conda activate ie-rlcd-vllm
  for SP in train validation; do
    SETS=$(ls $ROOT/data/ood_split_$SP/*.jsonl | xargs -n1 basename | sed 's/.jsonl$//' | tr '\n' ' ')
    CUDA_VISIBLE_DEVICES=1 python -m ierlcd.baselines.llm_ner_eval --backend gner --model $HFM/dyyyyyyyy/GNER-LLaMA-7B \
      --ood $ROOT/data/ood_split_$SP --sets $SETS --out_dir $OUT/$SP-GNER-LLaMA-7B > $OUT/$SP-gner.log 2>&1 || echo "FAILED GNER $SP"
  done ) &
(   # GPU 2: InstructUIE (positive control)
  conda activate ie-rlcd
  for SP in train validation; do
    SETS=$(ls $ROOT/data/ood_split_$SP/*.jsonl | xargs -n1 basename | sed 's/.jsonl$//' | tr '\n' ' ')
    CUDA_VISIBLE_DEVICES=2 python -m ierlcd.baselines.instructuie_eval --ood $ROOT/data/ood_split_$SP --model $HFM/ZWK/InstructUIE \
      --sets $SETS --out_dir $OUT/$SP-InstructUIE > $OUT/$SP-instructuie.log 2>&1 || echo "FAILED InstructUIE $SP"
  done ) &
(   # GPU 3: our round-4 finals, seeds 42 and 43
  conda activate ie-rlcd
  ENC=$HFM/Qwen/Qwen3-0.6B
  for SP in train validation; do for S in 42 43; do
    R=$ROOT/runs2/p2-z4-s$S
    for arm in "rlcd rlcd-w0.1-f1 rlcd-w0.1-f1-ent entity" "bce bce-f1 bce-f1 label"; do
      set -- $arm
      for V in desc name_only; do
        FLAG=""; [ $V = name_only ] && FLAG=--name_only
        CUDA_VISIBLE_DEVICES=3 python -m ierlcd.eval.ood_eval --ood $ROOT/data/ood_split_$SP --encoder $ENC --head $1 \
          --decode $4 $FLAG --ckpt $R/$2/best.pt --calibration $R/$3/calibration.json \
          --out $OUT/$SP-$3-s$S-$V.json > $OUT/$SP-$3-s$S-$V.log 2>&1 || echo "FAILED $SP $3 s$S $V"
      done
    done
  done; done ) &
wait
conda activate ie-rlcd
python -m ierlcd.eval.contam_summary --out $OUT/summary.txt && cat $OUT/summary.txt
