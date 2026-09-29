#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-spy
#SBATCH --output=logs/%j-spy.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# mneb/spy (legal) evaluation set: build + chunk, Qwen-authored descriptions, then
# every model on it: GLiNER2 / gliformer / GNER baselines and our round-4 finals
# (RLCD none weight 0.1 entity decoding, BCE; seeds 42-46; descriptions and names only).
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false HF_HOME=/scratch/u6sn/yangw.u6sn/hf_cache
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
P=$ROOT/data/ood_spy
HFM=/scratch/u6sn/yangw.u6sn/huggingface_models
conda activate ie-rlcd
python -m ierlcd.prepare_spy build 2>&1 | tail -1 | cut -c1-900
conda activate ie-rlcd-vllm
HF_HUB_OFFLINE=1 python -m ierlcd.label_bank --inventory $P/inventory.json \
  --model $HFM/Qwen/Qwen3.8-27B --out $P/label_bank.json > $P/label_bank.log 2>&1
tail -8 $P/label_bank.log | cut -c1-250
conda activate ie-rlcd
python -m ierlcd.prepare_spy finish 2>&1 | tail -1 | cut -c1-800
export HF_HUB_OFFLINE=1
OUT=$ROOT/runs2/spy; mkdir -p $OUT
(   # GPU 0: GLiNER2
  conda activate $CONDA/envs/gliner2
  for M in fastino/gliner2-base-v1 fastino/gliner2-large-v1; do
    for V in desc name_only; do
      FLAG=""; [ $V = name_only ] && FLAG=--name_only
      CUDA_VISIBLE_DEVICES=0 python -m ierlcd.gliner2_eval --ood $P --model $M $FLAG \
        --out $OUT/$(basename $M)-$V.json > $OUT/$(basename $M)-$V.log 2>&1 || echo "FAILED $M $V"
    done
  done ) &
(   # GPU 1: gliformer, then GNER
  conda activate $CONDA/envs/gliformer
  for M in knowledgator/gliformer-base-v1 knowledgator/gliformer-large-v1; do
    CUDA_VISIBLE_DEVICES=1 python -m ierlcd.gliner2_eval --backend gliformer --name_only --ood $P --model $M \
      --out $OUT/$(basename $M)-name_only.json > $OUT/$(basename $M)-name_only.log 2>&1 || echo "FAILED $M"
  done
  conda activate ie-rlcd-vllm
  CUDA_VISIBLE_DEVICES=1 python -m ierlcd.llm_ner_eval --backend gner --model $HFM/dyyyyyyyy/GNER-LLaMA-7B \
    --ood $P --sets spy_legal --out_dir $OUT/GNER-LLaMA-7B > $OUT/gner.log 2>&1 || echo "FAILED GNER" ) &
ENC=$HFM/Qwen/Qwen3-0.6B
for g in 2 3; do
  (
    conda activate ie-rlcd
    if [ $g = 2 ]; then HEAD=rlcd; ARM=rlcd-w0.1-f1; EVAL=rlcd-w0.1-f1-ent; DEC=entity
    else HEAD=bce; ARM=bce-f1; EVAL=bce-f1; DEC=label; fi
    for S in 42 43 44 45 46; do
      R=$ROOT/runs2/p2-z4-s$S
      for V in desc name_only; do
        FLAG=""; [ $V = name_only ] && FLAG=--name_only
        CUDA_VISIBLE_DEVICES=$g python -m ierlcd.ood_eval --ood $P --encoder $ENC --head $HEAD --decode $DEC $FLAG \
          --ckpt $R/$ARM/best.pt --calibration $R/$EVAL/calibration.json \
          --out $R/$EVAL/ood-spy-$V.json > $R/$EVAL-ood-spy-$V.log 2>&1 || echo "FAILED $EVAL s$S $V"
      done
    done ) &
done
wait
for f in $OUT/*.log; do echo "$(basename $f): $(tail -1 $f | cut -c1-200)"; done
grep -h "^spy_legal" $OUT/gner.log | cut -c1-200 || true
