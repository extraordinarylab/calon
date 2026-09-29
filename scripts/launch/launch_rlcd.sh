#!/usr/bin/env bash
# BCE vs RLCD, both trained from scratch for 12k updates on the mixA25 mixture with
# Few-NERD fine types only, for ModernBERT and bidirectional Qwen2.5-0.5B, 3 seeds.
# 12 jobs x 4 GPUs. Only sbatch calls: safe on the login node.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export MIXSPEC=finerweb:0.1875,fewnerd:0.225,pile:0.225,massive:0.1125,synth_a:0.25
QWEN=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen2.5-0.5B
IDS=()
for pair in "rl-mbert answerdotai/ModernBERT-base" "rl-qwen $QWEN"; do
  read EXP ENCODER <<< "$pair"
  export EXP ENCODER
  for HEAD in bce rlcd; do
    for S in 42 43 44; do
      IDS+=($(sbatch --parsable --export=ALL,SEED=$S,HEAD=$HEAD --job-name=$EXP-$HEAD-s$S scripts/train/rlcd_run.sh))
    done
  done
done
echo "jobs: ${IDS[*]}"
