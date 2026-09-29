#!/usr/bin/env bash
# Experiments 4 and 5 on the chosen zero-shot mixture (BEST = mixA25 | mixA50):
#   4  Qwen2.5-0.5B made bidirectional, full arm set (warm-up + 8 forks) x 3 seeds
#   5  spherical-score ablation: 6 forks (alpha 0.1) from BEST's ModernBERT warm-ups
# Only sbatch calls: safe on the login node.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
BEST=${BEST:?}
case $BEST in
  mixA25) SPEC=finerweb:0.1875,fewnerd:0.225,pile:0.225,massive:0.1125,synth_a:0.25 ;;
  mixA50) SPEC=finerweb:0.125,fewnerd:0.15,pile:0.15,massive:0.075,synth_a:0.50 ;;
  *) echo "unknown BEST $BEST"; exit 1 ;;
esac
QWEN=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen2.5-0.5B
EXP=qwen-$BEST MIXSPEC=$SPEC ENCODER=$QWEN bash scripts/launch/launch_exp.sh | cut -c1-80
EXP=$BEST-sph MIXSPEC=$SPEC ARMSET=spherical INIT_EXP=$BEST bash scripts/launch/launch_exp.sh | cut -c1-80
