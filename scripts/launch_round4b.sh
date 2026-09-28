#!/usr/bin/env bash
# Round 4 step 2 (only sbatch calls): new data (mix3 = mix2 + fill-in mentions + synth_g3).
# Phase 1 picked none weight 0.1 on in-domain dev (best of 1 / 0.3 / 0.1 on both seeds).
#   RLCD w0.1 and BCE, Few-NERD fine vs precedence, 5 seeds; RLCD w0.03 (trend check), 2 seeds.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
export ENCODER=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B DATA_DIR=$ROOT/data/mix3 EXP=p2-z4
export MIXSPEC=finerweb:0.06,fewnerd:0.10,pile:0.10,massive:0.03,synth_a:0.06,synth_g:0.15,synth_r:0.12,synth_g2:0.15,synth_g3:0.23
IDS=()
for S in 42 43 44 45 46; do
  for F in 1 2; do
    IDS+=($(sbatch --parsable --export=ALL,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=$F,UPDATES=6000,NONEW=0.1,NAME=rlcd-w0.1-f$F \
            --job-name=p2-rlcd-f$F-s$S scripts/train_mn.sh))
    IDS+=($(sbatch --parsable --export=ALL,SEED=$S,HEAD=bce,DDROP=0.3,FINE=$F,UPDATES=6000,NAME=bce-f$F \
            --job-name=p2-bce-f$F-s$S scripts/train_mn.sh))
  done
done
for S in 42 43; do
  IDS+=($(sbatch --parsable --export=ALL,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=1,UPDATES=6000,NONEW=0.03,NAME=rlcd-w0.03-f1 \
          --job-name=p2-rlcd-w0.03-s$S scripts/train_mn.sh))
done
echo "${IDS[*]}" | tee $ROOT/runs2/p2_ids.txt
