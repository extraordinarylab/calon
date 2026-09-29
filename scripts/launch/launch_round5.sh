#!/usr/bin/env bash
# Round 5 (only sbatch calls). Data: mix4 = mix3 + CrossNER/MIT-seeded synth_c (161k
# train rows) in the mixture; synthetic sources refined (Nemotron re-annotation,
# RLCD flags, gpt-oss judge). mix4u = the same data before refinement.
#   RLCD w0.1 on mix4 and mix4u (effect of refinement), RLCD w0.03 on mix4 (the
#   dev-selected candidate from round 4), BCE on mix4; 5 seeds each.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MK=$(sbatch --parsable -p workq --time=1:00:00 --cpus-per-task=8 -o logs/%j-mix4u.out --job-name=mix4u \
  --wrap "rm -rf $ROOT/data/mix4u && cp -r $ROOT/data/mix4 $ROOT/data/mix4u && \
for s in synth_a synth_r synth_g synth_g2 synth_g3 synth_c; do mv $ROOT/data/mix4u/\$s/train.jsonl.pre_refine $ROOT/data/mix4u/\$s/train.jsonl; done && echo ok")
export ENCODER=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B
export MIXSPEC=finerweb:0.053,fewnerd:0.088,pile:0.088,massive:0.026,synth_a:0.053,synth_g:0.132,synth_r:0.106,synth_g2:0.132,synth_g3:0.202,synth_c:0.12
IDS=()
for S in 42 43 44 45 46; do
  IDS+=($(sbatch --parsable --export=ALL,EXP=p5-z5,DATA_DIR=$ROOT/data/mix4,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=1,UPDATES=6000,NONEW=0.1,NAME=rlcd-w0.1 \
          --job-name=p5-rlcd-w0.1-s$S scripts/train/train_mn.sh))
  IDS+=($(sbatch --parsable --export=ALL,EXP=p5-z5,DATA_DIR=$ROOT/data/mix4,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=1,UPDATES=6000,NONEW=0.03,NAME=rlcd-w0.03 \
          --job-name=p5-rlcd-w0.03-s$S scripts/train/train_mn.sh))
  IDS+=($(sbatch --parsable --export=ALL,EXP=p5-z5,DATA_DIR=$ROOT/data/mix4,SEED=$S,HEAD=bce,DDROP=0.3,FINE=1,UPDATES=6000,NAME=bce \
          --job-name=p5-bce-s$S scripts/train/train_mn.sh))
  IDS+=($(sbatch --parsable --dependency=afterok:$MK --export=ALL,EXP=p5-z5u,DATA_DIR=$ROOT/data/mix4u,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=1,UPDATES=6000,NONEW=0.1,NAME=rlcd-w0.1 \
          --job-name=p5u-rlcd-w0.1-s$S scripts/train/train_mn.sh))
done
echo "${IDS[*]}" | tee $ROOT/runs2/p5_ids.txt
