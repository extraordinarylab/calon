#!/usr/bin/env bash
# Round 5b (only sbatch calls): 2 extra transformer layers (width 1024, 16 heads,
# FFN 4096, pre-norm) over the encoder output before the span / label pooling, as in
# Laya. Same data (mix4) and settings as round 5, RLCD none weight 0.1 and 0.03, 5 seeds.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
PRE="export TMPDIR=/tmp/\$USER-\$SLURM_JOB_ID; mkdir -p \$TMPDIR; source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh; conda activate ie-rlcd; cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd; export PYTHONPATH=src HF_HUB_OFFLINE=1"
T=$(sbatch --parsable -p workq --time=0:40:00 --gres=gpu:1 --cpus-per-task=8 -o logs/%j-tests.out --job-name=tests \
    --wrap="$PRE; QWEN_TEST_PATH=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B python -m pytest -q tests 2>&1 | tail -3")
export ENCODER=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B TRAIN_EXTRA="--head_layers 2"
export MIXSPEC=finerweb:0.053,fewnerd:0.088,pile:0.088,massive:0.026,synth_a:0.053,synth_g:0.132,synth_r:0.106,synth_g2:0.132,synth_g3:0.202,synth_c:0.12
IDS=()
for S in 42 43 44 45 46; do
  for W in 0.1 0.03; do
    IDS+=($(sbatch --parsable --dependency=afterok:$T --export=ALL,EXP=p5-z5,DATA_DIR=$ROOT/data/mix4,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=1,UPDATES=6000,NONEW=$W,NAME=rlcd-w$W-tf2 \
            --job-name=p5-tf2-w$W-s$S scripts/train/train_mn.sh))
  done
done
echo "tests $T"; echo "${IDS[*]}" | tee $ROOT/runs2/p5b_ids.txt
