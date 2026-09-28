#!/usr/bin/env bash
# Round 4 head / augmentation ablations (only sbatch calls), each against p2-z4 rlcd-w0.1-f1
# (same data mix3, RLCD none weight 0.1, Few-NERD fine types), seeds 42 and 43:
#   marker  label vector = <LABEL> marker state (GLiNER2 / GLiFormer)
#   both    marker + mean, concatenated
#   wide    wider MLPs (hidden 2048, out 512; GLiNER2 uses 4x-wide MLPs)
#   lower   30% of training rows lowercased (MIT text is all lowercase)
#   anon    20% of described labels renamed "entity k" (GLiNER2 augmentation)
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
PRE="export TMPDIR=/tmp/\$USER-\$SLURM_JOB_ID; mkdir -p \$TMPDIR; source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh; conda activate ie-rlcd; cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd; export PYTHONPATH=src HF_HUB_OFFLINE=1"
T=$(sbatch --parsable -p workq --time=0:40:00 --gres=gpu:1 --cpus-per-task=8 -o logs/%j-tests.out --job-name=tests \
    --wrap="$PRE; QWEN_TEST_PATH=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B python -m pytest -q tests 2>&1 | tail -3")
export ENCODER=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B DATA_DIR=$ROOT/data/mix3 EXP=p2-z4
export MIXSPEC=finerweb:0.06,fewnerd:0.10,pile:0.10,massive:0.03,synth_a:0.06,synth_g:0.15,synth_r:0.12,synth_g2:0.15,synth_g3:0.23
declare -A V=([marker]="--label_pool marker" [both]="--label_pool both" [wide]="--head_hidden 2048 --head_out 512"
              [lower]="--lowercase_aug 0.3" [anon]="--anon_labels 0.2")
IDS=()
for S in 42 43; do
  for K in marker both wide lower anon; do
    export TRAIN_EXTRA="${V[$K]}"
    IDS+=($(sbatch --parsable --dependency=afterok:$T --export=ALL,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=1,UPDATES=6000,NONEW=0.1,NAME=rlcd-w0.1-f1-$K \
            --job-name=p2-$K-s$S scripts/train_mn.sh))
  done
done
echo "tests $T"; echo "${IDS[*]}" | tee $ROOT/runs2/p2c_ids.txt
