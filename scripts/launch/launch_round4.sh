#!/usr/bin/env bash
# Round 4 (only sbatch calls: safe on the login node):
#   phase 1  fixed checkpoint selection + "none" weight {1, 0.3, 0.1}, 2 seeds, current data (mix2)
#   data     copy mix2 -> mix3; 370k general+specific synthetic texts (g3, 32 nodes x 4 GPUs);
#            missing-mention check of ~390k train rows (fill, 16 nodes x 4 GPUs); assemble into mix3
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
PRE="export TMPDIR=/tmp/\$USER-\$SLURM_JOB_ID; mkdir -p \$TMPDIR; source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh; conda activate ie-rlcd; cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd; export PYTHONPATH=src HF_HUB_OFFLINE=1"
sub() { sbatch --parsable "$@"; }
T=$(sub -p workq --time=0:40:00 --gres=gpu:1 --cpus-per-task=8 -o logs/%j-tests.out --job-name=tests \
    --wrap="$PRE; QWEN_TEST_PATH=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B python -m pytest -q tests 2>&1 | tail -3")
echo "tests $T"

export ENCODER=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B EXP=p1-z3
export MIXSPEC=finerweb:0.08,fewnerd:0.12,pile:0.12,massive:0.04,synth_a:0.08,synth_g:0.20,synth_r:0.16,synth_g2:0.20
P1=()
for S in 42 43; do
  for W in 1.0 0.3 0.1; do
    P1+=($(sub --dependency=afterok:$T --export=ALL,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=1,UPDATES=6000,NONEW=$W,NAME=rlcd-w$W \
           --job-name=p1-rlcd-w$W-s$S scripts/train/train_mn.sh))
  done
  P1+=($(sub --dependency=afterok:$T --export=ALL,SEED=$S,HEAD=bce,DDROP=0.3,FINE=1,UPDATES=6000,NAME=bce-u6000 \
         --job-name=p1-bce-s$S scripts/train/train_mn.sh))
done
echo "phase1: ${P1[*]}"; echo "${P1[*]}" > $ROOT/runs2/p1_ids.txt

export IE_MIX=$ROOT/data/mix3
C=$(sub -p workq --time=2:00:00 --gres=gpu:1 --cpus-per-task=16 -o logs/%j-mix3.out --job-name=mix3-prep --dependency=afterok:$T \
    --wrap="$PRE; rm -rf $ROOT/data/mix3; cp -r $ROOT/data/mix2 $ROOT/data/mix3; python -m ierlcd.synth.llm_gen2 prepare-g3 --n 370000; python -m ierlcd.synth.llm_gen2 prepare-fill")
G3=$(sub --dependency=afterok:$C --array=0-31 --export=ALL,TASK=g3,NSHARDS=128 --job-name=gen4-g3 scripts/synth/gen_node.sh)
FI=$(sub --dependency=afterok:$C --array=0-15 --export=ALL,TASK=fill,NSHARDS=64 --job-name=gen4-fill scripts/synth/gen_node.sh)
AF=$(sub --dependency=afterok:$FI --export=ALL,ARGS="apply-fill" --job-name=gen4-apply-fill scripts/synth/gen2_step.sh)
AG=$(sub --dependency=afterok:$G3:$AF --export=ALL,ARGS="assemble-g2 --task g3" --job-name=gen4-assemble-g3 scripts/synth/gen2_step.sh)
echo "data: mix3=$C g3=$G3 fill=$FI apply=$AF assemble=$AG"; echo "$AG" > $ROOT/runs2/mix3_ready_id.txt
