#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-rlcd
#SBATCH --output=logs/%j-rlcd.out
#SBATCH --partition=workq
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# One model trained from scratch for UPDATES steps with DDP on 4 GPUs (HEAD = bce |
# rlcd, Few-NERD fine types only), then calibration and zero-shot OOD eval.
# Environment: SEED, EXP, ENCODER, HEAD, MIXSPEC.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}; EXP=${EXP:?}; ENCODER=${ENCODER:?}; HEAD=${HEAD:?}; MIXSPEC=${MIXSPEC:?}
UPDATES=${UPDATES:-12000}
SOURCES=$(echo $MIXSPEC | sed 's/:[0-9.]*//g')
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
RUNS=$ROOT/runs/$EXP-s$SEED
NAME=$HEAD-12k
mkdir -p $RUNS
git rev-parse HEAD > $RUNS/git_commit.txt
torchrun --standalone --nproc_per_node=4 -m ierlcd.train --data $MIX --out $RUNS/$NAME --mix $MIXSPEC \
  --encoder $ENCODER --head $HEAD --fewnerd_fine_only --mode bce --updates $UPDATES \
  --eval_every 1000 --workers 12 --seed $SEED > $RUNS/$NAME.log 2>&1
grep -E "EVAL|done" $RUNS/$NAME.log | tail -3 | cut -c1-250
python -m ierlcd.calibrate --data $MIX --sources $SOURCES --encoder $ENCODER --head $HEAD \
  --fewnerd_fine_only --ckpt $RUNS/$NAME/best.pt --out $RUNS/$NAME/calibration.json \
  > $RUNS/$NAME-calib.log 2>&1
python -m ierlcd.ood_eval --ood $ROOT/data/ood --encoder $ENCODER --head $HEAD \
  --ckpt $RUNS/$NAME/best.pt --calibration $RUNS/$NAME/calibration.json \
  --out $RUNS/$NAME/ood.json > $RUNS/$NAME-ood.log 2>&1
echo "===== $EXP s$SEED $NAME DONE ====="
