#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mn
#SBATCH --output=logs/%j-mn.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# One model trained from scratch with DDP over NODES x 4 GPUs: global batch 128,
# 3,000 updates (the same number of rows as 12,000 x 32), learning rates scaled by
# sqrt(128/32) = 2. Then calibration and OOD eval with descriptions and names only.
# Environment: SEED, EXP, ENCODER, HEAD, MIXSPEC, DDROP.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}; EXP=${EXP:?}; ENCODER=${ENCODER:?}; HEAD=${HEAD:?}; MIXSPEC=${MIXSPEC:?}
DDROP=${DDROP:-0}; UPDATES=${UPDATES:-3000}
SOURCES=$(echo $MIXSPEC | sed 's/:[0-9.]*//g')
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix2
RUNS=$ROOT/runs2/$EXP-s$SEED
NAME=$HEAD-dd$DDROP
mkdir -p $RUNS
git rev-parse HEAD > $RUNS/git_commit.txt
MASTER=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
srun --kill-on-bad-exit=1 bash -c "export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p \$TMPDIR; \
  torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER:29500 -m ierlcd.train --data $MIX --out $RUNS/$NAME --mix $MIXSPEC \
  --encoder $ENCODER --head $HEAD --fewnerd_fine_only --desc_dropout $DDROP --mode bce \
  --batch 128 --micro 8 --updates $UPDATES --enc_lr 4e-5 --head_lr 2e-4 --eval_every 500 \
  --workers 12 --seed $SEED" > $RUNS/$NAME.log 2>&1
grep -E "EVAL|done" $RUNS/$NAME.log | tail -2 | cut -c1-250
python -m ierlcd.calibrate --data $MIX --sources $SOURCES --encoder $ENCODER --head $HEAD \
  --fewnerd_fine_only --ckpt $RUNS/$NAME/best.pt --out $RUNS/$NAME/calibration.json \
  > $RUNS/$NAME-calib.log 2>&1
for V in desc name_only; do
  FLAG=""; [ $V = name_only ] && FLAG=--name_only
  python -m ierlcd.ood_eval --ood $ROOT/data/ood --encoder $ENCODER --head $HEAD $FLAG \
    --ckpt $RUNS/$NAME/best.pt --calibration $RUNS/$NAME/calibration.json \
    --out $RUNS/$NAME/ood-$V.json > $RUNS/$NAME-ood-$V.log 2>&1
done
cp $RUNS/$NAME/ood-desc.json $RUNS/$NAME/ood.json
echo "===== $EXP s$SEED $NAME DONE ====="
