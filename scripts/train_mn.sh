#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mn
#SBATCH --output=logs/%j-mn.out
#SBATCH --partition=workq
#SBATCH --time=8:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# One model trained from scratch with DDP over NODES x 4 GPUs: global batch 128,
# UPDATES steps (3,000 = the rows of 12,000 x 32), learning rates scaled by
# sqrt(128/32) = 2. Then calibration and OOD eval with descriptions and names
# only; RLCD runs are evaluated with both decoding rules (<NAME>/ and <NAME>-ent/).
# Environment: SEED, EXP, ENCODER, HEAD, MIXSPEC, DDROP; optional UPDATES,
# BRIER (RLCD Brier weight, default 1), FINE (1 = Few-NERD fine types only,
# default 1), NAME (run directory name), NONEW (none-span weight), DATA_DIR,
# TRAIN_EXTRA (extra train.py flags, e.g. "--label_pool marker --lowercase_aug 0.3").
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}; EXP=${EXP:?}; ENCODER=${ENCODER:?}; HEAD=${HEAD:?}; MIXSPEC=${MIXSPEC:?}
DDROP=${DDROP:-0}; UPDATES=${UPDATES:-3000}; BRIER=${BRIER:-1.0}; FINE=${FINE:-1}
NAME=${NAME:-$HEAD-dd$DDROP}; NONEW=${NONEW:-1.0}
case "$FINE" in 1) FF="--fewnerd_mode fine";; 2) FF="--fewnerd_mode precedence";; *) FF="--fewnerd_mode both";; esac
SOURCES=$(echo $MIXSPEC | sed 's/:[0-9.]*//g')
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=${DATA_DIR:-$ROOT/data/mix2}
RUNS=$ROOT/runs2/$EXP-s$SEED
mkdir -p $RUNS
git rev-parse HEAD > $RUNS/git_commit.txt
MASTER=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
srun --kill-on-bad-exit=1 bash -c "export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p \$TMPDIR; \
  torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER:29500 -m ierlcd.train --data $MIX --out $RUNS/$NAME --mix $MIXSPEC \
  --encoder $ENCODER --head $HEAD $FF --brier_weight $BRIER --none_weight $NONEW --desc_dropout $DDROP --mode bce \
  --batch 128 --micro 8 --updates $UPDATES --enc_lr 4e-5 --head_lr 2e-4 --eval_every 500 \
  --workers 12 --seed $SEED ${TRAIN_EXTRA:-}" > $RUNS/$NAME.log 2>&1 || {
  # a rendezvous error while ranks shut down must not skip evaluation of a saved model
  grep -q "^done " $RUNS/$NAME.log && [ -s $RUNS/$NAME/final.pt ] || exit 1
  echo "srun exited non-zero after training finished; continuing"; }
grep -E "EVAL|done" $RUNS/$NAME.log | tail -2 | cut -c1-250
DECODES=label; [ "$HEAD" = rlcd ] && DECODES="label entity"
for DEC in $DECODES; do
  OUT=$RUNS/$NAME; [ $DEC = entity ] && OUT=$RUNS/$NAME-ent
  mkdir -p $OUT
  python -m ierlcd.calibrate --data $MIX --sources $SOURCES --encoder $ENCODER --head $HEAD $FF \
    --decode $DEC --ckpt $RUNS/$NAME/best.pt --out $OUT/calibration.json > $OUT-calib.log 2>&1
  for V in desc name_only; do
    FLAG=""; [ $V = name_only ] && FLAG=--name_only
    python -m ierlcd.ood_eval --ood $ROOT/data/ood --encoder $ENCODER --head $HEAD --decode $DEC $FLAG \
      --ckpt $RUNS/$NAME/best.pt --calibration $OUT/calibration.json \
      --out $OUT/ood-$V.json > $OUT-ood-$V.log 2>&1
  done
  cp $OUT/ood-desc.json $OUT/ood.json
  for EXT in pii spy new; do                             # separate evaluation sets, when built
    [ -s $ROOT/data/ood_$EXT/manifest.json ] || continue
    for V in desc name_only; do
      FLAG=""; [ $V = name_only ] && FLAG=--name_only
      python -m ierlcd.ood_eval --ood $ROOT/data/ood_$EXT --encoder $ENCODER --head $HEAD --decode $DEC $FLAG \
        --ckpt $RUNS/$NAME/best.pt --calibration $OUT/calibration.json \
        --out $OUT/ood-$EXT-$V.json > $OUT-ood-$EXT-$V.log 2>&1
    done
  done
done
echo "===== $EXP s$SEED $NAME DONE ====="
