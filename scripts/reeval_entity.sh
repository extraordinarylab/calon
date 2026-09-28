#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-reeval
#SBATCH --output=logs/%j-reeval.out
#SBATCH --partition=workq
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Re-evaluate a trained RLCD model with entity-level decoding (threshold on
# 1 - p(none), label = argmax), threshold tuned on the training sources' dev_calib.
# Results go to runs2/<EXP>-s<SEED>/<HEAD>-dd<DDROP>-ent/. No retraining.
# Environment: SEED, EXP, ENCODER, MIXSPEC, DDROP, FINE (1 = Few-NERD fine only).
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}; EXP=${EXP:?}; ENCODER=${ENCODER:?}; MIXSPEC=${MIXSPEC:?}; DDROP=${DDROP:-0}
NAME=${NAME:-rlcd-dd$DDROP}
FF=""; [ "${FINE:-1}" = 1 ] && FF=--fewnerd_fine_only
SOURCES=$(echo $MIXSPEC | sed 's/:[0-9.]*//g')
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
RUNS=$ROOT/runs2/$EXP-s$SEED
OUT=$RUNS/$NAME-ent; mkdir -p $OUT
python -m ierlcd.calibrate --data $ROOT/data/mix2 --sources $SOURCES --encoder $ENCODER --head rlcd $FF \
  --decode entity --ckpt $RUNS/$NAME/best.pt --out $OUT/calibration.json > $OUT/calib.log 2>&1
for V in desc name_only; do
  FLAG=""; [ $V = name_only ] && FLAG=--name_only
  python -m ierlcd.ood_eval --ood $ROOT/data/ood --encoder $ENCODER --head rlcd --decode entity $FLAG \
    --ckpt $RUNS/$NAME/best.pt --calibration $OUT/calibration.json --out $OUT/ood-$V.json > $OUT/ood-$V.log 2>&1
done
cp $OUT/ood-desc.json $OUT/ood.json
echo "===== reeval $EXP s$SEED $NAME entity DONE ====="
