#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-eval-only
#SBATCH --output=logs/%j-eval-only.out
#SBATCH --partition=workq
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Calibration and OOD eval (with descriptions and names only) of an already
# trained train_mn.sh run. Environment: SEED, EXP, ENCODER, HEAD, MIXSPEC, DDROP.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}; EXP=${EXP:?}; ENCODER=${ENCODER:?}; HEAD=${HEAD:?}; MIXSPEC=${MIXSPEC:?}; DDROP=${DDROP:-0}
SOURCES=$(echo $MIXSPEC | sed 's/:[0-9.]*//g')
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix2
RUNS=$ROOT/runs2/$EXP-s$SEED
NAME=$HEAD-dd$DDROP
python -m calon.train.calibrate --data $MIX --sources $SOURCES --encoder $ENCODER --head $HEAD \
  --fewnerd_fine_only --ckpt $RUNS/$NAME/best.pt --out $RUNS/$NAME/calibration.json \
  > $RUNS/$NAME-calib.log 2>&1
for V in desc name_only; do
  FLAG=""; [ $V = name_only ] && FLAG=--name_only
  python -m calon.eval.ood_eval --ood $ROOT/data/ood --encoder $ENCODER --head $HEAD $FLAG \
    --ckpt $RUNS/$NAME/best.pt --calibration $RUNS/$NAME/calibration.json \
    --out $RUNS/$NAME/ood-$V.json > $RUNS/$NAME-ood-$V.log 2>&1
done
cp $RUNS/$NAME/ood-desc.json $RUNS/$NAME/ood.json
echo "===== eval $EXP s$SEED $NAME DONE ====="
