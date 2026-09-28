#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mix-fork
#SBATCH --output=logs/%j-mix-fork.out
#SBATCH --partition=workq
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# One fork arm from a warm-up's best checkpoint, then calibration and OOD eval.
# Environment: SEED, MODE, SIGMA, EXP, MIXSPEC, ENCODER, ALPHA (spherical weight),
# INIT_EXP (experiment whose warm-up to fork from; default EXP), TAG (name suffix).
set -euo pipefail
# the submitting shell's TMPDIR (/local/user/...) does not exist on compute nodes
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}; MODE=${MODE:?}; SIGMA=${SIGMA:?}
EXP=${EXP:-mix}
MIXSPEC=${MIXSPEC:-finerweb:0.25,fewnerd:0.30,pile:0.30,massive:0.15}
ENCODER=${ENCODER:-answerdotai/ModernBERT-base}
ALPHA=${ALPHA:-0}
INIT_EXP=${INIT_EXP:-$EXP}
TAG=${TAG:-}
SOURCES=$(echo $MIXSPEC | sed 's/:[0-9.]*//g')
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
RUNS=$ROOT/runs/$EXP-s$SEED
INIT=$ROOT/runs/$INIT_EXP-s$SEED/warmup/best.pt
NAME=$MODE-s$SIGMA$TAG
mkdir -p $RUNS
python -m ierlcd.train --data $MIX --out $RUNS/$NAME --init $INIT --mix $MIXSPEC --encoder $ENCODER \
  --mode $MODE --sigma $SIGMA --alpha $ALPHA --updates 2000 --eval_every 500 --workers 14 \
  --seed $SEED > $RUNS/$NAME.log 2>&1
python -m ierlcd.calibrate --data $MIX --sources $SOURCES --encoder $ENCODER \
  --ckpt $RUNS/$NAME/final.pt --out $RUNS/$NAME/calibration.json > $RUNS/$NAME-calib.log 2>&1
python -m ierlcd.ood_eval --ood $ROOT/data/ood --encoder $ENCODER --ckpt $RUNS/$NAME/final.pt \
  --calibration $RUNS/$NAME/calibration.json --out $RUNS/$NAME/ood.json > $RUNS/$NAME-ood.log 2>&1
echo "===== FORK $EXP s$SEED $NAME DONE ====="
