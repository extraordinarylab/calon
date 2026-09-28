#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mix-fork
#SBATCH --output=logs/%j-mix-fork.out
#SBATCH --partition=workq
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# One fork arm (SEED, MODE, SIGMA from the environment) from that seed's warm-up
# best checkpoint, then its calibration and OOD eval.
set -euo pipefail
# the submitting shell's TMPDIR (/local/user/...) does not exist on compute nodes
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}; MODE=${MODE:?}; SIGMA=${SIGMA:?}
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
RUNS=$ROOT/runs/mix-s$SEED
NAME=$MODE-s$SIGMA
python -m ierlcd.train --data $MIX --out $RUNS/$NAME --init $RUNS/warmup/best.pt --mode $MODE \
  --sigma $SIGMA --updates 2000 --eval_every 500 --workers 14 --seed $SEED > $RUNS/$NAME.log 2>&1
python -m ierlcd.calibrate --data $MIX --ckpt $RUNS/$NAME/final.pt \
  --out $RUNS/$NAME/calibration.json > $RUNS/$NAME-calib.log 2>&1
python -m ierlcd.ood_eval --ood $ROOT/data/ood --ckpt $RUNS/$NAME/final.pt \
  --calibration $RUNS/$NAME/calibration.json --out $RUNS/$NAME/ood.json > $RUNS/$NAME-ood.log 2>&1
echo "===== FORK s$SEED $NAME DONE ====="
