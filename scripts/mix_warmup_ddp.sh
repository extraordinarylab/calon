#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mix-warmup
#SBATCH --output=logs/%j-mix-warmup.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# BCE warm-up for one experiment and seed with DDP over the node's 4 GPUs (global
# batch 32), then calibration and OOD eval of its best checkpoint.
# Environment: SEED, EXP (runs/<EXP>-s<SEED>), MIXSPEC, ENCODER.
set -euo pipefail
# the submitting shell's TMPDIR (/local/user/...) does not exist on compute nodes
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:?}
EXP=${EXP:-mix}
MIXSPEC=${MIXSPEC:-finerweb:0.25,fewnerd:0.30,pile:0.30,massive:0.15}
ENCODER=${ENCODER:-answerdotai/ModernBERT-base}
UPDATES=${UPDATES:-10000}
SOURCES=$(echo $MIXSPEC | sed 's/:[0-9.]*//g')
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
RUNS=$ROOT/runs/$EXP-s$SEED
mkdir -p $RUNS
git rev-parse HEAD > $RUNS/git_commit.txt
echo "$MIXSPEC $ENCODER" > $RUNS/config.txt
torchrun --standalone --nproc_per_node=4 -m ierlcd.train --data $MIX --out $RUNS/warmup \
  --mix $MIXSPEC --encoder $ENCODER --mode bce --updates $UPDATES --eval_every 1000 \
  --workers 12 --seed $SEED > $RUNS/warmup.log 2>&1
grep -E "EVAL|done" $RUNS/warmup.log | cut -c1-300
python -m ierlcd.calibrate --data $MIX --sources $SOURCES --encoder $ENCODER \
  --ckpt $RUNS/warmup/best.pt --out $RUNS/warmup/calibration.json > $RUNS/warmup-calib.log 2>&1
python -m ierlcd.ood_eval --ood $ROOT/data/ood --encoder $ENCODER --ckpt $RUNS/warmup/best.pt \
  --calibration $RUNS/warmup/calibration.json --out $RUNS/warmup/ood.json \
  > $RUNS/warmup-ood.log 2>&1
echo "===== WARMUP $EXP s$SEED DONE ====="
