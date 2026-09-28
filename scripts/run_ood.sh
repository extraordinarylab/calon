#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-ood
#SBATCH --output=logs/%j-ood.out
#SBATCH --partition=workq
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Zero-shot OOD evaluation (CrossNER x5, MIT-Movie, MIT-Restaurant) of the
# warm-up checkpoint and every fork's final checkpoint.
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1
SEED=${SEED:-42}
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
RUNS=$ROOT/runs/fw-s$SEED
OOD=$ROOT/data/ood
python -m ierlcd.prepare_ood --train $ROOT/data/finerweb/train.jsonl --out $OOD

CKPTS=("warmup best.pt")
for arm in bce-s0.2 bce_scaled-s0.2 direct-s0.2 pg-s0.2 direct-s1.0 pg-s1.0 direct-s2.0 pg-s2.0; do
  CKPTS+=("$arm final.pt")
done
i=0
for c in "${CKPTS[@]}"; do
  read NAME FILE <<< "$c"
  CUDA_VISIBLE_DEVICES=$((i % 4)) python -m ierlcd.ood_eval --ood $OOD --ckpt $RUNS/$NAME/$FILE \
    --calibration $RUNS/$NAME/calibration.json --out $RUNS/$NAME/ood.json \
    > $RUNS/$NAME-ood.log 2>&1 &
  i=$((i + 1))
  if (( i % 4 == 0 )); then wait; fi
done
wait
grep -L "macro_f1_all" $RUNS/*-ood.log || true
echo "===== OOD DONE ====="
