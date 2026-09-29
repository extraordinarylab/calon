#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-ood-diag
#SBATCH --output=logs/%j-ood-diag.out
#SBATCH --partition=workq
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# OOD diagnostics on existing checkpoints: rerun the default (with-description)
# evaluation to get the error breakdown, and a name-only variant.
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1
python -m pytest -q tests/test_ood.py
SEED=${SEED:-42}
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
RUNS=$ROOT/runs/fw-s$SEED
OOD=$ROOT/data/ood

JOBS=()
for c in "warmup best.pt" bce-s0.2 bce_scaled-s0.2 direct-s0.2 pg-s0.2 direct-s1.0 pg-s1.0 \
         direct-s2.0 pg-s2.0; do
  read NAME FILE <<< "$c"
  FILE=${FILE:-final.pt}
  JOBS+=("$NAME $FILE desc" "$NAME $FILE name_only")
done
i=0
for j in "${JOBS[@]}"; do
  read NAME FILE VARIANT <<< "$j"
  FLAG=""; [ "$VARIANT" = name_only ] && FLAG=--name_only
  CUDA_VISIBLE_DEVICES=$((i % 4)) python -m ierlcd.eval.ood_eval --ood $OOD --ckpt $RUNS/$NAME/$FILE \
    --calibration $RUNS/$NAME/calibration.json --out $RUNS/$NAME/ood-$VARIANT.json $FLAG \
    > $RUNS/$NAME-ood-$VARIANT.log 2>&1 &
  i=$((i + 1))
  if (( i % 4 == 0 )); then wait; fi
done
wait
grep -L "macro_f1_all" $RUNS/*-ood-desc.log $RUNS/*-ood-name_only.log || true
echo "===== OOD DIAG DONE ====="
