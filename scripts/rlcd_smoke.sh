#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-rlcd-smoke
#SBATCH --output=logs/%j-rlcd-smoke.out
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# RLCD head: tests, then 200 updates on the mixA25 mixture (Few-NERD fine types
# only), calibration and OOD eval, for both encoders. Compact output only.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
SPEC=finerweb:0.1875,fewnerd:0.225,pile:0.225,massive:0.1125,synth_a:0.25
SRC=finerweb,fewnerd,pile,massive,synth_a
python -m pytest -q tests 2>&1 | tail -2
for ENC in answerdotai/ModernBERT-base /scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen2.5-0.5B; do
  OUT=$ROOT/runs/rlcd-smoke/$(basename $ENC); rm -rf $OUT
  echo "== $ENC"
  python -m ierlcd.train --data $MIX --out $OUT --mix $SPEC --encoder $ENC --head rlcd \
    --fewnerd_fine_only --updates 200 --eval_every 200 --eval_limit 200 --workers 16 2>&1 \
    | grep -E "EVAL|done|Error|error" | cut -c1-250
  grep '"step": 200' $OUT/log.jsonl | cut -c1-250
  python -m ierlcd.calibrate --data $MIX --sources $SRC --encoder $ENC --head rlcd --fewnerd_fine_only \
    --ckpt $OUT/final.pt --limit 200 --out $OUT/calibration.json 2>&1 | grep -vE "filter_long" | tail -7
  python -m ierlcd.ood_eval --ood $ROOT/data/ood --encoder $ENC --head rlcd --ckpt $OUT/final.pt \
    --calibration $OUT/calibration.json --out $OUT/ood.json 2>&1 | grep -E "macro_f1_all|Error" | cut -c1-200
done
