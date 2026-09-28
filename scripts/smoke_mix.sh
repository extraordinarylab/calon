#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-smoke-mix
#SBATCH --output=logs/%j-smoke-mix.out
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# Round-1 mixture end to end on one GPU: tests, label inventory, Qwen3.8-27B
# authored descriptions, mixture build, a 200-update timed run, calibration and
# OOD eval of that run. Output is kept to short summaries.
set -euo pipefail
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
BANK=$MIX/label_bank_v1.json
QWEN=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.8-27B
RUN=$ROOT/runs/smoke-mix
step() { echo "== $1 $(date +%T)"; }

conda activate ie-rlcd
step tests;      python -m pytest -q tests 2>&1 | tail -3
step inventory;  python -m ierlcd.prepare_mix --stage inventory --out $MIX

if [ ! -s $BANK ]; then
  step label_bank
  conda activate lco-embed
  python -m ierlcd.label_bank --inventory $MIX/inventory.json --model $QWEN --out $BANK \
    > $ROOT/label_bank.log 2>&1 || { tail -20 $ROOT/label_bank.log; exit 1; }
  tail -8 $ROOT/label_bank.log
  conda activate ie-rlcd
fi

step build;      python -m ierlcd.prepare_mix --stage build --bank $BANK --out $MIX 2>&1 | tail -60
step train
rm -rf $RUN
python -m ierlcd.train --data $MIX --out $RUN --updates 200 --eval_every 100 --eval_limit 200 \
  2>&1 | grep -E "loaded|EVAL|done|Error|error" | cut -c1-400
tail -1 $RUN/log.jsonl | cut -c1-300
step calibrate
python -m ierlcd.calibrate --data $MIX --ckpt $RUN/final.pt --limit 200 \
  --out $RUN/calibration.json 2>&1 | tail -6
step ood
python -m ierlcd.ood_eval --ood $ROOT/data/ood --ckpt $RUN/final.pt \
  --calibration $RUN/calibration.json --out $RUN/ood.json 2>&1 | grep -E "macro|Error" | cut -c1-300
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
step done
