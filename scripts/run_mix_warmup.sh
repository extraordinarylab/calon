#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mix-warmup
#SBATCH --output=logs/%j-mix-warmup.out
#SBATCH --partition=workq
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
# Job A of the round-1 mixture: label bank v2 (Qwen3.8-27B), mixture rebuild,
# 10k-update BCE warm-up. Job B (run_mix.sh) forks from warmup/best.pt.
set -euo pipefail
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SEED=${SEED:-42}
UPDATES=${UPDATES:-10000}
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
BANK=$MIX/label_bank_v2.json
RUNS=$ROOT/runs/mix-s$SEED
QWEN=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.8-27B
mkdir -p $RUNS
git rev-parse HEAD > $RUNS/git_commit.txt

conda activate ie-rlcd
python -m ierlcd.prepare_mix --stage inventory --out $MIX
if [ ! -s $BANK ]; then
  conda activate ie-rlcd-vllm
  python -m ierlcd.label_bank --inventory $MIX/inventory.json --model $QWEN --out $BANK \
    > $RUNS/label_bank.log 2>&1 || { tail -20 $RUNS/label_bank.log; exit 1; }
  tail -8 $RUNS/label_bank.log
  conda activate ie-rlcd
fi
python -m ierlcd.prepare_mix --stage build --bank $BANK --out $MIX > $RUNS/build.log 2>&1
python -m ierlcd.train --data $MIX --out $RUNS/warmup --mode bce --updates $UPDATES \
  --eval_every 1000 --workers 48 --seed $SEED > $RUNS/warmup.log 2>&1
grep -E "EVAL|done" $RUNS/warmup.log | cut -c1-300
echo "===== WARMUP DONE ====="
