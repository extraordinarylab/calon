#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-mix-prep
#SBATCH --output=logs/%j-mix-prep.out
#SBATCH --partition=workq
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# Label bank v2 (Qwen3.8-27B) and mixture build, shared by every seed.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX=$ROOT/data/mix
BANK=$MIX/label_bank_v2.json
QWEN=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3.8-27B
conda activate ie-rlcd
python -m calon.data.prepare_mix --stage inventory --out $MIX
if [ ! -s $BANK ]; then
  conda activate ie-rlcd-vllm
  python -m calon.synth.label_bank --inventory $MIX/inventory.json --model $QWEN --out $BANK \
    > $MIX/label_bank_v2.log 2>&1 || { tail -20 $MIX/label_bank_v2.log; exit 1; }
  tail -8 $MIX/label_bank_v2.log
  conda activate ie-rlcd
fi
python -m calon.data.prepare_mix --stage build --bank $BANK --out $MIX > $MIX/build.log 2>&1
tail -3 $MIX/build.log
echo "===== PREP DONE ====="
