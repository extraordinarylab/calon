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
  # Existing vLLM envs do not work here: lco-embed/world carry a flash_attn built for
  # torch 2.9 under torch 2.11; drivelology-judge's vllm wheel targets CUDA 13, newer
  # than the node driver. Build a clean one with the cu129 aarch64 wheel.
  if [ ! -x $CONDA/envs/ie-rlcd-vllm/bin/python ]; then
    step build_vllm_env
    conda create -y -q -n ie-rlcd-vllm python=3.12 > /dev/null
    conda activate ie-rlcd-vllm
    pip install -q --no-cache-dir torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu129
    pip install -q --no-cache-dir \
      "https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_aarch64.whl"
  fi
  conda activate ie-rlcd-vllm
  # PyPI torchvision/torchaudio (pulled in by vllm) do not match torch cu129
  if ! ls -d $CONDA/envs/ie-rlcd-vllm/lib/python3.12/site-packages/torchvision-*+cu129.dist-info >/dev/null 2>&1; then
    step fix_torchvision
    pip install -q --no-cache-dir --no-deps --force-reinstall torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu129
  fi
  pip freeze > logs/env-vllm-freeze.txt
  step label_bank
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
