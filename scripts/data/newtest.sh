#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-newtest
#SBATCH --output=logs/%j-newtest.out
#SBATCH --partition=workq
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Fresh contamination test sets (data/ood_new), no Qwen anywhere: Nemotron classifies
# the new Wikipedia articles, gpt-oss writes MIT-style requests, then Nemotron,
# gpt-oss and Hunyuan-A13B each annotate every snippet; >= 2 of 3 must agree.
# Needs newtest/wiki_pages.jsonl from `newtest fetch-wiki`.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd-vllm
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
H=/scratch/u6sn/yangw.u6sn/huggingface_models
NEM=$H/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16; OSS=$H/openai/gpt-oss-120b; HUN=$H/tencent/Hunyuan-A13B-Instruct
L=/scratch/u6sn/yangw.u6sn/ie-rlcd/newtest/logs; mkdir -p $L
run() { local gpus=$1; shift; TRITON_CACHE_DIR=$TMPDIR/triton-${gpus//,/} CUDA_VISIBLE_DEVICES=$gpus python -m calon.eval.newtest "$@"; }
run 0 classify --model $NEM > $L/classify.log 2>&1 &
[ -s /scratch/u6sn/yangw.u6sn/ie-rlcd/newtest/snippets_mit.jsonl ] || run 1 gen-mit --model $OSS > $L/genmit.log 2>&1 &
wait
tail -1 $L/classify.log | cut -c1-300; tail -1 $L/genmit.log 2>/dev/null | cut -c1-300 || true
run 0 snippets 2>&1 | tail -1
run 0 annotate --model $NEM > $L/a-nem.log 2>&1 &
run 1 annotate --model $OSS > $L/a-oss.log 2>&1 &
wait
run 0,1,2,3 annotate --model $HUN --tp 4 > $L/a-hun.log 2>&1 || { echo "Hunyuan FAILED"; grep -E "Error" $L/a-hun.log | tail -3 | cut -c1-300; }
for f in $L/a-*.log; do echo "$(basename $f): $(tail -1 $f | cut -c1-200)"; done
conda activate ie-rlcd
python -m calon.eval.newtest assemble 2>&1 | tail -2 | cut -c1-1500
