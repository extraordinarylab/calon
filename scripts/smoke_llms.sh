#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-smoke-llms
#SBATCH --output=logs/%j-smoke-llms.out
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# Can vLLM run each candidate non-Qwen annotator, how fast, and how well does it
# follow the JSON annotation prompt? Also: can a compute node reach the Wikipedia API?
set -uo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd-vllm
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
H=/scratch/u6sn/yangw.u6sn/huggingface_models
OUT=/scratch/u6sn/yangw.u6sn/ie-rlcd/runs2/smoke-llms; mkdir -p $OUT
python - <<'EOF'
import json, urllib.request
u = ("https://en.wikipedia.org/w/api.php?action=query&list=logevents&letype=create&lenamespace=0"
     "&lestart=2026-01-01T00:00:00Z&ledir=older&lelimit=3&format=json")
try:
    req = urllib.request.Request(u, headers={"User-Agent": "ie-rlcd-research/0.1"})
    d = json.load(urllib.request.urlopen(req, timeout=20))
    print("wikipedia api ok:", [e["title"] for e in d["query"]["logevents"]])
except Exception as e:
    print("wikipedia api FAILED:", repr(e)[:200])
EOF
for M in google/gemma-4-26B-A4B-it tencent/Hunyuan-A13B-Instruct nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16; do
  timeout 3000 python -m ierlcd.smoke_llms --model $H/$M --out $OUT/$(basename $M).json > $OUT/$(basename $M).log 2>&1 \
    && tail -1 $OUT/$(basename $M).log | cut -c1-300 || { echo "FAILED $M"; grep -E "Error|error" $OUT/$(basename $M).log | tail -3 | cut -c1-300; }
done
