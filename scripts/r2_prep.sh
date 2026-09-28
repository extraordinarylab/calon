#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-r2-prep
#SBATCH --output=logs/%j-r2-prep.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
# Round 2 preparation: data/mix2 with a 1,024-token window, real-text passages,
# Qwen generation requests (g, c2, n), Qwen3-0.6B weights and bidirectionality
# tests, and a separate env for the GLiNER2 baseline. Compact output only.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false UV_CACHE_DIR=/scratch/u6sn/yangw.u6sn/uv-cache
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
MIX2=$ROOT/data/mix2
step() { echo "== $1 $(date +%T)"; }

step mix2
python -m ierlcd.prepare_mix --stage inventory --out $MIX2 | tail -1
python -m ierlcd.prepare_mix --stage build --bank $ROOT/data/mix/label_bank_v2.json --out $MIX2 \
  --max_tokens 1024 > $MIX2.build.log 2>&1; tail -2 $MIX2.build.log | cut -c1-300
rm -rf $MIX2/synth_a && cp -r $ROOT/data/mix/synth_a $MIX2/synth_a

step qwen3
Q3=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B
if [ ! -s $Q3/config.json ]; then
  python - <<PY
from huggingface_hub import HfApi, snapshot_download
sha = HfApi().model_info("Qwen/Qwen3-0.6B").sha
snapshot_download("Qwen/Qwen3-0.6B", revision=sha, local_dir="$Q3", max_workers=2)
open("$Q3/REVISION", "w").write(sha + "\n")
PY
fi
cat $Q3/REVISION
QWEN_TEST_PATH=$Q3 python -m pytest -q tests 2>&1 | tail -2

step real_text
python -m ierlcd.llm_gen2 fetch-real 2>&1 | tail -1 | cut -c1-500
step requests
python -m ierlcd.llm_gen2 prepare-g --n 120000 | tail -1
python -m ierlcd.llm_gen2 prepare-c2 --per-domain 2060 | tail -1
python -m ierlcd.llm_gen2 prepare-n | tail -3

step gliner2_env
G=$CONDA/envs/gliner2
if [ ! -x $G/bin/python ]; then
  conda create -y -q -p $G python=3.11 > /dev/null
  $CONDA/bin/uv pip install --python $G/bin/python --torch-backend=cu129 -q "gliner2[local]" || \
    $CONDA/bin/uv pip install --python $G/bin/python --torch-backend=cu129 -q gliner2
fi
ls -d $G/lib/python3.11/site-packages/{gliner2,torch,transformers}-*.dist-info 2>/dev/null | xargs -n1 basename
step done
