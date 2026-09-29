#!/usr/bin/env bash
# Build ie-rlcd-vllm with uv so torch / torchvision / torchaudio resolve together
# from the cu129 index (the node driver supports CUDA 12.x, not 13). Called from
# inside a compute job, never on the login node.
set -euo pipefail
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
ENV=$CONDA/envs/ie-rlcd-vllm
export UV_CACHE_DIR=/scratch/u6sn/yangw.u6sn/uv-cache
source $CONDA/etc/profile.d/conda.sh
rm -rf $ENV
conda create -y -q -p $ENV python=3.12 > /dev/null
$CONDA/bin/uv pip install --python $ENV/bin/python --torch-backend=cu129 -q \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0%2Bcu129-cp38-abi3-manylinux_2_28_aarch64.whl"
# torchcodec's .so fails to load here and vLLM does not catch OSError; it is only
# used for video inputs.
$CONDA/bin/uv pip uninstall --python $ENV/bin/python -q torchcodec || true
$CONDA/bin/uv pip freeze --python $ENV/bin/python > logs/env-vllm-freeze.txt
grep -iE "^(torch|torchvision|torchaudio|vllm|transformers)[=@ ]" logs/env-vllm-freeze.txt
