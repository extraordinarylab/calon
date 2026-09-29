#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-env
#SBATCH --output=logs/%j-env.out
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# Build the aarch64 `ie-rlcd` environment on a GH200 node, then check that
# ModernBERT loads and runs a bf16 forward pass on the GPU.
set -euo pipefail
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
if ! conda env list | grep -q '/ie-rlcd$'; then
  conda create -y -n ie-rlcd python=3.12
fi
conda activate ie-rlcd
pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir transformers==5.15.0 accelerate==1.14.0 datasets==5.0.0 \
  "numpy==2.2.6" safetensors==0.8.0 pytest scikit-learn
pip freeze > logs/env-freeze.txt
HF_HUB_OFFLINE=1 python - <<'PY'
import platform, torch, transformers
from transformers import AutoModel, AutoTokenizer
print("machine", platform.machine(), "torch", torch.__version__, "cuda", torch.version.cuda,
      "transformers", transformers.__version__, "gpu", torch.cuda.get_device_name(0))
name = "answerdotai/ModernBERT-base"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModel.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
enc = tok(["Acme hired Maya."], return_tensors="pt", return_offsets_mapping=True)
print("offsets", enc.pop("offset_mapping").tolist())
with torch.no_grad():
    h = m(**{k: v.cuda() for k, v in enc.items()}).last_hidden_state
print("hidden", tuple(h.shape), h.dtype, "attn", m.config._attn_implementation)
PY
echo "===== ENV DONE ====="
