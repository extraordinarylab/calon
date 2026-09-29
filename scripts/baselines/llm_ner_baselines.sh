#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-llm-ner
#SBATCH --output=logs/%j-llm-ner.out
#SBATCH --partition=workq
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Decoder-LLM NER baselines trained only on Pile-NER data (GNER-LLaMA-7B):
# download once, then every OOD set
# and the PII set. Results: runs2/llm-ner/<model>/<set>.json.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false
conda activate ie-rlcd
python -m pytest -q tests/test_llm_ner_parse.py 2>&1 | tail -2
conda activate ie-rlcd-vllm
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
HFM=/scratch/u6sn/yangw.u6sn/huggingface_models
# UniNER and SLIMER are not used: they ask one type per prompt, so one span can get
# several types; GNER tags each word once, like RLCD picks one label per span
MODELS=(gner:dyyyyyyyy/GNER-LLaMA-7B)
for mb in "${MODELS[@]}"; do
  id=${mb#*:}
  if [ ! -s $HFM/$id/config.json ] || ! ls $HFM/$id/*.safetensors $HFM/$id/*.bin >/dev/null 2>&1; then
    HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download as d; \
d('$id', local_dir='$HFM/$id', max_workers=2, allow_patterns=['*.json','*.model','*.safetensors','*.bin','*.txt','*.py']) " \
      > $TMPDIR/dl-$(basename $id).log 2>&1 &
  fi
done
wait
export HF_HUB_OFFLINE=1
OOD="crossner_ai crossner_literature crossner_music crossner_politics crossner_science mit_movie mit_restaurant"
g=0
for mb in "${MODELS[@]}"; do
  b=${mb%%:*}; id=${mb#*:}; OUT=$ROOT/runs2/llm-ner/$(basename $id); mkdir -p $OUT
  ls $HFM/$id > $OUT/files.txt
  ( CUDA_VISIBLE_DEVICES=$g python -m calon.baselines.llm_ner_eval --backend $b --model $HFM/$id --ood $ROOT/data/ood \
      --sets $OOD --out_dir $OUT > $OUT/ood.log 2>&1
    CUDA_VISIBLE_DEVICES=$g python -m calon.baselines.llm_ner_eval --backend $b --model $HFM/$id --ood $ROOT/data/ood_pii \
      --sets pii_ai4privacy --out_dir $OUT > $OUT/pii.log 2>&1 ) &
  g=$((g + 1))
done
wait
for mb in "${MODELS[@]}"; do
  id=${mb#*:}; echo "== $id"; grep -hE "^(crossner|mit|pii)|Error" $ROOT/runs2/llm-ner/$(basename $id)/*.log | cut -c1-220
done
