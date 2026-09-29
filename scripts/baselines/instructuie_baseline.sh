#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-instructuie
#SBATCH --output=logs/%j-instructuie.out
#SBATCH --partition=workq
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
# Zero-shot InstructUIE (ZWK/InstructUIE, Flan-T5 11B) baseline: downloads the
# weights once, then four GPUs each take some of the OOD / PII sets; names only.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false
ROOT=/scratch/u6sn/yangw.u6sn/ie-rlcd
M=/scratch/u6sn/yangw.u6sn/huggingface_models/ZWK/InstructUIE
if [ ! -s $M/pytorch_model-00005-of-00005.bin ]; then
  HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download as d; \
d('ZWK/InstructUIE', local_dir='$M', max_workers=2)"
fi
ls $M | head -20
OUT=$ROOT/runs2/instructuie; mkdir -p $OUT
SETG=("mit_movie" "crossner_politics crossner_science mit_restaurant" "crossner_ai crossner_literature crossner_music" "pii_ai4privacy")
for g in 0 1 2 3; do
  D=$ROOT/data/ood; [ "${SETG[$g]}" = pii_ai4privacy ] && D=$ROOT/data/ood_pii
  CUDA_VISIBLE_DEVICES=$g python -m ierlcd.baselines.instructuie_eval --ood $D --model $M --sets ${SETG[$g]} \
    --out_dir $OUT > $OUT/gpu$g.log 2>&1 &
done
wait
tail -n 3 $OUT/gpu*.log | cut -c1-300
python -m ierlcd.baselines.instructuie_eval merge --out_dir $OUT --out $OUT/instructuie-name_only.json \
  --sets crossner_ai crossner_literature crossner_music crossner_politics crossner_science mit_movie mit_restaurant \
  --pii_sets pii_ai4privacy
