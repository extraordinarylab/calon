#!/usr/bin/env bash
# Final zero-shot comparison, 5 seeds (only sbatch calls: safe on the login node):
# removes the 8 train rows the punctuation-insensitive audit flagged, then trains
#   RLCD  Brier 0, Few-NERD fine only, 6,000 updates, 30% description dropout
#         (chosen on the training sources' dev_calib; both decodes evaluated)
#   BCE   3,000 and 6,000 updates, 30% description dropout
# on the round-3 zero-shot mixture, each on 4 nodes, with OOD + PII evaluation.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
A=$(sbatch --parsable --export=ALL --job-name=audit-remove --wrap="export TMPDIR=/tmp/\$USER-\$SLURM_JOB_ID; mkdir -p \$TMPDIR; source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh; conda activate ie-rlcd; cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd; PYTHONPATH=src python -m calon.data.audit_leak --mix /scratch/u6sn/yangw.u6sn/ie-rlcd/data/mix2 --remove --out /scratch/u6sn/yangw.u6sn/ie-rlcd/runs2/leak_audit_removed.json" \
     -p workq --time=1:00:00 --gres=gpu:1 --cpus-per-task=16 -o logs/%j-audit-remove.out)
echo "audit-remove $A"
export ENCODER=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B EXP=final-z3
export MIXSPEC=finerweb:0.08,fewnerd:0.12,pile:0.12,massive:0.04,synth_a:0.08,synth_g:0.20,synth_r:0.16,synth_g2:0.20
IDS=()
for S in 42 43 44 45 46; do
  IDS+=($(sbatch --parsable --dependency=afterok:$A --export=ALL,SEED=$S,HEAD=rlcd,DDROP=0.3,BRIER=0,FINE=1,UPDATES=6000,NAME=rlcd-b0-f1-u6000-dd0.3 \
          --job-name=fin-rlcd-s$S scripts/train/train_mn.sh))
  for U in 3000 6000; do
    IDS+=($(sbatch --parsable --dependency=afterok:$A --export=ALL,SEED=$S,HEAD=bce,DDROP=0.3,UPDATES=$U,NAME=bce-u$U-dd0.3 \
            --job-name=fin-bce-u$U-s$S scripts/train/train_mn.sh))
  done
done
echo "final: ${IDS[*]}"
echo "${IDS[*]}" > /scratch/u6sn/yangw.u6sn/ie-rlcd/runs2/final_ids.txt
