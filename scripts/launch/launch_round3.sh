#!/usr/bin/env bash
# Round 3 (only sbatch calls: safe on the login node):
#   1. 40k music / literature / spoken-query synthetic texts (no target examples)
#   2. RLCD sweep on Qwen3 with the new data, seed 42: Brier weight {0, 0.25, 1} x
#      Few-NERD {fine only, coarse+fine} x updates {3000, 6000}; plus a BCE reference
#   3. seeds 45, 46 for the round-2 zero-shot BCE / RLCD runs (5 seeds in total)
# The final 5-seed runs of the selected RLCD setting are launched after the sweep
# (selection uses the training sources' dev_calib only).
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
NG2=${NG2:-32}
sub() { sbatch --parsable "$@"; }
P=$(sub --export=ALL,ARGS="prepare-g2 --n 40000" --job-name=gen3-prep scripts/synth/gen2_step.sh)
G2=$(sub --dependency=afterok:$P --array=0-$((NG2 - 1)) --export=ALL,TASK=g2,NSHARDS=$NG2 --job-name=gen3-g2 scripts/synth/gen2_run.sh)
AS=$(sub --dependency=afterok:$G2 --export=ALL,ARGS="assemble-g2" --job-name=gen3-assemble scripts/synth/gen2_step.sh)
echo "gen: prep=$P g2=$G2 assemble=$AS"

Q3=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B
Z=finerweb:0.10,fewnerd:0.15,pile:0.15,massive:0.05,synth_a:0.10,synth_g:0.25,synth_r:0.20
Z3=finerweb:0.08,fewnerd:0.12,pile:0.12,massive:0.04,synth_a:0.08,synth_g:0.20,synth_r:0.16,synth_g2:0.20
IDS=()
export ENCODER=$Q3 EXP=qwen3-z3 MIXSPEC=$Z3
for B in 0 0.25 1.0; do
  for F in 1 0; do
    for U in 3000 6000; do
      N=rlcd-b$B-f$F-u$U-dd0.3
      IDS+=($(sub --dependency=afterok:$AS --export=ALL,SEED=42,HEAD=rlcd,DDROP=0.3,BRIER=$B,FINE=$F,UPDATES=$U,NAME=$N \
              --job-name=r3-$N scripts/train/train_mn.sh))
    done
  done
done
IDS+=($(sub --dependency=afterok:$AS --export=ALL,SEED=42,HEAD=bce,DDROP=0.3 --job-name=r3-bce-dd0.3-s42 scripts/train/train_mn.sh))
echo "sweep: ${IDS[*]}"
echo "${IDS[*]}" > /scratch/u6sn/yangw.u6sn/ie-rlcd/runs2/sweep_ids.txt

export EXP=qwen3-zs MIXSPEC=$Z
EX=()
for HEAD in bce rlcd; do
  for S in 45 46; do
    EX+=($(sub --export=ALL,SEED=$S,HEAD=$HEAD,DDROP=0.3 --job-name=r3-zs-$HEAD-s$S scripts/train/train_mn.sh))
  done
done
echo "extra seeds: ${EX[*]}"
