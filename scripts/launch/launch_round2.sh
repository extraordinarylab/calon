#!/usr/bin/env bash
# Round 2 as one dependency graph (only sbatch calls: safe on the login node):
#   Qwen generation arrays -> prepare-r2 -> r2 array -> assemble ->
#   48 trainings (2 encoders x 2 heads x 2 data sets x 2 description-dropout rates
#   x 3 seeds, 4 nodes each) and the GLiNER2 baselines.
# GPU counts per array are set from the smoke-test throughput.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
S2=/scratch/u6sn/yangw.u6sn/ie-rlcd/synth2
NG=${NG:-40}; NR1=${NR1:-32}; NC2=${NC2:-8}; NN=${NN:-8}; NR2=${NR2:-24}
rm -f $S2/*/shard-*-of-10000.jsonl                   # smoke-test outputs
sub() { sbatch --parsable "$@"; }
G=$(sub --array=0-$((NG - 1)) --export=ALL,TASK=g,NSHARDS=$NG --job-name=gen2-g scripts/synth/gen2_run.sh)
R1=$(sub --array=0-$((NR1 - 1)) --export=ALL,TASK=r1,NSHARDS=$NR1 --job-name=gen2-r1 scripts/synth/gen2_run.sh)
C2=$(sub --array=0-$((NC2 - 1)) --export=ALL,TASK=c2,NSHARDS=$NC2 --job-name=gen2-c2 scripts/synth/gen2_run.sh)
N=$(sub --array=0-$((NN - 1)) --export=ALL,TASK=n,NSHARDS=$NN --job-name=gen2-n scripts/synth/gen2_run.sh)
P2=$(sub --dependency=afterok:$R1 --export=ALL,ARGS="prepare-r2" --job-name=gen2-prep-r2 scripts/synth/gen2_step.sh)
R2=$(sub --dependency=afterok:$P2 --array=0-$((NR2 - 1)) --export=ALL,TASK=r2,NSHARDS=$NR2 --job-name=gen2-r2 scripts/synth/gen2_run.sh)
AS=$(sub --dependency=afterok:$G:$R2:$C2:$N --export=ALL,ARGS="assemble" --job-name=gen2-assemble scripts/synth/gen2_step.sh)
echo "gen: g=$G r1=$R1 c2=$C2 n=$N prep_r2=$P2 r2=$R2 assemble=$AS"

Z=finerweb:0.10,fewnerd:0.15,pile:0.15,massive:0.05,synth_a:0.10,synth_g:0.25,synth_r:0.20
S=finerweb:0.08,fewnerd:0.12,pile:0.12,massive:0.04,synth_a:0.08,synth_g:0.20,synth_r:0.16,synth_c:0.20
Q3=/scratch/u6sn/yangw.u6sn/huggingface_models/Qwen/Qwen3-0.6B
IDS=()
for enc in "qwen3 $Q3" "mbert answerdotai/ModernBERT-base"; do
  read ETAG ENCODER <<< "$enc"
  for data in "zs $Z" "seeded $S"; do
    read DTAG MIXSPEC <<< "$data"
    export EXP=$ETAG-$DTAG ENCODER MIXSPEC
    for HEAD in bce rlcd; do
      for DDROP in 0 0.3; do
        for SEED in 42 43 44; do
          IDS+=($(sub --dependency=afterok:$AS --export=ALL,SEED=$SEED,HEAD=$HEAD,DDROP=$DDROP \
                  --job-name=r2-$EXP-$HEAD-dd$DDROP-s$SEED scripts/train/train_mn.sh))
        done
      done
    done
  done
done
echo "train: ${#IDS[@]} jobs ${IDS[0]}..${IDS[-1]}"
GB=$(sub scripts/baselines/gliner2_baseline.sh)
echo "gliner2 baseline: $GB"
SUM=$(sub --dependency=afterany:$(IFS=:; echo "${IDS[*]}"):$GB --export=ALL,EXPS="qwen3-zs qwen3-seeded mbert-zs mbert-seeded",ROOT_RUNS=runs2 \
      scripts/eval/summary_all.sh)
echo "summary: $SUM"
