#!/usr/bin/env bash
# Qwen data generation at scale, then experiments 2 (mixture + A) and 3 (+ C),
# as one dependency graph. Only sbatch calls: safe on the login node.
# Smoke-test throughput on one GPU: a1 4.1 passages/s, a2 8.4 passages/s,
# c 1.8 requests/s -> ~6.8, ~3.3 and ~1.1 GPU-hours for 100k / 100k / 7k.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
S=/scratch/u6sn/yangw.u6sn/ie-rlcd/synth
rm -rf $S/a1 $S/a2 $S/c                 # smoke-test outputs (different shard layout)
A1=$(sbatch --parsable --array=0-47 --export=ALL,TASK=a1,NSHARDS=48 --job-name=synth-a1 scripts/synth_run.sh)
C=$(sbatch --parsable --array=0-7 --export=ALL,TASK=c,NSHARDS=8 --job-name=synth-c scripts/synth_run.sh)
P2=$(sbatch --parsable --dependency=afterok:$A1 --export=ALL,ARGS="prepare-a2 --min-type-count 3" \
     --job-name=synth-prep-a2 scripts/synth_step.sh)
A2=$(sbatch --parsable --dependency=afterok:$P2 --array=0-23 --export=ALL,TASK=a2,NSHARDS=24 \
     --job-name=synth-a2 scripts/synth_run.sh)
AS=$(sbatch --parsable --dependency=afterok:$A2:$C --export=ALL,ARGS="assemble" \
     --job-name=synth-assemble scripts/synth_step.sh)
echo "a1=$A1 c=$C prep_a2=$P2 a2=$A2 assemble=$AS"
# round-1 base shares (0.25/0.30/0.30/0.15) scaled to what the synthetic sources leave
AFTER=$AS EXP=mixA25 MIXSPEC=finerweb:0.1875,fewnerd:0.225,pile:0.225,massive:0.1125,synth_a:0.25 \
  bash scripts/launch_exp.sh
AFTER=$AS EXP=mixA50 MIXSPEC=finerweb:0.125,fewnerd:0.15,pile:0.15,massive:0.075,synth_a:0.50 \
  bash scripts/launch_exp.sh
AFTER=$AS EXP=mixA25C15 MIXSPEC=finerweb:0.15,fewnerd:0.18,pile:0.18,massive:0.09,synth_a:0.25,synth_c:0.15 \
  bash scripts/launch_exp.sh
AFTER=$AS EXP=mixA50C15 MIXSPEC=finerweb:0.0875,fewnerd:0.105,pile:0.105,massive:0.0525,synth_a:0.50,synth_c:0.15 \
  bash scripts/launch_exp.sh
