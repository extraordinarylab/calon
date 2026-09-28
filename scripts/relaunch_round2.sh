#!/usr/bin/env bash
# Re-assemble the Qwen data (now filtered to one 512-token window) and resubmit
# experiments 2 and 3. Only sbatch calls: safe on the login node.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
rm -rf /scratch/u6sn/yangw.u6sn/ie-rlcd/runs/mixA*
AS=$(sbatch --parsable --export=ALL,ARGS="assemble" --job-name=synth-assemble scripts/synth_step.sh)
echo "assemble=$AS"
AFTER=$AS EXP=mixA25 MIXSPEC=finerweb:0.1875,fewnerd:0.225,pile:0.225,massive:0.1125,synth_a:0.25 \
  bash scripts/launch_exp.sh | cut -c1-60
AFTER=$AS EXP=mixA50 MIXSPEC=finerweb:0.125,fewnerd:0.15,pile:0.15,massive:0.075,synth_a:0.50 \
  bash scripts/launch_exp.sh | cut -c1-60
AFTER=$AS EXP=mixA25C15 MIXSPEC=finerweb:0.15,fewnerd:0.18,pile:0.18,massive:0.09,synth_a:0.25,synth_c:0.15 \
  bash scripts/launch_exp.sh | cut -c1-60
AFTER=$AS EXP=mixA50C15 MIXSPEC=finerweb:0.0875,fewnerd:0.105,pile:0.105,massive:0.0525,synth_a:0.50,synth_c:0.15 \
  bash scripts/launch_exp.sh | cut -c1-60
