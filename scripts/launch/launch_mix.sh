#!/usr/bin/env bash
# Submit the whole round-1 mixture as one dependency graph (only sbatch calls, safe
# on the login node): prep -> 3 DDP warm-ups (one node each) -> 24 single-GPU
# forks -> summary.
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
SEEDS=${SEEDS:-"42 43 44"}
ARMS=("bce 0.2" "bce_scaled 0.2" "direct 0.2" "pg 0.2" "direct 1.0" "pg 1.0" "direct 2.0" "pg 2.0")
DEP=""
if [ -z "${SKIP_PREP:-}" ]; then
  P=$(sbatch --parsable scripts/data/mix_prep.sh); echo "prep $P"; DEP="--dependency=afterok:$P"
fi
FORKS=()
for S in $SEEDS; do
  W=$(SEED=$S sbatch --parsable $DEP --export=ALL,SEED=$S scripts/train/mix_warmup_ddp.sh)
  echo "warmup s$S $W"
  for arm in "${ARMS[@]}"; do
    read MODE SIGMA <<< "$arm"
    F=$(sbatch --parsable --dependency=afterok:$W --export=ALL,SEED=$S,MODE=$MODE,SIGMA=$SIGMA \
        --job-name=fork-s$S-$MODE-$SIGMA scripts/train/mix_fork.sh)
    FORKS+=($F)
  done
done
echo "forks ${FORKS[*]}"
SUM=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${FORKS[*]}") scripts/data/mix_summary.sh)
echo "summary $SUM"
