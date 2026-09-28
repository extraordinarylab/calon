#!/usr/bin/env bash
# Submit one experiment (only sbatch calls; safe on the login node).
#   EXP=name MIXSPEC=src:p,... [ENCODER=...] [ARMSET=full|spherical] [INIT_EXP=...]
#   [SEEDS="42 43 44"] [AFTER=jobid:jobid] bash scripts/launch_exp.sh
# full:      per seed a 4-GPU DDP warm-up, then 8 single-GPU forks
# spherical: no warm-up; 6 forks (direct/pg x sigma 0.2/1/2, alpha 0.1) from
#            INIT_EXP's warm-ups, written into runs/<EXP>-s<seed>
# Prints the ids of every job it submitted as the last line "jobs: ...".
set -euo pipefail
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
EXP=${EXP:?}; MIXSPEC=${MIXSPEC:?}
ENCODER=${ENCODER:-answerdotai/ModernBERT-base}
ARMSET=${ARMSET:-full}
INIT_EXP=${INIT_EXP:-$EXP}
SEEDS=${SEEDS:-"42 43 44"}
DEP=${AFTER:+--dependency=afterok:$AFTER}
ALL=()
for S in $SEEDS; do
  COMMON="ALL,SEED=$S,EXP=$EXP,MIXSPEC=$MIXSPEC,ENCODER=$ENCODER,INIT_EXP=$INIT_EXP"
  if [ "$ARMSET" = full ]; then
    W=$(sbatch --parsable $DEP --export=$COMMON --job-name=wu-$EXP-s$S scripts/mix_warmup_ddp.sh)
    ALL+=($W)
    FDEP="--dependency=afterok:$W"
    ARMS=("bce 0.2 0" "bce_scaled 0.2 0" "direct 0.2 0" "pg 0.2 0" "direct 1.0 0" "pg 1.0 0" "direct 2.0 0" "pg 2.0 0")
  else
    FDEP=$DEP
    ARMS=("direct 0.2 0.1" "pg 0.2 0.1" "direct 1.0 0.1" "pg 1.0 0.1" "direct 2.0 0.1" "pg 2.0 0.1")
  fi
  for arm in "${ARMS[@]}"; do
    read MODE SIGMA ALPHA <<< "$arm"
    TAG=""; [ "$ALPHA" != 0 ] && TAG=-sph
    F=$(sbatch --parsable $FDEP --export=$COMMON,MODE=$MODE,SIGMA=$SIGMA,ALPHA=$ALPHA,TAG=$TAG \
        --job-name=fk-$EXP-s$S-$MODE-$SIGMA$TAG scripts/mix_fork.sh)
    ALL+=($F)
  done
done
echo "jobs: ${ALL[*]}"
