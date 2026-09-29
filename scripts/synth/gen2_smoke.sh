#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-gen2-smoke
#SBATCH --output=logs/%j-gen2-smoke.out
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# 256 items of each generation task on one GPU: throughput plus two parsed
# examples per task, written to a separate smoke directory.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
source /lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/etc/profile.d/conda.sh
conda activate ie-rlcd-vllm
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
for T in g c2 r1 n; do
  echo "== $T $(date +%T)"
  EXTRA=""; case $T in n) EXTRA="--max-tokens 768";; g) EXTRA="--max-tokens 2048";; esac
  python -m ierlcd.synth.llm_gen2 run --task $T --shard 0 --num-shards 10000 --limit 256 --chunk 256 $EXTRA \
    2>&1 | grep -E "^shard|^chunk|Error" | cut -c1-200
done
conda activate ie-rlcd
python - <<'PY'
import json, glob
from ierlcd.synth.llm_annotate import parse_json
for t in ("g", "c2", "r1", "n"):
    rows = [json.loads(l) for f in glob.glob(f"/scratch/u6sn/yangw.u6sn/ie-rlcd/synth2/{t}/shard-00000-of-10000.jsonl") for l in open(f)]
    ok = [r for r in rows if isinstance(parse_json(r["output"]), dict)]
    cut = sum(r["finish_reason"] == "length" for r in rows)
    print(f"--- {t}: {len(rows)} outputs, {len(ok)} parse, {cut} truncated")
    for r in ok[:2]:
        print("   ", r["output"][:600].replace("\n", " "))
PY
