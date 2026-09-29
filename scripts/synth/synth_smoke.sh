#!/usr/bin/env bash
#SBATCH --job-name=ie-rlcd-synth-smoke
#SBATCH --output=logs/%j-synth-smoke.out
#SBATCH --partition=interactive
#SBATCH --reservation=interactive
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
# Qwen data generation on one GPU: prepare the full A passage pool and C requests,
# run 300 A passages (both passes) and 100 C requests, assemble, and print speed
# plus a few examples.
set -euo pipefail
export TMPDIR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $TMPDIR
CONDA=/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3
source $CONDA/etc/profile.d/conda.sh
cd /home/u6sn/yangw.u6sn/workspace/ie-rlcd
export PYTHONPATH=src HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
S=/scratch/u6sn/yangw.u6sn/ie-rlcd/synth
step() { echo "== $1 $(date +%T)"; }

conda activate ie-rlcd
[ -s $S/a1_requests.jsonl ] || { step prepare_a; python -m ierlcd.synth.llm_annotate prepare-a --n 100000 2>&1 | tail -2 | cut -c1-600; }
[ -s $S/c_requests.jsonl ] || { step prepare_c; python -m ierlcd.synth.llm_annotate prepare-c --per-domain 1000 2>&1 | tail -2; }
conda activate ie-rlcd-vllm
export HF_HUB_OFFLINE=1
step run_a1; python -m ierlcd.synth.llm_annotate run --task a1 --limit 300 --chunk 300 2>&1 | grep -E "^shard|^chunk|Error" | cut -c1-200
step run_c;  python -m ierlcd.synth.llm_annotate run --task c --limit 100 --chunk 100 2>&1 | grep -E "^shard|^chunk|Error" | cut -c1-200
conda activate ie-rlcd
step prepare_a2; python -m ierlcd.synth.llm_annotate prepare-a2 --min-type-count 1 2>&1 | tail -1
conda activate ie-rlcd-vllm
step run_a2; python -m ierlcd.synth.llm_annotate run --task a2 --limit 300 --chunk 300 --max-tokens 768 2>&1 | grep -E "^shard|^chunk|Error" | cut -c1-200
conda activate ie-rlcd
step assemble; python -m ierlcd.synth.llm_annotate assemble 2>&1 | grep -E '"(rows|entities|stats|a1_|a2_|c_|synth)' | head -30
step examples
python - <<'PY'
import json
for src in ("synth_a", "synth_c"):
    rows = [json.loads(l) for l in open(f"/scratch/u6sn/yangw.u6sn/ie-rlcd/data/mix/{src}/train.jsonl")][:3]
    for r in rows:
        print(f"--- {src} {r.get('domain', '')}: {r['text'][:220]}")
        print("   positive:", [(r["text"][e["start"]:e["end"]], e["label"]) for e in r["entities"]][:8])
        print("   absent:", [q["label"] for q in r["queries"] if q["status"] in ("explicit_absent", "scope_assumed_absent")][:8])
        print("   descr:", [(q["label"], q["description"][:80]) for q in r["queries"] if q["status"] == "positive"][:2])
PY
step done
