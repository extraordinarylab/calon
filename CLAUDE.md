## 跟使用者溝通：用人話

使用者看不懂技術細節堆疊的說明。回報時：

- 用繁體中文白話講，不要中英夾雜；非用不可的術語（例如 F1、seed）第一次出現時用一句話解釋。
- 先講結論：現在在做什麼、結果是好是壞、下一步是什麼、大概多久。
- 用具體例子取代抽象描述（例如「模型把『Maya』判成人名的機率」，而不是「span–label logit」）。
- 不要列出 job ID、檔案路徑、參數名稱，除非使用者問起。
- 一次回報控制在幾句話或一張小表；細節等使用者問再講。

## Code layout

| Path | Contents |
| --- | --- |
| `src/ierlcd/model/` | span-label model, losses |
| `src/ierlcd/data/` | tokenization / candidates, collator, source adapters, mixture sampling, eval-set builders, leakage audit |
| `src/ierlcd/synth/` | LLM generation / annotation (Qwen, REBEL), label descriptions, refinement (re-annotation, RLCD flags, judge) |
| `src/ierlcd/train/` | DDP training, calibration |
| `src/ierlcd/eval/` | metrics, OOD evaluation, summaries, fresh contamination test sets |
| `src/ierlcd/baselines/` | GLiNER2 / gliformer, InstructUIE, GNER evaluation |
| `tools/` | one-off analyses and smoke tests (not part of the package; run as `python -m tools.<name>` from the repo root) |
| `scripts/<env,data,synth,train,eval,baselines,launch>/` | sbatch scripts by purpose; `launch/` scripts only call sbatch |

New code goes into the matching subpackage; nothing new at `src/ierlcd/` top level.

## Never compute on the login node

Development, smoke tests and debugging go to the **interactive** partition.
Full runs go to **workq**. The login node is for editing files, reading logs,
inspecting artifacts and submitting jobs — nothing that loads a model or
compiles.

```bash
# development / smoke test / environment build
sbatch --partition=interactive --reservation=interactive --time=8:00:00 \
  --gres=gpu:1 --cpus-per-task=32 <script>

# the real run
sbatch --partition=workq --time=23:59:00 --gres=gpu:4 --cpus-per-task=64 <script>
```

Environment builds (`conda create`, `pip install`, compiling CUDA extensions)
count as compute: submit them, do not run them on the login node.

### The login node runs nothing — not even "small" things

Commands on the login node have repeatedly got the Claude session killed and
crashed VS Code. The login node is only for:

- editing files (Read / Edit / Write), `git`
- `sbatch`, `squeue`, `sacct`, `scancel`
- `tail` / `cat` / `ls` of logs and small result files

Everything else goes into an sbatch script on the **interactive** partition,
however quick it looks. That includes:

- any `python` — data inspection, JSON summaries of results, `pytest`,
  one-line `python -c`
- importing libraries to check versions (`import vllm`, `import torch`):
  read `site-packages/*.dist-info` names with `ls` instead
- scanning datasets, `du` / `find` over large trees (weights, HF caches)
- `curl` / `hf` downloads — agents doing web research use WebSearch /
  WebFetch, not shell commands

For result tables, have the job itself write a small summary file (text or
JSON) and `cat` it.

Also:

- No background polling loops on the login node (`until sacct ...; sleep`,
  monitors). Submit, stop, and check `sacct` once when asked or when the user
  says the job is done.
- Only read compact summaries (tens of lines). Jobs must not dump long
  previews; never Read multi-MB or very-long-line files.
- Change files with Edit / Write, not shell `sed`, heredocs or `python`.

## Partition limits

| | interactive | workq |
| --- | --- | --- |
| Wall clock | 8 h | 24 h (`workq_qos`) |
| Jobs per user | **1**, counting queued | 256 running, 512 submitted |
| Dependency chains | impossible — the second link is always rejected | `sbatch --dependency=afterany:<id>` |

A run longer than 24 h is a chain on workq. Interactive cannot hold one at all,
so it is for single short jobs only. Nodes are aarch64 GH200, 4 GPUs, 288
cores, 460 GB.

## Scale by nodes, and measure before committing

Do a one-instance smoke test, get a real seconds-per-item number, then size the
allocation from it. Say the cost before starting something that runs for days.

Sharding is by data, not by rewriting upstream: give each worker its own slice
and let the pipeline's own resume skip what is already on disk. Ranges must not
overlap — two workers on the same slice write the same files concurrently.

## Conda environments

At `/lus/lfs1aip2/scratch/u6sn/yangw.u6sn/miniforge3/envs/`, reachable as
`/scratch/u6sn/yangw.u6sn/miniforge3`. There is **no** `miniforge3` under
`$HOME`, so `source ~/miniforge3/etc/profile.d/conda.sh` fails.

## Data, weights and results

Datasets, model weights and generated videos stay on
`/scratch/u6sn/yangw.u6sn/`, never in git. Only code, configs, pinned upstream
snapshots and dated reports are committed.

`hf download` gets OOM-killed on the login node with xet enabled: pass
`HF_HUB_DISABLE_XET=1 --max-workers 2`. Downloading with `--revision <sha>`
writes no `refs/main`, so an offline `from_pretrained` will not find the model;
write that file by hand.

## Git

Commits and pull requests go out as **penguinwang96825
<penguinwang@smail.nchu.edu.tw>**. Do not add a `Co-Authored-By: Claude`
trailer or a "Generated with Claude Code" line.

Check `git config user.email` before the first commit: this repository's
*local* config has been set to another contributor's identity, which silently
overrides the correct global one.

Branch before experimenting, and stage explicit paths rather than `git add -A`
— switching branches leaves other branches' build artifacts in the tree.
