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
