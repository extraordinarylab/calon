"""Download round-1 sources at a pinned revision and write a format summary.

Each repo is snapshot at its current commit sha (recorded in the summary), plus
its datasets-server parquet conversion when one exists, because Few-NERD and
MASSIVE are script-based on the main branch.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# repo -> allow_patterns for the parquet conversion (None = all)
REPOS = {"DFKI-SLT/few-nerd": ["supervised/*"], "Universal-NER/Pile-NER-type": None,
         "Universal-NER/Pile-NER-definition": None, "AmazonScience/massive": ["en-US/*"]}
MAX_PREVIEW = 1500


def preview_file(path):
    """First records of a json/jsonl/parquet/tsv file, truncated."""
    out = {"file": str(path), "bytes": path.stat().st_size}
    suffix = path.suffix.lower()
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            out["rows"] = pf.metadata.num_rows
            out["schema"] = str(pf.schema_arrow)[:MAX_PREVIEW]
            out["first"] = json.dumps(pf.read_row_group(0).slice(0, 2).to_pylist(),
                                      default=str)[:MAX_PREVIEW]
        elif suffix in {".json", ".jsonl"}:
            with open(path) as f:
                head = f.read(4096)
            if head.lstrip().startswith("["):
                data = json.load(open(path))
                out["rows"] = len(data)
                out["first"] = json.dumps(data[:2])[:MAX_PREVIEW]
            else:
                with open(path) as f:
                    out["rows"] = sum(1 for _ in f)
                out["first"] = head[:MAX_PREVIEW]
        elif suffix in {".txt", ".tsv", ".py", ".md"}:
            out["first"] = path.read_text(errors="replace")[:MAX_PREVIEW]
    except Exception as exc:                       # summary only; never fatal
        out["error"] = repr(exc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    api = HfApi()
    summary = {}
    for repo, parquet_patterns in REPOS.items():
        info = api.dataset_info(repo)
        rec = {"sha": info.sha, "gated": info.gated, "card_license": getattr(info.card_data, "license", None)}
        for rev, sub in ((info.sha, "main"), ("refs/convert/parquet", "parquet")):
            local = out / repo.replace("/", "__") / sub
            try:
                snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=local,
                                  allow_patterns=parquet_patterns if sub == "parquet" else None,
                                  max_workers=2)
            except Exception as exc:
                rec[f"{sub}_error"] = repr(exc)
                continue
            if sub == "parquet":
                refs = api.list_repo_refs(repo, repo_type="dataset")
                rec["parquet_sha"] = next((c.target_commit for c in refs.converts), None)
            files = sorted(p for p in local.rglob("*") if p.is_file() and ".cache" not in p.parts)
            rec[f"{sub}_files"] = [{"path": str(p.relative_to(local)), "bytes": p.stat().st_size}
                                   for p in files]
            rec[f"{sub}_previews"] = [preview_file(p) for p in files
                                      if p.suffix.lower() in {".parquet", ".json", ".jsonl", ".py", ".md"}
                                      and ("en" in p.name or "en-US" in str(p) or sub == "main"
                                           or "supervised" in str(p))][:12]
        summary[repo] = rec
        print(repo, rec["sha"], flush=True)
    (out / "fetch_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
