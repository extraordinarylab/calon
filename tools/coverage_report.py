"""Tokenizer coverage on each split: boundary alignment, width coverage, candidate recall."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from ierlcd.data.alignment import coverage, tokenize_text
from ierlcd.data.collator import load_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--W", type=int, default=32)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.encoder)
    report = {}
    for split in ("train", "dev_select", "dev_calib", "test"):
        tot, lens, widths, labels_per_row = collections.Counter(), [], [], []
        for r in load_rows(Path(args.data) / f"{split}.jsonl"):
            _, offsets = tokenize_text(tok, r["text"])
            lens.append(len(offsets))
            labels_per_row.append(len(r["queries"]))
            tot.update(coverage(r["text"], offsets, r["entities"], args.W))
            for ent in r["entities"]:
                widths.append(len(tokenize_text(tok, r["text"][ent["start"]:ent["end"]])[0]))
        g = max(tot["gold"], 1)
        report[split] = {
            "rows": len(lens), "gold": tot["gold"],
            "boundary_alignment_recall": tot["boundary"] / g,
            "width_coverage_given_boundary": tot["width"] / max(tot["boundary"], 1),
            "candidate_recall": tot["width"] / g,
            "text_tokens": {"mean": float(np.mean(lens)), "p99": float(np.percentile(lens, 99)),
                            "max": int(max(lens))},
            "entity_tokens_standalone": {"p99": float(np.percentile(widths, 99)),
                                         "max": int(max(widths))},
            "queries_per_row": {"mean": float(np.mean(labels_per_row)),
                                "max": int(max(labels_per_row))},
        }
    print(json.dumps(report, indent=2))
    (Path(args.data) / f"coverage_W{args.W}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
