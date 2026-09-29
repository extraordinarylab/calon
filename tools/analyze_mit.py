"""Why MIT Movie / Restaurant are weak: per-label F1 of our CalOn vs BCE vs gliformer,
error types, and how MIT mentions differ from training mentions (length, case)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd")
KEY = "calibrated@fw_tuned"
SETS = ["mit_movie", "mit_restaurant"]
OURS = {"calon": "rlcd-w0.1-ent", "bce": "bce-u6000"}


def rows(p, limit=None):
    with open(p) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                return
            yield json.loads(line)


def mention_stats(p, limit=None):
    words, lower, n = [], 0, 0
    for r in rows(p, limit):
        for e in r["entities"]:
            s = r["text"][e["start"]:e["end"]]
            words.append(len(s.split()))
            lower += s == s.lower()
            n += 1
    return {"mentions": n, "mean_words": float(np.mean(words)) if words else 0,
            "share_3plus_words": float(np.mean([w >= 3 for w in words])) if words else 0,
            "share_all_lowercase": lower / n if n else 0}


def main():
    runs = ROOT / "runs2"
    glif = json.loads((runs / "mit-analysis/gliformer-large-v1-name_only.json").read_text())["sets"]
    g2 = json.loads((runs / "mit-analysis/gliner2-large-v1-desc.json").read_text())["sets"]
    for s in SETS:
        gold = Counter(e["label"] for r in rows(ROOT / f"data/ood/{s}.jsonl") for e in r["entities"])
        per = defaultdict(dict)
        for name, arm in OURS.items():
            for mode in ("desc", "name_only"):
                vals = defaultdict(list)
                for seed in (42, 43):
                    o = json.loads((runs / f"p1-z3-s{seed}/{arm}/ood-{mode}.json").read_text())["sets"][s][KEY]
                    for lab, v in o["per_label"].items():
                        vals[lab].append(v)
                    if name == "calon" and mode == "desc" and seed == 42:
                        print(f"== {s} CalOn error types (seed 42, with descriptions):", json.dumps(o["errors"]))
                for lab, v in vals.items():
                    keys = ("precision", "recall", "f1")
                    per[lab][f"{name}-{mode}"] = tuple(np.mean([x[k] for x in v]) for k in keys)
        for lab, v in glif[s]["per_label"].items():
            per[lab]["glif-name"] = (v["precision"], v["recall"], v["f1"])
        for lab, v in g2[s]["per_label"].items():
            per[lab]["gliner2-desc"] = (v["precision"], v["recall"], v["f1"])
        cols = ["rlcd-desc", "rlcd-name_only", "bce-desc", "glif-name", "gliner2-desc"]
        print(f"== {s}: per-label P/R/F1   (columns: {' | '.join(cols)})")
        for lab in sorted(gold, key=lambda l: -gold[l]):
            cells = [("%.2f/%.2f/%.2f" % per[lab][c]) if c in per[lab] else "-" for c in cols]
            print(f"  {lab:22s} n={gold[lab]:5d}  " + " | ".join(cells))
        print(f"  mentions: {json.dumps(mention_stats(ROOT / f'data/ood/{s}.jsonl'))}")
    for src in ["finerweb", "fewnerd", "pile", "massive", "synth_g", "synth_g2", "synth_r"]:
        print(f"train {src}: {json.dumps(mention_stats(ROOT / f'data/mix2/{src}/train.jsonl', 5000))}")


if __name__ == "__main__":
    sys.exit(main())
