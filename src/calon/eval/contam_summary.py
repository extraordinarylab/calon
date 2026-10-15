"""Contamination table: F1 on each benchmark's train / validation split vs its test
split, per model. A clear train > test gap
suggests the model trained on that split; InstructUIE on MIT is the known case."""
import argparse
import json
from pathlib import Path

import numpy as np

from calon.data.roots import ROOT

R = ROOT / "runs2"
C = R / "contam"
KEY = "calibrated@fw_tuned"
SETS = ["crossner_ai", "crossner_literature", "crossner_music", "crossner_politics", "crossner_science",
        "mit_movie", "mit_restaurant"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def glin(test_file, split_file):
    def get(split, name):
        o = load(test_file if split == "test" else split_file.format(split=split))
        return o["sets"].get(name, {}).get("f1") if o else None
    return get


def per_set_dir(test_dir, split_dir):
    def get(split, name):
        o = load((Path(test_dir) if split == "test" else Path(split_dir.format(split=split))) / f"{name}.json")
        return o["f1"] if o else None
    return get


def ours(eval_name, mode):
    def get(split, name):
        vals = []
        for s in (42, 43):
            p = (R / f"p2-z4-s{s}/{eval_name}/ood-{mode}.json" if split == "test"
                 else C / f"{split}-{eval_name}-s{s}-{mode}.json")
            o = load(p)
            if o and name in o["sets"]:
                vals.append(o["sets"][name][KEY]["f1"])
        return float(np.mean(vals)) if vals else None
    return get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    models = {"CalOn (ours, desc)": ours("rlcd-w0.1-f1-ent", "desc"),
              "CalOn (ours, name)": ours("rlcd-w0.1-f1-ent", "name_only"),
              "BCE (ours, desc)": ours("bce-f1", "desc")}
    for m in ["gliner2-base-v1", "gliner2-large-v1"]:
        for v in ["desc", "name_only"]:
            models[f"{m} ({v})"] = glin(R / f"gliner2/{m}-{v}.json", str(C / ("{split}-" + f"{m}-{v}.json")))
    for m in ["gliformer-base-v1", "gliformer-large-v1"]:
        models[f"{m} (name_only)"] = glin(R / f"gliner2/{m}-name_only.json",
                                          str(C / ("{split}-" + f"{m}-name_only.json")))
    models["GNER-LLaMA-7B (name_only)"] = per_set_dir(R / "llm-ner/GNER-LLaMA-7B", str(C / "{split}-GNER-LLaMA-7B"))
    models["InstructUIE (name_only, positive control)"] = per_set_dir(R / "instructuie", str(C / "{split}-InstructUIE"))
    fmt = lambda v: "   -  " if v is None else f"{v:.3f}"
    lines = ["F1 by split. gap = train - test (CrossNER: also validation - test).", ""]
    for model, get in models.items():
        lines.append(f"== {model}")
        gaps_cn, gaps_mit = [], []
        for name in SETS:
            te, tr, va = get("test", name), get("train", name), get("validation", name)
            gap = None if te is None or tr is None else tr - te
            (gaps_mit if name.startswith("mit") else gaps_cn).append(gap)
            lines.append(f"  {name:20s} test {fmt(te)}  train {fmt(tr)}  valid {fmt(va)}  gap {fmt(gap)}")
        g = lambda xs: fmt(float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None)
        lines.append(f"  mean gap CrossNER {g(gaps_cn)}   MIT {g(gaps_mit)}")
    Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
