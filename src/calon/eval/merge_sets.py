"""Merge per-set evaluation results (one job per evaluation set, --sets) into the file one
whole-group run of ood_eval / re_eval / ee_eval writes: the union of "sets", the other keys of the
first part, and the group macros recomputed over all sets.

  python -m calon.eval.merge_sets --tool ood --out <file> <part.json> ...
"""
import argparse
import json
from pathlib import Path

import numpy as np


def macros(tool, res):
    sets = res["sets"]
    if tool == "ood":
        for key in res.get("settings", {}):
            cross = [sets[n][key]["f1"] for n in sets if n.startswith("crossner")]
            res[f"macro_f1_crossner[{key}]"] = float(np.mean(cross)) if cross else float("nan")
            res[f"macro_f1_all[{key}]"] = float(np.mean([v[key]["f1"] for v in sets.values()]))
    elif tool == "re":
        for mode in ("given", "end2end"):
            vals = [v[mode]["f1"] for v in sets.values() if mode in v]
            res[f"macro_f1_{mode}"] = float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=["ood", "re", "ee"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", type=int, default=0, help="fail unless this many sets were merged")
    ap.add_argument("parts", nargs="+")
    args = ap.parse_args()
    res = None
    for p in sorted(args.parts):
        part = json.loads(Path(p).read_text())
        if res is None:
            res = {k: v for k, v in part.items() if not k.startswith("macro")}
            res["sets"] = {}
        res["sets"].update(part["sets"])
    if args.expect and len(res["sets"]) != args.expect:
        raise SystemExit(f"{args.out}: {len(res['sets'])} sets merged, {args.expect} expected")
    macros(args.tool, res)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(args.out, len(res["sets"]), "sets", {k: round(v, 4) for k, v in res.items() if k.startswith("macro")})


if __name__ == "__main__":
    main()
