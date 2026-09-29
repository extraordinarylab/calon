"""Table for a separate evaluation directory (spy / new): our 5-seed round-4 finals
(CalOn entity decoding, BCE) and the baselines, per set: F1, precision, recall,
absent-query FPR, entity miss rate."""
import argparse
import json
from pathlib import Path

import numpy as np

R = Path("/scratch/u6sn/yangw.u6sn/ie-rlcd/runs2")
KEY = "calibrated@fw_tuned"
COLS = ["f1", "precision", "recall", "absent_query_fpr", "entity_miss_rate"]


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True, help="spy or new: files ood-<ext>-<mode>.json, runs2/<ext>/")
    ap.add_argument("--exp", default="p2-z4")
    ap.add_argument("--arms", default="rlcd-w0.1-f1-ent,bce-f1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = {}
    for arm in args.arms.split(","):
        for mode in ("desc", "name_only"):
            per = {}
            for s in range(42, 47):
                o = load(R / f"{args.exp}-s{s}/{arm}/ood-{args.ext}-{mode}.json")
                if not o:
                    continue
                for name, v in o["sets"].items():
                    per.setdefault(name, []).append([v[KEY].get(c, np.nan) for c in COLS])
            if per:
                rows[f"{arm} ({mode}, {len(next(iter(per.values())))} seeds)"] = {
                    n: (np.mean(v, 0), np.std(v, 0)) for n, v in per.items()}
    for f in sorted((R / args.ext).glob("*.json")):
        o = load(f)
        if o and "sets" in o:
            rows[f.stem] = {n: (np.array([v.get(c, np.nan) for c in COLS]), None) for n, v in o["sets"].items()}
    for d in sorted(p for p in (R / args.ext).iterdir() if p.is_dir()):
        per = {f.stem: load(f) for f in d.glob("*.json")}
        per = {n: (np.array([v.get(c, np.nan) for c in COLS]), None) for n, v in per.items() if v}
        if per:
            rows[f"{d.name} (name_only)"] = per
    sets = sorted({n for v in rows.values() for n in v})
    lines = []
    for name in sets:
        lines.append(f"== {name}   columns: " + " | ".join(COLS))
        for model, per in rows.items():
            if name not in per:
                continue
            m, s = per[name]
            cells = [f"{x:.3f}" + (f"±{y:.3f}" if s is not None and k == 0 else "") for k, (x, y) in
                     enumerate(zip(m, s if s is not None else m))]
            lines.append(f"  {model:48s} " + " | ".join(cells))
    Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
