"""Compact result tables for a run directory, next to a baseline run directory."""
import argparse
import json
from pathlib import Path

ARMS = ["warmup", "bce-s0.2", "bce_scaled-s0.2", "direct-s0.2", "pg-s0.2", "direct-s1.0",
        "pg-s1.0", "direct-s2.0", "pg-s2.0"]
OOD = ["crossner_ai", "crossner_literature", "crossner_music", "crossner_politics",
       "crossner_science", "mit_movie", "mit_restaurant"]
KEY = "calibrated@fw_tuned"


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def ood_rows(runs, tag):
    lines = []
    for a in ARMS:
        r = load(Path(runs) / a / "ood.json")
        if r is None:
            continue
        f = [r["sets"][s][KEY]["f1"] for s in OOD]
        fpr = [r["sets"][s][KEY]["absent_query_fpr"] for s in OOD]
        lines.append(f"{tag:4s} {a:16s} " + " ".join(f"{x:.3f}" for x in f)
                     + f" | {r[f'macro_f1_crossner[{KEY}]']:.3f} {r[f'macro_f1_all[{KEY}]']:.3f}"
                     + " | FPR " + " ".join(f"{x:.2f}" for x in fpr))
    return lines


def errors(runs, arm):
    r = load(Path(runs) / arm / "ood.json")
    if r is None:
        return []
    out = []
    for s in OOD:
        e = r["sets"][s][KEY]["errors"]
        fp = {k: v for k, v in e["fp"].items() if k != "wrong_type_extra_label"}
        nfp, nfn = max(sum(fp.values()), 1), max(sum(e["fn"].values()), 1)
        out.append(f"  {s:20s} FP={nfp:5d} type {fp.get('wrong_type', 0) / nfp:.2f} "
                   f"bnd+type {fp.get('boundary_and_type', 0) / nfp:.2f} "
                   f"spurious {fp.get('spurious', 0) / nfp:.2f} | FN={nfn:5d} "
                   f"missed {e['fn'].get('missed', 0) / nfn:.2f} "
                   f"other-type {e['fn'].get('found_as_other_type', 0) / nfn:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--baseline")
    args = ap.parse_args()
    print(f"== in-domain test F1 (calibrated@tuned on pooled dev_calib), {args.runs}")
    for a in ARMS:
        c = load(Path(args.runs) / a / "calibration.json")
        if c is None:
            continue
        per = {s: v["calibrated@tuned"] for s, v in c["test"].items()}
        print(f"{a:16s} T={c['temperature']:.2f} "
              + " ".join(f"{s}={v['f1']:.3f}" for s, v in per.items())
              + f" | macro {c['test_macro_f1[calibrated@tuned]']:.3f}"
              + " | NLL " + " ".join(f"{v['nll_decision']:.4f}" for v in per.values())
              + " | ECEpred " + " ".join(f"{v['ece_predicted']:.3f}" for v in per.values()))
    print("== OOD F1: " + " ".join(s.replace("crossner_", "cn_")[:7] for s in OOD)
          + " | macroCN macroAll | absent-query FPR")
    lines = ood_rows(args.runs, "mix")
    if args.baseline:
        lines += ood_rows(args.baseline, "fw")
    print("\n".join(lines))
    for arm in ("warmup", "bce-s0.2"):
        print(f"== OOD error breakdown, mix {arm}")
        print("\n".join(errors(args.runs, arm)))
    w = Path(args.runs) / "warmup" / "eval.jsonl"
    if w.exists():
        print("== warm-up dev_select trajectory (macro F1, macro NLL)")
        for line in w.read_text().splitlines():
            r = json.loads(line)
            print(f"  {r['step']:6d} {r['macro_f1']:.4f} {r['macro_nll']:.5f}")


if __name__ == "__main__":
    main()
