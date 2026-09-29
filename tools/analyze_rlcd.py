"""Why does RLCD lose to BCE on some OOD sets? Compact comparison from saved ood.json.

For each OOD set: precision / recall / F1 and absent-query FPR of both heads
(mean over seeds), the error breakdown, the per-label F1 with the largest
gaps, and the number of labels asked per row; plus the mean number of queries
per training row seen by the model.
"""
import argparse
import collections
import json
from pathlib import Path

KEY = "calibrated@fw_tuned"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/scratch/u6sn/yangw.u6sn/ie-rlcd/runs2")
    ap.add_argument("--exp", default="qwen3-zs")
    ap.add_argument("--arms", default="bce-dd0.3,rlcd-dd0.3")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--ood", default="/scratch/u6sn/yangw.u6sn/ie-rlcd/data/ood")
    ap.add_argument("--labels", help="set:label,... -> print per-label F1 of every arm, then exit")
    args = ap.parse_args()
    if args.labels:
        arms, seeds = args.arms.split(","), args.seeds.split(",")
        res = {a: [json.loads((Path(args.runs) / f"{args.exp}-s{s}" / a / "ood-desc.json").read_text())
                   for s in seeds] for a in arms}
        print("set:label " + " ".join(arms))
        for item in args.labels.split(","):
            name, lab = item.split(":")
            f = [mean([r["sets"][name][KEY]["per_label"][lab]["f1"] for r in res[a]]) for a in arms]
            print(f"{name}:{lab} " + " ".join(f"{x:.3f}" for x in f))
        return
    arms, seeds = args.arms.split(","), args.seeds.split(",")
    res = {a: [json.loads((Path(args.runs) / f"{args.exp}-s{s}" / a / "ood-desc.json").read_text())
               for s in seeds] for a in arms}
    cal = {a: [json.loads((Path(args.runs) / f"{args.exp}-s{s}" / a / "calibration.json").read_text())
               for s in seeds] for a in arms}
    for a in arms:
        print(f"{a}: T={mean([c['temperature'] for c in cal[a]]):.2f} "
              f"threshold={mean([c['dev_calib']['calibrated@tuned']['threshold'] for c in cal[a]]):.2f}")
    for a in arms:
        qs = collections.Counter()
        for s in seeds:
            p = Path(args.runs) / f"{args.exp}-s{s}" / a / "query_stats_rank0.json"
            if p.exists():
                qs.update(json.loads(p.read_text()))
        rows = sum(v for k, v in qs.items() if k.startswith("rows/"))
        qn = sum(v for k, v in qs.items() if "/" in k and not k.startswith(("rows/", "desc/")))
        print(f"{a}: training queries per row {qn / max(rows, 1):.1f}")
    sets = list(res[arms[0]][0]["sets"])
    for name in sets:
        n_labels = len(json.loads(open(Path(args.ood) / f"{name}.jsonl").readline())["queries"])
        print(f"\n== {name}  ({n_labels} labels asked per row)")
        for a in arms:
            m = [r["sets"][name][KEY] for r in res[a]]
            fp, fn = collections.Counter(), collections.Counter()
            for x in m:
                fp.update(x["errors"]["fp"])
                fn.update(x["errors"]["fn"])
            nfp = sum(v for k, v in fp.items() if k != "wrong_type_extra_label") or 1
            nfn = sum(fn.values()) or 1
            print(f"  {a:12s} P={mean([x['precision'] for x in m]):.3f} R={mean([x['recall'] for x in m]):.3f} "
                  f"F1={mean([x['f1'] for x in m]):.3f} absentFPR={mean([x['absent_query_fpr'] for x in m]):.3f} "
                  f"| FP/seed={nfp / len(m):.0f} type={fp['wrong_type'] / nfp:.2f} "
                  f"bnd+type={fp['boundary_and_type'] / nfp:.2f} "
                  f"spur={fp['spurious'] / nfp:.2f} | FN/seed={nfn / len(m):.0f} missed={fn['missed'] / nfn:.2f} "
                  f"other-type={fn['found_as_other_type'] / nfn:.2f} bnd={fn['boundary'] / nfn:.2f}")
        labs = res[arms[0]][0]["sets"][name][KEY]["per_label"].keys()
        gaps = []
        for lab in labs:
            f = {a: mean([r["sets"][name][KEY]["per_label"][lab]["f1"] for r in res[a]]) for a in arms}
            r_ = {a: mean([r["sets"][name][KEY]["per_label"][lab]["recall"] for r in res[a]]) for a in arms}
            p_ = {a: mean([r["sets"][name][KEY]["per_label"][lab]["precision"] for r in res[a]]) for a in arms}
            gaps.append((f[arms[1]] - f[arms[0]], lab, f, p_, r_))
        for g, lab, f, p_, r_ in sorted(gaps)[:5]:
            print(f"    {lab:22s} F1 {f[arms[0]]:.2f}->{f[arms[1]]:.2f}  P {p_[arms[0]]:.2f}->{p_[arms[1]]:.2f}  "
                  f"R {r_[arms[0]]:.2f}->{r_[arms[1]]:.2f}")


if __name__ == "__main__":
    main()
