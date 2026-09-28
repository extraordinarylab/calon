"""Compact result tables for a run directory, next to a baseline run directory."""
import argparse
import collections
import json
from pathlib import Path

ARMS = ["warmup", "bce-s0.2", "bce_scaled-s0.2", "direct-s0.2", "pg-s0.2", "direct-s1.0",
        "pg-s1.0", "direct-s2.0", "pg-s2.0",
        "direct-s0.2-sph", "pg-s0.2-sph", "direct-s1.0-sph", "pg-s1.0-sph", "direct-s2.0-sph",
        "pg-s2.0-sph", "bce-12k", "rlcd-12k", "bce-dd0", "bce-dd0.3", "rlcd-dd0", "rlcd-dd0.3",
        "rlcd-dd0-ent", "rlcd-dd0.3-ent"]
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


def arm_metrics(runs, arm):
    c = load(Path(runs) / arm / "calibration.json")
    o = load(Path(runs) / arm / "ood.json")
    if c is None or o is None:
        return None
    if isinstance(c.get("test"), dict) and "raw@0.5" in c["test"]:   # FiNERWeb-only format
        in_f1 = c["test"]["calibrated@tuned"]["f1"]
        in_nll = c["test"]["calibrated@tuned"]["nll_decision"]
    else:
        in_f1 = c["test_macro_f1[calibrated@tuned]"]
        in_nll = sum(v["calibrated@tuned"]["nll_decision"] for v in c["test"].values()) / len(c["test"])
    fpr = [o["sets"][s][KEY]["absent_query_fpr"] for s in OOD]
    ece = [o["sets"][s][KEY]["ece_predicted"] for s in OOD]
    miss = [1 - o["sets"][s][KEY]["recall"] for s in OOD]          # entity-level false negatives
    qfnr = [o["sets"][s][KEY].get("present_query_fnr") for s in OOD]
    return {"in_f1": in_f1, "in_nll": in_nll, "ood_cn": o[f"macro_f1_crossner[{KEY}]"],
            "ood_all": o[f"macro_f1_all[{KEY}]"], "mit_movie": o["sets"]["mit_movie"][KEY]["f1"],
            "mit_rest": o["sets"]["mit_restaurant"][KEY]["f1"],
            "ood_fpr": sum(fpr) / len(fpr), "ood_ece_pred": sum(ece) / len(ece),
            "ood_miss": sum(miss) / len(miss),
            **({"ood_qfnr": sum(qfnr) / len(qfnr)} if all(x is not None for x in qfnr) else {})}


def seeds_table(seed_dirs, baseline):
    import statistics
    cols = ["in_f1", "in_nll", "ood_cn", "ood_all", "mit_movie", "mit_rest", "ood_fpr", "ood_ece_pred"]
    print(f"== mean ± std over seeds {[Path(d).name for d in seed_dirs]} "
          "(in_* = in-domain test macro over sources; ood_* = zero-shot, T/thr from own dev_calib)")
    print(f"{'arm':16s} n " + " ".join(f"{c:>15s}" for c in cols))
    for a in ARMS:
        ms = [m for d in seed_dirs if (m := arm_metrics(d, a))]
        if not ms:
            continue
        cells = []
        for c in cols:
            xs = [m[c] for m in ms]
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            cells.append(f"{statistics.mean(xs):.4f}±{sd:.4f}")
        print(f"{a:16s} {len(ms)} " + " ".join(f"{x:>15s}" for x in cells))
    if baseline:
        print("-- FiNERWeb-only baseline (seed 42; in_* is FiNERWeb test only)")
        for a in ARMS:
            m = arm_metrics(baseline, a)
            if m:
                print(f"{a:16s} 1 " + " ".join(f"{m[c]:15.4f}" for c in cols))


def collect_all(root, exps, seeds):
    """{exp: {arm: {metric: [per-seed values]}}} including per-OOD-set F1 and FPR."""
    import statistics
    out = {}
    for exp in exps:
        dirs = [Path(root) / f"{exp}-s{s}" for s in seeds]
        dirs = [d for d in dirs if d.exists()]
        if exp == "fw":
            dirs = [Path(root) / "fw-s42"]
        arms = {}
        found = sorted({p.parent.name for d in dirs for p in d.glob("*/ood.json")})
        for a in [x for x in ARMS if x in found] + [x for x in found if x not in ARMS]:
            vals = collections.defaultdict(list)
            for d in dirs:
                m = arm_metrics(d, a)
                o = load(d / a / "ood.json")
                if m is None or o is None:
                    continue
                for k, v in m.items():
                    vals[k].append(v)
                c = load(d / a / "calibration.json")
                if c and not ("raw@0.5" in c.get("test", {})):
                    for src, t in c["test"].items():
                        vals[f"test/{src}"].append(t["calibrated@tuned"]["f1"])
                elif c:
                    vals["test/finerweb"].append(c["test"]["calibrated@tuned"]["f1"])
                pii = load(d / a / "ood-pii-desc.json")
                if pii is not None:                    # separate PII evaluation set
                    pm = pii["sets"]["pii_ai4privacy"][KEY]
                    vals["pii_f1"].append(pm["f1"])
                    vals["pii_fpr"].append(pm["absent_query_fpr"])
                    vals["pii_miss"].append(1 - pm["recall"])
                o2 = load(d / a / "ood-name_only.json")
                if o2 is not None:                     # same model, labels by name only
                    vals["no_desc_ood_all"].append(o2[f"macro_f1_all[{KEY}]"])
                    vals["no_desc_ood_cn"].append(o2[f"macro_f1_crossner[{KEY}]"])
                    vals["no_desc_fpr"].append(sum(o2["sets"][s][KEY]["absent_query_fpr"] for s in OOD) / len(OOD))
                for s in OOD:
                    vals[f"f1/{s}"].append(o["sets"][s][KEY]["f1"])
                    vals[f"fpr/{s}"].append(o["sets"][s][KEY]["absent_query_fpr"])
                    vals[f"ece_pred/{s}"].append(o["sets"][s][KEY]["ece_predicted"])
            if vals:
                arms[a] = {k: {"mean": statistics.mean(v),
                               "std": statistics.stdev(v) if len(v) > 1 else 0.0, "n": len(v)}
                           for k, v in vals.items()}
        out[exp] = arms
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs")
    ap.add_argument("--seeds", nargs="+")
    ap.add_argument("--baseline")
    ap.add_argument("--root", help="runs root, with --exps")
    ap.add_argument("--exps", nargs="+")
    ap.add_argument("--json_out")
    args = ap.parse_args()
    if args.exps:
        allres = collect_all(args.root, args.exps, [42, 43, 44])
        Path(args.json_out).write_text(json.dumps(allres, indent=1))
        cols = ["in_f1", "ood_cn", "ood_all", "mit_movie", "mit_rest", "ood_fpr", "ood_miss", "ood_qfnr",
                "ood_ece_pred", "no_desc_ood_all", "no_desc_fpr", "pii_f1", "pii_fpr", "pii_miss"]
        g = Path(args.root) / "gliner2"
        if g.exists():
            print("== GLiNER2 zero-shot baselines (default threshold, same test sets and scoring)")
            for f in sorted(g.glob("*.json")):
                r = json.loads(f.read_text())
                sets = " ".join(f"{s.replace('crossner_', 'cn_')[:7]}={v['f1']:.3f}" for s, v in r["sets"].items())
                miss = sum(1 - v["recall"] for v in r["sets"].values()) / len(r["sets"])
                print(f"  {f.stem:28s} all={r['macro_f1_all']:.4f} cn={r['macro_f1_crossner']:.4f} "
                      f"fpr={r['mean_absent_fpr']:.3f} miss={miss:.3f} | {sets}")
        print("exp/arm".ljust(28) + "".join(c.rjust(16) for c in cols))
        for exp, arms in allres.items():
            for a, m in arms.items():
                print(f"{exp + '/' + a:28s}" + "".join(
                    f"{m[c]['mean']:.4f}±{m[c]['std']:.4f}".rjust(16) if c in m else " " * 16
                    for c in cols))
        for arm in ("warmup", "bce-s0.2", "bce-12k", "rlcd-12k", "bce-dd0", "bce-dd0.3", "rlcd-dd0", "rlcd-dd0.3"):
            if not any(arm in arms for arms in allres.values()):
                continue
            print(f"== per OOD set F1, arm {arm} (mean over seeds)")
            print("exp".ljust(14) + "".join(s.replace("crossner_", "cn_")[:10].rjust(11) for s in OOD))
            for exp, arms in allres.items():
                if arm in arms:
                    print(exp.ljust(14) + "".join(f"{arms[arm]['f1/' + s]['mean']:11.4f}" for s in OOD))
            print(f"== in-domain test F1 per source, arm {arm}")
            srcs = ["finerweb", "fewnerd", "pile", "massive", "synth_a", "synth_c"]
            print("exp".ljust(14) + "".join(s.rjust(11) for s in srcs))
            for exp, arms in allres.items():
                if arm in arms:
                    print(exp.ljust(14) + "".join(
                        f"{arms[arm]['test/' + s]['mean']:11.4f}" if f"test/{s}" in arms[arm] else " " * 11
                        for s in srcs))
        return
    if args.seeds:
        seeds_table(args.seeds, args.baseline)
        return
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
