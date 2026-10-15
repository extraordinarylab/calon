"""mix15 (or MIX_NAME, with REFILTERED / ROUNDS from the environment): mix14 with the target-seeded data, every
source weighted by the square root of its rows.

  - mix14's sources, linked as they are, except synth_c / synth_cre / synth_cee: their train rows
    are the re-checked ones (data/seed_refiltered), their held-out splits mix14's
  - the seeded data of all rounds (data/seed, seed_r2, seed_r3) split by evaluation set into
    train-only sources seed_<set> (no dev_select / dev_calib: model selection and calibration
    stay on the other sources, as in ee5)
  - sampling share of a source proportional to sqrt(train rows) (a middle way between equal
    shares and shares by size)
Writes data/mix15/<source>/, data/mix15/mixspec.txt (the MIXSPEC string) and summary.json (rows,
shares, task shares). Only the seed_<set> train files are written; the rest are symlinks.
"""
import collections
import json
import math
import os

from calon.data.roots import ROOT

D = ROOT / "data"
OUT = D / os.environ.get("MIX_NAME", "mix15")
BASE = D / "mix14"
REFILTERED = D / os.environ.get("REFILTERED", "seed_refiltered")   # re-checked synth_c / synth_cre / synth_cee
ROUNDS = tuple(os.environ.get("ROUNDS", "seed,seed_r2,seed_r3").split(","))
SPLITS = ("train.jsonl", "dev_select.jsonl", "dev_calib.jsonl", "test.jsonl")
EE = {"synth_gee", "synth_cee", "synth_nest", "ccnews_ee", "maven_arg", "geneva", "rams", "phee", "synth_phr"}


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def count(path):
    with open(path) as f:
        return sum(1 for _ in f)


def task_of(name, sample):
    if name in EE or name.startswith("seed_") and sample.get("id", "").endswith((":trig", ":args")):
        return "EE"
    return "RE" if "rel_queries" in sample else "NER"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in BASE.glob("*.json"):            # inventory.json (Few-NERD type tree), manifest.json
        link(f, OUT / f.name)
    rows, task = {}, {}
    for d in sorted(p for p in BASE.iterdir() if (p / "train.jsonl").exists()):
        out = OUT / d.name
        out.mkdir(exist_ok=True)
        for sp in SPLITS:
            if (d / sp).exists():
                link(d / sp, out / sp)
        if (REFILTERED / d.name / "train.jsonl").exists():
            link(REFILTERED / d.name / "train.jsonl", out / "train.jsonl")
        rows[d.name] = count(out / "train.jsonl")
        with open(out / "train.jsonl") as f:
            task[d.name] = task_of(d.name, json.loads(f.readline()))
    by_set = collections.defaultdict(list)
    for rnd in ROUNDS:
        for src in ("synth_sner", "synth_sre", "synth_see"):
            p = D / rnd / src / "train.jsonl"
            if not p.exists():
                continue
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    by_set[r["domain"]].append(line if line.endswith("\n") else line + "\n")
    for name, lines in sorted(by_set.items()):
        src = f"seed_{name}"
        out = OUT / src
        out.mkdir(exist_ok=True)
        with open(out / "train.jsonl", "w") as f:
            f.writelines(lines)
        rows[src] = len(lines)
        task[src] = task_of(src, json.loads(lines[0]))
    w = {s: math.sqrt(n) for s, n in rows.items() if n}
    z = sum(w.values())
    share = {s: v / z for s, v in w.items()}
    spec = ",".join(f"{s}:{share[s]:.5f}" for s in sorted(share))
    (OUT / "mixspec.txt").write_text(spec + "\n")
    tasks = collections.Counter()
    groups = collections.Counter()
    for s, p in share.items():
        tasks[task[s]] += p
        groups["seeded (new)" if s.startswith("seed_") else
               "seeded (re-checked)" if s in ("synth_c", "synth_cre", "synth_cee") else "other"] += p
    summary = {"rows": rows, "share": share, "task_share": dict(tasks), "group_share": dict(groups),
               "sources": len(share)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("sources", len(share), "rows", sum(rows.values()))
    print("task share", {k: round(v, 3) for k, v in tasks.items()})
    print("group share", {k: round(v, 3) for k, v in groups.items()})
    for s in sorted(share, key=share.get, reverse=True)[:12]:
        print(f"  {s:28s} rows {rows[s]:8d} share {share[s]:.4f}")
    print("  ...")
    for s in sorted(share, key=share.get)[:6]:
        print(f"  {s:28s} rows {rows[s]:8d} share {share[s]:.4f}")


if __name__ == "__main__":
    main()
