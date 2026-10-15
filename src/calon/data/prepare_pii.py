"""PII evaluation set from ai4privacy/pii-masking-openpii-1.5m (validation, English).

Evaluation only, never trained on. Label codes become readable names; each row
asks the full label set, unmentioned labels are scope_assumed_absent
(benchmark_closed_world), like the other OOD sets. Descriptions are authored by
label_bank.py from the label name alone.
Stages: build (download, convert, write inventory) -> label_bank -> finish.
"""
import argparse
import collections
import json
import random
import re

from calon.data.roots import ROOT

OUT = ROOT / "data/ood_pii"
REPO = "ai4privacy/pii-masking-openpii-1.5m"
NAMES = {"GIVENNAME": "given name", "SURNAME": "surname", "TITLE": "personal title", "CITY": "city",
         "STREET": "street name", "BUILDINGNUM": "building number", "ZIPCODE": "postal code",
         "TELEPHONENUM": "telephone number", "EMAIL": "email address", "DATE": "date", "TIME": "time",
         "AGE": "age", "SEX": "sex", "GENDER": "gender", "IDCARDNUM": "identity card number",
         "PASSPORTNUM": "passport number", "DRIVERLICENSENUM": "driver's license number",
         "SOCIALNUM": "social security number", "TAXNUM": "tax identification number",
         "CREDITCARDNUMBER": "credit card number", "ACCOUNTNUM": "bank account number",
         "USERNAME": "username", "PASSWORD": "password", "IP": "IP address", "URL": "URL",
         "STATE": "state or province", "COUNTRY": "country", "COUNTY": "county"}


def readable(code):
    if code in NAMES:
        return NAMES[code]
    s = code.lower().replace("_", " ")
    return re.sub(r"num\b", " number", s).strip()


def build(args):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.dataset_info(REPO)
    files = ["data/validation.jsonl"]        # the repo ships jsonl only
    rows, stats, langs = [], collections.Counter(), collections.Counter()
    for f in files:
        path = hf_hub_download(REPO, f, repo_type="dataset", revision=info.sha, cache_dir=str(ROOT / "raw/hf"))
        for line in open(path):
            rec = json.loads(line)
            lang = str(rec.get("language", "")).lower()
            langs[lang] += 1
            if lang not in ("en", "english"):
                continue
            text, mask = rec["source_text"], rec["privacy_mask"]
            if isinstance(mask, str):
                mask = json.loads(mask)
            ents, bad = [], False
            for m in mask or []:
                s, e, lab = int(m["start"]), int(m["end"]), m["label"]
                if text[s:e] != m.get("value", text[s:e]):
                    bad = True
                    break
                ents.append({"start": s, "end": e, "code": lab})
            if bad:
                stats["offset_mismatch_rows"] += 1
                continue
            rows.append({"uid": rec.get("uid"), "text": text, "ents": ents})
            if len(rows) >= args.n * 5:
                break
        stats["english_rows_seen"] = len(rows)
    stats["top_languages"] = str(langs.most_common(5))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.n]
    codes = sorted({e["code"] for r in rows for e in r["ents"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_rows.json").write_text(json.dumps(rows))
    (OUT / "inventory.json").write_text(json.dumps(
        {"pii": [{"raw": c, "name": readable(c), "parent": None} for c in codes]}, indent=1))
    (OUT / "build_meta.json").write_text(json.dumps({"repo": REPO, "revision": info.sha, "files": files,
                                                     "n": len(rows), "codes": codes, "stats": dict(stats)}, indent=1))
    print(json.dumps({"n": len(rows), "labels": [readable(c) for c in codes], "stats": dict(stats)}))


def finish(args):
    bank = json.loads((OUT / "label_bank.json").read_text())["bank"]["pii"]
    rows = json.loads((OUT / "raw_rows.json").read_text())
    train_labels = set()
    for line in open(ROOT / "data/mix/finerweb/train.jsonl"):
        train_labels |= {q["label"].lower().replace("_", " ") for q in json.loads(line)["queries"]}
    out = []
    for k, r in enumerate(rows):
        ents = [{"start": e["start"], "end": e["end"], "label": bank[e["code"]]["name"]} for e in r["ents"]]
        present = {e["label"] for e in ents}
        queries = [{"label": v["name"], "description": v["description"],
                    "status": "positive" if v["name"] in present else "scope_assumed_absent",
                    "description_origin": v["description_origin"],
                    "supervision_origin": "source" if v["name"] in present else "benchmark_closed_world"}
                   for v in bank.values()]
        out.append({"id": f"pii_ai4privacy:validation:{r['uid'] or k}", "source": "pii_ai4privacy",
                    "text": r["text"], "queries": queries, "entities": ents})
    with open(OUT / "pii_ai4privacy.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels = [v["name"] for v in bank.values()]
    manifest = {"scope_policy": "benchmark_closed_world", "splits_used": "validation (English), sampled",
                "threshold_policy": "zero-shot: training sources' dev_calib T and threshold",
                "descriptions": "authored by label_bank.py from the label name only",
                "sets": {"pii_ai4privacy": {
                    "repo": REPO, "rows": len(out), "entities": sum(len(r["entities"]) for r in out),
                    "labels": labels, "seen_labels": [l for l in labels if l in train_labels],
                    "unseen_labels": [l for l in labels if l not in train_labels]}}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["sets"]["pii_ai4privacy"])[:800])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "finish"])
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"build": build, "finish": finish}[args.stage](args)


if __name__ == "__main__":
    main()
