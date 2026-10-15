"""LaTeX result tables for the paper from a round_summary JSON.

Three tables (NER, relations, events) in two versions: with type descriptions (w/ desc., main text)
and with type names only (w/o desc., appendix). In the main tables, baselines that accept names
only (GLiFormer, GNER) are shown with their names-only result and marked with a dagger. NER covers
CrossNER, MIT, the other NER sets and the personal-information sets (no contamination-check sets);
relations are scored with gold entities and end to end; events by argument classification with gold
triggers, and trigger and argument classification end to end. F1 x 100, mean over seeds, best per
row in bold, an average row per block when every model has every set. A dash marks a setting that
is not supported or not run; TBD (in \\rev colour) marks a run we still have to do.
"""
import argparse
import json
from pathlib import Path

from .report_html import EXCLUDE

NAMES = {"crossner_ai": "AI", "crossner_literature": "Literature", "crossner_music": "Music",
         "crossner_politics": "Politics", "crossner_science": "Science", "mit_movie": "Movie",
         "mit_restaurant": "Restaurant", "scier": "SciER", "scier_ood": "SciER (OOD)", "gentle": "GENTLE",
         "biored": "BioRED", "measeval": "MeasEval", "pii_ai4privacy": "ai4privacy", "spy_legal": "spy legal",
         "spy_medical": "spy medical",
         "crossre_ai": "AI", "crossre_literature": "Literature", "crossre_music": "Music", "crossre_news": "News",
         "crossre_politics": "Politics", "crossre_science": "Science", "aste_14lap": "14lap",
         "aste_14res": "14res", "aste_15res": "15res", "aste_16res": "16res", "biored_re": "BioRED", "gap": "GAP",
         "litbank": "LitBank", "maven_ere": "MAVEN-ERE", "measeval_re": "MeasEval", "retacred": "Re-TACRED",
         "casie": "CASIE", "mlee": "MLEE", "m2e2": "M2E2", "mailex": "MAILEX", "sentivent": "SENTiVENT",
         "speed_mpox": "SPEED++ mpox", "speed_zika": "SPEED++ zika"}
# (block title, sets); a block with more than one set gets an average row
NER_BLOCKS = [("CrossNER", ["crossner_ai", "crossner_literature", "crossner_music", "crossner_politics",
                            "crossner_science"]),
              ("MIT", ["mit_movie", "mit_restaurant"]),
              ("Other domains", ["scier", "scier_ood", "gentle", "biored", "measeval"]),
              ("Personal information", ["pii_ai4privacy", "spy_legal", "spy_medical"])]
RE_BLOCKS = [("CrossRE", ["crossre_ai", "crossre_literature", "crossre_music", "crossre_news", "crossre_politics",
                          "crossre_science"]),
             ("ASTE", ["aste_14lap", "aste_14res", "aste_15res", "aste_16res"]),
             ("Other tasks", ["scier", "scier_ood", "biored_re", "gap", "litbank", "maven_ere", "measeval_re",
                              "retacred"])]
EE_BLOCKS = [("", ["casie", "mlee", "m2e2", "mailex", "sentivent", "speed_mpox", "speed_zika"])]
BASELINES = ["GLiNER2", "GLiFormer", "GLiREL", "GNER", "InstructUIE", "GoLLIE"]
NAMES_ONLY = {"GLiFormer", "GLiREL", "GNER", "InstructUIE"}          # accept type names only
# runs still to do: (task, model) -> the column is shown, missing cells as TBD
TBD: set[tuple[str, str]] = set()
# test sets whose training split a baseline was trained on (InstructUIE's supervised IE INSTRUCTIONS
# mixture; GoLLIE's training data): the score is printed with SEEN_MARK and still competes for the bold
# GoLLIE: TACRED (slot filling)
SEEN = {"InstructUIE": {"mit_movie", "mit_restaurant", "casie"}, "GoLLIE": {"retacred"}}
SEEN_MARK = "$^\\ddagger$"
PREFIX = ""


def canon(m):
    s = m.lower()
    for key, name in (("gliner2-large", "GLiNER2"), ("gliformer-large", "GLiFormer"), ("glirel", "GLiREL"),
                      ("gner", "GNER"),
                      ("instructuie", "InstructUIE"), ("gollie", "GoLLIE"), ("gpt-oss-120b", "gpt-oss")):
        if key in s:
            return name
    return None


def merged(res, groups, exp, mode, task):
    """column -> {"set | metric": mean}. mode "desc": ours and GLiNER2 / gpt-oss with descriptions,
    names-only baselines as they are; mode "name": everything with type names only."""
    data = {}
    for g in groups:
        for m, c in res.get(g, {}).items():
            if not m.startswith("CalOn"):
                continue
            if m.startswith(f"CalOn {exp} ") and ("(desc" in m) == (mode == "desc"):
                data.setdefault("ours", {}).update({k: v[0] for k, v in c.items()})
        for m, c in res.get(g if mode == "desc" else f"{g}@name_only", {}).items():
            name = None if m.startswith("CalOn") else canon(m)
            if name:
                data.setdefault(name, {}).update({k: v[0] for k, v in c.items()})
    for col in data.values():
        for k in [k for k in col if k.split(" | ")[0] in EXCLUDE]:
            del col[k]
    cols = ["ours"] + [b for b in BASELINES if b in data or (task, b) in TBD]
    tbd = {b for b in cols if (task, b) in TBD and b not in data}
    return data, cols, tbd


class _NA:
    """A setting the dataset does not define (printed as a dash)."""


NA = _NA()
# settings a dataset does not define: end to end on the sets that annotate only some entity pairs
# (prepare_evalsets2.GIVEN_ONLY: concept-level BioRED, one pair per Re-TACRED sentence, the GAP
# pronoun, LitBank and MAVEN-ERE mention pairs), so predicted entities outside them cannot be scored
UNDEFINED = {"end2end": {"biored_re", "retacred", "gap", "litbank", "maven_ere"}}
# settings a model does not support (printed as a dash): GLiREL classifies relations between given entities only
UNSUPPORTED = {"GLiREL": {"end2end"}}


def cell(v, best):
    """v: a score, NA (undefined: dash) or None (not run yet: TBD)."""
    if v is NA:
        return "--"
    if v is None:
        return "\\rev{TBD}"
    s = f"{100 * v:.1f}"
    return f"\\textbf{{{s}}}" if best is not None and abs(v - best) < 5e-4 else s


def rows_for(data, cols, sets, metrics):
    """One row per set that ours has; values ordered metric-major, model-minor."""
    out = []
    for s in sets:
        if not any(f"{s} | {mt}" in data.get("ours", {}) for mt in metrics):
            continue
        out.append((NAMES.get(s, s), [[NA if s in UNDEFINED.get(mt, ()) or mt in UNSUPPORTED.get(c, ()) else
                                        data.get(c, {}).get(f"{s} | {mt}")
                                       for c in cols] for mt in metrics],
                    {j for j, c in enumerate(cols) if s in SEEN.get(c, ())}))
    return out


def avg_row(rows, n_metrics, n_cols):
    """Mean over the sets where the setting is defined; TBD when a model misses one of them."""
    vals = []
    for i in range(n_metrics):
        vals.append([])
        for j in range(n_cols):
            v = [r[1][i][j] for r in rows if r[1][i][j] is not NA]
            if not v:
                vals[-1].append(NA)
            else:
                vals[-1].append(sum(v) / len(v) if all(x is not None for x in v) else None)
    # a model is marked in a metric only if a set it was trained on counts towards that metric's average
    seen = [{j for r in rows for j in r[2] if r[1][i][j] is not NA} for i in range(n_metrics)]
    return ("Average", vals, seen)


def fmt_row(name, vals, seen, cols, tbd):
    """seen: column indices whose model trained on this set (marked; they still compete for the bold), or one
    such set per metric (average rows)."""
    cells = []
    for i, per_metric in enumerate(vals):
        s = seen[i] if isinstance(seen, list) else seen
        per_metric = [None if c in tbd and v is not NA else v for v, c in zip(per_metric, cols)]
        present = [v for v in per_metric if v is not None and v is not NA]
        best = max(present) if present else None
        cells += [cell(v, best) + SEEN_MARK if j in s and v is not None and v is not NA else cell(v, best)
                  for j, v in enumerate(per_metric)]
    return f"{name} & " + " & ".join(cells) + " \\\\"


def table(data, cols, tbd, blocks, metrics, metric_names, caption, label, mode, wide=True):
    n = len(cols)
    dag = "$^\\dagger$" if mode == "desc" else ""
    heads = ["\\textsc{CalOn}"] + [c + (dag if c in NAMES_ONLY else "") for c in cols[1:]]
    width = 1 + n * len(metrics)
    env = "table*" if wide else "table"
    # adjustbox shrinks the table only when it is wider than the page
    lines = [f"\\begin{{{env}}}[tp]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\begin{adjustbox}{max width=\\textwidth}",
             f"\\begin{{tabular}}{{l{'r' * n * len(metrics)}}}", "\\toprule"]
    if len(metrics) > 1:
        lines.append("\\multirow{2}{*}{Dataset} & "
                     + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{mn}}}" for mn in metric_names) + " \\\\")
        lines.append("".join(f"\\cmidrule(lr){{{2 + i * n}-{1 + (i + 1) * n}}}" for i in range(len(metrics))))
        lines.append(" & " + " & ".join(heads * len(metrics)) + " \\\\")
    else:
        lines.append("Dataset & " + " & ".join(heads) + " \\\\")
    lines.append("\\midrule")
    for title, sets in blocks:
        rows = rows_for(data, cols, sets, metrics)
        if not rows:
            continue
        if title:
            lines.append(f"\\multicolumn{{{width}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        lines += [fmt_row(*r, cols, tbd) for r in rows]
        if len(rows) > 1:
            lines.append(f"\\cmidrule(lr){{1-{width}}}")
            lines.append(fmt_row(*avg_row(rows, len(metrics), n), cols, tbd))
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{adjustbox}", f"\\caption{{{PREFIX}{caption}}}", f"\\label{{{label}}}",
              f"\\end{{{env}}}"]
    return "\n".join(lines)


def captions(mode, seeds):
    """Captions as in the paper's tables/ files (the names-only tables refer back to the main ones)."""
    tail = " \\rev{TBD marks a run still to be done.}"
    seen = f"{SEEN_MARK}: not zero-shot, as the model was trained on the training split of this dataset."
    if mode != "desc":
        seen2 = f"{SEEN_MARK}: the model was trained on the training split of this dataset."
        bold = "Best score per row in \\textbf{bold}."
        ner = ("Zero-shot NER without type descriptions, micro F1 in \\%. Same setting as Table~\\ref{tab:ner}, "
               "but every "
               "model receives only the type names (e.g., \\textit{programming language}), without descriptions. "
               f"{seen2} {bold}{tail}")
        re_ = ("Relation extraction without type descriptions, micro F1 in \\%. Same settings as Table~\\ref{tab:re} "
               "(gold entities and end-to-end), but every model receives only the relation type names. A dash marks "
               "datasets that label relations only for pre-selected entity pairs, so end-to-end F1 cannot be "
               "computed. "
               f"{seen2} {bold}{tail}")
        ee = ("Event extraction without type descriptions, micro F1 in \\%. Same settings as Table~\\ref{tab:ee}, "
              "but every "
              f"model receives only the names of the event types and roles. {seen2} {bold}{tail}")
        return ner, re_, ee
    ner = ("Zero-shot NER, micro F1 in \\%. For each sentence, every model receives the full type set of its dataset "
           "(e.g., all 14 types of CrossNER AI, including types that do not occur in the sentence), each with a short "
           "description; models marked $^\\dagger$ accept only type names and receive the names alone. "
           f"{seen} Best score per row in \\textbf{{bold}}; results with type names only for all models "
           f"are in Appendix~\\ref{{sec:appendix_wo_desc}}.{tail}")
    re_ = ("Relation extraction, micro F1 in \\%. \\emph{Gold entities}: the gold entity spans are given and only "
           "relations are predicted; for baselines that cannot take gold entities, we keep their end-to-end relations "
           "whose two spans are gold entities. \\emph{End-to-end}: the model first finds the entities itself and "
           "then the "
           "relations between them; a relation counts as correct only if both of its spans and its type are correct. "
           "A dash marks datasets that label relations only for pre-selected entity pairs (e.g., one pair per sentence "
           "in Re-TACRED), so end-to-end F1 cannot be computed. "
           "$^\\dagger$, "
           f"{SEEN_MARK} and bold as in Table~\\ref{{tab:ner}}; for Re-TACRED, {SEEN_MARK} marks training on TACRED, "
           f"which has the same sentences.{tail}")
    ee = ("Event extraction, micro F1 in \\%. \\emph{Arg.\\ (gold triggers)}: argument and role classification given "
          "the gold triggers; \\emph{Trigger} and \\emph{Arg.\\ (end to end)}: trigger classification, and argument "
          f"classification on the predicted triggers. $^\\dagger$, {SEEN_MARK} and bold as in "
          f"Table~\\ref{{tab:ner}}.{tail}")
    return ner, re_, ee


def build(res, exp, mode, seeds):
    sfx = "" if mode == "desc" else "_wo_desc"
    c_ner, c_re, c_ee = captions(mode, seeds)
    d, c, t = merged(res, ["ner", "ner_x", "ner_x2", "ner_pii", "ner_spy"], exp, mode, "ner")
    ner = table(d, c, t, NER_BLOCKS, ["F1"], ["F1"], c_ner, f"tab:ner{sfx}", mode)
    d, c, t = merged(res, ["re", "re_x", "re_x2"], exp, mode, "re")
    re_ = table(d, c, t, RE_BLOCKS, ["given", "end2end"], ["Gold entities", "End to end"], c_re, f"tab:re{sfx}",
                mode)
    d, c, t = merged(res, ["ee", "ee_x2"], exp, mode, "ee")
    ee = table(d, c, t, EE_BLOCKS, ["given AC", "e2e TC", "e2e AC"],
               ["Arg.\\ (gold triggers)", "Trigger (end to end)", "Arg.\\ (end to end)"], c_ee, f"tab:ee{sfx}", mode)
    # the TBD note only where a TBD cell is left
    note = " \\rev{TBD marks a run still to be done.}"
    tabs = [t if t.count("\\rev{TBD") > 1 else t.replace(note, "") for t in (ner, re_, ee)]
    return tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--exp", required=True, help="our version, e.g. ee5")
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--out_dir", required=True,
                    help="the paper's tables/ directory: {ner,re,ee}_{with,without}_description.tex")
    ap.add_argument("--prefix", default="", help="LaTeX put at the start of every caption (e.g. a pending note)")
    args = ap.parse_args()
    global PREFIX
    PREFIX = args.prefix
    res = json.loads(Path(args.summary).read_text())
    seeds = f"mean over {args.seeds} seeds" if args.seeds > 1 else "one seed"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for mode, sfx in (("desc", "with"), ("name", "without")):
        for task, tab in zip(("ner", "re", "ee"), build(res, args.exp, mode, seeds)):
            (out / f"{task}_{sfx}_description.tex").write_text(tab.rstrip("\n") + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
