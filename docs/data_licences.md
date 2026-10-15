# Licences of the datasets in CalOn (checked 2026-10-03)

Source of each entry: the dataset's GitHub LICENSE, Hugging Face card, LDC catalogue page or paper.
OPEN = redistribution allowed (with the stated conditions); RESTRICTED = redistribution not allowed
or only on request; UNCLEAR = no licence stated. Planned release repo in the last column
(huggingface.co/calon-ai; restricted and unclear sets go to the gated repo, by the authors' decision).

## Training sources

| Source | Class | Licence | Condition | Release repo |
| --- | --- | --- | --- | --- |
| Few-NERD | OPEN | CC BY-SA 4.0 | share-alike | calon-train-real |
| FinerWeb-10BT | OPEN | Apache-2.0 | attribution | calon-train-real |
| Pile-NER-type | OPEN | CC BY-NC 4.0 | non-commercial; annotated with GPT-3.5 | calon-train-real |
| MASSIVE | OPEN | CC BY 4.0 | attribution | calon-train-real |
| REBEL | OPEN | CC BY-NC-SA 4.0 | non-commercial, share-alike | calon-train-real |
| MAVEN-Arg | OPEN | CC BY-SA 4.0 (paper; no LICENSE file) | share-alike | calon-train-real |
| GENEVA | OPEN | CC BY 3.0 (FrameNet) | cite GENEVA and FrameNet | calon-train-real |
| RAMS | OPEN | CC BY-SA 4.0 | share-alike | calon-train-real |
| PHEE | OPEN | MIT | | calon-train-real |
| CC-News text (ccnews_ee) | UNCLEAR | none (Common Crawl terms) | publishers keep copyright | gated |
| synth_* (Qwen3 generations) | OPEN | Qwen3 Apache-2.0, no output restriction | check sources seeded from real data | calon-synth |

## Evaluation sets

| Set | Class | Licence | Condition | Release repo |
| --- | --- | --- | --- | --- |
| CrossNER | OPEN | MIT (repository) | | calon-eval |
| MIT Movie / Restaurant | UNCLEAR | none | | gated |
| SciER | OPEN | GPL-3.0 | copyleft | calon-eval |
| DynamicNER | OPEN | Apache-2.0 | | calon-eval |
| BUSTER | OPEN | Apache-2.0 | | calon-eval |
| GENTLE | OPEN | CC BY 4.0 | attribution | calon-eval |
| BioRED | OPEN | US Government work | cite | calon-eval |
| MeasEval | UNCLEAR | annotations unlicensed; texts CC BY | | gated |
| ai4privacy pii-masking | RESTRICTED | AI4Privacy licence | no redistribution without permission | gated |
| mneb/spy | UNCLEAR | none | | gated |
| CrossRE | OPEN | GPL-3.0 | copyleft | calon-eval |
| ASTE (SemEval ABSA) | RESTRICTED (likely) | META-SHARE NC, no redistribution (unconfirmed) | | gated |
| DWIE | RESTRICTED text | annotations GPL-3.0; text from Deutsche Welle | | gated |
| GAP | OPEN | Apache-2.0 | | calon-eval |
| LitBank | OPEN | CC BY 4.0 | attribution | calon-eval |
| MAVEN-ERE | OPEN | GPL-3.0 | copyleft | calon-eval |
| TACRED / Re-TACRED | RESTRICTED | LDC User Agreement (LDC2018T24) | LDC licence | gated |
| CASIE | UNCLEAR | none | | gated |
| MLEE | OPEN | CC BY-NC-SA 3.0 | non-commercial, share-alike | calon-eval |
| M2E2 | UNCLEAR | none (ACE parts need LDC) | | gated |
| WikiEvents | UNCLEAR | code MIT, no data licence | | gated |
| MAILEX | OPEN | CC BY-SA 4.0 | share-alike | calon-eval |
| SENTiVENT | OPEN | CC BY 4.0 | attribution | calon-eval |
| SPEED / SPEED++ | RESTRICTED | request form; Twitter policy | | gated |

LLM outputs: Qwen3 and gpt-oss-120b are Apache-2.0 with no stated output restriction.
