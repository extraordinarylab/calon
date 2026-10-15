# CalOn

<p align="center">
  <a href="https://arxiv.org/abs/2610.17113"><img alt="arXiv: 2610.17113" src="https://img.shields.io/badge/arXiv-2610.17113-b31b1b"></a>
  <a href="https://huggingface.co/calon-ai/calon-0.6b"><img alt="Hugging Face: calon-0.6b" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-calon--0.6b-ffd21e"></a>
  <a href="LICENSE.txt"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776ab">
</p>

CalOn extracts entities, relations, and events from text for any types you name, without examples from your dataset. It is a 0.6B-parameter encoder: you give it a text and a list of types (optionally with a short description of each), and it returns the matching character spans with their types and probabilities.

For every candidate span, or pair of spans for relations, CalOn makes one choice among the requested types and a "none" class, so deciding *whether* to extract a span is separated from deciding *which* type it is. Every result is an exact span of the input text.

Paper: [Calibrated Typed Decisions for Entity, Relation, and Event Extraction](https://arxiv.org/abs/2610.17113)

Model: [calon-ai/calon-0.6b](https://huggingface.co/calon-ai/calon-0.6b)

## Install

```bash
pip install git+https://github.com/extraordinarylab/calon
```

## Usage

```python
from calon.model.pipeline import CalOn

model = CalOn.from_pretrained("calon-ai/calon-0.6b")  # uses a GPU if available
```

### Entities

```python
model.extract_entities(
    text="Marie Curie won the Nobel Prize in Physics in 1903 for her research on radioactivity at the University of Paris.",
    types=["person", "award", "organisation", "field of study", "date"],
)
```

```python
[
    {"text": "Marie Curie", "label": "person", "start": 0, "end": 11, "score": 0.9705},
    {"text": "Nobel Prize in Physics", "label": "award", "start": 20, "end": 42, "score": 0.8488},
    {"text": "1903", "label": "date", "start": 46, "end": 50, "score": 0.9263},
    {"text": "radioactivity", "label": "field of study", "start": 71, "end": 84, "score": 0.5719},
    {"text": "University of Paris", "label": "organisation", "start": 92, "end": 111, "score": 0.8752},
]
```

### Relations

```python
relations = model.extract_relations(
    text="Satya Nadella, the chief executive of Microsoft, was born in Hyderabad.",
    entity_types=["person", "organisation", "location"],
    relation_types={
        "employer": "the organisation a person works for",
        "place of birth": "where a person was born",
    },
)
for r in relations:
    print(r["head"]["text"], "|", r["relation"], "|", r["tail"]["text"], "|", r["score"])
```

```
Satya Nadella | employer | Microsoft | 0.9544
Satya Nadella | place of birth | Hyderabad | 0.9832
```

Each relation also carries the full head and tail entities (text, label, offsets and score).

### Events

```python
model.extract_events(
    text="Apple hired Jane Smith as chief financial officer in March 2021.",
    event_types={
        "hiring": {
            "description": "a person is employed by an organisation",
            "roles": {
                "employer": "the organisation",
                "employee": "the person hired",
                "position": "the job",
                "time": "when",
            },
        },
    },
)
```

```python
[
    {
        "type": "hiring",
        "trigger": {"text": "hired", "start": 6, "end": 11},
        "score": 0.7733,
        "arguments": [
            {"role": "employer", "text": "Apple", "start": 0, "end": 5, "score": 0.9975},
            {"role": "employee", "text": "Jane Smith", "start": 12, "end": 22, "score": 0.9988},
            {"role": "position", "text": "chief financial officer", "start": 26, "end": 49, "score": 0.9995},
            {"role": "time", "text": "March 2021", "start": 53, "end": 63, "score": 0.9989},
        ],
    },
]
```

### Notes

- Types can be a list of names or a `{name: description}` mapping. A short description helps when a name is ambiguous.
- An event type maps to a list of role names, or to a description and its roles.
- `score` is the model's probability for the predicted type or role. The default thresholds were fitted on held-out data from the training sources after correcting for the down-weighted "none" class, so scores can be read as probabilities.
- Pass `threshold=` (between 0 and 1) to trade precision for recall.

## Training

The training code is in `calon.train`. See [tutorial/training_data.md](tutorial/training_data.md) for the data format and how to train on your own sources.

## Limitations

English only. Spans are at most 32 tokens and inputs at most 1,024 tokens, and relations are found within one input.

## Citation

```bibtex
@misc{wang2026calon,
    title={Calibrated Typed Decisions for Entity, Relation, and Event Extraction}, 
    author={Yang Wang and Hanlin Xiao and Yanan Ma and Chenghao Xiao and Chenghua Lin},
    year={2026},
    eprint={2610.17113},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2610.17113}, 
}
```

## License

Apache 2.0.
