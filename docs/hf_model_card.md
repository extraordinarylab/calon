---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: calon
pipeline_tag: token-classification
language:
- en
tags:
- information-extraction
- named-entity-recognition
- relation-extraction
- event-extraction
- zero-shot
---

# CalOn 0.6B

CalOn is a 0.6B-parameter encoder for zero-shot entity, relation, and event extraction. You give it a text and the types you want, optionally with a short description of each, and it returns character spans with their types and probabilities. It needs no examples from your dataset.

Paper: [Calibrated Typed Decisions for Entity, Relation, and Event Extraction](https://arxiv.org/abs/2610.17113) · Code: [extraordinarylab/calon](https://github.com/extraordinarylab/calon)

## Usage

```bash
pip install git+https://github.com/extraordinarylab/calon
```

```python
from calon.model.pipeline import CalOn

model = CalOn.from_pretrained("calon-ai/calon-0.6b")

model.extract_entities(
    text="Marie Curie won the Nobel Prize in Physics in 1903 for her research on radioactivity at the University of Paris.",
    types=["person", "award", "organisation", "field of study", "date"],
)

model.extract_relations(
    text="Satya Nadella, the chief executive of Microsoft, was born in Hyderabad.",
    entity_types=["person", "organisation", "location"],
    relation_types={
        "employer": "the organisation a person works for",
        "place of birth": "where a person was born",
    },
)

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

Types can be a list of names or a `{name: description}` mapping. Each result is a character span of the text with a `score`, the model's probability for the predicted type or role. The default thresholds were fitted on held-out data from the training sources after correcting for the down-weighted "none" class; pass `threshold=` to trade precision for recall. See the [repository](https://github.com/extraordinarylab/calon) for the outputs of these examples.

## Model

CalOn encodes the requested types together with the text using a bidirectional Qwen3-0.6B. Interaction layers let the types and candidate spans (or span pairs) attend to each other, and each candidate then makes a single choice among the requested types and a null class. A span is extracted when the probability that it is not null passes the threshold. Relations use the same choice over ordered pairs of entity spans, and events combine the two (triggers are spans, arguments are trigger-span pairs).

The model was trained on 3.1M instances from 84 sources of human-labelled, LLM-annotated, and synthetic data, and evaluated zero-shot on 40 entity, relation, and event benchmarks.

## Limitations

English only. Spans are at most 32 tokens and inputs at most 1,024 tokens; relations are extracted within one input. Much of the training data is annotated by language models.

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

Apache 2.0, as the Qwen3-0.6B backbone.
