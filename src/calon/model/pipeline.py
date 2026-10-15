"""Inference API for released CalOn checkpoints.

    from calon.model.pipeline import CalOn
    model = CalOn.from_pretrained("calon-ai/calon-0.6b")
    model.extract_entities("Maya Lin designed the Vietnam Veterans Memorial.", ["person", "artwork"])
    model.extract_relations(text, entity_types, relation_types)
    model.extract_events(text, {"acquisition": ["acquirer", "acquired company", "price", "time"]})

A checkpoint directory (or Hugging Face repo) holds model.safetensors, model_config.json (head options) and
calon_config.json: encoder name, maximum span width, the null-class prior correction (null_shift = log(1/omega), added
to the null logit to undo the down-weighting of null candidates in training) and the temperatures and thresholds fitted
on held-out data of the training sources after that correction.
Types are a list of names or a {name: description} mapping. Every returned mention is a character span of the text.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..data.collator import Collator, add_control_tokens, encode_row
from ..eval.evaluate import decode, decode_relations, predict, predict_re, re_temperature, with_temperature
from .model import SpanLabelModel


def _shift_entities(outs, shift):
    for o in outs:
        if shift and o["u"].size:
            o["u"] = o["u"].copy()
            o["u"][0] += shift
    return outs


def _shift_relations(outs, shift):
    """The same correction on relation choice probabilities: the no-relation odds times exp(shift)."""
    res = []
    for o in outs:
        p = o["p"]
        if shift and p.shape[1] and p.shape[0] > 1:
            p = p.copy()
            p[0] *= math.exp(shift)
            p = p / p.sum(0, keepdims=True)
        res.append(dict(o, p=np.asarray(p)))
    return res


def _queries(types):
    items = types.items() if isinstance(types, dict) else ((t, "") for t in types)
    # inference queries every type in a closed world: a type with no mention is absent
    return [{"label": t, "description": d or "", "status": "scope_assumed_absent"} for t, d in items]


class CalOn:
    def __init__(self, model, tokenizer, collate, config, device):
        self.model, self.tok, self.collate, self.cfg, self.device = model, tokenizer, collate, config, device

    @classmethod
    def from_pretrained(cls, path_or_repo, device=None):
        from transformers import AutoTokenizer
        path = Path(path_or_repo)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(path_or_repo)))
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = json.loads((path / "calon_config.json").read_text())
        head_cfg = json.loads((path / "model_config.json").read_text())
        tok = AutoTokenizer.from_pretrained(cfg["encoder"])
        lab_id, txt_id = add_control_tokens(tok)
        model = SpanLabelModel(cfg["encoder"], len(tok), W=cfg["max_span_tokens"], head="rlcd", **head_cfg)
        weights = path / "model.safetensors"
        if weights.exists():
            from safetensors.torch import load_model
            load_model(model, str(weights), device="cpu")
        else:
            model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        model = model.to(device).eval()
        return cls(model, tok, Collator(tok, lab_id, txt_id), cfg, device)

    def _entities(self, text, types, threshold):
        row = {"id": "q", "text": text, "queries": _queries(types), "entities": []}
        enc = encode_row(row, self.tok, self.cfg["max_span_tokens"])
        raw = _shift_entities(predict(self.model, [enc], self.collate, self.device), self.cfg.get("null_shift", 0.0))
        out = with_temperature(raw, self.cfg["entity_temperature"], "entity")[0]
        thr = self.cfg["entity_threshold"] if threshold is None else threshold
        found = decode(enc, out["dz"], math.log(thr / (1 - thr)), out["choice"])
        score = {(enc["cands"][c][0], enc["cands"][c][1], enc["labels"][m]): float(1 / (1 + math.exp(-out["z"][m, c])))
                 for m, c in zip(*out["choice"].nonzero())}
        return sorted(found), score

    def _relations(self, enc):
        outs = predict_re(self.model, [enc], self.collate, self.device)
        outs = _shift_relations(outs, self.cfg.get("null_shift", 0.0))
        return re_temperature(outs, self.cfg.get("relation_temperature", 1.0))[0]

    def extract_entities(self, text, types, threshold=None):
        """[{text, label, start, end, score}] for the mentions of the queried entity types; score is the probability
        of the predicted type. threshold: on the probability that a span is an entity of any queried type."""
        found, score = self._entities(text, types, threshold)
        return [{"text": text[s:e], "label": lab, "start": s, "end": e, "score": round(score[(s, e, lab)], 4)}
                for s, e, lab in found]

    def extract_events(self, text, event_types, threshold=None):
        """Events as [{type, trigger, score, arguments: [{role, text, start, end, score}]}]. event_types maps each event
        type to its roles, as a list of role names or {"description": ..., "roles": [names] or {name: description}}.
        Triggers are found by the entity decision over the event types; argument spans by the entity decision over the
        event types and their roles; each (trigger, span) pair then takes one role of the trigger's event type through
        the relation decision (as the evaluation in calon.eval.ee_eval)."""
        types, roles_of, role_desc = {}, {}, {}
        for t, spec in event_types.items():
            spec = {"roles": spec} if isinstance(spec, (list, tuple)) else spec
            types[t] = spec.get("description", "")
            rs = spec.get("roles", [])
            items = rs.items() if isinstance(rs, dict) else ((r, "") for r in rs)
            for r, d in items:
                roles_of.setdefault(t, []).append(r)
                role_desc[r] = role_desc.get(r) or d
        triggers = self.extract_entities(text, types)
        if not triggers:
            return []
        present = {e["label"] for e in triggers}
        asked = sorted({r for t in present for r in roles_of.get(t, [])})
        events = [{"type": e["label"], "trigger": {k: e[k] for k in ("text", "start", "end")}, "score": e["score"],
                   "arguments": []} for e in triggers]
        if not asked:
            return events
        queries = {t: types[t] for t in sorted(present)}
        queries.update({r: role_desc[r] for r in asked if r not in queries})
        cand = [e for e in self.extract_entities(text, queries, threshold) if e["label"] in set(asked)]
        tspans = list(dict.fromkeys((e["start"], e["end"]) for e in triggers))
        spans = tspans + [s for s in dict.fromkeys((e["start"], e["end"]) for e in cand) if s not in tspans]
        if len(spans) < 2:
            return events
        row = {"id": "q", "text": text, "queries": _queries(queries), "entities": [],
               "rel_queries": _queries({r: role_desc[r] for r in asked}), "relations": []}
        enc = encode_row(row, self.tok, self.cfg["max_span_tokens"], spans)
        p = self._relations(enc)["p"]
        thr = self.cfg["relation_threshold"] if threshold is None else threshold
        labels = enc["rel_labels"]
        for ev in events:
            h = (ev["trigger"]["start"], ev["trigger"]["end"])
            valid = [m for m, lab in enumerate(labels) if lab in roles_of.get(ev["type"], [])]
            for k, (a, b) in enumerate(enc["pair_char"]):
                if a != h or b == h or not valid or 1 - p[0, k] < thr:
                    continue
                m = max(valid, key=lambda m: p[1 + m, k])
                ev["arguments"].append({"role": labels[m], "text": text[b[0]:b[1]], "start": b[0], "end": b[1],
                                        "score": round(float(p[1 + m, k]), 4)})
            ev["arguments"].sort(key=lambda x: x["start"])
        return events

    def extract_relations(self, text, entity_types, relation_types, threshold=None):
        """End to end: entities of entity_types, then the relations of relation_types between them, as
        [{head, relation, tail, score}] with head / tail entities as returned by extract_entities."""
        ents = self.extract_entities(text, entity_types)
        if len(ents) < 2:
            return []
        spans = sorted({(e["start"], e["end"]) for e in ents})
        row = {"id": "q", "text": text, "queries": _queries(entity_types), "entities": [],
               "rel_queries": _queries(relation_types), "relations": []}
        enc = encode_row(row, self.tok, self.cfg["max_span_tokens"], spans)
        out = self._relations(enc)
        thr = self.cfg["relation_threshold"] if threshold is None else threshold
        by_span = {}
        for e in ents:
            by_span.setdefault((e["start"], e["end"]), e)
        p = out["p"]
        res = []
        for h, t, lab in sorted(decode_relations(enc, out, thr)):
            k = enc["pair_char"].index((h, t))
            m = enc["rel_labels"].index(lab)
            res.append({"head": by_span[h], "relation": lab, "tail": by_span[t], "score": round(float(p[1 + m, k]), 4)})
        return res
