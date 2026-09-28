"""Span-label scorer: one logit z per (label, candidate) (SPEC 3).

Encoders: ModernBERT (bidirectional as released) or a decoder LM made
bidirectional (SPEC 3, Qwen2.5-0.5B): every attention module's is_causal is
switched off and the model receives an explicit 4D padding-only mask (bool for
SDPA, additive float for eager), so right-context tokens reach left positions and
padded keys are never attended. The KV cache is disabled.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

DECODER_TYPES = {"qwen2", "qwen3", "llama", "mistral"}


def mlp(d_in, hidden, d_out, dropout):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, d_out))


HEAD_KEYS = ("hidden", "out", "label_pool")


def load_model(encoder, vocab_size, W, head, ckpt, device="cuda"):
    """Rebuild a trained model; head options come from model_config.json next to the
    checkpoint (absent for older runs, which used the defaults)."""
    cfg_path = Path(ckpt).parent / "model_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    model = SpanLabelModel(encoder, vocab_size, W=W, head=head, **cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def padding_only_mask(attention_mask, impl, dtype):
    """[B, T] 0/1 -> [B, 1, T, T]: every query sees every non-padded key."""
    keep = attention_mask[:, None, None, :].bool().expand(-1, 1, attention_mask.shape[1], -1)
    if impl == "sdpa":
        return keep
    return torch.zeros(keep.shape, dtype=dtype, device=keep.device).masked_fill(
        ~keep, torch.finfo(dtype).min)


class SpanLabelModel(nn.Module):
    def __init__(self, encoder_name, vocab_size, W=32, hidden=512, out=256, width_dim=32,
                 dropout=0.1, attn_implementation=None, head="bce", label_pool="mean"):
        super().__init__()
        self.head_type = head
        # label vector from the name+description tokens ("mean"), from the <LABEL>
        # marker's hidden state ("marker", as GLiNER2 / GLiFormer do), or both concatenated
        self.label_pool = label_pool
        kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
        # fp32 master weights (Qwen's config says bf16); compute runs under bf16 autocast
        self.encoder = AutoModel.from_pretrained(encoder_name, dtype=torch.float32, **kw)
        if vocab_size > self.encoder.get_input_embeddings().num_embeddings:
            self.encoder.resize_token_embeddings(vocab_size)
        self.bidirectional_decoder = self.encoder.config.model_type in DECODER_TYPES
        if self.bidirectional_decoder:
            self.encoder.config.use_cache = False
            self.encoder.config.is_causal = False
            for m in self.encoder.modules():
                if hasattr(m, "is_causal"):
                    m.is_causal = False
        d = self.encoder.config.hidden_size
        self.width = nn.Embedding(W, width_dim)
        self.span_mlp = mlp(3 * d + width_dim, hidden, out, dropout)
        self.label_mlp = mlp(2 * d if label_pool == "both" else d, hidden, out, dropout)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = 1 / math.sqrt(out)
        if head == "rlcd":
            # the "not an entity" candidate: a learned key in label space, so its
            # utility u_none = <span, none> / sqrt(r) depends on the span like any label
            self.none_key = nn.Parameter(torch.randn(out) * 0.02)

    def encode(self, input_ids, attention_mask):
        if not self.bidirectional_decoder:
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        impl = self.encoder.config._attn_implementation
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else \
            self.encoder.get_input_embeddings().weight.dtype
        mask = padding_only_mask(attention_mask, impl, dtype)
        return self.encoder(input_ids=input_ids, attention_mask=mask,
                            use_cache=False).last_hidden_state

    def forward(self, input_ids, attention_mask, label_mask, cand_i, cand_j):
        h = self.encode(input_ids, attention_mask)
        with torch.autocast(h.device.type, enabled=False):              # heads in fp32
            return self.head(h.float(), label_mask, cand_i, cand_j)

    def head(self, h, label_mask, cand_i, cand_j):
        # label vectors: mean over name+description tokens and/or the <LABEL> marker,
        # which the collator puts right before each label's tokens
        lm = label_mask.to(h.dtype)
        mean_lab = torch.einsum("bmt,btd->bmd", lm, h) / lm.sum(-1, keepdim=True).clamp_min(1)
        if self.label_pool == "mean":
            lab = mean_lab
        else:
            pos = (lm.argmax(-1) - 1).clamp_min(0)                      # [B,M]
            marker = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
            lab = marker if self.label_pool == "marker" else torch.cat([marker, mean_lab], -1)
        # span vectors: start, end, mean, width
        B, T, D = h.shape
        cs = torch.cat([h.new_zeros(B, 1, D), h.cumsum(1)], 1)          # [B,T+1,D]
        gather = lambda x, idx: x.gather(1, idx[..., None].expand(-1, -1, D))
        width = cand_j - cand_i
        mean = (gather(cs, cand_j + 1) - gather(cs, cand_i)) / (width + 1)[..., None].to(h.dtype)
        span = torch.cat([gather(h, cand_i), gather(h, cand_j), mean,
                          self.width(width.clamp(0, self.width.num_embeddings - 1))], -1)
        s = self.span_mlp(span)                                         # [B,C,O]
        l = self.label_mlp(lab)                                         # [B,M,O]
        z = torch.einsum("bmo,bco->bmc", l, s) * self.scale + self.bias
        if self.head_type != "rlcd":
            return z                                                    # [B,M,C] logits
        # RLCD: utilities for [none, label_1..label_M] per span -> [B,1+M,C]
        u_none = torch.einsum("o,bco->bc", self.none_key, s)[:, None, :] * self.scale
        return torch.cat([u_none, z], 1)
