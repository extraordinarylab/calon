"""ModernBERT span-label scorer: one logit z per (label, candidate) (SPEC 3)."""
import math

import torch
from torch import nn
from transformers import AutoModel


def mlp(d_in, hidden, d_out, dropout):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, d_out))


class SpanLabelModel(nn.Module):
    def __init__(self, encoder_name, vocab_size, W=32, hidden=512, out=256, width_dim=32,
                 dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.resize_token_embeddings(vocab_size)
        d = self.encoder.config.hidden_size
        self.width = nn.Embedding(W, width_dim)
        self.span_mlp = mlp(3 * d + width_dim, hidden, out, dropout)
        self.label_mlp = mlp(d, hidden, out, dropout)
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = 1 / math.sqrt(out)

    def forward(self, input_ids, attention_mask, label_mask, cand_i, cand_j):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        with torch.autocast(h.device.type, enabled=False):              # heads in fp32
            return self.head(h.float(), label_mask, cand_i, cand_j)

    def head(self, h, label_mask, cand_i, cand_j):
        # label vectors: mean over name+description tokens
        lm = label_mask.to(h.dtype)
        lab = torch.einsum("bmt,btd->bmd", lm, h) / lm.sum(-1, keepdim=True).clamp_min(1)
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
        return torch.einsum("bmo,bco->bmc", l, s) * self.scale + self.bias
