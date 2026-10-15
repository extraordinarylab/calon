"""Span-label scorer: one logit z per (label, candidate).

Encoders: ModernBERT (bidirectional as released) or a decoder LM made
bidirectional (Qwen): every attention module's is_causal is
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


HEAD_KEYS = ("hidden", "out", "label_pool", "head_layers", "head_heads", "relations", "head_arch", "attn_dim",
             "attn_heads", "attn_layers", "bilinear_rank")


class MixBlock(nn.Module):
    """Pre-norm transformer block, attention through the memory-efficient SDPA kernel (never [L, L] in memory): a row
    has up to ~30k candidates, and nn.MultiheadAttention with a padding mask materialises [B, heads, L, L]
    (36.8 GiB in the 2026-10-04 smoke test)."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.o = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                             # x [1, L, D]: one row, no padding, no mask
        from torch.nn.attention import SDPBackend, sdpa_kernel
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            a = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.n2(x)))


class Mixer(nn.Module):
    """Self-attention over one row's [none, labels, candidates] (or [no relation, relations, pairs]).
    Each row runs on its own real positions only (no padding, so no mask to expand); padded positions
    come back as zeros and are never scored."""

    def __init__(self, dim, heads, layers, dropout):
        super().__init__()
        self.blocks = nn.ModuleList(MixBlock(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, src_key_padding_mask):
        out = torch.zeros_like(x)
        for b in range(x.shape[0]):
            idx = (~src_key_padding_mask[b]).nonzero(as_tuple=True)[0]
            xb = x[b, idx][None]
            for blk in self.blocks:
                xb = blk(xb)
            out[b, idx] = self.norm(xb)[0]
        return out


def mixer(dim, heads, layers, dropout):
    return Mixer(dim, heads, layers, dropout)


def canonical_head(head):
    """Head name. The method is RLCD (the framework and package are CalOn); "calon" was used as the
    method's name for a while and is accepted as an alias."""
    return "rlcd" if head == "calon" else head


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
                 dropout=0.1, attn_implementation=None, head="bce", label_pool="mean",
                 head_layers=0, head_heads=16, relations=False, head_arch="mlp", attn_dim=256, attn_heads=4,
                 attn_layers=2, bilinear_rank=64):
        super().__init__()
        # head_arch "mlp": span and label vectors from two 2-layer MLPs, scored by a scaled dot product.
        # "attn": both projected to attn_dim, then [none | labels | candidates] go through attn_layers of
        # self-attention together (every position sees every other), scored by a low-rank bilinear form
        # s^T U V^T l / sqrt(rank); the relation head the same over [no relation | relations | pairs].
        self.head_arch = head_arch
        head = canonical_head(head)
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
        self.post = None
        if head_layers:
            layer = nn.TransformerEncoderLayer(d, nhead=head_heads, dim_feedforward=4 * d, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.post = nn.TransformerEncoder(layer, head_layers, norm=nn.LayerNorm(d),
                                              enable_nested_tensor=False)
        self.width = nn.Embedding(W, width_dim)
        lab_in = 2 * d if label_pool == "both" else d
        if head_arch == "attn":
            out = attn_dim                               # span vectors (also the relation head's input)
            self.span_proj = nn.Linear(3 * d + width_dim, attn_dim)
            self.label_proj = nn.Linear(lab_in, attn_dim)
            self.kind = nn.Embedding(3, attn_dim)        # 0 none, 1 label, 2 candidate
            self.none_tok = nn.Parameter(torch.randn(attn_dim) * 0.02)
            self.mix = mixer(attn_dim, attn_heads, attn_layers, dropout)
            self.U = nn.Linear(attn_dim, bilinear_rank, bias=False)
            self.V = nn.Linear(attn_dim, bilinear_rank, bias=False)
            self.scale = 1 / math.sqrt(bilinear_rank)
        else:
            self.span_mlp = mlp(3 * d + width_dim, hidden, out, dropout)
            self.label_mlp = mlp(lab_in, hidden, out, dropout)
            self.scale = 1 / math.sqrt(out)
        self.bias = nn.Parameter(torch.zeros(()))
        if head == "rlcd" and head_arch != "attn":
            # the "not an entity" candidate: a learned key in label space, so its
            # utility u_none = <span, none> / sqrt(r) depends on the span like any label
            # (head_arch "attn": none_tok, a position of the self-attention sequence)
            self.none_key = nn.Parameter(torch.randn(out) * 0.02)
        # relation head: an ordered entity pair (head span, tail span) chooses among
        # [no relation, queried relations]; pair = [s_h, s_t, s_h * s_t, text between]
        self.relations = relations
        if relations:
            self.between = nn.Linear(d, out)
            self.rel_bias = nn.Parameter(torch.zeros(()))
            if head_arch == "attn":
                self.pair_proj = nn.Linear(4 * out, attn_dim)
                self.rel_proj = nn.Linear(d, attn_dim)
                self.rel_kind = nn.Embedding(3, attn_dim)
                self.none_rel_tok = nn.Parameter(torch.randn(attn_dim) * 0.02)
                self.rel_mix = mixer(attn_dim, attn_heads, attn_layers, dropout)
                self.Ur = nn.Linear(attn_dim, bilinear_rank, bias=False)
                self.Vr = nn.Linear(attn_dim, bilinear_rank, bias=False)
            else:
                self.pair_mlp = mlp(4 * out, hidden, out, dropout)
                self.rel_mlp = mlp(d, hidden, out, dropout)
                self.none_rel = nn.Parameter(torch.randn(out) * 0.02)

    def mix_score(self, mix, kind, none_tok, U, V, keys, keys_pad, items, items_pad, bias):
        """Self-attention over [none | keys | items] (padded positions masked as keys), then a low-rank
        bilinear score of every (key, item) and (none, item): utilities [B, 1 + K, I], none first."""
        B, K, A = keys.shape
        none = (none_tok + kind.weight[0]).expand(B, 1, A)
        seq = torch.cat([none, keys + kind.weight[1], items + kind.weight[2]], 1)
        pad = torch.cat([keys_pad.new_zeros(B, 1), keys_pad, items_pad], 1)
        out = mix(seq, src_key_padding_mask=pad)
        n, k, it = out[:, :1], out[:, 1:1 + K], out[:, 1 + K:]
        ui = U(it)                                                      # [B,I,r]
        z = torch.einsum("bkr,bir->bki", V(k), ui) * self.scale + bias
        u_none = torch.einsum("bkr,bir->bki", V(n), ui) * self.scale
        return torch.cat([u_none, z], 1), it

    def encode(self, input_ids, attention_mask):
        if not self.bidirectional_decoder:
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        impl = self.encoder.config._attn_implementation
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else \
            self.encoder.get_input_embeddings().weight.dtype
        mask = padding_only_mask(attention_mask, impl, dtype)
        return self.encoder(input_ids=input_ids, attention_mask=mask,
                            use_cache=False).last_hidden_state

    def forward(self, input_ids, attention_mask, label_mask, cand_i, cand_j,
                rel_mask=None, pair_h=None, pair_t=None):
        """Entity utilities [B,1+M,C] (RLCD) or logits [B,M,C]; with relation inputs also
        relation utilities [B,1+R,P] over ordered candidate pairs."""
        h = self.contextualize(self.encode(input_ids, attention_mask), attention_mask)
        with torch.autocast(h.device.type, enabled=False):              # heads in fp32
            h = h.float()
            z, s = self.head(h, label_mask, cand_i, cand_j, return_spans=True)
            if rel_mask is None:
                return z
            return z, self.relation_head(h, s, rel_mask, cand_i, cand_j, pair_h, pair_t)

    def relation_head(self, h, s, rel_mask, cand_i, cand_j, pair_h, pair_t):
        B, T, D = h.shape
        O = s.shape[-1]
        gs = lambda idx: s.gather(1, idx[..., None].expand(-1, -1, O))
        sh, st = gs(pair_h), gs(pair_t)                                  # [B,P,O]
        # mean of the tokens strictly between the two spans (zeros when adjacent / nested)
        hi, hj, ti, tj = (cand_i.gather(1, pair_h), cand_j.gather(1, pair_h),
                          cand_i.gather(1, pair_t), cand_j.gather(1, pair_t))
        lo = torch.minimum(hj, tj) + 1
        up = torch.maximum(torch.maximum(hi, ti), lo)
        cs = torch.cat([h.new_zeros(B, 1, D), h.cumsum(1)], 1)
        g = lambda idx: cs.gather(1, idx[..., None].expand(-1, -1, D))
        between = (g(up) - g(lo)) / (up - lo).clamp_min(1)[..., None].to(h.dtype)
        feats = torch.cat([sh, st, sh * st, self.between(between)], -1)
        rm = rel_mask.to(h.dtype)
        rel_mean = torch.einsum("brt,btd->brd", rm, h) / rm.sum(-1, keepdim=True).clamp_min(1)
        if self.head_arch == "attn":
            # padded relation queries have no tokens; padded pairs are (0, 0)
            u, _ = self.mix_score(self.rel_mix, self.rel_kind, self.none_rel_tok, self.Ur, self.Vr,
                                  self.rel_proj(rel_mean), rm.sum(-1) == 0, self.pair_proj(feats),
                                  (pair_h == 0) & (pair_t == 0), self.rel_bias)
            return u
        pair = self.pair_mlp(feats)                                      # [B,P,O]
        rel = self.rel_mlp(rel_mean)
        zr = torch.einsum("bro,bpo->brp", rel, pair) * self.scale + self.rel_bias
        u_none = torch.einsum("o,bpo->bp", self.none_rel, pair)[:, None, :] * self.scale
        return torch.cat([u_none, zr], 1)

    def contextualize(self, h, attention_mask):
        """Optional extra transformer layers over the whole sequence (labels and text
        together) after the encoder, as in Laya's 2-layer head; identity when 0."""
        if self.post is None:
            return h
        return self.post(h, src_key_padding_mask=~attention_mask.bool())

    def head(self, h, label_mask, cand_i, cand_j, return_spans=False):
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
        if self.head_arch == "attn":
            # padded labels have no tokens; padded candidates start at token 0 (real ones after the labels)
            u, s = self.mix_score(self.mix, self.kind, self.none_tok, self.U, self.V, self.label_proj(lab),
                                  lm.sum(-1) == 0, self.span_proj(span), cand_i == 0, self.bias)
            z = u if self.head_type == "rlcd" else u[:, 1:]
            return (z, s) if return_spans else z
        s = self.span_mlp(span)                                         # [B,C,O]
        l = self.label_mlp(lab)                                         # [B,M,O]
        z = torch.einsum("bmo,bco->bmc", l, s) * self.scale + self.bias
        if self.head_type == "rlcd":
            # RLCD: utilities for [none, label_1..label_M] per span -> [B,1+M,C]
            u_none = torch.einsum("o,bco->bc", self.none_key, s)[:, None, :] * self.scale
            z = torch.cat([u_none, z], 1)
        return (z, s) if return_spans else z                            # BCE: [B,M,C] logits
