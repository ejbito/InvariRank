from __future__ import annotations

from enum import Enum
from typing import Any

from .spans import SpanInfo


class AttentionMaskMode(str, Enum):
    CAUSAL = "causal"
    BLOCK = "block"


def parse_attention_mask_mode(value: str | AttentionMaskMode) -> AttentionMaskMode:
    try:
        return AttentionMaskMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported attention mask mode: {value}") from exc


def make_4d_causal_mask_from_2d(attn_2d: Any, dtype: Any) -> Any:
    import torch

    batch, seq = attn_2d.shape
    device = attn_2d.device
    tril = torch.tril(torch.ones((seq, seq), device=device, dtype=torch.bool))
    key_allowed = attn_2d.bool().unsqueeze(1).unsqueeze(1)
    allowed = tril.view(1, 1, seq, seq) & key_allowed
    neg = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, seq, seq), device=device, dtype=dtype)
    return mask.masked_fill(~allowed, neg)


def make_span_item_block_mask(attn_2d: Any, span_info: SpanInfo, dtype: Any, *, span_causal: bool = True) -> Any:
    import torch

    batch, seq = attn_2d.shape
    if batch != 1:
        raise ValueError("Block mask construction currently expects batch_size=1.")

    device = attn_2d.device
    allowed = torch.zeros((seq, seq), dtype=torch.bool, device=device)

    s0, s1 = span_info.span_start, span_info.span_end
    span_len = s1 - s0
    if span_causal:
        allowed[s0:s1, s0:s1] = torch.tril(torch.ones((span_len, span_len), device=device, dtype=torch.bool))
    else:
        allowed[s0:s1, s0:s1] = True

    for c0, c1 in span_info.candidate_spans:
        allowed[c0:c1, s0:s1] = True
        allowed[c0:c1, c0:c1] = True

    key_allowed = attn_2d[0].bool().unsqueeze(0)
    allowed &= key_allowed

    neg = torch.finfo(dtype).min
    mask = torch.zeros((1, 1, seq, seq), device=device, dtype=dtype)
    return mask.masked_fill(~allowed, neg)


def build_attention_mask(attn_2d: Any, span_info: SpanInfo, cfg: Any, dtype: Any) -> Any:
    mode = parse_attention_mask_mode(getattr(cfg, "attention_mask", "causal"))
    if mode is AttentionMaskMode.CAUSAL:
        return make_4d_causal_mask_from_2d(attn_2d, dtype)
    if mode is AttentionMaskMode.BLOCK:
        return make_span_item_block_mask(
            attn_2d,
            span_info,
            dtype,
            span_causal=bool(getattr(cfg, "span_causal", True)),
        )
    raise AssertionError("unreachable")


def allowed_from_mask(mask: Any) -> Any:
    return mask == 0
