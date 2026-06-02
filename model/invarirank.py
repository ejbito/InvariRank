from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class SpanInfo:
    span_start: int
    span_end: int
    candidate_spans: list[tuple[int, int]]


class SpanExtractor:
    def __init__(self, tokenizer: Any, cfg: Any):
        self.span_start_id = tokenizer.convert_tokens_to_ids(cfg.span_start_token)
        self.span_end_id = tokenizer.convert_tokens_to_ids(cfg.span_end_token)
        self.item_start_id = tokenizer.convert_tokens_to_ids(cfg.item_start_token)
        self.item_end_id = tokenizer.convert_tokens_to_ids(cfg.item_end_token)
        self.unk_id = getattr(tokenizer, "unk_token_id", None)

    def __call__(self, input_ids: Any) -> SpanInfo:
        ids = input_ids[0].tolist() if getattr(input_ids, "ndim", 1) == 2 else input_ids.tolist()

        required = [self.span_start_id, self.span_end_id, self.item_start_id, self.item_end_id]
        if any(x is None or x < 0 or x == self.unk_id for x in required):
            raise ValueError("Special token IDs missing from tokenizer. Add special tokens before tokenizing.")

        try:
            span_start = ids.index(self.span_start_id)
            span_end = ids.index(self.span_end_id) + 1
        except ValueError as exc:
            raise ValueError("Shared [SPAN] markers were not found in input_ids.") from exc

        candidate_spans: list[tuple[int, int]] = []
        cursor = span_end
        while True:
            try:
                item_start = ids.index(self.item_start_id, cursor)
                item_end = ids.index(self.item_end_id, item_start + 1)
            except ValueError:
                break
            candidate_spans.append((item_start, item_end + 1))
            cursor = item_end + 1

        if not candidate_spans:
            raise ValueError("No [ITEM] candidate spans found.")

        return SpanInfo(span_start=span_start, span_end=span_end, candidate_spans=candidate_spans)


class AttentionMaskMode(str, Enum):
    CAUSAL = "causal"
    BLOCK = "block"


class PositionIdMode(str, Enum):
    STANDARD = "standard"
    SHARED = "shared"


def parse_attention_mask_mode(value: str | AttentionMaskMode) -> AttentionMaskMode:
    try:
        return AttentionMaskMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported attention mask mode: {value}") from exc


def parse_position_id_mode(value: str | PositionIdMode) -> PositionIdMode:
    try:
        return PositionIdMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported position ID mode: {value}") from exc


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


def make_shared_position_ids(input_ids: Any, span_info: SpanInfo) -> Any:
    import torch

    _, seq = input_ids.shape
    device = input_ids.device
    pos = torch.zeros(seq, dtype=torch.long, device=device)

    s0, s1 = span_info.span_start, span_info.span_end
    span_len = s1 - s0
    pos[s0:s1] = torch.arange(span_len, device=device)

    base = span_len
    for c0, c1 in span_info.candidate_spans:
        cand_len = c1 - c0
        pos[c0:c1] = torch.arange(base, base + cand_len, device=device)

    return pos.unsqueeze(0)


def build_position_ids(input_ids: Any, span_info: SpanInfo, cfg: Any) -> Any | None:
    mode = parse_position_id_mode(getattr(cfg, "position_ids", "standard"))
    if mode is PositionIdMode.STANDARD:
        return None
    if mode is PositionIdMode.SHARED:
        return make_shared_position_ids(input_ids, span_info)
    raise AssertionError("unreachable")


def validate_candidate_count(span_info: SpanInfo, expected: int) -> None:
    observed = len(span_info.candidate_spans)
    if observed != expected:
        raise ValueError(f"Expected {expected} candidate spans, found {observed}.")
