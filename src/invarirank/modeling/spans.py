from __future__ import annotations

from dataclasses import dataclass
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


def validate_candidate_count(span_info: SpanInfo, expected: int) -> None:
    observed = len(span_info.candidate_spans)
    if observed != expected:
        raise ValueError(f"Expected {expected} candidate spans, found {observed}.")

