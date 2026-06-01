from __future__ import annotations

from enum import Enum
from typing import Any

from .spans import SpanInfo


class PositionIdMode(str, Enum):
    STANDARD = "standard"
    SHARED = "shared"


def parse_position_id_mode(value: str | PositionIdMode) -> PositionIdMode:
    try:
        return PositionIdMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported position ID mode: {value}") from exc


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

