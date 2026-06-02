from __future__ import annotations

from typing import Any

from .base import candidate_id, extract_relevance_labels, format_candidate_item, format_user_history, load_template


def build_prompt(sample: dict[str, Any], permutation: list[int], cfg: Any) -> str:
    style = str(getattr(cfg, "prompt_style", "invarirank")).lower()
    if style == "invarirank":
        from .invarirank import build_prompt as build_invarirank_prompt

        return build_invarirank_prompt(sample, permutation, cfg)
    if style in {"zeroshot", "rankgpt"}:
        from .zeroshot import build_prompt as build_zeroshot_prompt

        return build_zeroshot_prompt(sample, permutation, cfg)
    raise ValueError(f"Unsupported prompt_style: {style}")


__all__ = [
    "build_prompt",
    "candidate_id",
    "extract_relevance_labels",
    "format_candidate_item",
    "format_user_history",
    "load_template",
]
