from __future__ import annotations

from typing import Any

from .base import format_candidate_item, format_user_history, load_template


def build_prompt(sample: dict[str, Any], permutation: list[int], cfg: Any) -> str:
    template = load_template(getattr(cfg, "prompt_template", None), "invarirank")
    span_start = getattr(cfg, "span_start_token", "[SPAN]")
    span_end = getattr(cfg, "span_end_token", "[/SPAN]")
    item_start = getattr(cfg, "item_start_token", "[ITEM]")
    item_end = getattr(cfg, "item_end_token", "[/ITEM]")

    instruction = template.get(
        "instruction",
        "Given the user's interaction history, rank the candidate items according to the user's preferences.",
    )
    history_header = template.get("history_header", "User history:")
    candidate_separator = template.get("candidate_separator", "")

    history_text = format_user_history(sample.get("history"), template)
    parts = [span_start, instruction, ""]
    if history_header:
        parts.append(history_header)
    if history_text:
        parts.append(history_text)
    parts.extend([span_end, ""])

    candidates = sample["candidates"]
    for idx in permutation:
        parts.append(item_start)
        parts.append(format_candidate_item(candidates[idx], template))
        parts.append(item_end)
        if candidate_separator:
            parts.append(candidate_separator)
        else:
            parts.append("")

    return "\n".join(parts)
