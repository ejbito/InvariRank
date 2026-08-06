from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from invarirank.prompts import format_candidate_item, format_user_history

PROMPT_OPTIONS = ("rankgpt", "marker")
RANKGPT_PROMPT_VERSION = "rankgpt-json-v1"


def validate_prompt(prompt: str) -> str:
    if prompt not in PROMPT_OPTIONS:
        raise ValueError(f"Unknown prompt '{prompt}'. Valid prompts: {list(PROMPT_OPTIONS)}")
    return prompt


def candidate_label(candidate_index: int) -> str:
    return f"C{int(candidate_index)}"


def build_rankgpt_prompt(
    sample: Mapping[str, Any],
    permutation: Sequence[int],
    *,
    output_count: int | None = None,
) -> str:
    labels = [candidate_label(index) for index in permutation]
    count = len(labels) if output_count is None else min(int(output_count), len(labels))
    if count < 1:
        raise ValueError("output_count must be positive when provided.")

    parts = [
        "You are a ranking assistant. Rank the candidate items for the user based on the user history.",
        "",
        "User history:",
    ]
    history_text = format_user_history(sample.get("history"), _rankgpt_template())
    parts.append(history_text if history_text else "No user history is available.")
    parts.extend(["", "Candidate items:"])

    candidates = sample["candidates"]
    for index, label in zip(permutation, labels, strict=True):
        parts.append(f"[{label}] {format_candidate_item(candidates[index], _rankgpt_template())}")

    if count == len(labels):
        example = labels
        parts.extend(
            [
                "",
                "Rank all candidate items from most to least relevant.",
                f'Return only JSON in this format: {{"rank_order": {json.dumps(example)}}}.',
            ]
        )
    else:
        example = [f"C<number-{index + 1}>" for index in range(count)]
        noun = "item" if count == 1 else "items"
        parts.extend(
            [
                "",
                f"Return only the top {count} {noun}, ordered from most to least relevant.",
                f'Return only JSON in this format: {{"rank_order": {json.dumps(example)}}}.',
            ]
        )
    return "\n".join(parts)


def _rankgpt_template() -> dict[str, Any]:
    return {
        "history_item_format": "{title}{year_text}{rating_text}{genres_text}",
        "candidate_item_format": "{title}{year_text}{genres_text}",
    }
