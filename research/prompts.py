"""Versioned research prompts and deterministic generated-output parsing."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from invarirank.prompts import format_candidate_item, format_user_history

PROMPT_VERSIONS = {"rankgpt": "rankgpt-json-v1"}

RANKGPT_TEMPLATE: dict[str, Any] = {
    "name": "rankgpt",
    "instruction": "You are a ranking assistant. Rank the candidate items for the user based on the user history.",
    "history_header": "User history:",
    "candidates_header": "Candidate items:",
    "history_item_format": "{title}{year_text}{rating_text}{genres_text}",
    "candidate_item_format": "{title}{year_text}{genres_text}",
}


@dataclass(frozen=True)
class ParsedRanking:
    """A validated generated selection expanded to a complete candidate order."""

    labels: tuple[str, ...]
    order: tuple[int, ...]
    unknown_labels: tuple[str, ...]
    duplicate_labels: tuple[str, ...]
    unreturned_labels: tuple[str, ...]
    status: str
    repaired: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "parsed_labels": list(self.labels),
            "unknown_labels": list(self.unknown_labels),
            "duplicate_labels": list(self.duplicate_labels),
            "missing_labels": list(self.unreturned_labels),
            "parse_status": self.status,
            "repaired": self.repaired,
        }


def candidate_label(candidate_index: int) -> str:
    """Return an unambiguous local output label for one candidate index."""

    return f"C{int(candidate_index)}"


def build_research_prompt(
    sample: Mapping[str, Any],
    permutation: Sequence[int],
    *,
    output_count: int | None = None,
) -> str:
    if output_count is not None and output_count < 1:
        raise ValueError("output_count must be positive or omitted.")
    values = RANKGPT_TEMPLATE
    labels = [candidate_label(index) for index in permutation]
    parts = [values["instruction"], "", values["history_header"]]
    if history_text := format_user_history(sample.get("history"), values):
        parts.append(history_text)
    parts.extend(["", values["candidates_header"]])
    candidates = sample["candidates"]
    for index, label in zip(permutation, labels):
        parts.append(f"[{label}] {format_candidate_item(candidates[index], values)}")

    count = len(labels) if output_count is None else min(output_count, len(labels))
    if count < len(labels):
        noun = "item" if count == 1 else "items"
        example = [f"C<number-{index + 1}>" for index in range(count)]
        parts.extend(
            [
                "",
                f"Return only the top {count} {noun}, ordered from most to least relevant.",
                f'Return only JSON in this format: {{"rank_order": {json.dumps(example)}}}.',
            ]
        )
    else:
        parts.extend(
            [
                "",
                "Rank all candidate items from most to least relevant.",
                f'Return only JSON in this format: {{"rank_order": {json.dumps(labels)}}}.',
            ]
        )
    return "\n".join(parts)


def parse_generated_ranking(
    text: str,
    permutation: Sequence[int],
    *,
    expected_count: int | None = None,
    incomplete_output: str = "append_input_order",
    allow_fenced_json: bool = True,
) -> ParsedRanking:
    """Parse generated candidate labels and return a complete candidate order."""

    if incomplete_output not in {"append_input_order", "error"}:
        raise ValueError(f"Unsupported incomplete-output policy: {incomplete_output}")
    expected_labels = [candidate_label(index) for index in permutation]
    label_to_index = dict(zip(expected_labels, permutation))
    required_count = (
        int(expected_count)
        if expected_count is not None
        else len(expected_labels)
    )
    if not 1 <= required_count <= len(expected_labels):
        raise ValueError("expected_count must be between one and the number of candidates.")
    try:
        payload = _parse_json_object(text, allow_fenced_json=allow_fenced_json)
    except ValueError:
        if incomplete_output == "error":
            raise
        return ParsedRanking(
            labels=(),
            order=tuple(permutation),
            unknown_labels=(),
            duplicate_labels=(),
            unreturned_labels=tuple(expected_labels),
            status="failed",
            repaired=True,
        )
    raw_labels = payload.get("rank_order")
    if not isinstance(raw_labels, list):
        if incomplete_output == "error":
            raise ValueError("Generated output must contain a JSON list under 'rank_order'.")
        return ParsedRanking(
            labels=(),
            order=tuple(permutation),
            unknown_labels=(),
            duplicate_labels=(),
            unreturned_labels=tuple(expected_labels),
            status="failed",
            repaired=True,
        )

    valid: list[str] = []
    unknown: list[str] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for raw_label in raw_labels:
        label = str(raw_label).strip()
        if label not in label_to_index:
            unknown.append(label)
        elif label in seen:
            duplicates.append(label)
        elif len(valid) < required_count:
            valid.append(label)
            seen.add(label)
        else:
            unknown.append(label)

    completion_labels = [label for label in expected_labels if label not in seen]
    invalid = bool(unknown or duplicates or len(valid) != required_count)
    if invalid and incomplete_output == "error":
        raise ValueError(
            "Generated ranking is invalid: "
            f"expected {required_count} valid labels, received {len(valid)}; "
            f"unknown={unknown}, duplicates={duplicates}."
        )
    completed = valid + completion_labels
    unreturned = completion_labels if len(valid) != required_count else []
    repaired = invalid
    return ParsedRanking(
        labels=tuple(valid),
        order=tuple(label_to_index[label] for label in completed),
        unknown_labels=tuple(unknown),
        duplicate_labels=tuple(duplicates),
        unreturned_labels=tuple(unreturned),
        status="repaired" if repaired else "valid",
        repaired=repaired,
    )


def _parse_json_object(text: str, *, allow_fenced_json: bool) -> Mapping[str, Any]:
    candidates = [text.strip()]
    if allow_fenced_json:
        fenced = re.finditer(r"```(?:json)?\s*(.*?)```", text, re.I | re.S)
        candidates.extend(match.group(1).strip() for match in fenced)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError("Generated output does not contain a valid JSON object.")


__all__ = [
    "PROMPT_VERSIONS",
    "RANKGPT_TEMPLATE",
    "ParsedRanking",
    "build_research_prompt",
    "candidate_label",
    "parse_generated_ranking",
]
