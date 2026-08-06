from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiments.reranking.prompt_builder import candidate_label

JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedRanking:
    labels: tuple[str, ...]
    order: tuple[int, ...]
    unknown_labels: tuple[str, ...]
    duplicate_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    parse_status: str
    repaired: bool
    parse_error: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "parsed_labels": list(self.labels),
            "unknown_labels": list(self.unknown_labels),
            "duplicate_labels": list(self.duplicate_labels),
            "missing_labels": list(self.missing_labels),
            "parse_status": self.parse_status,
            "repaired": self.repaired,
            "parse_error": self.parse_error,
        }


def candidate_item_ids(user_record: Mapping[str, Any]) -> list[str]:
    candidates = user_record.get("candidates", {})
    return [str(record["item_id"]) for _, record in sorted(candidates.items(), key=lambda item: int(item[0]))]


def parse_rankgpt_output(
    text: str,
    permutation: Sequence[int],
    *,
    expected_count: int | None = None,
    incomplete_output: str = "append_input_order",
    allow_fenced_json: bool = True,
) -> ParsedRanking:
    if incomplete_output not in {"append_input_order", "error"}:
        raise ValueError(f"Unsupported incomplete-output policy: {incomplete_output}")

    expected_labels = [candidate_label(index) for index in permutation]
    label_to_index = dict(zip(expected_labels, permutation, strict=True))
    required_count = len(expected_labels) if expected_count is None else int(expected_count)
    if not 1 <= required_count <= len(expected_labels):
        raise ValueError("expected_count must be between one and the number of candidates.")

    try:
        payload = _parse_json_object(text, allow_fenced_json=allow_fenced_json)
        raw_labels = payload.get("rank_order")
        if not isinstance(raw_labels, list):
            raise ValueError("Generated output must contain a JSON list under 'rank_order'.")
    except ValueError as exc:
        if incomplete_output == "error":
            raise
        return ParsedRanking(
            labels=(),
            order=tuple(permutation),
            unknown_labels=(),
            duplicate_labels=(),
            missing_labels=tuple(expected_labels),
            parse_status="failed",
            repaired=True,
            parse_error=str(exc),
        )

    valid: list[str] = []
    unknown: list[str] = []
    duplicate: list[str] = []
    seen: set[str] = set()
    for raw_label in raw_labels:
        label = str(raw_label).strip()
        if label not in label_to_index:
            unknown.append(label)
        elif label in seen:
            duplicate.append(label)
        elif len(valid) < required_count:
            valid.append(label)
            seen.add(label)
        else:
            unknown.append(label)

    completion = [label for label in expected_labels if label not in seen]
    invalid = bool(unknown or duplicate or len(valid) != required_count)
    if invalid and incomplete_output == "error":
        raise ValueError(
            f"Generated ranking is invalid: expected {required_count} labels, got {len(valid)} valid labels."
        )
    completed = valid + completion
    return ParsedRanking(
        labels=tuple(valid),
        order=tuple(label_to_index[label] for label in completed),
        unknown_labels=tuple(unknown),
        duplicate_labels=tuple(duplicate),
        missing_labels=tuple(completion if len(valid) != required_count else ()),
        parse_status="repaired" if invalid else "valid",
        repaired=invalid,
    )


def validate_ranking(
    ranked_item_ids: Sequence[str],
    valid_candidate_ids: Sequence[str],
    parse_error: str | None = None,
) -> dict[str, Any]:
    valid_ids = [str(item_id) for item_id in valid_candidate_ids]
    valid_id_set = set(valid_ids)
    seen = set()
    reranked = []
    invalid = []
    duplicate = []
    for item_id in [str(item_id) for item_id in ranked_item_ids]:
        if item_id in seen:
            duplicate.append(item_id)
            continue
        seen.add(item_id)
        if item_id not in valid_id_set:
            invalid.append(item_id)
            continue
        reranked.append(item_id)
    missing = [item_id for item_id in valid_ids if item_id not in reranked]
    reranked.extend(missing)
    return {
        "reranked_item_ids": reranked,
        "invalid_item_ids": invalid,
        "duplicate_item_ids": duplicate,
        "missing_candidate_item_ids": missing,
        "parse_error": parse_error,
        "num_invalid_item_ids": len(invalid),
        "num_duplicate_item_ids": len(duplicate),
        "num_missing_candidate_item_ids": len(missing),
    }


def _parse_json_object(text: str, *, allow_fenced_json: bool) -> Mapping[str, Any]:
    candidates = [text.strip()]
    if allow_fenced_json:
        candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.I | re.S))
    match = JSON_OBJECT_PATTERN.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError("Generated output does not contain a valid JSON object.")
