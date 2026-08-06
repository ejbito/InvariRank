from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.utils.progress import progress


def load_item_metadata(
    processed_dir: str | Path,
    show_progress: bool = True,
) -> dict[int, dict[str, Any]]:
    path = Path(processed_dir) / "items.csv"
    if not path.exists():
        return {}

    items = pd.read_csv(path)
    metadata = {}
    records = items.to_dict(orient="records")
    for record in progress(
        records,
        desc="Loading item metadata",
        total=len(records),
        enabled=show_progress,
    ):
        item_id = int(record["item_id"])
        metadata[item_id] = _clean_record(record)
    return metadata


def format_llm_candidates(
    recommendations: Mapping[int, Sequence[int]],
    item_metadata: Mapping[int, Mapping[str, Any]] | None = None,
    ground_truth: Mapping[int, Sequence[int]] | None = None,
    retriever_name: str | None = None,
    split: str | None = None,
    require_ground_truth_in_candidates: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    item_metadata = item_metadata or {}
    ground_truth = ground_truth or {}
    users = {}
    num_users_before_filter = len(recommendations)
    num_users_with_ground_truth_in_candidates = 0

    user_ids = sorted(recommendations)
    for user_id in progress(
        user_ids,
        desc="Formatting LLM candidates",
        total=len(user_ids),
        enabled=show_progress,
    ):
        candidate_item_ids = [int(item_id) for item_id in recommendations[user_id]]
        ground_truth_item_ids = [int(item_id) for item_id in ground_truth.get(user_id, [])]
        has_ground_truth_candidate = bool(set(candidate_item_ids) & set(ground_truth_item_ids))
        num_users_with_ground_truth_in_candidates += int(has_ground_truth_candidate)
        if require_ground_truth_in_candidates and not has_ground_truth_candidate:
            continue

        candidates = {}
        for rank, internal_item_id in enumerate(candidate_item_ids, start=1):
            record = _llm_item_record(internal_item_id, item_metadata.get(internal_item_id, {}))
            candidates[str(rank)] = _clean_record(record)

        user_record: dict[str, Any] = {
            "user_id": int(user_id),
            "candidates": candidates,
        }
        if ground_truth_item_ids:
            user_record["ground_truth_item_ids"] = [
                _display_item_id(item_id, item_metadata) for item_id in ground_truth_item_ids
            ]
        users[str(user_id)] = user_record

    num_users_after_filter = len(users)
    return {
        "retriever": retriever_name,
        "split": split,
        "candidate_filter": {
            "require_ground_truth_in_candidates": require_ground_truth_in_candidates,
            "num_users_before_filter": num_users_before_filter,
            "num_users_after_filter": num_users_after_filter,
            "num_users_with_ground_truth_in_candidates": num_users_with_ground_truth_in_candidates,
            "ground_truth_coverage": (
                num_users_with_ground_truth_in_candidates / num_users_before_filter
                if num_users_before_filter
                else 0.0
            ),
        },
        "users": users,
    }


def _clean_record(record: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        if isinstance(value, list):
            cleaned[key] = value
        elif isinstance(value, str) and _looks_like_json_container(value):
            cleaned[key] = _parse_json_container(value)
        elif pd.isna(value):
            cleaned[key] = None
        elif hasattr(value, "item"):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def _llm_item_record(
    internal_item_id: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    record = dict(metadata)
    record.pop("raw_item_id", None)
    record["item_id"] = _display_item_id(internal_item_id, {internal_item_id: metadata})

    if "genres" in record and isinstance(record["genres"], str):
        record["genres"] = [genre for genre in record["genres"].split("|") if genre]

    ordered = {"item_id": record.pop("item_id")}
    ordered.update(record)
    return ordered


def make_llm_item_record(
    internal_item_id: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return _llm_item_record(internal_item_id, metadata)


def _display_item_id(
    internal_item_id: int,
    item_metadata: Mapping[int, Mapping[str, Any]],
) -> Any:
    raw_item_id = item_metadata.get(internal_item_id, {}).get("raw_item_id")
    return raw_item_id if raw_item_id is not None else internal_item_id


def _looks_like_json_container(value: str) -> bool:
    stripped = value.strip()
    return (stripped.startswith("[") and stripped.endswith("]")) or (
        stripped.startswith("{") and stripped.endswith("}")
    )


def _parse_json_container(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
