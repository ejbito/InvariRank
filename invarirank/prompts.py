from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

INVARIRANK_TEMPLATE: dict[str, Any] = {
    "name": "invarirank",
    "instruction": (
        "Given the user's interaction history, rank the candidate items according to the user's preferences."
    ),
    "history_header": "User history:",
    "history_item_format": "title: {title}{year_text}{rating_text}{genres_text}",
    "candidate_item_format": "{title}{year_text}{genres_text}",
    "candidate_separator": "",
}


def load_template(name_or_path: str | None, default_name: str) -> dict[str, Any]:
    name = name_or_path or default_name
    if name == "invarirank":
        return dict(INVARIRANK_TEMPLATE)

    path = Path(name)
    if not path.exists() and not path.suffix:
        candidate = path.with_suffix(".json")
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {name}")
    with path.open("r", encoding="utf-8") as handle:
        template = json.load(handle)
    if not isinstance(template, dict):
        raise ValueError(f"Prompt template must be a JSON object: {path}")
    return template


def candidate_id(item: Mapping[str, Any], fallback: int) -> str:
    for key in ("item_id", "id", "asin", "movie_id"):
        if key in item:
            return str(item[key])
    return str(fallback)


def item_fields(item: Mapping[str, Any]) -> dict[str, Any]:
    title = item.get("title", item.get("name", ""))
    year = item.get("year", "")
    rating = item.get("rating", "")
    genres = item.get("genres", item.get("categories", []))
    genre_text = ", ".join(map(str, genres)) if genres else ""
    return {
        "item_id": candidate_id(item, 0),
        "title": str(title),
        "year": year,
        "rating": rating,
        "genres": genre_text,
        "year_text": f" ({year})" if year else "",
        "rating_text": f", rating: {rating}" if rating else "",
        "genres_text": f", genres: {genre_text}" if genre_text else "",
    }


def format_user_history(
    history: Sequence[Mapping[str, Any]] | None,
    template: Mapping[str, Any] | None = None,
) -> str:
    template = template or {}
    item_format = template.get("history_item_format", "title: {title}{year_text}{rating_text}{genres_text}")
    return "\n".join(item_format.format(**item_fields(item)) for item in history or [])


def format_candidate_item(item: Mapping[str, Any], template: Mapping[str, Any] | None = None) -> str:
    template = template or {}
    item_format = template.get("candidate_item_format", "{title}{year_text}{genres_text}")
    return item_format.format(**item_fields(item))


def extract_relevance_labels(sample: Mapping[str, Any], permutation: Sequence[int]) -> list[int]:
    return [int(sample["candidates"][index]["relevance"]) for index in permutation]


def build_prompt(sample: Mapping[str, Any], permutation: Sequence[int], cfg: Any) -> str:
    return build_invarirank_prompt(sample, permutation, cfg)


def build_invarirank_prompt(sample: Mapping[str, Any], permutation: Sequence[int], cfg: Any) -> str:
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
    for index in permutation:
        parts.extend(
            [
                item_start,
                format_candidate_item(candidates[index], template),
                item_end,
                candidate_separator,
            ]
        )
    return "\n".join(parts)


__all__ = [
    "INVARIRANK_TEMPLATE",
    "build_invarirank_prompt",
    "build_prompt",
    "candidate_id",
    "extract_relevance_labels",
    "format_candidate_item",
    "format_user_history",
    "item_fields",
    "load_template",
]
