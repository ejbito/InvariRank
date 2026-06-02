from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def load_template(name_or_path: str | None, default_name: str) -> dict[str, Any]:
    name = name_or_path or default_name
    path = Path(name)
    if not path.suffix:
        path = TEMPLATES_DIR / f"{name}.json"
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        fallback = TEMPLATES_DIR / f"{name}.json"
        if fallback.exists():
            path = fallback
    with path.open("r", encoding="utf-8") as f:
        template = json.load(f)
    if not isinstance(template, dict):
        raise ValueError(f"Prompt template must be a JSON object: {path}")
    return template


def format_user_history(history: list[dict[str, Any]] | None, template: dict[str, Any] | None = None) -> str:
    template = template or {}
    item_format = template.get("history_item_format", "title: {title}{year_text}{rating_text}{genres_text}")
    lines: list[str] = []
    for item in history or []:
        fields = item_fields(item)
        lines.append(item_format.format(**fields))
    return "\n".join(lines)


def format_candidate_item(item: dict[str, Any], template: dict[str, Any] | None = None) -> str:
    template = template or {}
    item_format = template.get("candidate_item_format", "{title}{year_text}{genres_text}")
    return item_format.format(**item_fields(item))


def item_fields(item: dict[str, Any]) -> dict[str, Any]:
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


def extract_relevance_labels(sample: dict[str, Any], permutation: list[int]) -> list[int]:
    return [int(sample["candidates"][idx].get("relevance", 0)) for idx in permutation]


def candidate_id(item: dict[str, Any], fallback: int) -> str:
    for key in ("item_id", "id", "asin", "movie_id"):
        if key in item:
            return str(item[key])
    return str(fallback)
